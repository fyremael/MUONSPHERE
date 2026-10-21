#!/usr/bin/env python3
"""GCL natural sink emergence from random initialization; AdamW only.

Baseline vs true Schatten-4^4 attention deviation penalty vs
Schatten-4^4 + excess column penalty. GPT2-124M architecture, no
pretrained weights, no injected logit bias, FineWeb-Edu GPT2 BPE.
"""
import argparse, hashlib, json, math, os, random, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS=("lm","tnlm","tnlm_s4","tnlm_s4_col","fsm_tnlm","fsm_tnlm_s4","fsm_tnlm_s4_col")
EOS=50256

def tnlm_targets(x, objective):
    """GPT2 internally shifts: logits[:,0] predicts labels[:,1]."""
    if objective == "lm":
        return x
    if objective not in ("tnlm", "fsm_tnlm"):
        raise ValueError(objective)
    labels=x.clone()
    labels[:,1]=-100
    return labels

def install_force_sink_mask(model,k):
    """Fixed nonlearned P0 channel partition at GPT2 pre-LN inputs.

    P0 retains channels [0,k); all positions >=1 retain [k,d).
    Hook applies before every attention and MLP pre-normalization.
    This structural intervention is NOT a direct attention-logit bias.
    """
    assert 0 < k < model.config.n_embd
    def hook(module,args):
        z=args[0]
        assert z.ndim==3
        left=torch.cat((z[:,:1,:k],torch.zeros_like(z[:,:1,k:])),dim=-1)
        right=torch.cat((torch.zeros_like(z[:,1:,:k]),z[:,1:,k:]),dim=-1)
        return (torch.cat((left,right),dim=1),)
    hooks=[]
    for block in model.transformer.h:
        hooks.extend([block.ln_1.register_forward_pre_hook(hook),
                      block.ln_2.register_forward_pre_hook(hook)])
    return hooks

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class Corpus:
    def __init__(self,root,seq):
        root=Path(root)
        self.manifest=json.loads((root/"corpus_manifest.json").read_text())
        self.train=np.memmap(root/"train_tokens.uint16",dtype=np.uint16,mode="r",
            shape=(self.manifest["train_tokens"],))
        self.valid=np.memmap(root/"valid_tokens.uint16",dtype=np.uint16,mode="r",
            shape=(self.manifest["valid_tokens"],))
        self.seq=seq
    def batch(self,bs,rng,split="train"):
        data=self.train if split=="train" else self.valid
        starts=rng.integers(0,len(data)-self.seq,size=bs)
        x=np.empty((bs,self.seq),dtype=np.int64)
        x[:,0]=EOS
        for i,j in enumerate(starts):
            x[i,1:]=data[j:j+self.seq-1]
        return torch.tensor(x,device="cuda",dtype=torch.long)

def attention_losses(attentions,min_q=64,max_q=128):
    """Computes exact S4^4 of an attention DEVIATION submatrix.

    E is attention deviation from causal uniform, rows selected by fixed
    equally-spaced indices among later queries, every key retained.
    tr((E E^T)^2)/Q^2 = ||E||_S4^4/Q^2, not a surrogate.
    Plus separate targeted excess-column concentration, normalized
    to avoid scaling with context length.
    """
    s4=col=0.
    for a in attentions:
        a=a.float()
        length=a.shape[-1]
        begin=min(min_q,length//4)
        idx=torch.linspace(begin,length-1,
            steps=min(max_q,length-begin),device=a.device).long().unique()
        subset=a.index_select(-2,idx)
        n=len(idx)
        keyidx=torch.arange(length,device=a.device)
        fair=(keyidx[None,:]<=idx[:,None]).float()
        fair=fair/(idx+1).float()[:,None]
        deviation=subset-fair[None,None,:,:]
        gram=deviation@deviation.transpose(-2,-1)
        # exact Schatten-4 norm fourth power of the 128-by-T matrix
        # normalized for number of sampled queries
        s4=s4+gram.square().sum(dim=(-1,-2)).mean()/(n*n)
        positive=deviation.clamp_min(0.)
        key_mass=positive.square().mean(dim=-2)
        col=col+key_mass.square().sum(dim=-1).mean()
    return s4/len(attentions),col/len(attentions)

@torch.no_grad()
def diagnostics(model,val,head_threshold=0.3,attn_batches=2):
    model.eval()
    ce=[];bos=[];excess=[];maxcol=[];headh=[];entropy=[]
    keyhead=[];earlyhead=[];peak_at_zero=[];early_mass=[]
    layer_bos=[];layer_sink=[];layer_maxcol=[]
    n=0
    for x in val:
        with torch.autocast("cuda",dtype=torch.bfloat16):
            y=model(x,labels=x,output_attentions=(n<attn_batches),
                    return_dict=True,use_cache=False)
        ce.append(y.loss.float().item())
        if n<attn_batches:
            this_bos=[];this_sink=[];this_max=[]
            for a in y.attentions:
                a=a.float()
                length=a.size(-1)
                q=torch.arange(length//4,length,device=a.device)
                late=a[:,:,length//4:,:]
                real=late.sum(-1).clamp_min(1e-8)
                normalized=late/real[...,None]
                bos_probs=normalized[...,0]
                bos.append(bos_probs.mean().item())
                excess.append((bos_probs-1./(q+1)[None,None,:]).mean().item())
                columns=normalized.mean(-2)
                peaks, peak_indexes=columns.max(-1)
                maxcol.append(peaks.mean().item())
                keyhead.append((peaks>0.3).float().mean().item())
                earlyhead.append(((peaks>0.3)&(peak_indexes<=4)).float().mean().item())
                peak_at_zero.append((peak_indexes==0).float().mean().item())
                early_mass.append(normalized[...,:5].sum(-1).mean().item())
                headh.append((bos_probs.mean(-1)>head_threshold).float().mean().item())
                this_bos.append(bos_probs.mean().item())
                this_sink.append((bos_probs.mean(-1)>head_threshold).float().mean().item())
                this_max.append(peaks.mean().item())
                entropy.append((-(normalized*normalized.clamp_min(1e-9).log())
                                 .sum(-1)).mean().item())
            layer_bos.append(this_bos)
            layer_sink.append(this_sink)
            layer_maxcol.append(this_max)
        n+=1
    layer_avg=lambda lists: [sum(arr[i] for arr in lists)/len(lists)
          for i in range(len(lists[0]))]
    layer_sink_mean=layer_avg(layer_sink)
    return dict(bos_by_layer=layer_avg(layer_bos),
        sink_fraction_by_layer=layer_sink_mean,
        max_column_by_layer=layer_avg(layer_maxcol),
        strongest_layer_sink_fraction=max(layer_sink_mean),
        strongly_sinking_layers=sum(v>=0.25 for v in layer_sink_mean),
        valid_ce=sum(ce)/len(ce),valid_ppl=math.exp(min(20,sum(ce)/len(ce))),
        bos_share=sum(bos)/len(bos),
        bos_excess=sum(excess)/len(excess),
        max_column_share=sum(maxcol)/len(maxcol),
        sink_head_fraction=sum(headh)/len(headh),
        any_key_sink_head_fraction=sum(keyhead)/len(keyhead),
        early_key_sink_head_fraction=sum(earlyhead)/len(earlyhead),
        peak_key_is_bos_fraction=sum(peak_at_zero)/len(peak_at_zero),
        first_five_key_mass=sum(early_mass)/len(early_mass),
        attention_entropy=sum(entropy)/len(entropy))

def make_model():
    from transformers import GPT2Config,GPT2LMHeadModel
    cfg=GPT2Config(vocab_size=50257,n_positions=1024,n_ctx=1024,
        n_embd=768,n_layer=12,n_head=12,
        resid_pdrop=0.,attn_pdrop=0.,embd_pdrop=0.,
        bos_token_id=EOS,eos_token_id=EOS,
        use_cache=False)
    model=GPT2LMHeadModel(cfg)
    model.config._attn_implementation="eager"
    return model.to("cuda")

def learning_rate(step,steps,warmup,peak):
    if step<=warmup:return peak*step/max(1,warmup)
    progress=(step-warmup)/max(1,steps-warmup)
    return peak*(0.1+0.9*0.5*(1+math.cos(math.pi*min(1.,progress))))

def logjson(prefix,record):
    print(prefix,json.dumps(record,sort_keys=True),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--data",default="/content/gcl_s4_scratch_corpus")
    ap.add_argument("--out",default="/content/gcl_s4_scratch_results")
    ap.add_argument("--stage",default="pilot")
    ap.add_argument("--steps",type=int,default=800)
    ap.add_argument("--batch",type=int,default=2)
    ap.add_argument("--accum",type=int,default=4)
    ap.add_argument("--context",type=int,default=512)
    ap.add_argument("--peak-lr",type=float,default=3e-4)
    ap.add_argument("--warmup",type=int,default=60)
    ap.add_argument("--lambda-s4",type=float,default=0.15)
    ap.add_argument("--lambda-column",type=float,default=0.15)
    ap.add_argument("--force-sink-k",type=int,default=16)
    ap.add_argument("--eval-batches",type=int,default=4)
    ap.add_argument("--eval-every",type=int,default=100)
    ap.add_argument("--seeds",nargs="+",type=int,default=[101,202,303])
    ap.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    ap.add_argument("--wandb",choices=["offline","disabled","online"],default="offline")
    ap.add_argument("--save-weights",action="store_true")
    ap.add_argument("--long-eval",action="store_true")
    opts,_=ap.parse_known_args()
    assert opts.context in (256,512,1024)
    assert 0<=opts.lambda_s4 and 0<=opts.lambda_column
    assert opts.steps>0 and opts.batch>0 and opts.accum>0
    assert len(set(opts.seeds))==len(opts.seeds)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    assert torch.cuda.is_available(), "An NVIDIA GPU is required"
    out=Path(opts.out);out.mkdir(parents=True,exist_ok=True)
    data=Corpus(opts.data,opts.context)
    config=vars(opts).copy()
    config.update(torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(0),start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        model="GPT2Config-random-init-12L-12H-768D-124M",
        pretraining_checkpoint=None,
        tokenizer="GPT2 BPE 50257",corpus=data.manifest)
    (out/"config.json").write_text(json.dumps(config,indent=2))
    logjson("RUN_CONFIG",config)
    w=None
    if opts.wandb!="disabled":
        try:
            import wandb
            wandb.init(project="gcl-attention-sinks",name="gpt2-scratch-s4-a100",
                group="fineweb-from-scratch-s4",mode=opts.wandb,
                config=vars(opts),dir=str(out),reinit=True)
            w=wandb
        except Exception as exc: logjson("WANDB_ERROR",{"error":str(exc)})
    all_results=[]
    count=0
    with (out/"metrics.jsonl").open("w",buffering=1) as stream:
      for seed in opts.seeds:
        valid_rng=np.random.default_rng(500001)
        valid=[data.batch(min(2,opts.batch),valid_rng,"valid")
               for _ in range(opts.eval_batches)]
        for arm in opts.arms:
            seed_all(seed)
            model=make_model()
            hook_handles=install_force_sink_mask(model,opts.force_sink_k) if arm.startswith("fsm_tnlm") else []
            n_params=sum(p.numel() for p in model.parameters())
            initial_model_sha=hashlib.sha256(torch.cat([
                p.detach().flatten()[:min(16,p.numel())].float().cpu()
                for p in model.parameters()]).numpy().tobytes()).hexdigest()
            optimizer=torch.optim.AdamW(
                model.parameters(),lr=opts.peak_lr,betas=(0.9,0.95),
                weight_decay=0.1,fused=True)
            training_rng=np.random.default_rng(seed+12017)
            start_metrics=diagnostics(model,valid)
            logjson("INITIAL",dict(seed=seed,arm=arm,parameters=n_params,
                sampled_init_sha256=initial_model_sha,**start_metrics))
            with (out/"initial.jsonl").open("a",buffering=1) as initlog:
                initlog.write(json.dumps(dict(seed=seed,arm=arm,
                    model_sha256_sample=initial_model_sha,**start_metrics))+"\n")
            started=time.monotonic()
            last=None
            for step in range(1,opts.steps+1):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                lr=learning_rate(step,opts.steps,opts.warmup,opts.peak_lr)
                for group in optimizer.param_groups:group["lr"]=lr
                tr_ce=tr_s4=tr_col=0.
                for micro in range(opts.accum):
                    x=data.batch(opts.batch,training_rng)
                    objective="lm" if arm=="lm" else "tnlm"
                    labels=tnlm_targets(x,objective)
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        output=model(x,labels=labels,output_attentions=("s4" in arm),
                            use_cache=False,return_dict=True)
                        ce=output.loss
                        if "s4" not in arm:
                            s4=ce.new_zeros(())
                            column=ce.new_zeros(())
                        else:
                            s4,column=attention_losses(output.attentions)
                        loss=ce+opts.lambda_s4*s4*("s4" in arm)
                        if arm.endswith("s4_col"):
                            loss=loss+opts.lambda_column*column
                    if not torch.isfinite(loss).item():
                        raise RuntimeError(f"nonfinite step {step} arm {arm} seed {seed}")
                    (loss/opts.accum).backward()
                    tr_ce+=ce.detach().float().item()/opts.accum
                    tr_s4+=s4.detach().float().item()/opts.accum
                    tr_col+=column.detach().float().item()/opts.accum
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                count+=1
                if step==1 or step%opts.eval_every==0 or step==opts.steps:
                    metrics=diagnostics(model,valid)
                    if opts.long_eval and step==opts.steps:
                        long_data=Corpus(opts.data,1024)
                        long_valid_rng=np.random.default_rng(500001)
                        long_valid=[long_data.batch(1,long_valid_rng,"valid")
                            for _ in range(2)]
                        metrics.update({"long_"+k:v for k,v in
                            diagnostics(model,long_valid).items()})
                    last=dict(seed=seed,arm=arm,step=step,lr=lr,
                        stage=opts.stage,train_ce=tr_ce,train_s4=tr_s4,
                        train_column=tr_col,params=n_params,
                        tokens=step*opts.batch*opts.accum*opts.context,
                        elapsed_seconds=round(time.monotonic()-started,2),
                        **metrics)
                    stream.write(json.dumps(last,sort_keys=True)+"\n")
                    logjson("METRIC",last)
                    if w:w.log({f"{arm}/seed{seed}/{k}":v
                          for k,v in last.items() if isinstance(v,(int,float))},step=count)
            if opts.save_weights:
                torch.save({"model":model.cpu().state_dict(),
                            "seed":seed,"arm":arm,
                            "global_step":opts.steps},
                         out/f"weights_{seed}_{arm}.pt")
            all_results.append(dict(seed=seed,arm=arm,step=opts.steps,
                initial_sha256=initial_model_sha,
                completed=True,last=last))
            (out/"summary.json").write_text(json.dumps(all_results,indent=2))
            for h in hook_handles: h.remove()
            del model,optimizer
            torch.cuda.empty_cache()
    if w:w.finish()
    expected=len(opts.seeds)*len(opts.arms)
    status={"complete":len(all_results)==expected,"count":len(all_results),
            "expected":expected,
            "baseline_natural_sink_emerged":all(
             next(r for r in all_results if r["seed"]==seed and r["arm"]=="lm")
                ["last"]["bos_excess"]>0.05 for seed in opts.seeds
             ) if "lm" in opts.arms else None}
    (out/"run_status.json").write_text(json.dumps(status,indent=2))
    logjson("RUN_END",status)

if __name__=="__main__":main()
