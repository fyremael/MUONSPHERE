#!/usr/bin/env python3
"""Natural-language sink continuation experiment; authentic pretrained GPT2 124M.

FineWeb-Edu sample-10BT, document-disjoint split, GPT2 BPE, exact shared base
checkpoint, no injected biases. Records conditional key concentration and
held-out LM loss under paired seeds. No artificial stressor.
"""
import argparse,csv,hashlib,json,math,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ARMS=("adamw","muon_frob","muon_s4","muon_s4_conc","adamw_conc")
VOCAB=50257
BASE="openai-community/gpt2"

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def polar_ns(g,kind,steps=5):
    x=g.float()
    swap=x.shape[0]>x.shape[1]
    if swap:x=x.T
    if kind=="s4":
        divisor=torch.linalg.vector_norm(x@x.T).sqrt()
    else:divisor=torch.linalg.vector_norm(x)
    x=x/divisor.clamp_min(1e-8)
    for _ in range(steps):
        a=x@x.T
        x=3.4445*x + (-4.775*a + 2.0315*(a@a))@x
    if swap:x=x.T
    return x.to(g.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self,params,lr=0.001,kind="s4",momentum=0.95):
        super().__init__(params,dict(lr=lr,kind=kind,momentum=momentum))
    @torch.no_grad()
    def step(self,closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:continue
                state=self.state[p]
                if "m" not in state:state["m"]=torch.zeros_like(p)
                m=state["m"]
                m.mul_(group["momentum"]).add_(p.grad)
                z=polar_ns(p.grad+group["momentum"]*m,group["kind"])
                scale=math.sqrt(max(1.,p.shape[0]/p.shape[1]))
                p.mul_(1-group["lr"]*0.01)
                p.add_(z,alpha=-group["lr"]*scale)

class TokenData:
    def __init__(self,root,seq=512):
        self.root=Path(root); self.seq=seq
        meta=json.loads((self.root/"corpus_manifest.json").read_text())
        self.train=np.memmap(self.root/"train_tokens.uint16",dtype=np.uint16,mode="r",
                             shape=(meta["train_tokens"],))
        self.valid=np.memmap(self.root/"valid_tokens.uint16",dtype=np.uint16,mode="r",
                             shape=(meta["valid_tokens"],))
        self.meta=meta
    def batch(self,bs,rng,train=True):
        a=self.train if train else self.valid
        starts=rng.integers(0,len(a)-self.seq,size=bs)
        out=np.empty((bs,self.seq),dtype=np.int32)
        out[:,0]=50256
        for i,j in enumerate(starts):out[i,1:]=a[j:j+self.seq-1]
        return torch.from_numpy(out.astype(np.int64)).pin_memory().to("cuda",non_blocking=True)
    def validation(self,num,bs=2):
        rg=np.random.default_rng(71289)
        return [self.batch(bs,rg,False) for _ in range(num)]

def regularizer(att,seq_start=128):
    """S4-inspired fourth-order *column* regularizer, causal-aware.

    NOT a Schatten-4 matrix norm. Always condition on real-token allocation.
    """
    loss=0.
    for layer in att:
        a=layer.float()
        t=a.shape[-1]
        start=min(seq_start,t//2)
        q=torch.arange(start,t,device=a.device)
        k=torch.arange(t,device=a.device)
        fair=(k[None,:]<=q[:,None]).float()[None,None,:,:]/(q+1)[None,None,:,None]
        late=a[:,:,start:,:]
        fair=fair*late.sum(-1,keepdim=True)
        excess=(late-fair).relu()
        col=excess.square().mean(-2)
        loss=loss+col.square().sum(-1).mean()*t
    return loss/len(att)

@torch.no_grad()
def evaluate(model,batches,sample_for_attention=2):
    model.eval()
    sums={}
    for i,tokens in enumerate(batches):
        with torch.autocast("cuda",dtype=torch.bfloat16):
            out=model(tokens,labels=tokens,
                      output_attentions=(i<sample_for_attention),
                      return_dict=True,use_cache=False)
        rec={"valid_ce":out.loss.float().item()}
        if i<sample_for_attention:
            att=out.attentions
            if not att or att[0] is None:
                raise RuntimeError("Attention diagnostics missing")
            key_masses=[]; cols=[]; firsts=[]; excesses=[]; heads=[]; entropy=[]
            for a in att:
                a=a.float()
                start=a.size(-1)//4
                q=torch.arange(start,a.size(-2),device=a.device)
                late=a[:,:,start:,:]
                real=late.sum(-1).clamp_min(1e-8)
                normal=late/real[...,None]
                bos=normal[...,0]
                firsts.append(bos.mean())
                excesses.append((bos-1./(q+1)[None,None,:]).mean())
                cols.append(normal.mean(-2).max(-1).values.mean())
                heads.append((bos.mean(-1)>0.30).float().mean())
                key_masses.append((bos.mean(-1)>0.5).float().mean())
                entropy.append((-(normal*normal.clamp_min(1e-10).log()).sum(-1)).mean())
            rec.update(bos_real_share=torch.stack(firsts).mean().item(),
                       bos_excess=torch.stack(excesses).mean().item(),
                       max_column_real_share=torch.stack(cols).mean().item(),
                       head_sink_rate_30=torch.stack(heads).mean().item(),
                       head_sink_rate_50=torch.stack(key_masses).mean().item(),
                       attention_entropy=torch.stack(entropy).mean().item())
        for key,value in rec.items():
            sums.setdefault(key,[]).append(value)
    return {key:sum(values)/len(values) for key,values in sums.items()}

def group_optimizers(model,arm,adam_lr,muon_lr):
    if arm.startswith("adamw"):
        return [torch.optim.AdamW(model.parameters(),lr=adam_lr,weight_decay=0.01)]
    matrices,others=[],[]
    for name,p in model.named_parameters():
        is_matrix=p.ndim==2 and ".h." in name and (
            ".attn.c_attn.weight" in name or
            ".attn.c_proj.weight" in name or
            ".mlp.c_fc.weight" in name or
            ".mlp.c_proj.weight" in name)
        (matrices if is_matrix else others).append(p)
    assert matrices and others
    mode="frob" if arm=="muon_frob" else "s4"
    return [Muon(matrices,lr=muon_lr,kind=mode),
            torch.optim.AdamW(others,lr=adam_lr,weight_decay=0.01)]

def init_wandb(out,opts):
    if opts.wandb_mode=="disabled":return None
    try:
        import wandb
        wandb.init(project="gcl-attention-sinks",name="natural-gpt2-124m-a100",
                   group="fineweb-edu-no-logit-bias",mode=opts.wandb_mode,
                   dir=str(out),config=vars(opts),reinit=True)
        return wandb
    except Exception as e:
        print("WANDB_FAILED",repr(e),flush=True)
        return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data",default="/content/gcl_natural_sink")
    p.add_argument("--out",default="/content/gcl_natural_sink_result")
    p.add_argument("--context",type=int,default=512)
    p.add_argument("--microbatch",type=int,default=2)
    p.add_argument("--accum",type=int,default=4)
    p.add_argument("--steps",type=int,default=200)
    p.add_argument("--eval-every",type=int,default=25)
    p.add_argument("--eval-batches",type=int,default=6)
    p.add_argument("--seeds",nargs="+",type=int,default=[101,202,303])
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    p.add_argument("--lambda-conc",type=float,default=0.03)
    p.add_argument("--adam-lr",type=float,default=3e-5)
    p.add_argument("--muon-lr",type=float,default=0.001)
    p.add_argument("--wandb-mode",choices=["offline","online","disabled"],default="offline")
    p.add_argument("--max-training-minutes",type=float,default=0.)
    p.add_argument("--stage",default="pilot")
    p.add_argument("--require-natural-sink",action="store_true")
    p.add_argument("--eval-long-context",action="store_true")
    args,_=p.parse_known_args()
    assert args.context in (128,256,512,1024)
    assert args.steps>0 and args.accum>0
    assert len(set(args.seeds))==len(args.seeds)
    assert len(set(args.arms))==len(args.arms)
    assert torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    data=TokenData(args.data,args.context)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    config=vars(args).copy()
    config.update(device_name=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__,
                  pretraining_checkpoint=BASE,
                  corpus_sha=data.meta["manifest_sha256"],
                  corpus_train_sha=data.meta["train_sha256"],
                  corpus_valid_sha=data.meta["valid_sha256"],
                  start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    (out/"config.json").write_text(json.dumps(config,indent=2))
    print("RUN_CONFIG",json.dumps(config,sort_keys=True),flush=True)
    wandb=init_wandb(out,args)
    from transformers import GPT2LMHeadModel
    summary=[]
    run_start=time.time()
    total_global=0
    for seed in args.seeds:
        valid=data.validation(args.eval_batches,args.microbatch)
        for arm in args.arms:
            seed_all(seed)
            model=GPT2LMHeadModel.from_pretrained(BASE,attn_implementation="eager").to("cuda")
            # Disable dropout across all arms so the first step starts from the exact same function.
            for module in model.modules():
                if isinstance(module,nn.Dropout):module.p=0.
            n_params=sum(x.numel() for x in model.parameters())
            optimizers=group_optimizers(model,arm,args.adam_lr,args.muon_lr)
            sample_rng=np.random.default_rng(seed+193)
            baseline_metrics=evaluate(model,valid,sample_for_attention=2)
            if args.require_natural_sink:
                assert baseline_metrics["bos_excess"]>0.15 and baseline_metrics["head_sink_rate_30"]>0.2, baseline_metrics
            if arm==args.arms[0]:
                print("PRETRAINED_BASELINE",json.dumps(
                    dict(seed=seed,parameters=n_params,**baseline_metrics)),flush=True)
            print("ARM_STARTED",json.dumps({"seed":seed,"arm":arm,
                "param_count":n_params,"baseline_metrics":baseline_metrics}),flush=True)
            (out/"initial_metrics.jsonl").open("a").write(
                json.dumps(dict(seed=seed,arm=arm,**baseline_metrics))+"\n")
            started=time.time()
            for step in range(1,args.steps+1):
                model.train()
                for optimizer in optimizers:optimizer.zero_grad(set_to_none=True)
                loss_total=0.
                conc_total=0.
                for micro in range(args.accum):
                    x=data.batch(args.microbatch,sample_rng)
                    with torch.autocast("cuda",dtype=torch.bfloat16):
                        output=model(x,labels=x,
                          output_attentions=("conc" in arm),return_dict=True,
                          use_cache=False)
                        language_loss=output.loss
                        if "conc" in arm:
                            penalty=regularizer(output.attentions,seq_start=args.context//4)
                        else:
                            penalty=language_loss.new_zeros(())
                        loss=language_loss+args.lambda_conc*penalty
                    if not torch.isfinite(loss).item():
                        raise RuntimeError(f"non-finite loss {seed=} {arm=} {step=}")
                    (loss/args.accum).backward()
                    loss_total+=language_loss.detach().float().item()/args.accum
                    conc_total+=penalty.detach().float().item()/args.accum
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                for optimizer in optimizers:optimizer.step()
                total_global+=1
                if step==1 or step%args.eval_every==0 or step==args.steps:
                    metrics=evaluate(model,valid,sample_for_attention=2)
                    if args.eval_long_context and step==args.steps:
                        long_data=TokenData(args.data,1024)
                        long_val=long_data.validation(2,1)
                        long_metrics=evaluate(model,long_val,sample_for_attention=2)
                        metrics.update({"long_"+k:v for k,v in long_metrics.items()})
                    record=dict(stage=args.stage,seed=seed,arm=arm,step=step,
                      train_ce=loss_total,train_concentration=conc_total,
                      tokens_seen=step*args.context*args.microbatch*args.accum,
                      elapsed_s=round(time.time()-started,2),**metrics)
                    print("METRIC",json.dumps(record,sort_keys=True),flush=True)
                    with (out/"metrics.jsonl").open("a",buffering=1) as stream:
                        stream.write(json.dumps(record,sort_keys=True)+"\n")
                    if wandb:
                        wandb.log({f"{arm}/seed{seed}/{k}":v for k,v in record.items()
                           if isinstance(v,(int,float))},step=total_global)
                if args.max_training_minutes>0 and (
                    time.time()-run_start>args.max_training_minutes*60):
                    print("TIME_BUDGET_STOP",seed,arm,step,flush=True)
                    break
            completed=(step==args.steps)
            summary.append(dict(stage=args.stage,seed=seed,arm=arm,
                completed=completed,steps_completed=step,n_params=n_params,
                baseline_metrics=baseline_metrics,last_metrics=metrics))
            (out/"summary.json").write_text(json.dumps(summary,indent=2))
            del model,optimizers
            torch.cuda.empty_cache()
            if not completed:break
        if not completed:break
    if wandb:wandb.finish()
    (out/"run_status.json").write_text(json.dumps(
        dict(status="complete" if all(x["completed"] for x in summary) else "partial",
             completed=len(summary),expected=len(args.arms)*len(args.seeds)),indent=2))
    print("RUN_END",json.dumps({"n":len(summary),
        "all_complete":all(x["completed"] for x in summary),
        "out":str(out)}),flush=True)

if __name__=="__main__":main()
