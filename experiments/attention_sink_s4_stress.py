#!/usr/bin/env python3
"""Multi-seed, larger-LM, calibrated attention-sink stress test on Colab.

Intervention is explicitly synthetic: additive key-zero logit bias. It tests
resistance to an induced sink, NOT the natural genesis of attention sinks.
"""
import argparse, csv, json, math, os, random, time
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import torch
from torch import nn
from torch.nn import functional as F

# Import the versioned baseline optimizer and concentrated-column statistic.
import attention_sink_s4 as baseline

ARMS = baseline.ARMS
VOCAB, BOS, SEP, EOS = baseline.VOCAB, baseline.BOS, baseline.SEP, baseline.EOS

class StressAttention(nn.Module):
    def __init__(self, d, heads, use_null):
        super().__init__()
        self.heads, self.dk, self.use_null = heads, d // heads, use_null
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        if use_null:
            self.null_bias = nn.Parameter(torch.ones(heads))

    def forward(self, x, bos_logit_bias=0., ablate_bos=False):
        b, t, d = x.shape
        q, k, v = self.qkv(x).view(b,t,3,self.heads,self.dk).unbind(2)
        q, k, v = (z.transpose(1,2) for z in (q,k,v))
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.dk)
        mask = torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        if ablate_bos:
            # First query can only access BOS unless null is available.
            # Exclude it from the ablation; evaluate only late-copy queries.
            late = scores[...,1:,:]
            late = torch.cat((late[...,:1].fill_(torch.finfo(scores.dtype).min),late[...,1:]),-1)
            scores = torch.cat((scores[...,:1,:],late),-2)
        elif bos_logit_bias:
            # Bias only late queries; do not change BOS itself or source encoding.
            late = scores[...,1:,:]
            late = torch.cat((late[...,:1]+bos_logit_bias,late[...,1:]),-1)
            scores = torch.cat((scores[...,:1,:],late),-2)
        if self.use_null:
            null = self.null_bias[None,:,None,None].expand(b,self.heads,t,1)
            a = F.softmax(torch.cat((scores,null),-1),-1)[...,:t]
        else:
            a = F.softmax(scores,-1)
        z = (a @ v).transpose(1,2).contiguous().view(b,t,d)
        return self.out(z), a

class StressBlock(nn.Module):
    def __init__(self,d,h,use_null):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d),nn.LayerNorm(d)
        self.attn = StressAttention(d,h,use_null)
        self.up, self.down = nn.Linear(d,4*d),nn.Linear(4*d,d)
    def forward(self,x,bias=0.,ablate_bos=False):
        z,a = self.attn(self.n1(x),bias,ablate_bos)
        x = x + z
        return x + self.down(F.gelu(self.up(self.n2(x)))),a

class StressLM(nn.Module):
    def __init__(self,span,width,layers,heads,use_null):
        super().__init__()
        self.emb = nn.Embedding(VOCAB,width)
        self.pos = nn.Embedding(2*span+2,width)
        self.blocks = nn.ModuleList([StressBlock(width,heads,use_null)
                                     for _ in range(layers)])
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width,VOCAB,bias=False)
    def forward(self,tokens,bias=0.,ablate_bos=False):
        t=tokens.shape[1]
        x=self.emb(tokens)+self.pos(torch.arange(t,device=tokens.device))
        aa=[]
        for block in self.blocks:
            x,a=block(x,bias,ablate_bos)
            aa.append(a)
        return self.head(self.norm(x)),aa

def sample(bs,span,rng,device):
    return baseline.sample(bs,span,rng,device)

def make_optimizers(model,arm,adam_lr,muon_lr):
    if arm=="adamw":
        return [torch.optim.AdamW(model.parameters(),lr=adam_lr,weight_decay=0.01)]
    mats,other=[],[]
    for name,p in model.named_parameters():
        (mats if name.startswith("blocks.") and name.endswith("weight")
         and p.ndim==2 else other).append(p)
    assert mats and other
    kind="frob" if arm=="muon_frob" else "s4"
    return [baseline.Muon(mats,lr=muon_lr,mode=kind),
            torch.optim.AdamW(other,lr=adam_lr,weight_decay=0.01)]

def extra_metrics(aa,span):
    a=torch.stack(aa,0)[:,:,:,span+1:,:]
    t=a.shape[-1]
    q=torch.arange(span+1,t,device=a.device)
    real=a.sum(-1).clamp_min(1e-8)
    p0=a[...,0]
    p0share=p0/real
    maxcol=(a/real[...,None]).mean(-2).max(-1).values
    source_idx=(q-span).view(1,1,1,-1,1).expand(*a.shape[:-1],1)
    source=a.gather(-1,source_idx).squeeze(-1)
    source_share=(source/real)
    expected=1.0/(q+1)
    return {
        "bos_mass":p0.mean().item(),
        "bos_excess":(p0-real*expected).mean().item(),
        "bos_real_share":p0share.mean().item(),
        "nonbos_real_share":(1-p0share).mean().item(),
        "null_mass":(1-real).clamp_min(0).mean().item(),
        "max_column_real_share":maxcol.mean().item(),
        "source_key_mass":source.mean().item(),
        "source_key_real_share":source_share.mean().item(),
        "fourth_order":baseline.fourth_order_penalty(aa,span).item(),
        "high_sink_head_fraction":(p0share.mean(-1)>0.5).float().mean().item(),
    }

@torch.no_grad()
def evaluate(model,validation,span,bias,ablate_bos=False):
    model.eval()
    metrics=[]
    for x,y in validation:
        logits,aa=model(x,bias,ablate_bos)
        ce=F.cross_entropy(logits.flatten(0,1),y.flatten()).item()
        copy=(logits[:,span+1:2*span+1].argmax(-1)==
              y[:,span+1:2*span+1]).float().mean().item()
        copy_ce=F.cross_entropy(
            logits[:,span+1:2*span+1].reshape(-1,VOCAB),
            y[:,span+1:2*span+1].reshape(-1)).item()
        met={"valid_ce":ce,"copy_ce":copy_ce,"copy_acc":copy}
        met.update(extra_metrics(aa,span))
        metrics.append(met)
    return {k:sum(z[k] for z in metrics)/len(metrics) for k in metrics[0]}


def verify_induction(model, validation, span, bias):
    clean=evaluate(model,validation,span,0.)
    induced=evaluate(model,validation,span,bias)
    delta=induced["bos_mass"]-clean["bos_mass"]
    # Calibrated positive-control gate: token zero is deliberately made salient.
    assert delta>0.35 and induced["bos_real_share"]>0.5, {
        "error":"failed_to_induce_sink", "clean":clean, "induced":induced}
    return {"clean_bos":clean["bos_mass"],
            "stressed_bos":induced["bos_mass"],
            "delta_bos":delta,
            "stressed_bos_real_share":induced["bos_real_share"]}

def evaluate_three(model,validation,span,bias):
    stressed=evaluate(model,validation,span,bias)
    clean=evaluate(model,validation,span,0.)
    ablated=evaluate(model,validation,span,bias,ablate_bos=True)
    return {**stressed,
            "clean_valid_ce":clean["valid_ce"],
            "clean_copy_acc":clean["copy_acc"],
            "clean_bos_mass":clean["bos_mass"],
            "ablated_valid_ce":ablated["valid_ce"],
            "ablated_copy_acc":ablated["copy_acc"],
            "ablation_delta_ce":ablated["valid_ce"]-stressed["valid_ce"],
            "stressor_delta_ce":stressed["valid_ce"]-clean["valid_ce"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--width",type=int,default=384)
    ap.add_argument("--layers",type=int,default=6)
    ap.add_argument("--heads",type=int,default=8)
    ap.add_argument("--span",type=int,default=24)
    ap.add_argument("--batch",type=int,default=16)
    ap.add_argument("--steps",type=int,default=120)
    ap.add_argument("--log-every",type=int,default=30)
    ap.add_argument("--eval-batches",type=int,default=3)
    ap.add_argument("--seeds",type=int,nargs="+",default=[101,202,303])
    ap.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    ap.add_argument("--bos-bias",type=float,default=7.)
    ap.add_argument("--lambda-conc",type=float,default=10.)
    ap.add_argument("--adam-lr",type=float,default=0.001)
    ap.add_argument("--muon-lr",type=float,default=0.02)
    ap.add_argument("--out",default="/content/gcl_attention_stress")
    ap.add_argument("--wandb-mode",choices=["offline","online","disabled"],
                    default="offline")
    ap.add_argument("--self-test",action="store_true")
    args=ap.parse_args()
    assert args.bos_bias>0 and args.steps>0 and args.span>=4
    assert args.width%args.heads==0 and len(set(args.seeds))==len(args.seeds)
    if args.self_test:
        assert args.steps<=3
    torch.set_num_threads(min(4,os.cpu_count() or 2))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    meta=vars(args).copy()
    meta.update(device=str(device),torch_version=torch.__version__,
                gpu=(torch.cuda.get_device_name(0) if device.type=="cuda" else "CPU"),
                utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    (out/"config.json").write_text(json.dumps(meta,indent=2))
    test=baseline.self_test(device,args.span)
    (out/"positive_control.json").write_text(json.dumps(test,indent=2))
    print("CONFIG",json.dumps(meta,sort_keys=True),flush=True)
    result=[]
    calibration=[]
    wandb=baseline.get_wandb(out,meta,args.wandb_mode)
    print("POSITIVE_CONTROL",json.dumps(test),flush=True)
    global_step=0
    with (out/"metrics.jsonl").open("w",buffering=1) as stream:
      for seed in args.seeds:
        vg=torch.Generator(device="cpu").manual_seed(seed+31337)
        validation=[sample(args.batch,args.span,vg,device)
                    for _ in range(args.eval_batches)]
        for arm in args.arms:
            baseline.set_seed(seed)
            use_null="null" in arm
            model=StressLM(args.span,args.width,args.layers,args.heads,
                           use_null).to(device)
            params=sum(p.numel() for p in model.parameters())
            if seed==args.seeds[0] and arm==args.arms[0]:
                print("PARAMETERS",params,flush=True)
            # Calibrate before training every arm/seed, independent of learned model.
            calib=verify_induction(model,validation,args.span,args.bos_bias)
            calib.update(seed=seed,arm=arm)
            calibration.append(calib)
            (out/"calibration.json").write_text(json.dumps(calibration,indent=2))
            optim=make_optimizers(model,arm,args.adam_lr,args.muon_lr)
            tg=torch.Generator(device="cpu").manual_seed(seed+123)
            lam=args.lambda_conc if "conc" in arm else 0.
            start=time.monotonic()
            for step in range(1,args.steps+1):
                model.train()
                x,y=sample(args.batch,args.span,tg,device)
                for opt in optim: opt.zero_grad(set_to_none=True)
                logits,aa=model(x,args.bos_bias)
                lm=F.cross_entropy(logits.flatten(0,1),y.flatten())
                cp=F.cross_entropy(
                    logits[:,args.span+1:2*args.span+1].reshape(-1,VOCAB),
                    y[:,args.span+1:2*args.span+1].reshape(-1))
                conc=(baseline.fourth_order_penalty(aa,args.span)
                      if lam else lm.new_zeros(()))
                loss=lm+cp+lam*conc
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss {seed=} {arm=} {step=}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                for opt in optim: opt.step()
                if step==1 or step%args.log_every==0 or step==args.steps:
                    if step==args.steps:
                        metrics=evaluate_three(model,validation,args.span,args.bos_bias)
                    else:
                        metrics=evaluate(model,validation,args.span,args.bos_bias)
                    rec=dict(seed=seed,arm=arm,step=step,params=params,
                             train_ce=lm.item(),train_copy_ce=cp.item(),
                             train_penalty=conc.item(),
                             elapsed_s=round(time.monotonic()-start,2),**metrics)
                    assert all(math.isfinite(v) for v in rec.values()
                               if isinstance(v,(int,float))),rec
                    stream.write(json.dumps(rec,sort_keys=True)+"\n")
                    global_step+=1
                    if wandb:
                        wandb.log({f"{arm}/seed{seed}/{k}":v for k,v in rec.items()
                                   if isinstance(v,(int,float))},step=global_step)
                    print("METRIC",json.dumps(rec,sort_keys=True),flush=True)
            result.append(rec)
            (out/"summary.json").write_text(json.dumps(result,indent=2))
            del model,optim
            if device.type=="cuda":torch.cuda.empty_cache()
    if wandb:wandb.finish()
    fields=sorted(set().union(*(r.keys() for r in result)))
    with (out/"summary.csv").open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(result)
    with ZipFile(out/"results.zip","w",ZIP_DEFLATED) as archive:
        for p in out.rglob("*"):
            if p.is_file() and p.name!="results.zip":
                archive.write(p,p.relative_to(out))
    print("COMPLETE",json.dumps({"evaluations":len(result),
          "seeds":args.seeds,"arms":args.arms,"output":str(out)}),flush=True)

if __name__=="__main__":
    main()
