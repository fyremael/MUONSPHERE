#!/usr/bin/env python3
"""Attention sink x Muon/Schatten-4 ablation. Designed for Colab T4."""
import argparse, csv, json, math, os, random, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

ARMS = ("adamw", "muon_frob", "muon_s4", "muon_s4_conc",
        "muon_s4_null", "muon_s4_null_conc")
VOCAB, BOS, SEP, EOS = 35, 32, 33, 34

def set_seed(s):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def sample(bs, span, rng, device):
    a = torch.randint(0, 32, (bs, span), generator=rng)
    one = lambda token: torch.full((bs, 1), token, dtype=torch.long)
    seq = torch.cat((one(BOS), a, one(SEP), a, one(EOS)), 1).to(device)
    return seq[:, :-1], seq[:, 1:]

def polar_ns(m, mode="s4", steps=5):
    x = m.float()
    transpose = x.shape[0] > x.shape[1]
    if transpose: x = x.T
    if mode == "s4":
        norm = torch.linalg.vector_norm(x @ x.T).sqrt()
    elif mode == "frob":
        norm = torch.linalg.vector_norm(x)
    else: raise ValueError(mode)
    x = x / norm.clamp_min(1e-7)
    for _ in range(steps):
        a = x @ x.T
        x = 3.4445*x + (-4.775*a + 2.0315*(a@a))@x
    return (x.T if transpose else x).to(m.dtype)

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, mode="s4", beta=0.95):
        super().__init__(params, dict(lr=lr, mode=mode, beta=beta))
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                st = self.state[p]
                if "m" not in st: st["m"] = torch.zeros_like(p)
                m = st["m"]
                m.mul_(group["beta"]).add_(p.grad)
                z = polar_ns(p.grad + group["beta"]*m, group["mode"])
                scale = math.sqrt(max(1., p.shape[0]/p.shape[1]))
                p.mul_(1-group["lr"]*0.01)
                p.add_(z, alpha=-group["lr"]*scale)

class Attn(nn.Module):
    def __init__(self, d, heads, null):
        super().__init__()
        self.h, self.dh, self.null = heads, d//heads, null
        self.qkv = nn.Linear(d, 3*d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        if null: self.null_bias = nn.Parameter(torch.ones(heads))
    def forward(self, x):
        b,t,d = x.shape
        q,k,v = self.qkv(x).view(b,t,3,self.h,self.dh).unbind(2)
        q,k,v = (z.transpose(1,2) for z in (q,k,v))
        logits = (q@k.transpose(-2,-1))/math.sqrt(self.dh)
        mask = torch.ones(t,t,device=x.device,dtype=torch.bool).tril()
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        if self.null:
            null = self.null_bias[None,:,None,None].expand(b,self.h,t,1)
            a = F.softmax(torch.cat((logits,null),-1),-1)[...,:t]
        else: a = F.softmax(logits,-1)
        return self.proj((a@v).transpose(1,2).reshape(b,t,d)), a

class Block(nn.Module):
    def __init__(self,d,h,null):
        super().__init__()
        self.n1,self.n2 = nn.LayerNorm(d),nn.LayerNorm(d)
        self.attn = Attn(d,h,null)
        self.up,self.down = nn.Linear(d,4*d),nn.Linear(4*d,d)
    def forward(self,x):
        y,a = self.attn(self.n1(x))
        x = x + y
        return x + self.down(F.gelu(self.up(self.n2(x)))), a

class Model(nn.Module):
    def __init__(self,span,d=96,h=4,depth=2,null=False):
        super().__init__()
        self.emb = nn.Embedding(VOCAB,d)
        self.pos = nn.Embedding(2*span+2,d)
        self.blocks = nn.ModuleList([Block(d,h,null) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d,VOCAB,bias=False)
    def forward(self,tokens):
        t = tokens.shape[1]
        x = self.emb(tokens) + self.pos(torch.arange(t,device=tokens.device))
        attentions = []
        for block in self.blocks:
            x,a = block(x)
            attentions.append(a)
        return self.head(self.norm(x)), attentions

def fourth_order_penalty(att, span):
    """Penalize fourth power of excess column energy, relative to causal uniform."""
    total = 0.
    for a in att:
        t = a.shape[-1]
        q = torch.arange(span+1,t,device=a.device)
        k = torch.arange(t,device=a.device)
        visible = (k[None,:] <= q[:,None]).to(a.dtype)
        # Null mass is not penalized; compare to uniform allocation of real mass.
        real = a[:,:,span+1:,:].sum(-1,keepdim=True)
        fair = visible[None,None,:,:] * real/(q+1)[None,None,:,None]
        positive_excess = (a[:,:,span+1:,:]-fair).relu()
        per_key_energy = positive_excess.square().mean(-2)
        total = total + per_key_energy.square().sum(-1).mean()*t
    return total / len(att)

@torch.no_grad()
def eval_model(model, val, span):
    model.eval()
    sums = {}
    for x,y in val:
        logits,aa = model(x)
        ce = F.cross_entropy(logits.flatten(0,1), y.flatten()).item()
        acc = (logits[:,span+1:2*span+1].argmax(-1) ==
               y[:,span+1:2*span+1]).float().mean().item()
        a = torch.stack(aa,0)[:,:,:,span+1:,:]
        q = torch.arange(span+1,x.shape[1],device=x.device)
        real_mass = a.sum(-1)
        expected_first = real_mass/(q+1)[None,None,None,:]
        p0 = a[...,0]
        col = a.mean(-2)
        metrics = {
            "valid_ce":ce, "valid_ppl":math.exp(min(ce,20)),
            "copy_acc":acc, "bos_mass":p0.mean().item(),
            "bos_excess":(p0-expected_first).mean().item(),
            "max_col_mass":col.max(-1).values.mean().item(),
            "null_mass":(1-real_mass).clamp_min(0).mean().item(),
            "attention_entropy":(-a*a.clamp_min(1e-12).log()).sum(-1).mean().item(),
            "fourth_order_penalty":fourth_order_penalty(aa,span).item()
        }
        for k,v in metrics.items(): sums[k] = sums.get(k,0)+v
    return {k:v/len(val) for k,v in sums.items()}

def optimizers(model,arm,adam_lr,muon_lr):
    if arm=="adamw":
        return [torch.optim.AdamW(model.parameters(),lr=adam_lr,weight_decay=0.01)]
    matrices,others=[],[]
    for name,p in model.named_parameters():
        (matrices if p.ndim==2 and name.startswith("blocks.") and
         name.endswith("weight") else others).append(p)
    assert matrices and others
    mu = Muon(matrices,lr=muon_lr,mode="frob" if arm=="muon_frob" else "s4")
    return [mu,torch.optim.AdamW(others,lr=adam_lr,weight_decay=0.01)]

def self_test(device,span):
    n=16
    concentration=torch.zeros(n,n,device=device)
    concentration[:,0]=1
    uniform=torch.ones(n,n,device=device)/n
    s4=lambda a: torch.linalg.svdvals(a).pow(4).sum().pow(.25).item()
    assert s4(concentration)>3.9*s4(uniform)
    for mode in ("frob","s4"):
        assert torch.isfinite(polar_ns(torch.randn(16,32,device=device),mode)).all()
    # Causal uniform attention should have zero positive excess.
    q=torch.ones(span*2+2,span*2+2,device=device).tril()
    q=q/q.sum(-1,keepdim=True)
    assert fourth_order_penalty([q[None,None]],span)<1e-10
    return {"sink_s4":s4(concentration),"uniform_s4":s4(uniform),"passed":True}

def get_wandb(out,cfg,mode):
    if mode=="disabled":return None
    try:
        import wandb
        wandb.init(project="gcl-attention-sinks",
                   name="muon-s4-20261021",
                   group="s4-attention-sink-ablation",
                   config=cfg,dir=str(out),mode=mode,reinit=True)
        return wandb
    except Exception as e:
        print("WANDB_UNAVAILABLE",repr(e),flush=True)
        return None


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--steps",type=int,default=100)
    p.add_argument("--batch",type=int,default=24)
    p.add_argument("--span",type=int,default=20)
    p.add_argument("--width",type=int,default=96)
    p.add_argument("--layers",type=int,default=2)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--seed",type=int,default=20261021)
    p.add_argument("--log-every",type=int,default=20)
    p.add_argument("--eval-batches",type=int,default=3)
    p.add_argument("--arms",nargs="+",choices=ARMS,default=list(ARMS))
    p.add_argument("--lambda-conc",type=float,default=0.05)
    p.add_argument("--adam-lr",type=float,default=0.001)
    p.add_argument("--muon-lr",type=float,default=0.02)
    p.add_argument("--out",default="/content/gcl_s4_sink_results")
    p.add_argument("--wandb-mode",choices=["offline","online","disabled"],
                   default="offline")
    p.add_argument("--self-test",action="store_true")
    args=p.parse_args()
    assert args.span>=4 and args.width%args.heads==0
    assert args.steps>0 and args.batch>0
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4,os.cpu_count() or 2))
    set_seed(args.seed)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    cfg=vars(args).copy()
    cfg.update(device=str(device),torch_version=torch.__version__,
               gpu=torch.cuda.get_device_name(0) if device.type=="cuda" else None,
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()))
    (out/"config.json").write_text(json.dumps(cfg,indent=2))
    control=self_test(device,args.span)
    (out/"positive_control.json").write_text(json.dumps(control,indent=2))
    print("CONFIG",json.dumps(cfg,sort_keys=True),flush=True)
    print("CONTROL",json.dumps(control),flush=True)
    if args.self_test: assert args.steps<=5
    vg=torch.Generator().manual_seed(args.seed+31337)
    val=[sample(args.batch,args.span,vg,device) for _ in range(args.eval_batches)]
    wandb=get_wandb(out,cfg,args.wandb_mode)
    summary=[]
    with (out/"metrics.jsonl").open("w",buffering=1) as logfile:
        for arm in args.arms:
            set_seed(args.seed)
            model=Model(args.span,args.width,args.heads,args.layers,
                        "null" in arm).to(device)
            opts=optimizers(model,arm,args.adam_lr,args.muon_lr)
            tg=torch.Generator().manual_seed(args.seed+123)
            coeff=args.lambda_conc if "conc" in arm else 0.
            began=time.monotonic()
            for step in range(1,args.steps+1):
                model.train()
                x,y=sample(args.batch,args.span,tg,device)
                for opt in opts:opt.zero_grad(set_to_none=True)
                logits,aa=model(x)
                lm=F.cross_entropy(logits.flatten(0,1),y.flatten())
                copy=F.cross_entropy(
                    logits[:,args.span+1:2*args.span+1].reshape(-1,VOCAB),
                    y[:,args.span+1:2*args.span+1].reshape(-1))
                penalty=fourth_order_penalty(aa,args.span) if coeff else lm.new_zeros(())
                loss=lm+copy+coeff*penalty
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                for opt in opts:opt.step()
                if step==1 or step%args.log_every==0 or step==args.steps:
                    scores=eval_model(model,val,args.span)
                    rec=dict(arm=arm,step=step,train_lm_ce=lm.item(),
                             train_copy_ce=copy.item(),
                             train_penalty=penalty.item(),
                             elapsed_s=round(time.monotonic()-began,2),**scores)
                    assert all(math.isfinite(v) for v in rec.values()
                               if isinstance(v,(float,int))),rec
                    logfile.write(json.dumps(rec,sort_keys=True)+"\n")
                    if wandb:
                        wandb.log({arm+"/"+k:v for k,v in rec.items()
                                   if isinstance(v,(int,float))},
                                  step=ARMS.index(arm)*args.steps+step)
                    print("METRIC",json.dumps(rec,sort_keys=True),flush=True)
            summary.append(rec)
            (out/"summary.json").write_text(json.dumps(summary,indent=2))
            del model,opts
            if device.type=="cuda":torch.cuda.empty_cache()
    with (out/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    if wandb:wandb.finish()
    from zipfile import ZipFile, ZIP_DEFLATED
    with ZipFile(out/"results.zip", "w", compression=ZIP_DEFLATED) as z:
        for item in out.rglob("*"):
            if item.is_file() and item.name!="results.zip":
                z.write(item, item.relative_to(out))
    print("COMPLETE",json.dumps({"directory":str(out),"arms":len(summary)}),
          flush=True)


import types as _gcl_types
baseline = _gcl_types.SimpleNamespace(
 ARMS=ARMS, VOCAB=VOCAB, BOS=BOS, SEP=SEP, EOS=EOS,
 set_seed=set_seed, sample=sample, Muon=Muon, self_test=self_test,
 fourth_order_penalty=fourth_order_penalty, get_wandb=get_wandb)

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
# Baseline frozen below as a local namespace.

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


#!/usr/bin/env python3
"""Hostile controls for attention sink metrics; returns a nonzero code on failure."""
import json
import torch
from torch.nn import functional as F
import types as _stypes
stress=_stypes.SimpleNamespace(extra_metrics=extra_metrics)

def fabricate(kind,span=24):
    t=2*span+2
    logits=torch.zeros(1,1,t,t)
    mask=torch.ones(t,t,dtype=torch.bool).tril()
    logits=logits.masked_fill(~mask,-1e9)
    if kind in ("sink","relocated","null_hide"):
        logits[...,1:,0]+=8.
    if kind=="relocated":
        logits[...,2:,1]+=12.
    if kind=="null_hide":
        nulllogits=torch.full((1,1,t,1),16.)
        probs=F.softmax(torch.cat((logits,nulllogits),dim=-1),dim=-1)[...,:t]
    else: probs=F.softmax(logits,dim=-1)
    return stress.extra_metrics([probs],span)

def test_adversaries():
    result={name:fabricate(name) for name in ("sink","relocated","null_hide","diffuse")}
    s,r,n,d=(result[name] for name in ("sink","relocated","null_hide","diffuse"))
    assert s["bos_real_share"]>0.95 and s["max_column_real_share"]>0.95
    assert r["bos_real_share"]<0.05 and r["max_column_real_share"]>0.9, r
    assert n["bos_mass"]<0.05 and n["null_mass"]>0.9, n
    assert n["bos_real_share"]>0.6, n
    assert d["bos_real_share"]<0.1 and d["max_column_real_share"]<0.1, d
    assert n["source_key_mass"]<d["source_key_mass"], (n,d)
    print("HOSTILE_METRIC_CONTROLS",json.dumps(result,sort_keys=True),flush=True)
    print("PASSED adversarial sink relocation, null mass hiding, and diffuse control")
    return result

if __name__=="__main__":
    test_adversaries()
