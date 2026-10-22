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

if __name__=="__main__":
    main()
