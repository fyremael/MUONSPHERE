#!/usr/bin/env python3
"""Paired zero-shot retrieval evaluations for tiny-from-scratch causal LMs.

Two independent probes:
1. Natural-corpus span copy: evidence vs corrupted evidence, at 512/1024.
2. Passkey four-way choice: natural filler, four distinct candidate tokens,
   correct answer in early context vs replaced with another answer.

No answers from validation corpus appear in training examples of the probe.
Reports full conditional token-loss, exact token accuracy, and retrieval lift.
"""
import math
import numpy as np
import torch
import tiktoken

def _score(model,x,positions):
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        logits=model(x[None],use_cache=False,return_dict=True).logits[0].float()
    j=torch.as_tensor(positions,device=x.device,dtype=torch.long)
    targets=x[j]
    pred=logits[j-1]
    lp=pred.log_softmax(dim=-1)
    nll=(-lp.gather(-1,targets[:,None])).mean().item()
    acc=(pred.argmax(-1)==targets).float().mean().item()
    return nll,acc,pred

def measure_retrieval(model,valid_memmap,seed=101,n=24,contexts=(512,1024)):
    """A held-out, immutable, deterministic probe shared across all arms."""
    from numpy.random import default_rng
    rng=default_rng(705019)
    enc=tiktoken.get_encoding("gpt2")
    marker=enc.encode(" The exact marked phrase was:",disallowed_special=())
    src_marker=enc.encode(" [marked phrase]:",disallowed_special=())
    choices=tuple(enc.encode(" "+x)[0] for x in ("amber","violet","copper","silver"))
    assert len(set(choices))==4
    output={}
    for length in contexts:
        records=[]
        # A dozen paired 1024-token forward passes per condition is small
        # relative to a full training run, yet sufficient for mean estimates.
        for sample in range(n):
            start=int(rng.integers(0,len(valid_memmap)-length-4096))
            raw=np.asarray(valid_memmap[start:start+length],dtype=np.int64).copy()
            source_at=24
            needle=np.asarray(valid_memmap[start+length+37:start+length+45],dtype=np.int64)
            distract=np.asarray(valid_memmap[start+length+103:start+length+111],dtype=np.int64)
            base=raw.copy()
            base[0]=50256
            base[8:8+len(src_marker)]=src_marker
            target_at=length-12
            base[target_at-len(marker):target_at]=marker
            base[target_at:target_at+8]=needle
            good=base.copy()
            bad=base.copy()
            good[source_at:source_at+8]=needle
            bad[source_at:source_at+8]=distract
            g=torch.tensor(good,dtype=torch.long,device="cuda")
            b=torch.tensor(bad,dtype=torch.long,device="cuda")
            positions=range(target_at,target_at+8)
            clean_nll,clean_acc,_=_score(model,g,positions)
            bad_nll,bad_acc,_=_score(model,b,positions)
            # One-token passkey retrieval with natural filler and four
            # equally presented answer candidates; do not call this trained
            # instruction following, it is a zero-shot context-use probe.
            answer=sample%4
            other=(answer+1)%4
            pass_prompt=enc.encode(" The passkey is",disallowed_special=())
            query=enc.encode(" The passkey is",disallowed_special=())
            pos_answer=30
            target=length-4
            p=raw.copy()
            p[0]=50256
            p[10:10+len(pass_prompt)]=pass_prompt
            p[pos_answer]=choices[answer]
            p[target-len(query):target]=query
            # target token is scored, never explicitly supplied to model
            p[target]=choices[answer]
            q=p.copy()
            q[pos_answer]=choices[other]
            pt=torch.tensor(p,dtype=torch.long,device="cuda")
            qt=torch.tensor(q,dtype=torch.long,device="cuda")
            _,_,lg=_score(model,pt,[target])
            _,_,lb=_score(model,qt,[target])
            ix=torch.tensor(choices,device="cuda",dtype=torch.long)
            good_scores=lg[0,ix]
            corrupted_scores=lb[0,ix]
            forced_accuracy=(int(good_scores.argmax().item())==answer)
            four_rank=(good_scores[answer]-torch.logsumexp(good_scores,-1)).item()
            chance_correct_lift=(good_scores[answer]-corrupted_scores[answer]).item()
            records.append(dict(
                natural_copy_nll=clean_nll,natural_copy_ablate_nll=bad_nll,
                natural_copy_top1=clean_acc,natural_copy_ablate_top1=bad_acc,
                passkey_fourway_correct=1. if forced_accuracy else 0.,
                passkey_logprob=four_rank,
                passkey_ablation_true_logit_lift=chance_correct_lift))
        agg={}
        for key in records[0]:
            vals=np.asarray([v[key] for v in records])
            agg[key]=float(vals.mean())
            agg[key+"_std"]=float(vals.std(ddof=1))
        agg["natural_copy_lift_nats"]=agg["natural_copy_ablate_nll"]-agg["natural_copy_nll"]
        agg["n"]=len(records)
        agg["context"]=length
        agg["passkey_chance"]=0.25
        output[str(length)]=agg
    return output

