#!/usr/bin/env python3
"""Hostile controls for attention sink metrics; returns a nonzero code on failure."""
import json
import torch
from torch.nn import functional as F
import attention_sink_s4_stress as stress

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
