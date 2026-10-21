#!/usr/bin/env python3
"""Prepare disjoint FineWeb-Edu GPT2-tokenized train/validation streams on Colab."""
import argparse, hashlib, json, os, time
from pathlib import Path
import numpy as np
import tiktoken
from datasets import load_dataset

p=argparse.ArgumentParser()
p.add_argument("--train-tokens",type=int,default=18_000_000)
p.add_argument("--valid-tokens",type=int,default=1_500_000)
p.add_argument("--out",default="/content/gcl_natural_sink")
args,_=p.parse_known_args()
out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
enc=tiktoken.get_encoding("gpt2")
train=np.memmap(out/"train_tokens.uint16",mode="w+",dtype=np.uint16,shape=(args.train_tokens,))
valid=np.memmap(out/"valid_tokens.uint16",mode="w+",dtype=np.uint16,shape=(args.valid_tokens,))
train_pos=valid_pos=0
counts={"train_docs":0,"valid_docs":0,"skipped_docs":0}
idhash=hashlib.sha256()
source="HuggingFaceFW/fineweb-edu"
start=time.time()
ds=load_dataset(source,name="sample-10BT",split="train",streaming=True)
for i,doc in enumerate(ds):
    key=str(doc.get("id") or doc.get("url") or i)
    txt=doc.get("text") or ""
    if len(txt)<200:
        counts["skipped_docs"]+=1;continue
    group=int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%11
    if group==0:
        arr,pos,limit,split=valid,valid_pos,args.valid_tokens,"valid"
    else:
        arr,pos,limit,split=train,train_pos,args.train_tokens,"train"
    if pos>=limit:continue
    tokens=[50256]+enc.encode(txt,disallowed_special=())
    n=min(len(tokens),limit-pos)
    arr[pos:pos+n]=np.asarray(tokens[:n],dtype=np.uint16)
    idhash.update((key+"\t"+split+"\t"+str(n)+"\n").encode())
    if group==0: valid_pos+=n;counts["valid_docs"]+=1
    else: train_pos+=n;counts["train_docs"]+=1
    if i%1500==0:
        print("DATA_PROGRESS",json.dumps({"docs_seen":i,"train_tokens":train_pos,
            "valid_tokens":valid_pos,"seconds":round(time.time()-start,2)}),flush=True)
    if train_pos>=args.train_tokens and valid_pos>=args.valid_tokens:break
train.flush();valid.flush()
assert train_pos>=args.train_tokens and valid_pos>=args.valid_tokens,(train_pos,valid_pos)
metadata=dict(source=source,config="sample-10BT",tokenizer="tiktoken/gpt2",
    tokenizer_vocab=enc.n_vocab,train_tokens=train_pos,valid_tokens=valid_pos,
    split="document_id_sha256_mod11_0_validation_other_train",counts=counts,
    manifest_sha256=idhash.hexdigest(),elapsed_seconds=round(time.time()-start,2),
    train_sha256=hashlib.sha256((out/"train_tokens.uint16").read_bytes()).hexdigest(),
    valid_sha256=hashlib.sha256((out/"valid_tokens.uint16").read_bytes()).hexdigest())
(out/"corpus_manifest.json").write_text(json.dumps(metadata,indent=2))
print("CORPUS_READY",json.dumps(metadata,sort_keys=True),flush=True)
