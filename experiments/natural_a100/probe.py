import sys, subprocess, os, time
import torch
print("TORCH",torch.__version__,flush=True)
print("A100?", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU", flush=True)
if torch.cuda.is_available(): print("MEM_GB",torch.cuda.get_device_properties(0).total_memory/1e9,flush=True)
for lib in ("datasets","tiktoken","wandb"):
 try:
  m=__import__(lib);print("EXISTS",lib,getattr(m,'__version__',''),flush=True)
 except ImportError:
  print("INSTALL",lib,flush=True)
  subprocess.run([sys.executable,"-m","pip","install","--disable-pip-version-check","-q",lib],check=True,timeout=200)
from datasets import load_dataset
import tiktoken
print("ENCODER",tiktoken.get_encoding("gpt2").n_vocab,flush=True)
print("LOADING_STREAM",flush=True)
ds=load_dataset("HuggingFaceFW/fineweb-edu",name="sample-10BT",split="train",streaming=True)
item=next(iter(ds))
print("DOCUMENT",item.keys(), "chars",len(item.get("text","")), "excerpt",item.get("text","")[:120].replace("\n"," "),flush=True)
