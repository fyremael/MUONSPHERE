import sys,subprocess
try:
 import transformers
 print("TRANSFORMERS",transformers.__version__)
except ImportError:
 subprocess.run([sys.executable,"-m","pip","install","-q","transformers>=4.48,<5"],check=True,timeout=180)
from transformers import GPT2LMHeadModel
print("LOADING openai-community/gpt2",flush=True)
m=GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
print("GPT2_READY",sum(p.numel() for p in m.parameters()),m.config.n_layer,m.config.n_embd,flush=True)
