import torch,numpy as np
from transformers import GPT2Config,GPT2LMHeadModel
from retrieval_eval import measure_retrieval
c=GPT2Config(vocab_size=50257,n_positions=1024,n_ctx=1024,n_embd=768,n_layer=12,n_head=12, resid_pdrop=0.,attn_pdrop=0.,embd_pdrop=0.)
m=GPT2LMHeadModel(c).to('cuda').eval()
data=np.memmap('/content/gcl_tnlm_fineweb/valid_tokens.uint16',dtype=np.uint16,mode='r',shape=(4000000,))
print('SMOKE_RETRIEVAL',measure_retrieval(m,data,n=2,contexts=(512,1024)),flush=True)
