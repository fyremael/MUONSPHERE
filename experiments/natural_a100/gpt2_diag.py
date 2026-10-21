import torch, json
from transformers import GPT2LMHeadModel
m=GPT2LMHeadModel.from_pretrained("openai-community/gpt2",attn_implementation="eager")
print("MODEL", sum(p.numel() for p in m.parameters()))
print("ATTENTION",m.transformer.h[0].attn.__class__.__name__,m.config._attn_implementation)
print("NAMES",[n for n,_ in list(m.named_parameters())[:25]])
m.to("cuda")
with torch.no_grad():
 out=m(torch.randint(0,50257,(1,128),device="cuda"),output_attentions=True,return_dict=True)
print("SHAPES",tuple(out.logits.shape),len(out.attentions),tuple(out.attentions[0].shape))
a=torch.stack(out.attentions,0)
print("BOS_ATT",a[:,:,:,32:,0].mean().item(),"BOS_FIRSTHEAD",a[0,0,0,32:,0].mean().item())
del m
torch.cuda.empty_cache()
