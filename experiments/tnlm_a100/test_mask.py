"""Test TNLM shift, loss and gradients, and fixed-channel partition."""
import importlib.util
import torch
import torch.nn.functional as F
spec=importlib.util.spec_from_file_location("tnlm","/content/gcl_tnlm_train.py")
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
x=torch.tensor([[50256,17,23,13,51],[50256,7,14,29,39]],device='cuda')
assert x.shape==(2,5)
full=m.tnlm_targets(x,'lm')
masked=m.tnlm_targets(x,'tnlm')
assert torch.equal(full,x)
assert torch.equal(masked[:,0],x[:,0])
assert torch.all(masked[:,1]==-100)
assert torch.equal(masked[:,2:],x[:,2:])
v=64
logits=torch.randn(2,5,v,device='cuda',requires_grad=True)
labels=x.clamp_max(v-1)
labels[:,0]=1
lm=F.cross_entropy(logits[:,:-1,:].reshape(-1,v),labels[:,1:].reshape(-1))
tnlm_labels=labels.clone();tnlm_labels[:,1]=-100
tnlm=F.cross_entropy(logits[:,:-1,:].reshape(-1,v),tnlm_labels[:,1:].reshape(-1),
                      ignore_index=-100)
manual=F.cross_entropy(logits[:,1:-1,:].reshape(-1,v),labels[:,2:].reshape(-1))
assert torch.allclose(tnlm,manual,rtol=1e-5),(tnlm,manual)
g=torch.autograd.grad(tnlm,logits)[0]
assert g[:,0,:].abs().max().item()==0
assert g[:,1:-1,:].abs().max().item()>1e-8
print('MASK_VALIDATED',{'lm_ce':lm.item(),'tnlm_ce':tnlm.item(),
 'first_query_grad_max':g[:,0,:].abs().max().item()},flush=True)
import types
from torch import nn
model=types.SimpleNamespace(
    config=types.SimpleNamespace(n_embd=768),
    transformer=types.SimpleNamespace(h=[
        types.SimpleNamespace(ln_1=nn.LayerNorm(768).cuda(),
                              ln_2=nn.LayerNorm(768).cuda())
        for _ in range(2)]))
hooks=m.install_force_sink_mask(model,16)
z=torch.randn(1,8,768,device='cuda')
capture={}
def inspect(mod,inp):
    capture['input']=inp[0].detach().clone()
h=model.transformer.h[0].ln_1.register_forward_pre_hook(inspect)
model.transformer.h[0].ln_1(z)
u=capture['input']
assert torch.all(u[:,0,16:]==0) and torch.all(u[:,1:,:16]==0)
assert torch.allclose(u[:,0,:16],z[:,0,:16])
assert torch.allclose(u[:,1:,16:],z[:,1:,16:])
h.remove()
for h in hooks:h.remove()
print('FORCE_MASK_VALIDATED',flush=True)
