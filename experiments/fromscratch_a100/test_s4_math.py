"""Exact Schatten-4 implementation, positive control, hostile relocation test."""
import importlib.util
import torch
p="/content/gcl_s4_scratch_train.py"
spec=importlib.util.spec_from_file_location("gclscratch",p)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
t=32
device="cuda" if torch.cuda.is_available() else "cpu"
mask=torch.ones(t,t,device=device).tril()
fair=mask/mask.sum(-1,keepdim=True)
z=2*torch.ones((1,1,t,t),device=device)
z=z.masked_fill(mask[None,None]==0,-1e9)
uniform=torch.softmax(z,-1)
a,b=m.attention_losses([uniform])
assert abs(a.item())<1e-10 and abs(b.item())<1e-10,(a,b)
q=torch.arange(t,device=device)
s=torch.zeros((t,t),device=device)
s[:,0]=1.
sink=s[None,None].clone()
s4,col=m.attention_losses([sink])
start=min(64,t//4)
idx=torch.linspace(start,t-1,steps=min(128,t-start),device=device).long().unique()
D=sink[:,:,idx,:]-fair[idx][None,None,:,:]
n=len(idx)
by_svd=torch.linalg.svdvals(D.reshape(n,t)).pow(4).sum()/(n*n)
assert torch.allclose(s4,by_svd,rtol=1e-4,atol=1e-5),(s4,by_svd)
assert s4.item()>0.05 and col.item()>0.01,(s4,col)
# Pathological sink changes first column but could simply relocate;
# the spectral and concentration penalties should also detect that.
reloc=torch.zeros_like(sink)
reloc[:,:,:,1]=1
reloc[:,:,0,0]=1
rs,rc=m.attention_losses([reloc])
assert rs.item()>0.05 and rc.item()>0.01,(rs,rc)
# Confirm gradients propagate through the penalty.
logits=torch.randn(1,1,t,t,device=device,requires_grad=True)
x=torch.softmax(logits.masked_fill(mask[None,None]==0,-1e9),-1)
pen,cp=m.attention_losses([x])
(pen+cp).backward()
assert logits.grad is not None and torch.isfinite(logits.grad).all()
assert logits.grad.abs().max().item()>0
print("S4_TEST_PASSED",{
 "uniform_s4":a.item(),"uniform_column":b.item(),
 "sink_s4":s4.item(),"sink_s4_by_svd":by_svd.item(),
 "sink_column":col.item(),"relocated_s4":rs.item(),
 "relocated_column":rc.item(),
 "grad_abs_max":logits.grad.abs().max().item(),
})
