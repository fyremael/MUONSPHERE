from pathlib import Path
import os,torch
p=Path('/content/gcl_tnlm_fineweb/corpus_manifest.json')
print("PING_OK",p.exists(),torch.cuda.is_available(),torch.cuda.get_device_name(0),flush=True)
