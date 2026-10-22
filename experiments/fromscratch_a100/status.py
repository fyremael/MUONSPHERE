from pathlib import Path
import json
for folder in ("/content/gcl_s4_scratch_corpus","/content/gcl_s4_scratch_pilot","/content/gcl_s4_scratch_full"):
 p=Path(folder)
 print("FOLDER",folder, [(x.name,x.stat().st_size) for x in p.glob('*') if x.is_file()][:8],flush=True)
 for name in ("prepare.log","run.log"):
  f=p/name
  if f.exists():
   logs=f.read_text(errors="replace").splitlines()
   print(name,"lines",len(logs),"LAST",*logs[-3:],sep="\n",flush=True)
