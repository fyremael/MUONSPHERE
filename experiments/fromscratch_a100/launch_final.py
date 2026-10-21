import pathlib,subprocess,sys
out=pathlib.Path('/content/gcl_s4_scratch_final');out.mkdir(exist_ok=True,parents=True)
manifest=pathlib.Path('/content/gcl_s4_scratch_corpus/corpus_manifest.json')
assert manifest.is_file(), 'FineWeb-Edu dataset not prepared yet'
cmd=[sys.executable,'/content/gcl_s4_scratch_train.py',
 '--data','/content/gcl_s4_scratch_corpus','--out',str(out),
 '--stage','random_init_s4_plus_full_replay',
 '--steps','4000','--batch','2','--accum','4','--context','512',
 '--peak-lr','0.0003','--warmup','100',
 '--eval-every','250','--eval-batches','4',
 '--seeds','101','--arms','s4_plus_penalty',
 '--lambda-s4','0.15','--lambda-column','0.15',
 '--long-eval','--wandb','offline']
with open(out/'run.log','w',buffering=1) as f:
 p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
print('FINAL_COMPARISON_DISPATCHED',p.pid,repr(cmd),flush=True)

