import pathlib,zipfile,json
p=pathlib.Path('/content/gcl_fsm_s4only_direct')
assert json.loads((p/'run_status.json').read_text())['complete']
out=pathlib.Path('/content/gcl_fsm_s4only_private.zip')
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for x in p.rglob('*'):
  if x.is_file():z.write(x,x.relative_to('/content'))
 z.write('/content/gcl_tnlm_fineweb/corpus_manifest.json','corpus_manifest.json')
print('ARCHIVE',out.stat().st_size)
