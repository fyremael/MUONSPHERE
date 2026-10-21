from pathlib import Path
p=Path('/content/gcl_natural_sink')
print('DATA_READY', (p/'corpus_manifest.json').exists())
print('FILES',[(x.name,x.stat().st_size) for x in p.glob('*')][:8])
if (p/'prepare.log').exists():
 print('LAST_LOG','\n'.join((p/'prepare.log').read_text().splitlines()[-10:]))
