from pathlib import Path
p=Path('/content/gcl_natural_sink_pilot')
print('FILES',[(x.name,x.stat().st_size) for x in p.glob('*')])
f=p/'run.log'
if f.exists():print('RECENT','\n'.join(f.read_text().splitlines()[-12:])[-3500:])
