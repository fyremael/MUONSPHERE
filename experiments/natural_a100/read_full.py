from pathlib import Path
p=Path('/content/gcl_natural_sink_full')
log=p/'run.log'
print('FILES',[(x.name,x.stat().st_size) for x in p.glob('*')][:12])
if log.exists():
 lines=log.read_text(errors='replace').splitlines()
 print('METRICS',sum(x.startswith('METRIC ') for x in lines),
       'STARTS',sum(x.startswith('ARM_STARTED ') for x in lines))
 print('TAIL','\n'.join(lines[-4:])[-1500:])
