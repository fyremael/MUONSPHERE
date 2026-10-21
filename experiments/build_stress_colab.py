#!/usr/bin/env python3
"""Generate the single-file Colab entrypoint from the two versioned modules."""
from pathlib import Path
root = Path(__file__).resolve().parent
base = (root / 'attention_sink_s4.py').read_text()
stress = (root / 'attention_sink_s4_stress.py').read_text()
assert 'if __name__=="__main__":' in base
base = base.split('if __name__=="__main__":')[0]
assert 'import attention_sink_s4 as baseline' in stress
stress = stress.replace('import attention_sink_s4 as baseline', '# Baseline frozen below as a local namespace.')
header = '''\nimport types as _gcl_types\nbaseline = _gcl_types.SimpleNamespace(\n ARMS=ARMS, VOCAB=VOCAB, BOS=BOS, SEP=SEP, EOS=EOS,\n set_seed=set_seed, sample=sample, Muon=Muon, self_test=self_test,\n fourth_order_penalty=fourth_order_penalty, get_wandb=get_wandb)\n'''
# Full module has two headers, but Python imports are idempotent.
out = root / 'attention_sink_s4_stress_colab.py'
out.write_text((base + header + '\n' + stress).rstrip()+'\n')
print(out, 'bytes', out.stat().st_size)
