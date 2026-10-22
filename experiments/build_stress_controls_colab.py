#!/usr/bin/env python3
"""Create standalone Colab adversarial metric regression from pinned sources."""
from pathlib import Path
root=Path(__file__).resolve().parent
program=(root/'attention_sink_s4_stress_colab.py').read_text()
program=program.rsplit('if __name__=="__main__":',1)[0]
tests=(root/'test_attention_sink_stress_adversaries.py').read_text()
tests=tests.replace('import attention_sink_s4_stress as stress','import types as _stypes\nstress=_stypes.SimpleNamespace(extra_metrics=extra_metrics)')
out=root/'test_attention_sink_stress_adversaries_colab.py'
out.write_text((program+'\n'+tests).rstrip()+'\n')
print(out,'bytes',out.stat().st_size)
