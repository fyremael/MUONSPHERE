.PHONY: install test smoke-muon smoke-peft

install:
	pip install -e .[dev]

test:
	pytest -q

smoke-muon:
	python examples/train_muonsphere_demo.py --steps 5 --atomic_qkv_per_head

smoke-peft:
	python examples/train_peft_lora_demo.py --steps 5 --retract_every 1
