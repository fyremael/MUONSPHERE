"""MUONSPHERE: spectral-sphere optimizer and PEFT constraints for transformer experiments."""

from .atom import Atomic2DView, AtomBuilderRegistry
from .optim import MuonSphere, MuonSphereStats, msign_polar
from .peft import LoRALinear, PeftSpectralConstraintManager, inject_lora

__all__ = [
    "Atomic2DView",
    "AtomBuilderRegistry",
    "MuonSphere",
    "MuonSphereStats",
    "msign_polar",
    "LoRALinear",
    "PeftSpectralConstraintManager",
    "inject_lora",
]

__version__ = "0.1.0"
