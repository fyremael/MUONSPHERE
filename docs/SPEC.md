# MUONSPHERE Technical Spec

## Purpose

MUONSPHERE packages the nGPT/MODULUS spectral-operator control path into a practitioner scaffold.

The core invariant is:

```text
||W||_2 approximately R, where R = c * sqrt(d_out / d_in)
```

For PEFT, the monitored operator is:

```text
W_eff = W0 + s * Delta_W
```

## Components

### Atomic2DView

A live view into a 2D parameter or parameter slice. It exposes view(), grad_view(), and key().

### AtomBuilderRegistry

Architecture-specific mapping from model modules to atomic operators. This keeps the optimizer architecture-free.

### MuonSphere

Per atom, MuonSphere performs:

1. power iteration for top singular triplet;
2. pre-update spectral retraction;
3. Muon-style polar update;
4. optional tangent correction;
5. post-update spectral retraction.

### PEFT spectral manager

The PEFT path starts with LoRA. PeftSpectralConstraintManager estimates the spectral norm of W_eff using matvec/rmatvec and reduces delta_scale when the effective operator exceeds its configured radius.

## Non-goals

This is not yet a full distributed-training library. It is a working scaffold intended for controlled experiments, ablation, and integration into mature nGPT/MODULUS trainers.
