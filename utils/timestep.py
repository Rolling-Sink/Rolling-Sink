"""Helpers for decisions that must stay outside compiled model graphs."""


def is_zero_timestep(timestep) -> bool:
    """Return whether a scalar timestep is zero as a host-side boolean.

    Inference pipelines call this before entering the diffusion model.  Keeping
    scalar extraction at that boundary prevents ``Tensor.item()`` from breaking
    a compiled model graph while preserving the existing zero-timestep
    semantics.
    """

    item = getattr(timestep, "item", None)
    value = item() if item is not None else timestep
    return bool(value == 0)
