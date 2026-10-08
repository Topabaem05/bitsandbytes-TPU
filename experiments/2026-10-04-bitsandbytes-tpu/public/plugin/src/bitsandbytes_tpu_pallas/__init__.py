"""Separate, unqualified public Pallas-forward backend. Import creates no runtime."""
__version__ = "0.1.1"
SOURCE_VARIANT = "pallas-forward-mosaic7-gather-bf16-fp32-v1"


def register():
    from .registration import register as register_backend
    return register_backend()
