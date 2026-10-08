"""An experimental bitsandbytes backend. Import does not start a runtime."""

__version__ = "0.1.0"


def register():
    """Register the supported XLA operators when bitsandbytes loads this plugin."""
    from .registration import register as register_kernels
    return register_kernels()
