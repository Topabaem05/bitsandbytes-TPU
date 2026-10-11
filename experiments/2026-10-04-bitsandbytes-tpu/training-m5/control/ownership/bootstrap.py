"""Portable exception-only bootstrap; legacy parent stop is deliberately disabled."""
class ProviderError(RuntimeError):
    pass

def stop_group(process):
    raise RuntimeError("Legacy cleanup forbidden; use the exact V5 Ownership override")
