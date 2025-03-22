from ._version import __version__  # noqa: I001

from .llmma import LLMMA
from .provider import Provider
from . import provider, result
from ._lazy import use_local_provider, PROVIDERS

__all__ = [
    "__version__",
    "LLMMA",
    "Provider",
    "use_local_provider",
    "provider",
    "result",
    "PROVIDERS",
]
