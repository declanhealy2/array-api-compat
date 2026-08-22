from typing import Final

from .._internal import clone_module

__all__ = clone_module("mlx.core", globals())

from . import _aliases, _overrides
from ._aliases import *  # type: ignore[assignment,no-redef] # noqa: F403
from ._overrides import *  # type: ignore[assignment,no-redef] # noqa: F403
from ._info import __array_namespace_info__

# Import the compatibility submodules explicitly so they replace the native
# ``mlx.core.fft`` and ``mlx.core.linalg`` objects cloned above.
fft = __import__(__spec__.parent + ".fft", fromlist=["fft"])
linalg = __import__(__spec__.parent + ".linalg", fromlist=["linalg"])

__array_api_version__: Final = "2025.12"

__all__ = sorted(
    set(__all__)
    | set(_aliases.__all__)
    | set(_overrides.__all__)
    | {
        "__array_api_version__",
        "__array_namespace_info__",
        "fft",
        "linalg",
    }
)


def __dir__() -> list[str]:
    return __all__
