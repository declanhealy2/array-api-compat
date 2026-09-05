from typing import Final

import mlx.core as _mx

from .._internal import clone_module

bool_: Final = _mx.bool_
int8: Final = _mx.int8
int16: Final = _mx.int16
int32: Final = _mx.int32
int64: Final = _mx.int64
uint8: Final = _mx.uint8
uint16: Final = _mx.uint16
uint32: Final = _mx.uint32
uint64: Final = _mx.uint64
float32: Final = _mx.float32
float64: Final = _mx.float64
complex64: Final = _mx.complex64

__all__ = clone_module("mlx.core", globals())

from . import _aliases, _overrides
from ._aliases import *  # type: ignore[assignment,no-redef] # noqa: F403
from ._overrides import *  # type: ignore[assignment,no-redef] # noqa: F403
from ._info import __array_namespace_info__ as __array_namespace_info__

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
