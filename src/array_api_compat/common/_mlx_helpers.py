"""Lazy MLX integration for the public helper functions.

This module intentionally does not import MLX merely because
``array_api_compat`` is imported. MLX is imported only after an actual
``mlx.core.array`` or MLX namespace has been supplied.
"""

from __future__ import annotations

import sys
from types import ModuleType
from typing import Any

from . import _helpers as _base

_SCALAR_TYPES = (bool, int, float, complex, type(None))


def is_mlx_array(x: object) -> bool:
    """Return whether *x* is an MLX array without importing MLX."""
    module = sys.modules.get("mlx.core")
    if module is None:
        return False
    array_type = getattr(module, "array", None)
    return array_type is not None and isinstance(x, array_type)


def is_mlx_namespace(xp: ModuleType) -> bool:
    """Return whether *xp* is MLX or the array-api-compat MLX wrapper."""
    return xp.__name__ in {"mlx.core", "array_api_compat.mlx"}


def array_namespace(
    *xs: Any,
    api_version: str | None = None,
    use_compat: bool | None = None,
) -> ModuleType:
    """Return the Array API namespace, including the MLX compat wrapper."""
    mlx_inputs = [x for x in xs if is_mlx_array(x)]
    if not mlx_inputs:
        return _base.array_namespace(
            *xs,
            api_version=api_version,
            use_compat=use_compat,
        )

    for x in xs:
        if isinstance(x, _SCALAR_TYPES):
            continue
        if not is_mlx_array(x):
            raise TypeError("Multiple namespaces for array inputs: MLX and another backend")

    _base._check_api_version(api_version)
    if use_compat is False:
        import mlx.core as mx

        return mx

    from .. import mlx as mlx_compat

    return mlx_compat


get_namespace = array_namespace


def is_array_api_obj(x: object) -> bool:
    return is_mlx_array(x) or _base.is_array_api_obj(x)


def device(x: Any, /) -> Any:
    if not is_mlx_array(x):
        return _base.device(x)

    import mlx.core as mx

    # MLX arrays use unified memory and do not carry per-array residency.
    # The execution default is the only meaningful device value MLX exposes.
    return mx.default_device()


def to_device(
    x: Any,
    device: Any,
    /,
    *,
    stream: int | Any | None = None,
) -> Any:
    if not is_mlx_array(x):
        return _base.to_device(x, device, stream=stream)
    if stream is not None:
        raise NotImplementedError("MLX does not expose Array API stream handles")

    import mlx.core as mx

    if not isinstance(device, mx.Device):
        raise TypeError(f"expected an mlx.core.Device, got {type(device).__name__}")
    return mx.copy(x, stream=device)


def is_lazy_array(x: object) -> bool:
    if is_mlx_array(x):
        return True
    return _base.is_lazy_array(x)


__all__ = [
    "array_namespace",
    "device",
    "get_namespace",
    "is_array_api_obj",
    "is_lazy_array",
    "is_mlx_array",
    "is_mlx_namespace",
    "to_device",
]


def __dir__() -> list[str]:
    return __all__
