"""Array API inspection support for MLX."""

from __future__ import annotations

from typing import TYPE_CHECKING

import mlx.core as mx

from ..common._typing import (
    Capabilities,
    DTypeKind,
    DTypesAny,
    DefaultDTypes,
)
from ._typing import Device

if TYPE_CHECKING:
    from ._typing import DType


def _gpu_available() -> bool:
    """Return whether an MLX GPU backend is available."""
    try:
        if mx.metal.is_available():
            return True
    except (AttributeError, RuntimeError):
        pass
    try:
        return bool(mx.cuda.is_available())
    except (AttributeError, RuntimeError):
        return False


def _validate_device(device: Device | None) -> Device:
    if device is None:
        return mx.default_device()
    if not isinstance(device, mx.Device):
        raise TypeError(f"expected an mlx.core.Device, got {type(device).__name__}")
    if device == mx.gpu and not _gpu_available():
        raise ValueError("the MLX GPU device is not available")
    return device


_ALL_DTYPES: dict[str, DType] = {
    "bool": mx.bool_,
    "int8": mx.int8,
    "int16": mx.int16,
    "int32": mx.int32,
    "int64": mx.int64,
    "uint8": mx.uint8,
    "uint16": mx.uint16,
    "uint32": mx.uint32,
    "uint64": mx.uint64,
    "float16": mx.float16,
    "float32": mx.float32,
    "float64": mx.float64,
    "complex64": mx.complex64,
}

_KIND_NAMES: dict[str, tuple[str, ...]] = {
    "bool": ("bool",),
    "signed integer": ("int8", "int16", "int32", "int64"),
    "unsigned integer": ("uint8", "uint16", "uint32", "uint64"),
    "integral": (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ),
    "real floating": ("float16", "float32", "float64"),
    "complex floating": ("complex64",),
    "numeric": (
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "complex64",
    ),
}


class __array_namespace_info__:
    """Inspection interface for :mod:`array_api_compat.mlx`."""

    __module__ = "array_api_compat.mlx"

    def capabilities(self) -> Capabilities:
        # MLX intentionally does not expose data-dependent output shapes.
        return {
            "boolean indexing": False,
            "data-dependent shapes": False,
            "max dimensions": None,  # type: ignore[typeddict-item]
        }

    def default_device(self) -> Device:
        return mx.default_device()

    def default_dtypes(self, *, device: Device | None = None) -> DefaultDTypes:
        _validate_device(device)
        return {
            "real floating": mx.float32,
            "complex floating": mx.complex64,
            "integral": mx.int32,
            "indexing": mx.int32,
        }

    def dtypes(
        self,
        *,
        device: Device | None = None,
        kind: DTypeKind | None = None,
    ) -> DTypesAny:
        selected_device = _validate_device(device)
        names: tuple[str, ...]
        if kind is None:
            names = tuple(_ALL_DTYPES)
        elif isinstance(kind, tuple):
            unknown = [item for item in kind if item not in _KIND_NAMES]
            if unknown:
                raise ValueError(f"unsupported kind: {unknown[0]!r}")
            names = tuple(
                dict.fromkeys(
                    name
                    for item in kind
                    for name in _KIND_NAMES[item]
                )
            )
        else:
            try:
                names = _KIND_NAMES[kind]
            except KeyError:
                raise ValueError(f"unsupported kind: {kind!r}") from None

        # MLX exposes float64 for CPU execution only.
        if selected_device == mx.gpu:
            names = tuple(name for name in names if name != "float64")
        return {name: _ALL_DTYPES[name] for name in names}

    def devices(self) -> tuple[Device, ...]:
        devices: list[Device] = [mx.cpu]
        if _gpu_available():
            devices.append(mx.gpu)
        return tuple(devices)


__all__ = ["__array_namespace_info__"]


def __dir__() -> list[str]:
    return __all__
