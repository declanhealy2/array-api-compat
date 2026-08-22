"""Array API compatibility shims for :mod:`mlx.core`.

The wrappers in this module deliberately stay small. They adapt signatures,
keyword names, return containers, and dtype behavior while leaving array
execution to MLX. They do not monkeypatch ``mlx.core.array`` and never route
array operations through NumPy.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import mlx.core as mx

from ..common._typing import NestedSequence, SupportsBufferProtocol
from ._info import _validate_device
from ._typing import Array, Device, DType


def _stream(device: Device | None) -> Device | None:
    if device is None:
        return None
    return _validate_device(device)


def _copy_array(x: Array, *, device: Device | None = None) -> Array:
    # ``mlx.core.copy`` is a C++ API but is not currently public in Python.
    # ``full_like`` with the input itself as the fill array creates a native
    # MLX copy and lets us select the execution device without any host
    # conversion.
    return mx.full_like(
        x,
        x,
        dtype=x.dtype,
        stream=_stream(device),
    )


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise IndexError(f"axis {axis} is out of bounds for an array of dimension {ndim}")
    return normalized


def asarray(
    obj: Array | complex | NestedSequence[complex] | SupportsBufferProtocol,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    copy: bool | None = None,
) -> Array:
    """Convert an object to an MLX array with Array API copy semantics."""
    if copy is False and device is not None:
        raise ValueError("MLX cannot guarantee copy=False for an explicit device")

    if copy is False and isinstance(obj, mx.array):
        if dtype is None or dtype == obj.dtype:
            return obj
        raise ValueError("Unable to avoid copy while changing the dtype")

    result = mx.asarray(obj, dtype=dtype, copy=copy)
    if device is None:
        return result

    # MLX uses unified memory; executing an explicit native copy on the
    # requested device is the closest meaningful implementation of a
    # creation-device request without inventing residency metadata.
    return _copy_array(result, device=device)


def from_dlpack(
    x: Any,
    /,
    *,
    device: Device | None = None,
    copy: bool | None = None,
) -> Array:
    if copy is False and device is not None:
        raise ValueError("MLX cannot guarantee copy=False for an explicit device")
    result = mx.from_dlpack(x, copy=copy)
    if device is None:
        return result
    return _copy_array(result, device=device)


def arange(
    start: int | float,
    /,
    stop: int | float | None = None,
    step: int | float = 1,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    return mx.arange(start, stop, step, dtype=dtype, stream=_stream(device))


def empty(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    return mx.empty(shape, dtype=dtype, stream=_stream(device))


def empty_like(
    x: Array,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    return mx.empty_like(x, dtype=dtype, stream=_stream(device))


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    columns = n_rows if n_cols is None else n_cols
    if n_rows < 0 or columns < 0:
        raise ValueError("negative dimensions are not allowed")
    if n_rows == 0 or columns == 0 or k >= columns or k <= -n_rows:
        return mx.zeros(
            (n_rows, columns),
            dtype=mx.float32 if dtype is None else dtype,
            stream=_stream(device),
        )
    return mx.eye(
        n_rows,
        columns,
        k,
        dtype=mx.float32 if dtype is None else dtype,
        stream=_stream(device),
    )


def full(
    shape: int | tuple[int, ...],
    fill_value: complex,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    return mx.full(
        shape,
        fill_value,
        dtype=dtype,
        stream=_stream(device),
    )


def full_like(
    x: Array,
    /,
    fill_value: complex,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    return mx.full_like(
        x,
        fill_value,
        dtype=dtype,
        stream=_stream(device),
    )


def linspace(
    start: int | float,
    stop: int | float,
    /,
    num: int,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: bool = True,
) -> Array:
    return mx.linspace(
        start,
        stop,
        num,
        endpoint,
        dtype,
        stream=_stream(device),
    )


def meshgrid(*arrays: Array, indexing: str = "xy") -> tuple[Array, ...]:
    return tuple(mx.meshgrid(*arrays, indexing=indexing))


def ones(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    return mx.ones(shape, dtype=dtype, stream=_stream(device))


def ones_like(
    x: Array,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    return mx.ones_like(x, dtype=dtype, stream=_stream(device))


def zeros(
    shape: int | tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    return mx.zeros(shape, dtype=dtype, stream=_stream(device))


def zeros_like(
    x: Array,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    return mx.zeros_like(x, dtype=dtype, stream=_stream(device))


def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: bool = True,
) -> Array:
    if x.dtype == dtype:
        return _copy_array(x) if copy else x
    return mx.astype(x, dtype)


def broadcast_arrays(*arrays: Array) -> tuple[Array, ...]:
    return tuple(mx.broadcast_arrays(*arrays))


def broadcast_to(x: Array, shape: tuple[int, ...], /) -> Array:
    return mx.broadcast_to(x, shape)


def _prepend_identity(
    result: Array,
    *,
    axis: int,
    identity: int,
) -> Array:
    shape = list(result.shape)
    shape[axis] = 1
    initial = mx.full(tuple(shape), identity, dtype=result.dtype)
    return mx.concatenate((initial, result), axis=axis)


def cumulative_prod(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Array:
    if axis is None:
        result = mx.cumprod(mx.reshape(x, (-1,)), axis=0, dtype=dtype)
        normalized_axis = 0
    else:
        normalized_axis = _normalize_axis(axis, x.ndim)
        result = mx.cumprod(x, axis=normalized_axis, dtype=dtype)
    if include_initial:
        result = _prepend_identity(result, axis=normalized_axis, identity=1)
    return result


def cumulative_sum(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Array:
    if axis is None:
        result = mx.cumsum(mx.reshape(x, (-1,)), axis=0, dtype=dtype)
        normalized_axis = 0
    else:
        normalized_axis = _normalize_axis(axis, x.ndim)
        result = mx.cumsum(x, axis=normalized_axis, dtype=dtype)
    if include_initial:
        result = _prepend_identity(result, axis=normalized_axis, identity=0)
    return result


def concat(arrays: Sequence[Array], /, *, axis: int | None = 0) -> Array:
    return mx.concatenate(arrays, axis=axis)


def diff(x: Array, /, *, axis: int = -1) -> Array:
    return mx.diff(x, axis=axis)


def expand_dims(x: Array, /, *, axis: int) -> Array:
    return mx.expand_dims(x, axis=axis)


def flip(x: Array, /, *, axis: int | tuple[int, ...] | None = None) -> Array:
    return mx.flip(x, axis=axis)


def matrix_transpose(x: Array, /) -> Array:
    if x.ndim < 2:
        raise ValueError("matrix_transpose requires an array with at least two dimensions")
    return mx.swapaxes(x, -1, -2)


def moveaxis(
    x: Array,
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
    /,
) -> Array:
    return mx.moveaxis(x, source, destination)
