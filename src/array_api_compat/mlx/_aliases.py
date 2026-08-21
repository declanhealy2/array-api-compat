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

    result = mx.asarray(obj, dtype=dtype, copy=copy)
    if device is None:
        return result

    # MLX uses unified memory; executing an explicit copy on the requested
    # device is the closest meaningful implementation of a creation-device
    # request without inventing per-array residency metadata.
    return mx.copy(result, stream=_stream(device))


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
    return mx.copy(result, stream=_stream(device))


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
        return mx.copy(x) if copy else x
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
    if isinstance(source, int):
        if not isinstance(destination, int):
            raise ValueError("source and destination must have the same number of axes")
        return mx.moveaxis(x, source, destination)

    if isinstance(destination, int):
        raise ValueError("source and destination must have the same number of axes")
    if len(source) != len(destination):
        raise ValueError("source and destination must have the same number of axes")

    normalized_source = tuple(_normalize_axis(axis, x.ndim) for axis in source)
    normalized_destination = tuple(
        _normalize_axis(axis, x.ndim) for axis in destination
    )
    if len(set(normalized_source)) != len(normalized_source):
        raise ValueError("repeated axis in source")
    if len(set(normalized_destination)) != len(normalized_destination):
        raise ValueError("repeated axis in destination")

    order = [axis for axis in range(x.ndim) if axis not in normalized_source]
    for destination_axis, source_axis in sorted(
        zip(normalized_destination, normalized_source),
    ):
        order.insert(destination_axis, source_axis)
    return mx.transpose(x, order)


def permute_dims(x: Array, axes: tuple[int, ...], /) -> Array:
    return mx.transpose(x, axes)


def repeat(
    x: Array,
    repeats: int | Array,
    /,
    *,
    axis: int | None = None,
) -> Array:
    if isinstance(repeats, int):
        return mx.repeat(x, repeats, axis=axis)
    if isinstance(repeats, mx.array) and repeats.ndim == 0:
        return mx.repeat(x, int(repeats.item()), axis=axis)
    raise NotImplementedError(
        "MLX cannot represent the data-dependent output shape produced by "
        "a non-scalar repeats array"
    )


def reshape(
    x: Array,
    shape: tuple[int, ...],
    /,
    *,
    copy: bool | None = None,
) -> Array:
    result = mx.reshape(x, shape)
    return mx.copy(result) if copy is True else result


def roll(
    x: Array,
    shift: int | tuple[int, ...],
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    return mx.roll(x, shift, axis=axis)


def squeeze(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    return mx.squeeze(x, axis=axis)


def stack(arrays: Sequence[Array], /, *, axis: int = 0) -> Array:
    return mx.stack(arrays, axis=axis)


def unstack(x: Array, /, *, axis: int = 0) -> tuple[Array, ...]:
    return tuple(mx.unstack(x, axis=axis))


def all(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    return mx.all(x, axis=axis, keepdims=keepdims)


def any(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    return mx.any(x, axis=axis, keepdims=keepdims)


def max(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    return mx.max(x, axis=axis, keepdims=keepdims)


def mean(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    return mx.mean(x, axis=axis, keepdims=keepdims)


def min(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    return mx.min(x, axis=axis, keepdims=keepdims)


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    if dtype is not None and x.dtype != dtype:
        x = mx.astype(x, dtype)
    return mx.prod(x, axis=axis, keepdims=keepdims)


def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    if not float(correction).is_integer():
        raise ValueError("MLX supports only integral correction values")
    return mx.std(x, axis=axis, keepdims=keepdims, ddof=int(correction))


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    if dtype is not None and x.dtype != dtype:
        x = mx.astype(x, dtype)
    return mx.sum(x, axis=axis, keepdims=keepdims)


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    if not float(correction).is_integer():
        raise ValueError("MLX supports only integral correction values")
    return mx.var(x, axis=axis, keepdims=keepdims, ddof=int(correction))


def argmax(
    x: Array,
    /,
    *,
    axis: int | None = None,
    keepdims: bool = False,
) -> Array:
    return mx.argmax(x, axis=axis, keepdims=keepdims).astype(mx.int32)


def argmin(
    x: Array,
    /,
    *,
    axis: int | None = None,
    keepdims: bool = False,
) -> Array:
    return mx.argmin(x, axis=axis, keepdims=keepdims).astype(mx.int32)


def count_nonzero(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Array:
    return mx.count_nonzero(x, axis=axis, keepdims=keepdims).astype(mx.int32)


def _descending_key(x: Array) -> Array:
    if x.dtype == mx.bool_:
        return mx.logical_not(x)
    if mx.issubdtype(x.dtype, mx.unsignedinteger):
        return mx.subtract(mx.array(mx.iinfo(x.dtype).max, dtype=x.dtype), x)
    return mx.negative(x)


def argsort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Array:
    # MLX sorting is stable; a stable sort also satisfies stable=False.
    key = _descending_key(x) if descending else x
    return mx.argsort(key, axis=axis).astype(mx.int32)


def sort(
    x: Array,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Array:
    result = mx.sort(x, axis=axis)
    return mx.flip(result, axis=axis) if descending else result


def take(x: Array, indices: Array, /, *, axis: int | None = None) -> Array:
    return mx.take(x, indices, axis=axis)


def take_along_axis(
    x: Array,
    indices: Array,
    /,
    *,
    axis: int,
) -> Array:
    return mx.take_along_axis(x, indices, axis=axis)


def clip(
    x: Array,
    /,
    min: int | float | Array | None = None,
    max: int | float | Array | None = None,
) -> Array:
    if min is None and max is None:
        raise ValueError("at least one of min or max must be specified")
    return mx.clip(x, min, max)


def tril(x: Array, /, *, k: int = 0) -> Array:
    return mx.tril(x, k=k)


def triu(x: Array, /, *, k: int = 0) -> Array:
    return mx.triu(x, k=k)


__all__ = [
    "all",
    "any",
    "arange",
    "argmax",
    "argmin",
    "argsort",
    "asarray",
    "astype",
    "broadcast_arrays",
    "broadcast_to",
    "clip",
    "concat",
    "count_nonzero",
    "cumulative_prod",
    "cumulative_sum",
    "diff",
    "empty",
    "empty_like",
    "expand_dims",
    "eye",
    "flip",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "matrix_transpose",
    "max",
    "mean",
    "meshgrid",
    "min",
    "moveaxis",
    "ones",
    "ones_like",
    "permute_dims",
    "prod",
    "repeat",
    "reshape",
    "roll",
    "sort",
    "squeeze",
    "stack",
    "std",
    "sum",
    "take",
    "take_along_axis",
    "tril",
    "triu",
    "unstack",
    "var",
    "zeros",
    "zeros_like",
]


def __dir__() -> list[str]:
    return __all__
