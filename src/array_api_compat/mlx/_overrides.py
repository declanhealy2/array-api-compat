"""Focused Array API semantic overrides for the MLX compatibility namespace."""

from __future__ import annotations

import math
from collections.abc import Sequence
from itertools import zip_longest
from typing import Any, NamedTuple

import mlx.core as mx

from ._aliases import _copy_array
from ._info import _validate_device
from ._typing import Array, Device, DType

_DTYPE_TYPE = type(mx.float32)
_INT_BITS = {
    mx.int8: 8,
    mx.uint8: 8,
    mx.int16: 16,
    mx.uint16: 16,
    mx.int32: 32,
    mx.uint32: 32,
    mx.int64: 64,
    mx.uint64: 64,
}


class FInfo(NamedTuple):
    bits: int
    eps: float
    max: float
    min: float
    smallest_normal: float
    dtype: DType


class IInfo(NamedTuple):
    bits: int
    max: int
    min: int
    dtype: DType


class UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array


class UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array


class UniqueInverseResult(NamedTuple):
    values: Array
    inverse_indices: Array


def _dtype_of(value: Any) -> DType:
    if isinstance(value, mx.array):
        return value.dtype
    if isinstance(value, _DTYPE_TYPE):
        return value
    raise TypeError(f"expected an MLX array or dtype, got {type(value).__name__}")


def _as_array(value: Any, *, like: Array | None = None) -> Array:
    if isinstance(value, mx.array):
        return value
    return mx.asarray(value, dtype=None if like is None else like.dtype)


def _default_reduction_dtype(dtype: DType) -> DType:
    if dtype == mx.bool_ or mx.issubdtype(dtype, mx.signedinteger):
        return mx.int32
    if mx.issubdtype(dtype, mx.unsignedinteger):
        return mx.uint32
    return dtype


def _reduction_size(x: Array, axis: int | tuple[int, ...] | None) -> int:
    if axis is None:
        return math.prod(x.shape)
    axes = (axis,) if isinstance(axis, int) else axis
    normalized = tuple(a + x.ndim if a < 0 else a for a in axes)
    return math.prod(x.shape[a] for a in normalized)


def astype(
    x: Array,
    dtype: DType,
    /,
    *,
    copy: bool = True,
    device: Device | None = None,
) -> Array:
    target_device = None if device is None else _validate_device(device)
    if x.dtype == dtype:
        if not copy and target_device is None:
            return x
        return _copy_array(x, device=target_device)
    result = mx.astype(x, dtype)
    return result if target_device is None else _copy_array(result, device=target_device)


def broadcast_shapes(*shapes: tuple[int, ...]) -> tuple[int, ...]:
    if not shapes:
        return ()
    result: list[int] = []
    for dimensions in zip_longest(*(reversed(shape) for shape in shapes), fillvalue=1):
        output = max(dimensions)
        if any(dimension not in (1, output) for dimension in dimensions):
            raise ValueError(f"shapes {shapes!r} are not broadcastable")
        result.append(output)
    return tuple(reversed(result))


def broadcast_to(x: Array, /, shape: tuple[int, ...]) -> Array:
    return mx.broadcast_to(x, shape)


def can_cast(from_: DType | Array, to: DType, /) -> bool:
    source = _dtype_of(from_)
    try:
        return result_type(source, to) == to
    except ValueError:
        return False


def finfo(type_: DType | Array, /) -> FInfo:
    dtype = _dtype_of(type_)
    real_dtype = mx.float32 if dtype == mx.complex64 else dtype
    info = mx.finfo(real_dtype)
    bits = 32 if real_dtype == mx.float32 else 64
    smallest_normal = getattr(info, "smallest_normal", getattr(info, "tiny", None))
    return FInfo(
        bits=bits,
        eps=float(info.eps),
        max=float(info.max),
        min=float(info.min),
        smallest_normal=float(smallest_normal),
        dtype=real_dtype,
    )


def iinfo(type_: DType | Array, /) -> IInfo:
    dtype = _dtype_of(type_)
    info = mx.iinfo(dtype)
    return IInfo(
        bits=_INT_BITS[dtype],
        max=int(info.max),
        min=int(info.min),
        dtype=dtype,
    )


def result_type(*arrays_and_dtypes: Any) -> DType:
    if not arrays_and_dtypes:
        raise ValueError("result_type requires at least one argument")
    strong: list[DType] = []
    weak: list[Any] = []
    for value in arrays_and_dtypes:
        if isinstance(value, mx.array):
            strong.append(value.dtype)
        elif isinstance(value, _DTYPE_TYPE):
            strong.append(value)
        elif isinstance(value, (bool, int, float, complex)):
            weak.append(value)
        else:
            raise TypeError(f"unsupported result_type input {type(value).__name__}")
    if strong:
        return mx.result_type(*strong)
    return mx.result_type(*(mx.asarray(value).dtype for value in weak))


def cumulative_sum(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Array:
    if axis is None:
        if x.ndim > 1:
            raise ValueError("axis must be specified for arrays with more than one dimension")
        axis = 0
    output_dtype = _default_reduction_dtype(x.dtype) if dtype is None else dtype
    result = mx.cumsum(x, axis=axis, dtype=output_dtype)
    if include_initial:
        shape = list(result.shape)
        shape[axis] = 1
        result = mx.concatenate(
            (mx.zeros(tuple(shape), dtype=result.dtype), result), axis=axis
        )
    return result


def cumulative_prod(
    x: Array,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Array:
    if axis is None:
        if x.ndim > 1:
            raise ValueError("axis must be specified for arrays with more than one dimension")
        axis = 0
    output_dtype = _default_reduction_dtype(x.dtype) if dtype is None else dtype
    result = mx.cumprod(x, axis=axis, dtype=output_dtype)
    if include_initial:
        shape = list(result.shape)
        shape[axis] = 1
        result = mx.concatenate(
            (mx.ones(tuple(shape), dtype=result.dtype), result), axis=axis
        )
    return result


def sum(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    output_dtype = _default_reduction_dtype(x.dtype) if dtype is None else dtype
    if x.dtype != output_dtype:
        x = mx.astype(x, output_dtype)
    return mx.sum(x, axis=axis, keepdims=keepdims)


def prod(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Array:
    output_dtype = _default_reduction_dtype(x.dtype) if dtype is None else dtype
    if x.dtype != output_dtype:
        x = mx.astype(x, output_dtype)
    return mx.prod(x, axis=axis, keepdims=keepdims)


def var(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    base = mx.var(x, axis=axis, keepdims=keepdims, ddof=0)
    count = _reduction_size(x, axis)
    numerator = mx.array(count, dtype=base.dtype)
    denominator = mx.array(count - correction, dtype=base.dtype)
    return base * (numerator / denominator)


def std(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Array:
    return mx.sqrt(var(x, axis=axis, correction=correction, keepdims=keepdims))


def diff(
    x: Array,
    /,
    n: int = 1,
    axis: int = -1,
    prepend: Array | None = None,
    append: Array | None = None,
) -> Array:
    if n < 0:
        raise ValueError("n must be non-negative")
    result = x
    if prepend is not None:
        result = mx.concatenate((prepend, result), axis=axis)
    if append is not None:
        result = mx.concatenate((result, append), axis=axis)
    for _ in range(n):
        result = mx.diff(result, axis=axis)
    return result


def expand_dims(
    x: Array,
    /,
    axis: int | tuple[int, ...],
) -> Array:
    axes = (axis,) if isinstance(axis, int) else axis
    output_ndim = x.ndim + len(axes)
    normalized = tuple(a + output_ndim if a < 0 else a for a in axes)
    if any(a < 0 or a >= output_ndim for a in normalized):
        raise IndexError("axis is out of bounds")
    if len(set(normalized)) != len(normalized):
        raise ValueError("repeated axis")
    source = iter(x.shape)
    shape = tuple(1 if i in normalized else next(source) for i in range(output_ndim))
    return mx.reshape(x, shape)


def permute_dims(x: Array, /, axes: tuple[int, ...]) -> Array:
    return mx.transpose(x, axes)


def reshape(
    x: Array,
    /,
    shape: tuple[int, ...],
    *,
    copy: bool | None = None,
) -> Array:
    result = mx.reshape(x, shape)
    return _copy_array(result) if copy is True else result


def roll(
    x: Array,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Array:
    return mx.roll(x, shift, axis=axis)


def squeeze(x: Array, /, axis: int | tuple[int, ...]) -> Array:
    axes = (axis,) if isinstance(axis, int) else axis
    normalized = tuple(a + x.ndim if a < 0 else a for a in axes)
    if any(a < 0 or a >= x.ndim for a in normalized):
        raise IndexError("axis is out of bounds")
    if len(set(normalized)) != len(normalized):
        raise ValueError("repeated axis")
    if any(x.shape[a] != 1 for a in normalized):
        raise ValueError("cannot squeeze an axis whose size is not one")
    shape = tuple(size for index, size in enumerate(x.shape) if index not in normalized)
    return mx.reshape(x, shape)


def take_along_axis(
    x: Array,
    indices: Array,
    /,
    *,
    axis: int = -1,
) -> Array:
    return mx.take_along_axis(x, indices, axis=axis)


def searchsorted(
    x1: Array,
    x2: Array | int | float,
    /,
    *,
    side: str = "left",
    sorter: Array | None = None,
) -> Array:
    sequence = x1 if sorter is None else mx.take(x1, sorter)
    values = _as_array(x2, like=x1)
    return mx.searchsorted(sequence, values, side=side).astype(mx.int32)


def isin(
    x1: Array,
    x2: Array | int | float | complex | bool,
    /,
    *,
    assume_unique: bool = False,
    invert: bool = False,
) -> Array:
    del assume_unique
    values = _as_array(x2, like=x1).reshape((-1,))
    if values.size == 0:
        result = mx.zeros(x1.shape, dtype=mx.bool_)
    else:
        result = mx.any(x1[..., None] == values, axis=-1)
    return mx.logical_not(result) if invert else result


def nonzero(x: Array, /) -> tuple[Array, ...]:
    raise NotImplementedError("MLX cannot represent data-dependent nonzero shapes")


def unique_all(x: Array, /) -> UniqueAllResult:
    raise NotImplementedError("MLX cannot represent data-dependent unique shapes")


def unique_counts(x: Array, /) -> UniqueCountsResult:
    raise NotImplementedError("MLX cannot represent data-dependent unique shapes")


def unique_inverse(x: Array, /) -> UniqueInverseResult:
    raise NotImplementedError("MLX cannot represent data-dependent unique shapes")


def unique_values(x: Array, /) -> Array:
    raise NotImplementedError("MLX cannot represent data-dependent unique shapes")


def clip(
    x: Array,
    /,
    min: int | float | Array | None = None,
    max: int | float | Array | None = None,
) -> Array:
    if min is None and max is None:
        return _copy_array(x)
    return mx.clip(x, min, max)


def atan2(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    if isinstance(x1, mx.array):
        return mx.arctan2(x1, _as_array(x2, like=x1))
    if isinstance(x2, mx.array):
        return mx.arctan2(_as_array(x1, like=x2), x2)
    return mx.arctan2(mx.asarray(x1), mx.asarray(x2))


def hypot(x1: Array | int | float, x2: Array | int | float, /) -> Array:
    if isinstance(x1, mx.array):
        a, b = x1, _as_array(x2, like=x1)
    elif isinstance(x2, mx.array):
        a, b = _as_array(x1, like=x2), x2
    else:
        a, b = mx.asarray(x1), mx.asarray(x2)
    return mx.sqrt(a * a + b * b)


def copysign(x1: Array, x2: Array | int | float, /) -> Array:
    raise NotImplementedError("MLX does not yet expose IEEE sign-bit operations")


def nextafter(x1: Array, x2: Array | int | float, /) -> Array:
    raise NotImplementedError("MLX does not yet expose nextafter")


def signbit(x: Array, /) -> Array:
    raise NotImplementedError("MLX does not yet expose IEEE sign-bit operations")


def expm1(x: Array, /) -> Array:
    if x.dtype == mx.complex64:
        return mx.exp(x) - mx.ones_like(x)
    return mx.expm1(x)


def sign(x: Array, /) -> Array:
    if x.dtype == mx.complex64:
        magnitude = mx.abs(x)
        return mx.where(magnitude == 0, mx.zeros_like(x), x / magnitude)
    result = mx.sign(x)
    if mx.issubdtype(x.dtype, mx.floating):
        result = mx.where(mx.isnan(x), x, result)
    return result


def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    result = mx.left_shift(x1, x2)
    return mx.where(x2 >= _INT_BITS[x1.dtype], mx.zeros_like(result), result)


def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    result = mx.right_shift(x1, x2)
    width = _INT_BITS[x1.dtype]
    if mx.issubdtype(x1.dtype, mx.signedinteger):
        fill = mx.where(x1 < 0, -mx.ones_like(result), mx.zeros_like(result))
    else:
        fill = mx.zeros_like(result)
    return mx.where(x2 >= width, fill, result)


def floor_divide(x1: Array, x2: Array, /) -> Array:
    quotient = mx.floor_divide(x1, x2)
    if mx.issubdtype(x1.dtype, mx.integer):
        remainder = x1 - quotient * x2
        adjust = (remainder != 0) & ((x1 < 0) != (x2 < 0))
        quotient = mx.where(adjust, quotient - 1, quotient)
    return quotient


def remainder(x1: Array, x2: Array, /) -> Array:
    result = x1 - floor_divide(x1, x2) * x2
    positive_zero = mx.zeros_like(result)
    negative_zero = -positive_zero
    return mx.where(
        result == 0,
        mx.where(x2 < 0, negative_zero, positive_zero),
        result,
    )


__all__ = [
    "FInfo",
    "IInfo",
    "UniqueAllResult",
    "UniqueCountsResult",
    "UniqueInverseResult",
    "astype",
    "atan2",
    "bitwise_left_shift",
    "bitwise_right_shift",
    "broadcast_shapes",
    "broadcast_to",
    "can_cast",
    "clip",
    "copysign",
    "cumulative_prod",
    "cumulative_sum",
    "diff",
    "expand_dims",
    "expm1",
    "finfo",
    "floor_divide",
    "hypot",
    "iinfo",
    "isin",
    "nextafter",
    "nonzero",
    "permute_dims",
    "prod",
    "remainder",
    "reshape",
    "result_type",
    "roll",
    "searchsorted",
    "sign",
    "signbit",
    "squeeze",
    "std",
    "sum",
    "take_along_axis",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "var",
]
