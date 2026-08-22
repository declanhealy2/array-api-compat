"""Array API linear-algebra namespace for MLX."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal, NamedTuple

import mlx.core as mx

from .._internal import clone_module
from ._aliases import matrix_transpose
from ._typing import Array, DType

__all__ = clone_module("mlx.core.linalg", globals())


class EighResult(NamedTuple):
    eigenvalues: Array
    eigenvectors: Array


class QRResult(NamedTuple):
    Q: Array
    R: Array


class SlogdetResult(NamedTuple):
    sign: Array
    logabsdet: Array


class SVDResult(NamedTuple):
    U: Array
    S: Array
    Vh: Array


def cholesky(x: Array, /, *, upper: bool = False) -> Array:
    return mx.linalg.cholesky(x, upper=upper, stream=mx.cpu)


def cross(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    return mx.cross(x1, x2, axis=axis)


def diagonal(x: Array, /, *, offset: int = 0) -> Array:
    return mx.diagonal(x, offset=offset, axis1=-2, axis2=-1)


def eigh(x: Array, /) -> EighResult:
    return EighResult(*mx.linalg.eigh(x, stream=mx.cpu))


def matrix_norm(
    x: Array,
    /,
    *,
    keepdims: bool = False,
    ord: int | float | Literal["fro", "nuc"] = "fro",
) -> Array:
    return mx.linalg.norm(
        x,
        ord=ord,
        axis=(-2, -1),
        keepdims=keepdims,
    )


def matrix_rank(
    x: Array,
    /,
    *,
    rtol: float | Array | None = None,
) -> Array:
    if x.ndim < 2:
        raise ValueError("matrix_rank requires an array with at least two dimensions")
    singular_values = svdvals(x)
    largest = mx.max(singular_values, axis=-1, keepdims=True)
    if rtol is None:
        threshold = largest * max(x.shape[-2:]) * mx.finfo(
            singular_values.dtype
        ).eps
    else:
        threshold = largest * mx.asarray(rtol)[..., None]
    return mx.count_nonzero(
        singular_values > threshold,
        axis=-1,
    ).astype(mx.int32)


def outer(x1: Array, x2: Array, /) -> Array:
    return mx.outer(x1, x2)


def pinv(
    x: Array,
    /,
    *,
    rtol: float | Array | None = None,
) -> Array:
    if rtol is None:
        return mx.linalg.pinv(x, stream=mx.cpu)

    u, singular_values, vh = mx.linalg.svd(x, stream=mx.cpu)
    largest = mx.max(singular_values, axis=-1, keepdims=True)
    cutoff = largest * mx.asarray(rtol)[..., None]
    reciprocal = mx.where(
        singular_values > cutoff,
        mx.reciprocal(singular_values),
        mx.zeros_like(singular_values),
    )
    v = mx.conjugate(matrix_transpose(vh))
    uh = mx.conjugate(matrix_transpose(u))
    return (v * reciprocal[..., None, :]) @ uh


def qr(
    x: Array,
    /,
    *,
    mode: Literal["reduced", "complete"] = "reduced",
) -> QRResult:
    if mode not in ("reduced", "complete"):
        raise ValueError("mode must be 'reduced' or 'complete'")
    if mode == "complete" and x.shape[-2] > x.shape[-1]:
        raise NotImplementedError(
            "MLX currently provides reduced QR for tall matrices only"
        )
    return QRResult(*mx.linalg.qr(x, stream=mx.cpu))


def slogdet(x: Array, /) -> SlogdetResult:
    return SlogdetResult(*mx.linalg.slogdet(x, stream=mx.cpu))


def svd(
    x: Array,
    /,
    *,
    full_matrices: bool = True,
) -> SVDResult:
    if full_matrices and x.shape[-2] != x.shape[-1]:
        raise NotImplementedError(
            "MLX currently provides reduced SVD for rectangular matrices"
        )
    return SVDResult(*mx.linalg.svd(x, stream=mx.cpu))


def svdvals(x: Array, /) -> Array:
    return mx.linalg.svd(x, compute_uv=False, stream=mx.cpu)


def tensordot(
    x1: Array,
    x2: Array,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> Array:
    return mx.tensordot(x1, x2, axes=axes)


def trace(
    x: Array,
    /,
    *,
    offset: int = 0,
    dtype: DType | None = None,
) -> Array:
    return mx.trace(
        x,
        offset=offset,
        axis1=-2,
        axis2=-1,
        dtype=dtype,
    )


def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    return mx.vecdot(x1, x2, axis=axis)


def _normalize_axes(
    axis: int | tuple[int, ...] | None,
    ndim: int,
) -> tuple[int, ...]:
    if axis is None:
        axes = tuple(range(ndim))
    elif isinstance(axis, int):
        axes = (axis,)
    else:
        axes = axis

    normalized: list[int] = []
    for item in axes:
        current = item + ndim if item < 0 else item
        if current < 0 or current >= ndim:
            raise IndexError(
                f"axis {item} is out of bounds for an array of dimension {ndim}"
            )
        normalized.append(current)
    if len(set(normalized)) != len(normalized):
        raise ValueError("repeated axis")
    return tuple(normalized)


def vector_norm(
    x: Array,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: int | float = 2,
) -> Array:
    axes = _normalize_axes(axis, x.ndim)
    remaining = tuple(index for index in range(x.ndim) if index not in axes)

    if not axes:
        result = mx.abs(x)
    else:
        order = remaining + axes
        permuted = mx.transpose(x, order) if order != tuple(range(x.ndim)) else x
        reduced_size = math.prod(x.shape[index] for index in axes)
        reduced_shape = tuple(x.shape[index] for index in remaining) + (
            reduced_size,
        )
        flattened = mx.reshape(permuted, reduced_shape)
        result = mx.linalg.norm(flattened, ord=ord, axis=-1)

    if not keepdims:
        return result

    target_shape = [
        1 if index in axes else x.shape[index] for index in range(x.ndim)
    ]
    return mx.reshape(result, tuple(target_shape))


__all__ = sorted(
    set(__all__)
    | {
        "EighResult",
        "QRResult",
        "SVDResult",
        "SlogdetResult",
        "cholesky",
        "cross",
        "diagonal",
        "eigh",
        "matrix_norm",
        "matrix_rank",
        "matrix_transpose",
        "outer",
        "pinv",
        "qr",
        "slogdet",
        "svd",
        "svdvals",
        "tensordot",
        "trace",
        "vecdot",
        "vector_norm",
    }
)


def __dir__() -> list[str]:
    return __all__
