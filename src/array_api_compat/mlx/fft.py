"""Array API FFT namespace for MLX."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeAlias

import mlx.core as mx

from .._internal import clone_module
from ._aliases import _copy_array
from ._typing import Array, Device, DType

__all__ = clone_module("mlx.core.fft", globals())

_Norm: TypeAlias = Literal["backward", "ortho", "forward"]


def fft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    return mx.fft.fft(x, n=n, axis=axis, norm=norm)


def ifft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    return mx.fft.ifft(x, n=n, axis=axis, norm=norm)


def fftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    return mx.fft.fftn(x, s=s, axes=axes, norm=norm)


def ifftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    return mx.fft.ifftn(x, s=s, axes=axes, norm=norm)


def rfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    return mx.fft.rfft(x, n=n, axis=axis, norm=norm)


def irfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    return mx.fft.irfft(x, n=n, axis=axis, norm=norm)


def rfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    return mx.fft.rfftn(x, s=s, axes=axes, norm=norm)


def irfftn(
    x: Array,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: _Norm = "backward",
) -> Array:
    return mx.fft.irfftn(x, s=s, axes=axes, norm=norm)


def _opposite_norm(norm: _Norm) -> _Norm:
    if norm == "backward":
        return "forward"
    if norm == "forward":
        return "backward"
    return "ortho"


def hfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    if n is None:
        n = 2 * (x.shape[axis] - 1)
    return mx.fft.irfft(
        mx.conjugate(x),
        n=n,
        axis=axis,
        norm=_opposite_norm(norm),
    )


def ihfft(
    x: Array,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: _Norm = "backward",
) -> Array:
    return mx.conjugate(
        mx.fft.rfft(
            x,
            n=n,
            axis=axis,
            norm=_opposite_norm(norm),
        )
    )


def fftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    result = mx.fft.fftfreq(n, d=d)
    if dtype is not None:
        result = result.astype(dtype)
    if device is not None:
        result = _copy_array(result, device=device)
    return result


def rfftfreq(
    n: int,
    /,
    *,
    d: float = 1.0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Array:
    result = mx.fft.rfftfreq(n, d=d)
    if dtype is not None:
        result = result.astype(dtype)
    if device is not None:
        result = _copy_array(result, device=device)
    return result


def fftshift(
    x: Array,
    /,
    *,
    axes: int | Sequence[int] | None = None,
) -> Array:
    return mx.fft.fftshift(x, axes=axes)


def ifftshift(
    x: Array,
    /,
    *,
    axes: int | Sequence[int] | None = None,
) -> Array:
    return mx.fft.ifftshift(x, axes=axes)


__all__ = sorted(
    set(__all__)
    | {
        "fft",
        "ifft",
        "fftn",
        "ifftn",
        "rfft",
        "irfft",
        "rfftn",
        "irfftn",
        "hfft",
        "ihfft",
        "fftfreq",
        "rfftfreq",
        "fftshift",
        "ifftshift",
    }
)


def __dir__() -> list[str]:
    return __all__
