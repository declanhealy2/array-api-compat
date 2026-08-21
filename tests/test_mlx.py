import inspect

import pytest

mx = pytest.importorskip("mlx.core")

import array_api_compat
import array_api_compat.mlx as xp

mx.set_default_device(mx.cpu)


def test_namespace_dispatch():
    x = mx.arange(3)
    assert array_api_compat.array_namespace(x) is xp
    assert array_api_compat.array_namespace(x, use_compat=True) is xp
    assert array_api_compat.array_namespace(x, use_compat=False) is mx
    assert array_api_compat.is_mlx_array(x)
    assert array_api_compat.is_mlx_namespace(xp)
    assert array_api_compat.is_mlx_namespace(mx)


def test_namespace_does_not_patch_array_type():
    before_getitem = mx.array.__getitem__
    before_namespace = mx.array.__array_namespace__
    # Importing the wrapper must be observational only for the native type.
    __import__("array_api_compat.mlx")
    assert mx.array.__getitem__ is before_getitem
    assert mx.array.__array_namespace__ is before_namespace


def test_inspection_namespace():
    info = xp.__array_namespace_info__()
    assert info.capabilities() == {
        "boolean indexing": False,
        "data-dependent shapes": False,
        "max dimensions": None,
    }
    assert isinstance(info.devices(), tuple)
    assert info.default_device() in info.devices()
    defaults = info.default_dtypes()
    assert defaults["real floating"] == mx.float32
    assert defaults["complex floating"] == mx.complex64
    assert defaults["integral"] == mx.int32
    assert defaults["indexing"] == mx.int32
    assert "complex128" not in info.dtypes()
    assert info.dtypes(kind="bool") == {"bool": mx.bool_}


def test_creation_signatures_and_copy():
    assert "device" in inspect.signature(xp.asarray).parameters
    assert "copy" in inspect.signature(xp.asarray).parameters
    x = mx.arange(4)
    assert xp.asarray(x, copy=False) is x
    copied = xp.asarray(x, copy=True)
    assert copied is not x
    assert copied.tolist() == x.tolist()

    assert xp.arange(0, 4, 1).tolist() == [0, 1, 2, 3]
    assert xp.eye(0).shape == (0, 0)
    assert xp.eye(2, 3, k=5).shape == (2, 3)
    assert isinstance(xp.meshgrid(mx.arange(2), mx.arange(3)), tuple)


def test_manipulation_wrappers():
    x = mx.arange(24).reshape((2, 3, 4))
    assert xp.matrix_transpose(x).shape == (2, 4, 3)
    assert xp.moveaxis(x, (0, 2), (2, 0)).shape == (4, 3, 2)
    assert xp.moveaxis(x, (), ()).shape == x.shape
    assert isinstance(xp.unstack(x), tuple)
    assert xp.reshape(x, (6, 4), copy=True).shape == (6, 4)


def test_reduction_wrappers():
    x = mx.arange(6).reshape((2, 3))
    assert xp.sum(x, axis=0, dtype=mx.float32).dtype == mx.float32
    assert xp.prod(x + 1, axis=1, dtype=mx.float32).dtype == mx.float32
    assert xp.std(x.astype(mx.float32), correction=1).shape == ()
    assert xp.var(x.astype(mx.float32), correction=1).shape == ()

    cs = xp.cumulative_sum(mx.array([1, 2, 3]), include_initial=True)
    cp = xp.cumulative_prod(mx.array([2, 3]), include_initial=True)
    assert cs.tolist() == [0, 1, 3, 6]
    assert cp.tolist() == [1, 2, 6]


def test_searching_and_sorting_wrappers():
    x = mx.array([3, 1, 1, 2])
    assert xp.argmax(x).dtype == mx.int32
    assert xp.argmin(x).dtype == mx.int32
    assert xp.argsort(x, descending=True).tolist() == [0, 3, 1, 2]
    assert xp.sort(x, descending=True).tolist() == [3, 2, 1, 1]


def test_fft_namespace():
    x = mx.arange(8).astype(mx.float32)
    spectrum = xp.fft.rfft(x)
    recovered = xp.fft.irfft(spectrum, n=8)
    assert mx.allclose(recovered, x, atol=1e-5).item()

    hermitian = mx.array([1 + 0j, 2 + 1j, 3 + 0j], dtype=mx.complex64)
    h = xp.fft.hfft(hermitian, n=4)
    ih = xp.fft.ihfft(h, n=4)
    assert mx.allclose(ih, hermitian, atol=1e-5).item()


def test_linalg_namespace():
    x = mx.array([[2.0, 0.0], [0.0, 1.0]])
    result = xp.linalg.eigh(x)
    assert result.eigenvalues.shape == (2,)
    assert result.eigenvectors.shape == (2, 2)
    assert xp.linalg.matrix_norm(x).shape == ()
    assert xp.linalg.vector_norm(x, axis=(0, 1)).shape == ()
