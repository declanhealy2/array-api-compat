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
    __import__("array_api_compat.mlx")
    assert mx.array.__getitem__ is before_getitem
    assert mx.array.__array_namespace__ is before_namespace


def test_namespace_exports_complete_override_surface():
    required = {
        "astype",
        "broadcast_shapes",
        "can_cast",
        "cumulative_prod",
        "cumulative_sum",
        "diff",
        "finfo",
        "iinfo",
        "isin",
        "result_type",
        "searchsorted",
        "sum",
        "var",
    }
    assert required <= set(xp.__all__)
    assert all(hasattr(xp, name) for name in required)


def test_inspection_namespace():
    info = xp.__array_namespace_info__()
    assert info.capabilities() == {
        "boolean indexing": False,
        "data-dependent shapes": False,
        "max dimensions": 10,
    }
    assert isinstance(info.devices(), tuple)
    assert info.default_device() in info.devices()
    defaults = info.default_dtypes()
    assert defaults["real floating"] == mx.float32
    assert defaults["complex floating"] == mx.complex64
    assert defaults["integral"] == mx.int32
    assert defaults["indexing"] == mx.int32
    assert "float16" not in info.dtypes()
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


def test_asarray_converts_dtype_on_the_requested_device():
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        with mx.stream(mx.cpu):
            source = mx.asarray([1.25, 2.5], dtype=mx.float64)
        same_cpu_array = xp.asarray(source, dtype=mx.float64, device=mx.cpu)
        inferred = xp.asarray(source, dtype=mx.float32)
        converted = xp.asarray(source, dtype=mx.float32, device=mx.gpu)
        created = xp.asarray([1.25, 2.5], dtype=mx.float64, device=mx.cpu)
        promoted = xp.asarray(mx.asarray([1.25, 2.5]), dtype=mx.float64, device=mx.cpu)
        mx.eval(inferred)
        mx.eval(converted)
        mx.eval(created)
        mx.eval(promoted)
    finally:
        mx.set_default_device(previous_device)

    assert same_cpu_array is source
    assert inferred.dtype == mx.float32
    assert inferred.tolist() == [1.25, 2.5]
    assert converted.dtype == mx.float32
    assert converted.tolist() == [1.25, 2.5]
    assert created.dtype == mx.float64
    assert created.tolist() == [1.25, 2.5]
    assert promoted.dtype == mx.float64
    assert promoted.tolist() == [1.25, 2.5]


def test_dtype_helpers():
    assert xp.broadcast_shapes() == ()
    assert xp.broadcast_shapes((2, 1), (1, 3)) == (2, 3)
    assert xp.can_cast(mx.int8, mx.int16)
    assert xp.result_type(mx.int8, mx.int16, 1) == mx.int16

    float_info = xp.finfo(mx.asarray(1, dtype=mx.float32))
    assert float_info.bits == 32
    assert float_info.dtype == mx.float32
    assert float_info.eps > 0

    int_info = xp.iinfo(mx.asarray(1, dtype=mx.uint16))
    assert int_info.bits == 16
    assert int_info.dtype == mx.uint16
    assert int_info.min == 0


def test_manipulation_wrappers():
    x = mx.arange(24).reshape((2, 3, 4))
    assert xp.matrix_transpose(x).shape == (2, 4, 3)
    assert xp.moveaxis(x, (0, 2), (2, 0)).shape == (4, 3, 2)
    assert xp.moveaxis(x, (), ()).shape == x.shape
    assert isinstance(xp.unstack(x), tuple)
    assert xp.reshape(x, (6, 4), copy=True).shape == (6, 4)
    assert xp.expand_dims(mx.arange(3), 1).shape == (3, 1)
    assert xp.squeeze(mx.zeros((1, 3, 1)), (0, 2)).shape == (3,)
    assert xp.diff(mx.array([0, 1, 3]), n=2).tolist() == [1]


def test_reduction_wrappers():
    x = mx.arange(6).reshape((2, 3))
    assert xp.sum(x, axis=0, dtype=mx.float32).dtype == mx.float32
    assert xp.prod(x + 1, axis=1, dtype=mx.float32).dtype == mx.float32
    assert xp.sum(mx.array([1], dtype=mx.uint8)).dtype == mx.uint32
    assert xp.prod(mx.array([1], dtype=mx.int8)).dtype == mx.int32
    assert xp.std(x.astype(mx.float32), correction=1.5).shape == ()
    assert xp.var(x.astype(mx.float32), correction=0.5).shape == ()

    cs = xp.cumulative_sum(mx.array([1, 2, 3]), include_initial=True)
    cp = xp.cumulative_prod(mx.array([2, 3]), include_initial=True)
    assert cs.tolist() == [0, 1, 3, 6]
    assert cp.tolist() == [1, 2, 6]
    assert xp.cumulative_sum(mx.array([1], dtype=mx.uint8)).dtype == mx.uint32


def test_searching_sorting_and_set_wrappers():
    x = mx.array([3, 1, 1, 2])
    assert xp.argmax(x).dtype == mx.int32
    assert xp.argmin(x).dtype == mx.int32
    assert xp.argsort(x, descending=True).tolist() == [0, 3, 1, 2]
    assert xp.sort(x, descending=True).tolist() == [3, 2, 1, 1]
    assert xp.searchsorted(mx.array([1, 3, 5]), 3).item() == 1
    assert xp.isin(mx.array([1, 2, 3]), mx.array([2, 4])).tolist() == [
        False,
        True,
        False,
    ]


def test_elementwise_overrides():
    assert xp.floor_divide(mx.array(-1), mx.array(2)).item() == -1
    assert xp.remainder(mx.array(-1), mx.array(2)).item() == 1
    assert xp.hypot(mx.array(3.0), 4.0).item() == pytest.approx(5.0)
    complex_zero = mx.array(0 + 0j, dtype=mx.complex64)
    assert xp.expm1(complex_zero).item() == 0j


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


def test_complex_linalg_uses_supported_cpu_stream():
    previous_device = mx.default_device()
    mx.set_default_device(mx.gpu)
    try:
        matrix = mx.array([[2 + 0j, 0j], [0j, 4 + 0j]], dtype=mx.complex64)
        right_hand_side = mx.array([2 + 0j, 8 + 0j], dtype=mx.complex64)
        solution = xp.linalg.solve(matrix, right_hand_side)
        eigenvalues = xp.linalg.eigvalsh(matrix)
        mx.eval(solution, eigenvalues)
    finally:
        mx.set_default_device(previous_device)

    assert solution.tolist() == [1 + 0j, 2 + 0j]
    assert eigenvalues.tolist() == [2.0, 4.0]
