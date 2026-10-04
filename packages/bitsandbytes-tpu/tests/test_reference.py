"""CPU controls. These tests do not establish XLA or TPU execution."""
import importlib
import importlib.util

import pytest
import torch

CODE = (-1.0, -0.6961928009986877, -0.5250730514526367,
        -0.39491748809814453, -0.28444138169288635,
        -0.18477343022823334, -0.09105003625154495, 0.0,
        0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
        0.33791524171829224, 0.44070982933044434,
        0.5626170039176941, 0.7229568362236023, 1.0)


@pytest.fixture
def reference():
    assert importlib.util.find_spec("bitsandbytes_tpu.reference") is not None, "NF4 reference is not implemented"
    return importlib.import_module("bitsandbytes_tpu.reference")


def original_quant(a):
    import bitsandbytes
    return torch.ops.bitsandbytes.quantize_4bit.default(a, 64, "nf4", torch.uint8)


def original_dequant(p, s, shape, dtype):
    import bitsandbytes
    return torch.ops.bitsandbytes.dequantize_4bit.default(p, s, 64, "nf4", shape, dtype)


def test_codebook_packing_and_tail():
    assert importlib.util.find_spec("bitsandbytes_tpu.reference") is not None, "NF4 reference is not implemented"
    reference = importlib.import_module("bitsandbytes_tpu.reference")
    a = torch.tensor(CODE)
    p, s = reference.quantize_4bit(a, 64, "nf4", torch.uint8)
    # High nibble comes first. Both signs and all 16 symbols are present.
    assert p.tolist() == [[1], [35], [69], [103], [137], [171], [205], [239]]
    assert s.tolist() == [1.0]
    assert torch.equal(reference.dequantize_4bit(p, s, 64, "nf4", [16], torch.float32), a)
    odd, scales = reference.quantize_4bit(torch.tensor([1.0]), 64, "nf4", torch.uint8)
    assert odd.tolist() == [[247]]  # The unused tail is the zero code.
    assert scales.tolist() == [1.0]


@pytest.mark.parametrize("n", [1, 2, 63, 64, 65, 127, 128, 129])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_quant_dequant_matches_pinned_original(reference, n, dtype):
    a = torch.randn(n, generator=torch.Generator().manual_seed(431 + n)).to(dtype)
    want_p, want_s = original_quant(a)
    p, s = reference.quantize_4bit(a, 64, "nf4", torch.uint8)
    assert torch.equal(p, want_p)
    assert torch.equal(s, want_s)
    got = reference.dequantize_4bit(p, s, 64, "nf4", [n], dtype)
    assert torch.equal(got, original_dequant(p, s, [n], dtype))
    assert p.dtype == torch.uint8 and s.dtype == torch.float32
    assert p.device == a.device and s.device == a.device and got.device == a.device


def test_midpoint_ties_and_neighbors(reference):
    code = torch.tensor(CODE)
    bounds = (code[:-1] + code[1:]) / 2
    lo = torch.nextafter(bounds, torch.full_like(bounds, -torch.inf))
    hi = torch.nextafter(bounds, torch.full_like(bounds, torch.inf))
    # The endpoints keep the normalization scale exactly one.
    a = torch.cat((torch.tensor([-1., 1.]), bounds, lo, hi))
    want_p, want_s = original_quant(a)
    p, s = reference.quantize_4bit(a, 64, "nf4", torch.uint8)
    assert torch.equal(p, want_p) and torch.equal(s, want_s)
    codes = torch.stack((p[:, 0] >> 4, p[:, 0] & 15), dim=1).flatten()[:a.numel()]
    assert torch.equal(codes[2:17], torch.arange(15, dtype=torch.uint8))
    assert torch.equal(codes[17:32], torch.arange(15, dtype=torch.uint8))
    assert torch.equal(codes[32:47], torch.arange(1, 16, dtype=torch.uint8))


@pytest.mark.parametrize("n", [1, 64, 65])
def test_zero_blocks_keep_original_scale_and_zero_codes(reference, n):
    a = torch.zeros(n)
    p, s = reference.quantize_4bit(a, 64, "nf4", torch.uint8)
    want_p, want_s = original_quant(a)
    assert torch.equal(p, want_p) and torch.equal(s, want_s)
    assert torch.count_nonzero(reference.dequantize_4bit(p, s, 64, "nf4", [n], torch.float32)) == 0


@pytest.mark.parametrize("a", [
    torch.tensor([1e-40, -1e-40, 0., 1e-38]),
    torch.tensor([3.4028234663852886e38, -3.4028234663852886e38, 1e38, 0.]),
    torch.cat((torch.full((64,), 1e-40), torch.tensor([1e-40]))),
])
def test_finite_dynamic_range_uses_original_normalization(reference, a):
    p, s = reference.quantize_4bit(a, 64, "nf4", torch.uint8)
    want_p, want_s = original_quant(a)
    assert torch.equal(p, want_p) and torch.equal(s, want_s)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shape", [(5, 7), (2, 3, 7)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_gemm_shape_bias_and_dtype_match_original(reference, dtype, shape, with_bias):
    w = torch.randn(9, 7, generator=torch.Generator().manual_seed(25)).to(dtype)
    p, s = original_quant(w)
    a = torch.randn(shape, generator=torch.Generator().manual_seed(29)).to(dtype)
    bias = torch.linspace(-1, 1, 9).to(dtype) if with_bias else None
    want = torch.ops.bitsandbytes.gemm_4bit.default(a, p, [9, 7], s, 64, "nf4", bias)
    got = reference.gemm_4bit(a, p, [9, 7], s, 64, "nf4", bias)
    assert torch.equal(got, want)
    assert got.shape == (*shape[:-1], 9) and got.dtype == dtype


def test_noncontiguous_float_input_is_supported(reference):
    a = torch.arange(20, dtype=torch.float32).reshape(4, 5).t()
    p, s = reference.quantize_4bit(a, 64, "nf4", torch.uint8)
    want_p, want_s = original_quant(a)
    assert torch.equal(p, want_p) and torch.equal(s, want_s)


def test_out_mutates_supplied_tensor_and_returns_none(reference):
    p, s = original_quant(torch.arange(6, dtype=torch.float32))
    out = torch.full((2, 3), -99.)
    pointer = out.data_ptr()
    result = reference.dequantize_4bit_out(p, s, 64, "nf4", [2, 3], torch.float32, out)
    assert result is None and out.data_ptr() == pointer
    assert torch.equal(out, original_dequant(p, s, [2, 3], torch.float32))


@pytest.mark.parametrize("change", ["fp4", "block", "storage", "dtype", "empty", "nan", "inf"])
def test_quantizer_rejects_unsupported_input(reference, change):
    a = torch.ones(65); block = 64; kind = "nf4"; storage = torch.uint8
    if change == "fp4": kind = "fp4"
    if change == "block": block = 128
    if change == "storage": storage = torch.float32
    if change == "dtype": a = a.half()
    if change == "empty": a = a[:0]
    if change == "nan": a[0] = torch.nan
    if change == "inf": a[0] = torch.inf
    with pytest.raises((ValueError, NotImplementedError, RuntimeError)):
        reference.quantize_4bit(a, block, kind, storage)


@pytest.mark.parametrize("change", ["row", "flat", "size", "scales", "scale_dtype", "negative", "shape", "out"])
def test_dequantizer_rejects_invalid_layout_or_state(reference, change):
    p, s = original_quant(torch.ones(65)); shape = [65]; dtype = torch.float32
    if change == "row": p = p.t()
    if change == "flat": p = p.flatten()
    if change == "size": p = p[:-1]
    if change == "scales": s = s[:-1]
    if change == "scale_dtype": s = s.to(torch.uint8)
    if change == "negative": s[0] = -1
    if change == "shape": shape = [0]
    with pytest.raises((ValueError, NotImplementedError, RuntimeError)):
        if change == "out":
            reference.dequantize_4bit_out(p, s, 64, "nf4", shape, dtype, torch.empty(64))
        else:
            reference.dequantize_4bit(p, s, 64, "nf4", shape, dtype)


@pytest.mark.parametrize("change", ["nested_code", "nested_scale", "nested_offset", "rank", "inner", "bias_shape", "bias_dtype", "fp4"])
def test_gemm_rejects_unsupported_or_inconsistent_arguments(reference, change):
    a = torch.ones(2, 7); p, s = original_quant(torch.ones(9, 7)); bias = None; kw = {}; kind = "nf4"
    if change == "nested_code": kw['absmax_code'] = torch.ones(256)
    if change == "nested_scale": kw['absmax_8bit'] = torch.ones(1, dtype=torch.uint8)
    if change == "nested_offset": kw['absmax_offset'] = torch.tensor(0.)
    if change == "rank": a = a[0]
    if change == "inner": a = a[:, :-1]
    if change == "bias_shape": bias = torch.ones(8)
    if change == "bias_dtype": bias = torch.ones(9).bfloat16()
    if change == "fp4": kind = "fp4"
    with pytest.raises((ValueError, NotImplementedError, RuntimeError)):
        reference.gemm_4bit(a, p, [9, 7], s, 64, kind, bias, **kw)


def test_reference_does_not_call_host_transfer_methods(reference, monkeypatch):
    p, s = original_quant(torch.ones(2, 7))
    def forbid(*args, **kwargs):
        raise AssertionError('A host transfer method was called')
    for name in ('cpu', 'numpy', 'item'):
        monkeypatch.setattr(torch.Tensor, name, forbid)
    got_p, got_s = reference.quantize_4bit(torch.ones(2, 7), 64, 'nf4', torch.uint8)
    decoded = reference.dequantize_4bit(got_p, got_s, 64, 'nf4', [2, 7], torch.float32)
    result = reference.gemm_4bit(torch.ones(3, 7), p, [2, 7], s, 64, 'nf4')
    assert decoded.shape == (2, 7) and result.shape == (3, 2)


@pytest.mark.parametrize('change', ['scales_device', 'out_device', 'out_dtype', 'input_device', 'noncontiguous_packed'])
def test_state_device_and_output_errors_do_not_publish_output(reference, change):
    p, s = original_quant(torch.ones(2, 7)); out = torch.full((2, 7), -99.)
    if change == 'scales_device': s = torch.empty(1, device='meta')
    if change == 'out_device': out = torch.empty((2, 7), device='meta')
    if change == 'out_dtype': out = out.bfloat16()
    if change == 'noncontiguous_packed':
        expanded = torch.zeros(p.shape[0] * 2, 1, dtype=torch.uint8)
        expanded[::2] = p; p = expanded[::2]
    with pytest.raises((ValueError, RuntimeError)):
        if change == 'input_device':
            reference.gemm_4bit(torch.empty(3, 7, device='meta'), p, [2, 7], s, 64, 'nf4')
        else:
            reference.dequantize_4bit_out(p, s, 64, 'nf4', [2, 7], torch.float32, out)
    if out.device.type != 'meta': assert torch.equal(out, torch.full_like(out, -99.))
