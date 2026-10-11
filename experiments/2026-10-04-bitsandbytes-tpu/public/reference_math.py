"""Versioned exact supplement. The qualified CPU row remains the primary result."""
import hashlib
import torch
import contract as C
import oracle_math as M
import dyadic as D
import audit as A

SEMANTICS = {
    'version': 'PUBLIC15_QUALIFIED_CPU_PLUS_EXACT_DYADIC_V1',
    'primary': 'ORIGINAL_DEFAULT_DECODE_F_LINEAR_AUTOGRAD_QUALIFIED_LINUX',
    'supplement': 'EXACT_INTEGER_DYADIC_DOT_THEN_ONCE_RNE_OUTPUT_DTYPE',
    'decode': 'ORIGINAL_NF4_HIGH_FIRST_FP32_MULTIPLY_THEN_CASE_DTYPE',
    'accumulated_gradient': 'TWICE_FIRST_ROUNDED_FP32_GRADIENT',
    'primary_comparison': 'UNCHANGED_ORIGINAL_DEVICE_TOLERANCE',
    'host_cpu_gemm_bit_equality': 'NOT_REQUIRED',
    'device_comparison': 'UNCHANGED_QUALIFIED_LINUX_PRIMARY_ROWS',
}


def code_words(code):
    C.require(code.dtype == torch.float32 and list(code.shape) == [16], 'REFERENCE_CODEBOOK')
    return [int(w) & 0xffffffff for w in code.view(torch.int32).tolist()]


def supplement(case, decode, code):
    inputs = M.input_record(case)
    result = D.expected(case, inputs, code_words(code))
    activation, packed, scales, _, _ = M.inputs(case)
    dense = decode(packed.flatten(), scales, code, 64, case['weight_shape'], activation.dtype)
    raw = dense.float().contiguous().view(torch.uint8).numpy().tobytes()
    C.require(hashlib.sha256(raw).hexdigest() == result['decoded_dense_fp32_words_sha256'],
              'EXACT_ORIGINAL_DECODE_BOUNDARY')
    return result


def verify(primary, retained, case, decode, code, protocol):
    expected = supplement(case, decode, code)
    C.require(retained == expected, 'INDEPENDENT_EXACT_SUPPLEMENT:' + case['id'])
    key = 'fp32_forward_gradient' if case['dtype'] == 'float32' else 'bf16_forward_unit_scale'
    return A.primary_math(primary, expected, case, protocol['tolerances'][key])
