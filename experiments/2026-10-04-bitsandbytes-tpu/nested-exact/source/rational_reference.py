"""Independent scalar rational reference. No tensor or candidate arithmetic imports."""
from fractions import Fraction
from functools import lru_cache
import math
import struct

def require(value, message):
    if not value:
        raise ValueError(message)

def word(value, dtype='float32'):
    raw = struct.unpack('<I', struct.pack('<f', value))[0]
    require(raw & 0x7f800000 != 0x7f800000, 'FINITE_SOURCE_VALUE')
    if dtype == 'bfloat16':
        raw = ((raw + 0x7fff + ((raw >> 16) & 1)) >> 16) << 16
    return raw

@lru_cache(maxsize=65536)
def value(raw):
    raw &= 0xffffffff
    exponent, fraction = (raw >> 23) & 255, raw & 0x7fffff
    require(exponent != 255, 'FINITE_ARITHMETIC_WORD')
    numerator = fraction if exponent == 0 else fraction + (1 << 23)
    shift = -149 if exponent == 0 else exponent - 150
    result = Fraction(numerator << shift) if shift >= 0 else Fraction(numerator, 1 << -shift)
    return -result if raw >> 31 else result

def round_rational(target, negative_zero=False):
    sign = 0x80000000 if target < 0 or target == 0 and negative_zero else 0
    target = abs(target)
    if target == 0:
        return sign
    n, d = target.numerator, target.denominator
    exponent = n.bit_length() - d.bit_length()
    if (n << max(0, -exponent)) < (d << max(0, exponent)):
        exponent -= 1
    shift = 149 if exponent < -126 else 23 - exponent
    a, b = (n << shift, d) if shift >= 0 else (n, d << -shift)
    q, r = divmod(a, b)
    if 2 * r > b or 2 * r == b and q & 1:
        q += 1
    if exponent < -126:
        return sign | q
    if q == 1 << 24:
        q >>= 1
        exponent += 1
    if exponent > 127:
        return sign | 0x7f800000
    return sign | ((exponent + 127) << 23) | (q - (1 << 23))

@lru_cache(maxsize=262144)
def add(a, b):
    return round_rational(value(a) + value(b), a == b == 0x80000000)

@lru_cache(maxsize=262144)
def sub(a, b):
    return add(a, b ^ 0x80000000)

@lru_cache(maxsize=262144)
def mul(a, b):
    return round_rational(value(a) * value(b), bool((a ^ b) & 0x80000000))

@lru_cache(maxsize=262144)
def div(a, b):
    require(value(b) != 0, 'NONZERO_DIVISOR')
    return round_rational(value(a) / value(b), bool((a ^ b) & 0x80000000))

def source_sum(words):
    """Inspected Torch2.9 AVX2 rank1 sum order, extent1..513."""
    require(1 <= len(words) <= 513, 'SOURCE_SUM_DOMAIN')
    width = 1 if len(words) < 8 else 8
    full = len(words) // width
    vectors = [words[i * width:(i + 1) * width] for i in range(full)]
    ilp = full // 4
    acc = [[[0] * width for _ in range(4)] for _ in range(4)]
    power = max(4, ((ilp - 1).bit_length() if ilp else 0) // 4)
    step, mask, i = 1 << power, (1 << power) - 1, 0
    while i + step <= ilp:
        for _ in range(step):
            for k in range(4):
                acc[0][k] = [add(a, b) for a, b in zip(acc[0][k], vectors[4 * i + k])]
            i += 1
        for level in range(1, 4):
            for k in range(4):
                acc[level][k] = [add(a, b) for a, b in zip(acc[level][k], acc[level - 1][k])]
                acc[level - 1][k] = [0] * width
            if i & (mask << (level * power)):
                break
    while i < ilp:
        for k in range(4):
            acc[0][k] = [add(a, b) for a, b in zip(acc[0][k], vectors[4 * i + k])]
        i += 1
    for level in range(1, 4):
        for k in range(4):
            acc[0][k] = [add(a, b) for a, b in zip(acc[0][k], acc[level][k])]
    for i in range(ilp * 4, full):
        acc[0][0] = [add(a, b) for a, b in zip(acc[0][0], vectors[i])]
    for k in range(1, 4):
        acc[0][0] = [add(a, b) for a, b in zip(acc[0][0], acc[0][k])]
    if width == 1:
        return acc[0][0][0]
    result = 0
    for x in words[full * width:]:
        result = add(result, x)
    for x in acc[0][0]:
        result = add(result, x)
    return result


def quantize(words, mapping, blocksize):
    bounds = [div(add(a, b), 0x40000000) for a, b in zip(mapping, mapping[1:])]
    codes, scales = [], []
    for start in range(0, len(words), blocksize):
        block = words[start:start + blocksize]
        maximum = max(x & 0x7fffffff for x in block)
        denominator = max(maximum, 0x006ce3ee)
        full = len(block) == blocksize
        scales.append(maximum if full else denominator)
        reciprocal = div(0x3f800000, denominator) if full else None
        for x in block:
            z = mul(x, reciprocal) if full else div(x, denominator)
            z_value = min(Fraction(1), max(Fraction(-1), value(z)))
            codes.append(sum(z_value > value(midpoint) for midpoint in bounds))
    return codes, scales

def decode(codes, scales, mapping, blocksize):
    return [mul(mapping[x], scales[i // blocksize]) for i, x in enumerate(codes)]

def restored(codes, second, mapping, offset):
    return [add(x, offset) for x in decode(codes, second, mapping, 256)]

def expected_words(case, data):
    dynamic = [word(x) for x in data['dynamic_map']]
    nf4 = [word(x) for x in data['nf4_map']]
    output = {'dynamic_map': dynamic}
    if case['kind'] == 'block':
        codes, second = quantize([word(x) for x in case['data']], dynamic, 256)
        decoded = decode(codes, second, dynamic, 256)
        return dict(output, codes=codes, second_scales=second, decoded=decoded, decoded_out=decoded)
    if case['kind'] in ('signed_known', 'positive_known'):
        codes = case['codes']
        second = [word(x) for x in case['second_scales']]
        offset = word(case['offset'])
        scales = restored(codes, second, dynamic, offset)
        if case['kind'] == 'signed_known':
            return dict(output, codes=codes, second_scales=second, offset=[offset], restored_scales=scales)
        packed = [0x78] * (math.prod(case['weight_shape']) // 2)
    else:
        indices, first = quantize([word(x, case['dtype']) for x in case['weight']], nf4, 64)
        if len(indices) % 2:
            indices.append(7)
        packed = [(a << 4) | b for a, b in zip(indices[::2], indices[1::2])]
        offset = div(source_sum(first), word(len(first)))
        codes, second = quantize([sub(x, offset) for x in first], dynamic, 256)
        scales = restored(codes, second, dynamic, offset)
    count = math.prod(case['weight_shape'])
    indices = [x for byte in packed for x in (byte >> 4, byte & 15)][:count]
    decoded = decode(indices, scales, nf4, 64)
    if case['dtype'] == 'bfloat16':
        decoded = [((x + 0x7fff + ((x >> 16) & 1)) >> 16) << 16 for x in decoded]
    return dict(output, packed=packed, scale_codes=codes, offset=[offset], second_scales=second,
                nf4_map=nf4, restored_scales=scales, decoded=decoded, decoded_out=decoded)

def audit_row(case, data, raw):
    expected = expected_words(case, data)
    for name, wanted in expected.items():
        observed = raw['outputs'][name]
        got = observed['values'] if observed['dtype'] == 'uint8' else [word(x) for x in observed['values']]
        require(got == wanted, 'INDEPENDENT_RATIONAL_' + name)
    return {'quantized_array_count': len(expected), 'rational_arithmetic': 'PASS'}
