"""Pinned torch lazy DataHash. Little-endian uint128 layout only."""
MASK = (1 << 128) - 1
SEED = 0xc2b2ae3d27d4eb4f
MULT = 0xc6a4a7935bd1e995

def data_hash(data):
    if type(data) is not bytes:
        raise ValueError('HASH_BYTES_REQUIRED')
    value = SEED ^ ((len(data) * MULT) & MASK)
    for offset in range(0, len(data), 16):
        block = int.from_bytes(data[offset:offset + 16], 'little')
        block = (block * MULT) & MASK
        block ^= block >> 47
        block = (block * MULT) & MASK
        value = ((value ^ block) * MULT) & MASK
    value ^= value >> 47
    value = (value * MULT) & MASK
    return value ^ (value >> 47)

def hash_to_string(value):
    if type(value) is not int or not 0 <= value <= MASK:
        raise ValueError('HASH_UINT128_REQUIRED')
    return format(value >> 64, 'x') + format(value & ((1 << 64) - 1), '016x')

def computation_hash(data):
    return hash_to_string(data_hash(data))
