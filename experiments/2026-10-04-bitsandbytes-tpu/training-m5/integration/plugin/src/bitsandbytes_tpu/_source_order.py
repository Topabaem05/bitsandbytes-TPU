"""Private PyTorch2.9 CPU AVX2 source-order sum, with exact word addition.

Restricted rank1 arrays contain 1..513 or exactly1024 admitted finite FP32 words. This is
the reviewed four-level ILP4 order, not an arbitrary mathematical reduction.
No sum/mean tensor reduction or floating-point arithmetic occurs here.
"""
import torch

def vector_row_sum(vectors, zero, add_bits):
    size = len(vectors)
    ilp = size // 4
    acc = [[zero.clone() for _ in range(4)] for _ in range(4)]
    power = max(4, ((ilp - 1).bit_length() if ilp else 0) // 4)
    step, mask, i = 1 << power, (1 << power) - 1, 0
    while i + step <= ilp:
        for _ in range(step):
            for k in range(4):
                acc[0][k] = add_bits(acc[0][k], vectors[4 * i + k])
            i += 1
        for level in range(1, 4):
            for k in range(4):
                acc[level][k] = add_bits(acc[level][k], acc[level - 1][k])
                acc[level - 1][k] = zero.clone()
            if i & (mask << (level * power)):
                break
    while i < ilp:
        for k in range(4):
            acc[0][k] = add_bits(acc[0][k], vectors[4 * i + k])
        i += 1
    for level in range(1, 4):
        for k in range(4):
            acc[0][k] = add_bits(acc[0][k], acc[level][k])
    for i in range(ilp * 4, size):
        acc[0][0] = add_bits(acc[0][0], vectors[i])
    for k in range(1, 4):
        acc[0][0] = add_bits(acc[0][0], acc[0][k])
    return acc[0][0]

def cpu_sum8_bits(words, add_bits):
    if words.dtype != torch.int32 or words.ndim != 1 or not (1 <= words.shape[0] <= 513 or words.shape[0] == 1024):
        raise ValueError('FIXED_RANK1_FINITE_WORDS_REQUIRED')
    count = words.shape[0]
    if count < 8:
        return vector_row_sum([words[i:i + 1] for i in range(count)],
                              torch.zeros_like(words[:1]), add_bits)[0]
    full = count // 8 * 8
    partial = vector_row_sum([words[i:i + 8] for i in range(0, full, 8)],
                             torch.zeros_like(words[:8]), add_bits)
    total = torch.zeros_like(words[0])
    for i in range(full, count):
        total = add_bits(total, words[i])
    for i in range(8):
        total = add_bits(total, partial[i])
    return total
