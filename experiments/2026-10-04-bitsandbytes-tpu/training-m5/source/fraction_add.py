"""Independent exact Fraction oracle; no candidate or tensor imports."""
from fractions import Fraction

def signed(word):
    return word - (1 << 32) if word & (1 << 31) else word

def value(word):
    if type(word) is not int or not -(1 << 31) <= word < (1 << 31):
        raise ValueError('SIGNED32_WORD_REQUIRED')
    raw = word & 0xffffffff
    exponent = (raw >> 23) & 255
    fraction = raw & 0x7fffff
    if exponent == 255:
        raise ValueError('FINITE_WORD_REQUIRED')
    significand = fraction if exponent == 0 else fraction + (1 << 23)
    power = -149 if exponent == 0 else exponent - 150
    exact = Fraction(significand << power) if power >= 0 else Fraction(significand, 1 << -power)
    return -exact if raw >> 31 else exact

def positive_nearest(target):
    # The overflow midpoint is max-finite + 0.5 ULP of max-finite.
    if target >= Fraction((1 << 128) - (1 << 103)):
        return 0x7f800000
    low, high = 0, 0x7f7fffff
    while low < high:
        middle = (low + high + 1) // 2
        if value(middle) <= target:
            low = middle
        else:
            high = middle - 1
    if low == 0x7f7fffff or value(low) == target:
        return low
    upper = low + 1
    below, above = target - value(low), value(upper) - target
    if below == above:
        return low if low % 2 == 0 else upper
    return low if below < above else upper

def add(left, right):
    target = value(left) + value(right)
    if target == 0:
        return signed(0x80000000) if left < 0 and right < 0 else 0
    return signed((0x80000000 if target < 0 else 0) | positive_nearest(abs(target)))

def subtract(left, right):
    value(right)
    return add(left, signed((right & 0xffffffff) ^ 0x80000000))
