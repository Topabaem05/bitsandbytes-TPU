"""Private tensor integer FP32 word operations; finite inputs are a precondition.

No float arithmetic, integer division, host value read, or expected-output table.
The caller admits finite words before device execution. Count is shape metadata.
CPU int32 controls are preparation only; TPU execution is NOT_RUN.
"""
import torch

SIGN=-2147483648
FRACTION=0x7fffff

def require_words(words):
    if words.dtype!=torch.int32: raise ValueError('INT32_WORDS_REQUIRED')

def scalar_word(word):
    if type(word) is not int or not -(1<<31)<=word<(1<<31): raise ValueError('SIGNED32_BOUND_REQUIRED')
    if ((word&0x7fffffff)>>23)==255: raise ValueError('FINITE_BOUND_REQUIRED')
    return word

def scalar_key(word):
    return SIGN-word if word<0 else word

def clamp_bits(words,minimum,maximum=0x7f7fffff):
    """Finite FP32 clamp, with input bits retained on equal-value comparisons."""
    require_words(words); minimum=scalar_word(minimum); maximum=scalar_word(maximum)
    low=scalar_key(minimum); high=scalar_key(maximum)
    if low>high: raise ValueError('ORDERED_BOUNDS_REQUIRED')
    keys=torch.where(words<0,-(words&0x7fffffff),words)
    result=torch.where(keys<low,torch.full_like(words,minimum),words)
    return torch.where(keys>high,torch.full_like(words,maximum),result)

def divide_count_bits(words,count):
    """Round finite FP32/count to nearest, ties to even; preserve zero signs.

    Count is an exact positive integer in [1, 2**24]. All arithmetic is int32.
    The restoring division has 24 input bits and at most 24 appended zeros.
    Remainders stay below count; shifted remainders stay below 2**25.
    """
    require_words(words)
    if type(count) is not int or not 1<=count<=1<<24: raise ValueError('EXACT_POSITIVE_COUNT_REQUIRED')
    zero=torch.zeros_like(words); one=torch.ones_like(words)
    magnitude=words&0x7fffffff; exponent=magnitude>>23; fraction=magnitude&FRACTION
    mantissa=torch.where(exponent==0,fraction,fraction|(1<<23))
    power=torch.where(exponent==0,torch.full_like(words,-149),exponent-150)
    for _ in range(23):
        active=(mantissa>0)&(mantissa<(1<<23))
        mantissa=torch.where(active,mantissa<<1,mantissa)
        power=torch.where(active,power-1,power)
    log_count=count.bit_length()-1
    threshold=count<<(23-log_count) if log_count<=23 else count>>1
    low_ratio=mantissa<threshold
    result_power=power+23-log_count-low_ratio.to(torch.int32)
    normal=result_power>=-126
    shift=torch.where(normal,torch.full_like(words,log_count)+low_ratio.to(torch.int32),power+149)
    quotient=zero; remainder=zero
    for bit in range(23,-1,-1):
        remainder=(remainder<<1)|((mantissa>>bit)&1)
        take=remainder>=count
        remainder=torch.where(take,remainder-count,remainder)
        quotient=(quotient<<1)|take.to(torch.int32)
    for bit in range(24):
        active=shift>bit
        trial=remainder<<1; take=trial>=count
        remainder=torch.where(active,torch.where(take,trial-count,trial),remainder)
        quotient=torch.where(active,(quotient<<1)|take.to(torch.int32),quotient)
    twice=remainder<<1
    rounded=quotient+((twice>count)|((twice==count)&((quotient&1)!=0))).to(torch.int32)
    right=torch.where(shift<0,-shift,zero)
    half_shift=torch.where(right>0,right-1,zero)
    half=one<<half_shift; lower_mask=half-1
    truncated=quotient>>right
    half_set=(quotient&half)!=0
    above_half=((quotient&lower_mask)!=0)|(remainder!=0)
    right_rounded=truncated+(half_set&(above_half|((truncated&1)!=0))).to(torch.int32)
    rounded=torch.where(shift<0,right_rounded,rounded)
    carry=rounded==(1<<24)
    normal_mantissa=torch.where(carry,rounded>>1,rounded)
    normal_power=result_power+carry.to(torch.int32)
    encoded_normal=((normal_power+127)<<23)|(normal_mantissa&FRACTION)
    encoded=torch.where(normal,encoded_normal,rounded)
    encoded=torch.where(magnitude==0,zero,encoded)
    return (words&SIGN)|encoded
