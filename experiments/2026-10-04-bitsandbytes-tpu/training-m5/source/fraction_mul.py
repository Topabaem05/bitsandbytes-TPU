"""Independent Fraction rounding oracle; no tensor/candidate dependency."""
from fractions import Fraction

def signed(word):return word-(1<<32) if word&(1<<31) else word
def value(word):
    if type(word) is not int or not -(1<<31)<=word<(1<<31):raise ValueError('SIGNED32_WORD_REQUIRED')
    raw=word&0xffffffff;exponent=(raw>>23)&255;fraction=raw&0x7fffff
    if exponent==255:raise ValueError('FINITE_WORD_REQUIRED')
    significand=fraction if exponent==0 else fraction+(1<<23)
    power=-149 if exponent==0 else exponent-150
    result=Fraction(significand<<power) if power>=0 else Fraction(significand,1<<-power)
    return -result if raw>>31 else result

def nearest(target):
    if target>=Fraction((1<<128)-(1<<103)):return 0x7f800000
    low,high=0,0x7f7fffff
    while low<high:
        middle=(low+high+1)//2
        if value(middle)<=target:low=middle
        else:high=middle-1
    if low==0x7f7fffff or value(low)==target:return low
    upper=low+1;below=target-value(low);above=value(upper)-target
    if below==above:return low if low%2==0 else upper
    return low if below<above else upper

def multiply(left,right):
    target=abs(value(left)*value(right));sign=(left^right)&0x80000000
    return signed(sign|nearest(target))

def divide(left,right):
    l,r=value(left),value(right)
    if r==0:raise ValueError('NONZERO_DIVISOR_REQUIRED')
    target=abs(l/r);sign=(left^right)&0x80000000
    return signed(sign|nearest(target))
