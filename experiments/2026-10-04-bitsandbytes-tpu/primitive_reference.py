"""Independent standard-library FP32/IEEE witness for fixed primitive inputs.

PyTorch2.9 CPU SumKernel.cpp/DispatchStub.h source order: AVX2 sum fallback
also applies on AVX512 capability. This model is restricted to the fixed
rank1 arrays (at most 513 elements), without parallel-reduction ambiguity.
"""
import hashlib
import math
import struct

EXPRESSIONS=('mean','sum_div','sum_reciprocal','ordered_sum8','ordered_div8')
ARITHMETIC=('add_zero','multiply_one','divide_one','absolute','clamp','maximum')

def f32(x):return struct.unpack('<f',struct.pack('<f',x))[0]
def bits(x):return struct.unpack('<i',struct.pack('<f',x))[0]
def from_bits(x):return struct.unpack('<f',struct.pack('<i',x))[0]
def array(values,shape,dtype):
    values=list(values);raw=b''.join(struct.pack('<i' if dtype=='int32' else '<f',v) for v in values)
    return {'values':values,'shape':list(shape),'dtype':dtype,'bytes_sha256':hashlib.sha256(raw).hexdigest()}
def bit_array(values,shape):return array([bits(v) for v in values],shape,'int32')

def first_scales(case):
    values=[]
    for v in case['weight']:
        x=f32(v)
        if case['dtype']=='bfloat16':
            b=struct.unpack('<I',struct.pack('<f',x))[0]
            b=((b+0x7fff+((b>>16)&1))&0xffff0000)&0xffffffff
            x=struct.unpack('<f',struct.pack('<I',b))[0]
        values.append(x)
    full=len(values)//64*64
    scales=[max(abs(v) for v in values[i:i+64]) for i in range(0,full,64)]
    if full<len(values):scales.append(max(max(abs(v) for v in values[full:]),f32(1e-38)))
    return scales

def vector_row_sum(v,width):
    size=len(v);ilp=size//4
    acc=[[[0.]*width for _ in range(4)] for _ in range(4)]
    power=max(4,((ilp-1).bit_length() if ilp else 0)//4);step=1<<power;mask=step-1;i=0
    def add(left,right):return [f32(a+b) for a,b in zip(left,right)]
    while i+step<=ilp:
        for _ in range(step):
            for k in range(4):acc[0][k]=add(acc[0][k],v[4*i+k])
            i+=1
        for level in range(1,4):
            for k in range(4):acc[level][k]=add(acc[level][k],acc[level-1][k]);acc[level-1][k]=[0.]*width
            if i&(mask<<(level*power)):break
    while i<ilp:
        for k in range(4):acc[0][k]=add(acc[0][k],v[4*i+k])
        i+=1
    for level in range(1,4):
        for k in range(4):acc[0][k]=add(acc[0][k],acc[level][k])
    for i in range(ilp*4,size):acc[0][0]=add(acc[0][0],v[i])
    for k in range(1,4):acc[0][0]=add(acc[0][0],acc[0][k])
    return acc[0][0]

def cpu_sum8(v):
    if len(v)<8:return vector_row_sum([[x] for x in v],1)[0]
    full=len(v)//8*8;partial=vector_row_sum([v[i:i+8] for i in range(0,full,8)],8)
    total=0.
    for x in v[full:]:total=f32(total+x)
    for x in partial:total=f32(total+x)
    return total

def expected(spec):
    ints=spec['bit_patterns_signed32'];values=[from_bits(x) for x in ints]
    result={'host-float-bits':{'bits':array(ints,[len(ints)],'int32')},
            'device-int-float':{'float':array(values,[len(ints)],'float32')},
            'native-bitcast':{'bits':array(ints,[len(ints)],'int32'),'float':array(values,[len(ints)],'float32')}}
    arithmetic={'add_zero':[f32(x+0.) for x in values],'multiply_one':[f32(x*1.) for x in values],
                'divide_one':[f32(x/1.) for x in values],'absolute':[abs(x) for x in values],
                'clamp':[max(x,f32(1e-38)) for x in values],'maximum':[max(abs(x) for x in values)]}
    result['float-arithmetic']={k:bit_array(v,[] if k=='maximum' else [len(v)]) for k,v in arithmetic.items()}
    for c in spec['mean_cases']:
        v=c['first_scales'];n=c['count'];total=cpu_sum8(v)
        mean=f32(total/f32(n));reciprocal=f32(total*f32(1./f32(n)))
        outputs={k:bit_array([x],[]) for k,x in zip(EXPRESSIONS,(mean,mean,reciprocal,total,mean))}
        outputs.update(input_bits=bit_array(v,[n]),count_bits=bit_array([f32(n)],[]))
        result['mean-'+c['case_id']]=outputs
    return result
