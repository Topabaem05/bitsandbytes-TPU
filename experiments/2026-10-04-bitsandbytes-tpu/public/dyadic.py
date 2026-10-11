"""Exact Python integer reference. No Torch, NumPy, BLAS or device dependency."""
import hashlib,json,math,struct
SEMANTICS='PUBLIC15_QUALIFIED_CPU_PLUS_EXACT_DYADIC_V1'
def require(v,label):
 if not v:raise ValueError(label)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def float_word(v):
 require(type(v)in(int,float)and math.isfinite(v),'FINITE_NUMBER')
 word=struct.unpack('<I',struct.pack('<f',v))[0]
 require(struct.unpack('<f',struct.pack('<I',word))[0]==v,'EXACT_FP32_REPRESENTATION')
 return word
def parts(w):
 require(type(w)is int and 0<=w<1<<32 and ((w>>23)&255)!=255,'FINITE_FP32_WORD')
 exponent=(w>>23)&255;sign=-1 if w>>31 else 1
 return sign*((w&0x7fffff)+(1<<23 if exponent else 0)),exponent-150 if exponent else-149
def shift_round(n,count):
 if count<=0:return n<<(-count)
 quotient,remainder=divmod(n,1<<count);half=1<<(count-1)
 return quotient+int(remainder>half or(remainder==half and quotient&1))
def encode(n,unit,precision=24,*,negative_zero=False):
 require(type(n)is int and type(unit)is int and precision in(8,24),'ENCODE_ARGUMENTS')
 sign=0x80000000 if n<0 or(n==0 and negative_zero)else 0;n=abs(n)
 if n==0:return sign if precision==24 else sign>>16
 power=unit+n.bit_length()-1
 if power<-126:
  q=shift_round(n,-126-(precision-1)-unit)
  word=sign|(q<<(24-precision))
 else:
  q=shift_round(n,n.bit_length()-precision)
  if q==1<<precision:q>>=1;power+=1
  require(power<=127,'FINITE_RESULT_BOUND')
  word=sign|((power+127)<<23)|((q-(1<<(precision-1)))<<(24-precision))
 return word if precision==24 else word>>16
def multiply_word(left,right):
 l,le=parts(left);r,re=parts(right)
 return encode(l*r,le+re,negative_zero=bool((left^right)>>31))
def cast_bf16(word):
 parts(word)
 return ((word+0x7fff+((word>>16)&1))>>16)&0xffff
def double_word(word):
 n,e=parts(word)
 return encode(2*n,e,negative_zero=bool(word>>31))
def values(record):
 require(record['dtype']in('float32','bfloat16')and type(record['words'])is list and len(record['words'])==math.prod(record['shape']),'TYPED_WORD_RECORD')
 result=[]
 for w in record['words']:
  require(type(w)is int and 0<=w<(1<<32 if record['dtype']=='float32'else 1<<16),'WORD_RANGE')
  raw=w if record['dtype']=='float32'else w<<16;parts(raw);result.append(struct.unpack('<f',struct.pack('<I',raw))[0])
 return result
def units(words):
 canonical=[]
 for w in words:
  n,e=parts(w)
  if n:
   shift=(abs(n)&-abs(n)).bit_length()-1;n>>=shift;e+=shift
  canonical.append((n,e))
 nonzero=[e for n,e in canonical if n]
 unit=min(nonzero)if nonzero else 0
 coefficients=[]
 for n,e in canonical:coefficients.append(n<<(e-unit)if n else 0)
 return coefficients,unit
def dot_rows(left,right,left_exp,right_exp,precision,shape,bias=None):
 # Cache identical operand rows only. Each cached dot remains an exact
 # integer product/sum, independent of BLAS ordering or accumulator width.
 require(left and right and all(len(r)==len(left[0])for r in left+right),'DOT_RECTANGULAR')
 k=len(left[0]);cache={};words=[];unit=left_exp+right_exp
 bias_coeff=[]
 if bias is not None:
  for w in bias:
   n,e=parts(w);require(e>=unit or n==0,'BIAS_COMMON_UNIT');bias_coeff.append(n<<(e-unit)if n else 0)
 for l in left:
  for j,r in enumerate(right):
   key=(l,r)
   if key not in cache:cache[key]=sum(x*y for x,y in zip(l,r))
   words.append(encode(cache[key]+(bias_coeff[j]if bias is not None else 0),unit,precision))
 lm=max(abs(x)for row in left for x in row);rm=max(abs(x)for row in right for x in row)
 bound=k*lm*rm+(max(map(abs,bias_coeff))if bias_coeff else 0)
 proof={'left_unit_exp':left_exp,'right_unit_exp':right_exp,'contraction':k,'left_max_coefficient':lm,'right_max_coefficient':rm,'bias_max_coefficient':max(map(abs,bias_coeff))if bias_coeff else 0,'absolute_integer_sum_bound':bound,'bound_bits':bound.bit_length(),'float64_exact_sum_bound_proven':bound<1<<53,'reference_uses_float64':False,'unique_exact_dots':len(cache)}
 return {'dtype':'float32'if precision==24 else'bfloat16','shape':shape,'words':words},proof
def expected(case,inputs,code_words):
 # Fixed scientific input digest binds this mathematical supplement to the
 # original case; caller independently checks original admission/row seals.
 require(len(code_words)==16,'NF4_TABLE16')
 n,k=case['weight_shape'];shape=case['activation_shape'];m=math.prod(shape[:-1]);dtype=case['dtype']
 require(n==256 and k==512 and m in(128,129)and dtype in('float32','bfloat16'),'FIXED_PUBLIC_DOMAIN')
 require(set(inputs)==({'activation','packed','scales','dy','bias'}if case['bias']else{'activation','packed','scales','dy'}),'INPUT_KEYS')
 for name,d,sh in [('activation',dtype,shape),('packed','uint8',[n*k//2,1]),('scales','float32',[n*k//64]),('dy','float32',[*shape[:-1],n])]:
  require(inputs[name]['dtype']==d and inputs[name]['shape']==sh and len(inputs[name]['values'])==math.prod(sh),'INPUT_METADATA:'+name)
 av=[float_word(v)for v in inputs['activation']['values']];dy=[float_word(v)for v in inputs['dy']['values']];s=[float_word(v)for v in inputs['scales']['values']]
 require(all(type(v)is int and 0<=v<=255 for v in inputs['packed']['values']),'PACKED_BYTE_DOMAIN')
 decode_cache={};dense=[]
 for i in range(n*k):
  packed=inputs['packed']['values'][i//2];idx=packed>>4 if i%2==0 else packed&15;key=(idx,s[i//64])
  if key not in decode_cache:
   w=multiply_word(code_words[idx],key[1]);decode_cache[key]=w if dtype=='float32'else cast_bf16(w)<<16
  dense.append(decode_cache[key])
 aa,ae=units(av);ww,we=units(dense);left=[tuple(aa[i*k:(i+1)*k])for i in range(m)];right=[tuple(ww[i*k:(i+1)*k])for i in range(n)]
 bias=None
 if case['bias']:
  require(inputs['bias']['dtype']=='float32'and inputs['bias']['shape']==[n]and len(inputs['bias']['values'])==n,'BIAS_METADATA');bias=[float_word(v)for v in inputs['bias']['values']]
 y,yp=dot_rows(left,right,ae,we,24 if dtype=='float32'else 8,[*shape[:-1],n],bias);outputs={'y':y};proof={'y':yp}
 if case['gradients']!='NOT_RUN':
  require(dtype=='float32','FP32_GRADIENT_SCOPE');dd,de=units(dy);dleft=[tuple(dd[i*n:(i+1)*n])for i in range(m)];dright=[tuple(ww[j*k+i]for j in range(n))for i in range(k)]
  dx,dxp=dot_rows(dleft,dright,de,we,24,shape);outputs['dX']=dx;proof['dX']=dxp
  outputs['dX_accumulated']={**dx,'words':[double_word(w)for w in dx['words']]}
  if bias is not None:
   sums=[sum(dd[i*n+j]for i in range(m))for j in range(n)];db={'dtype':'float32','shape':[n],'words':[encode(v,de)for v in sums]};outputs['db']=db;outputs['db_accumulated']={**db,'words':[double_word(w)for w in db['words']]};bound=m*max(map(abs,dd));proof['db']={'input_unit_exp':de,'contraction':m,'absolute_integer_sum_bound':bound,'bound_bits':bound.bit_length(),'reference_uses_float64':False}
 return {'semantics':SEMANTICS,'case':case['id'],'inputs_sha256':digest(inputs),'decoded_dense_fp32_words_sha256':hashlib.sha256(struct.pack('<'+'I'*len(dense),*dense)).hexdigest(),'outputs':outputs,'precision_proof':proof}
def verify_supplement(record,case,inputs,code_words):
 require(record==expected(case,inputs,code_words),'EXACT_DYADIC_SUPPLEMENT_REPLAY')
 return record
