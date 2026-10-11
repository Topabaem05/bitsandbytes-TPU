"""Independent exact-rational oracle for the pinned 1024-term reduction tree."""
import fraction_add as A
import fraction_mul as M
ONE=0x3f800000
FLOOR=0x006ce3ee
NEG_ONE=-1082130432
def fold(values):
 out=0
 for value in values:out=A.add(out,value)
 return out
def sum1024(words):
 if len(words)!=1024:raise ValueError('EXACT1024_REQUIRED')
 # Eight lanes, four ILP accumulators, two chunks of sixteen.
 # Load index: chunk*512+j*32+k*8+lane. Direct fixed index expansion,
 # independent of the candidate's four-level carry loop.
 lanes=[]
 for lane in range(8):
  groups=[]
  for k in range(4):
   left=fold(words[j*32+k*8+lane]for j in range(16))
   right=fold(words[512+j*32+k*8+lane]for j in range(16))
   merged=A.add(A.add(0,left),right)
   groups.append(A.add(A.add(A.add(0,merged),0),0))
  value=groups[0]
  for v in groups[1:]:value=A.add(value,v)
  lanes.append(value)
 return fold(lanes)
def mean_center(words):
 total=sum1024(words);offset=M.divide(total,0x44800000)
 return total,offset,[A.subtract(w,offset)for w in words]
def encode(centered,dynamic):
 if len(centered)!=1024 or len(dynamic)!=256:raise ValueError('ENCODING_SHAPE')
 values=[A.value(w)for w in dynamic]
 if values!=sorted(values):raise ValueError('SORTED_DYNAMIC_MAP')
 bounds=[M.divide(A.add(dynamic[i],dynamic[i+1]),0x40000000)for i in range(255)]
 bounds=[A.value(v)for v in bounds];codes=[];second=[]
 for block in range(4):
  chunk=centered[block*256:(block+1)*256]
  maximum=max((w&0x7fffffff)for w in chunk);second.append(maximum)
  denominator=maximum if A.value(maximum)>=A.value(FLOOR)else FLOOR
  reciprocal=M.divide(ONE,denominator)
  for w in chunk:
   normalized=M.multiply(w,reciprocal);v=A.value(normalized)
   normalized=NEG_ONE if v<-1 else ONE if v>1 else normalized
   codes.append(sum(A.value(normalized)>b for b in bounds))
 return codes,second
def nested(words,dynamic):
 total,offset,centered=mean_center(words);codes,second=encode(centered,dynamic)
 restored=[A.add(M.multiply(dynamic[code],second[i//256]),offset)for i,code in enumerate(codes)]
 return {'sum':total,'offset':offset,'centered':centered,'codes':codes,'second':second,'restored':restored}
def serial_mean(words):return M.divide(fold(words),0x44800000)
