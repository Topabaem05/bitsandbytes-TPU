"""Exact supplement replay and unchanged primary-CPU mathematical gates."""
import math
import dyadic as D
def primary_math(primary,supplement,case,tolerance):
 expected_tolerance={'atol':1e-5,'rtol':1e-4}if case['dtype']=='float32'else{'atol':0.02,'rtol':0.02,'nrmse_max':0.02}
 D.require(tolerance==expected_tolerance,'UNCHANGED_ORIGINAL_TOLERANCE_REQUIRED')
 D.require(set(primary)=={'case','inputs_sha256','outputs'}and primary['case']==supplement['case']==case['id']and primary['inputs_sha256']==supplement['inputs_sha256'],'PRIMARY_INPUT_IDENTITY')
 D.require(set(primary['outputs'])==set(supplement['outputs']),'PRIMARY_OUTPUT_KEYS');gates={}
 for k,v in supplement['outputs'].items():
  x=primary['outputs'][k];D.require(set(x)=={'shape','dtype','values'}and x['shape']==v['shape']and x['dtype']==v['dtype']and len(x['values'])==len(v['words']),'PRIMARY_METADATA')
  for value in x['values']:
   word=D.float_word(value);D.require(x['dtype']!='bfloat16'or word&0xffff==0,'EXACT_BF16_VALUE_REQUIRED')
  target=D.values(v);errors=[abs(a-b)for a,b in zip(x['values'],target)]
  passed=all(e<=tolerance['atol']+tolerance['rtol']*abs(b)for e,b in zip(errors,target))
  rmse=math.sqrt(sum(e*e for e in errors)/len(errors));nrmse=rmse/max(math.sqrt(sum(b*b for b in target)/len(target)),1e-12)
  if 'nrmse_max'in tolerance:passed=passed and nrmse<=tolerance['nrmse_max']
  gates[k]={'status':'PASS'if passed else'FAIL','max_abs':max(errors),'rmse':rmse,'nrmse':nrmse};D.require(passed,'ORIGINAL_DEVICE_GATE:'+k)
 return gates
def verify(primary,supplement,case,inputs,code_words,tolerance):
 D.verify_supplement(supplement,case,inputs,code_words)
 return primary_math(primary,supplement,case,tolerance)
