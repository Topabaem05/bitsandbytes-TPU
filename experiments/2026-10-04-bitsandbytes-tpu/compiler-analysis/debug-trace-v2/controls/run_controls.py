"""Isolated CPU controls. Synthetic module bytes are not actual HLO records."""
import copy
import hashlib
import json
import pathlib
import shutil
import subprocess
import sys
import tempfile
sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from lazy_hash_v2 import computation_hash, hash_to_string, data_hash
from debug_trace_v2 import verify, TraceError, BAR, DASH, VERSION, MAX_FILE
NAMES = ('prepare','target','reference')
# Minimal schema-valid protobuf fixtures. They are not executable HLO modules.
def vint(n):
    out = bytearray()
    while n > 127: out.append((n & 127) | 128); n >>= 7
    out.append(n); return bytes(out)
def message(field, body):
    return vint((field << 3) | 2) + vint(len(body)) + body
ARRAY = b'\x10\x0b'  # ShapeProto.element_type=F32.
def tuple_shape(*children):
    return b'\x10\x0d' + b''.join(message(4,c) for c in children)
def program(inputs=1, result=ARRAY):
    return b''.join(message(1,ARRAY) for _ in range(inputs)) + message(2,result)
def module(name, inputs=1, result=ARRAY, helper_inputs=0):
    helper = message(3, message(1,b'helper')+message(4,program(helper_inputs))) if helper_inputs else b''
    return message(1,name.encode())+helper+message(4,program(inputs,result))+b'\x28\x01'
MODULES = tuple(module('fixture-'+n) for n in NAMES)
KEYS = ('1'+'0'*16, '2'+'0'*16, '3'+'0'*16)

def graph(kind,name,key,inputs=1,outputs=1):
    p = kind + ' Analysis: '
    values = (BAR, 'Compilation Cause' if kind == 'Compilation' else 'Execution Cause',
        '  fixture synchronization', 'Graph Info: ', '  Graph Name: '+name,
        '  Graph Hash: '+key, '  Number of Graph Inputs: '+str(inputs), '  Number of Graph Outputs: '+str(outputs),
        'Python Frame Triggered Execution: ', '  fixture (fixture.py:1)', DASH, BAR)
    return '\n' + '\n'.join(p+v for v in values) + '\n'

def post(h):
    p = 'Post Compilation Analysis: '
    fields = ('Graph input size','Graph output size','Aliased Input size','Intermediate tensor size','Compiled program size')
    return '\nComputation hash: '+h+'\n\n'+p+BAR+'\n'+''.join(p+n+': 0.000001 GB\n' for n in fields)+p+DASH+'\n'+p+BAR+'\n'

def cold(i):
    return graph('Compilation',NAMES[i],KEYS[i])+post(computation_hash(MODULES[i]))+graph('Execution',NAMES[i],KEYS[i])

def make(parts=None):
    parts = [cold(i) for i in range(3)] if parts is None else parts
    raw=b''.join(p.encode() for p in parts); start=0; windows=[]
    for i,part in enumerate(parts):
        end=start+len(part.encode()); module=MODULES[i]
        windows.append({'phase':NAMES[i],'graph_name':NAMES[i],'start':start,'end':end,
            'sync_completed':True,'readback_completed':True,'module_format':'DETERMINISTIC_FULL_HLO_MODULE_PROTO',
            'module_bytes':module,'module_sha256':hashlib.sha256(module).hexdigest(),
            'submitted_module_bytes':module,'submitted_module_sha256':hashlib.sha256(module).hexdigest(),
            'submitted_module_format':'DETERMINISTIC_FULL_HLO_MODULE_PROTO'});start=end
    return {'kind':VERSION,'scope':{'fresh_process':True,'single_device':True,'serial':True,'spmd':False,
        'eager':False,'persistent_cache':False,'debug_level':2},'process_nonce':'a'*32,
        'started_epoch':100.,'finished_epoch':110.,'deadline_epoch':120.,'parent_deadline_epoch':120.,
        'raw_log':raw,'raw_log_sha256':hashlib.sha256(raw).hexdigest(),'windows':windows}

def run():
    checks=[]
    def good(name,fn):
        fn();checks.append({'name':name,'status':'PASS'})
    def rejects(name,request,label,now=111.):
        try:verify(request,now_epoch=now)
        except TraceError as e:
            assert str(e)==label,(name,str(e),label)
        else:raise AssertionError(name+' accepted invalid record')
        checks.append({'name':name,'status':'PASS','rejection':label})
    build=pathlib.Path(tempfile.mkdtemp(prefix='hash-build-',dir=str(ROOT/'controls')))
    try:
        exe=build/'hash-reference'
        subprocess.run(['c++','-std=c++17','-O2',str(ROOT/'controls/hash_reference.cpp'),'-o',str(exe)],check=True,capture_output=True)
        vectors=[b'',bytes(range(15)),bytes(range(16)),bytes(range(17))]
        leading=next(bytes([i]) for i in range(256) if data_hash(bytes([i])) >> 124 == 0)
        vectors.append(leading)
        vector_rows=[]
        for b in vectors:
            observed=subprocess.check_output([str(exe),b.hex()],text=True).strip()
            expected=computation_hash(b)
            assert observed==expected
            vector_rows.append({'input_hex':b.hex(),'bytes':len(b),'cpp':observed,'python':expected})
            checks.append({'name':'hash_'+str(len(b))+'B_'+b.hex(),'status':'PASS'})
        for h in (0,1,1<<64,(0x0123456789abcdef<<64)|1,(1<<128)-1):
            observed=subprocess.check_output([str(exe),'--format',format(h,'x')],text=True).strip()
            assert observed==hash_to_string(h)
            checks.append({'name':'format_'+format(h,'x'),'status':'PASS'})
        assert hash_to_string(1<<64)=='10000000000000000'
        assert computation_hash(b'x') != hashlib.sha256(b'x').hexdigest()
        assert int.from_bytes(bytes(range(16)),'little').to_bytes(16,'little').hex() != hash_to_string(int.from_bytes(bytes(range(16)),'little'))
        checks.append({'name':'wrong_hash_and_native_byte_format','status':'PASS'})
    finally:
        shutil.rmtree(build)
    good('complete_cold_execute_all_phases',lambda:verify(make(),now_epoch=111.))
    good('retrospective_review_now1000_finished110_deadline120',lambda:verify(make(),now_epoch=1000.))
    future=make();future['finished_epoch']=112.
    rejects('future_finish',future,'ORIGINAL_DEADLINE')
    late=make();late['finished_epoch']=121.
    rejects('late_finish_review_now1000',late,'ORIGINAL_DEADLINE',now=1000.)
    def custom(submitted,optimized,ci=1,co=1,ei=1,eo=1,warm=False):
        target=graph('Compilation','target',KEYS[1],ci,co)+post(computation_hash(optimized))+graph('Execution','target',KEYS[1],ei,eo)
        if warm: target+=graph('Execution','target',KEYS[1],ei,eo)
        request=make([cold(0),target,cold(2)])
        request['windows'][1].update(module_bytes=optimized,module_sha256=hashlib.sha256(optimized).hexdigest(),
            submitted_module_bytes=submitted,submitted_module_sha256=hashlib.sha256(submitted).hexdigest())
        return request
    changed=custom(module('submitted',inputs=2),module('optimized'),ci=2)
    observed=verify(changed,now_epoch=1000.)['windows'][1]['events'][0]
    assert observed['compilation_arity']=={'inputs':2,'outputs':1}
    assert observed['execution_arity']=={'inputs':1,'outputs':1}
    checks.append({'name':'valid_separate_submitted_and_optimized_arities','status':'PASS'})
    nested=tuple_shape(tuple_shape(ARRAY,ARRAY),ARRAY)
    good('nested_tuple_top_level_two_outputs_not_three_leaves',lambda:verify(custom(module('submitted',result=nested),module('optimized',result=nested),co=2,eo=2),now_epoch=1000.))
    good('valid_empty_tuple_zero_outputs',lambda:verify(custom(module('submitted',result=tuple_shape()),module('optimized',result=tuple_shape()),co=0,eo=0),now_epoch=1000.))
    good('ignore_internal_helper_parameters',lambda:verify(custom(module('submitted',helper_inputs=4),module('optimized',helper_inputs=7)),now_epoch=1000.))
    for name,kwargs,label in [
        ('corrupt_cold_inputs999',{'ei':999},'EXECUTION_ARITY_MISMATCH'),
        ('corrupt_cold_outputs999',{'eo':999},'EXECUTION_ARITY_MISMATCH'),
        ('corrupt_compilation_inputs999',{'ci':999},'COMPILATION_ARITY_MISMATCH'),
        ('corrupt_compilation_outputs999',{'co':999},'COMPILATION_ARITY_MISMATCH'),
    ]: rejects(name,custom(MODULES[1],MODULES[1],**kwargs),label)
    for field in ('Inputs','Outputs'):
        parts=[cold(i) for i in range(3)];parts[1]+=graph('Execution','target',KEYS[1],999 if field=='Inputs' else 1,999 if field=='Outputs' else 1)
        rejects('corrupt_warm_'+field.lower(),make(parts),'EXECUTION_ARITY_MISMATCH')
    rejects('tuple_recursive_leaf_count_rejected',custom(module('s',result=nested),module('o',result=nested),co=2,eo=3),'EXECUTION_ARITY_MISMATCH')
    parts=[cold(i) for i in range(3)];parts[1]=parts[1].replace('  fixture synchronization','  Number of Graph Inputs: 999',1)
    rejects('duplicate_arity_in_cause',make(parts),'DUPLICATE_ARITY_FIELDS')
    request=make();request['windows'][1].pop('submitted_module_bytes')
    rejects('missing_submitted_evidence_unsupported',request,'UNSUPPORTED_SUBMITTED_MODULE_REQUIRED')
    rejects('missing_optimized_host_shape_unsupported',custom(MODULES[1],message(1,b'missing-host')),'UNSUPPORTED_MODULE_HOST_PROGRAM_SHAPE')
    rejects('missing_submitted_host_shape_unsupported',custom(message(1,b'missing-host'),MODULES[1]),'UNSUPPORTED_MODULE_HOST_PROGRAM_SHAPE')
    request=make();request['windows'][1]['submitted_module_sha256']='0'*64
    rejects('submitted_module_bad_seal',request,'SUBMITTED_MODULE_SEAL')
    request=make();request['windows'][1]['submitted_module_format']='HLO_PROTO_ENVELOPE'
    rejects('submitted_module_envelope_format',request,'SUBMITTED_MODULE_FORMAT')

    parts=[cold(i) for i in range(3)];parts[1]+=graph('Execution','target',KEYS[1])
    report=verify(make(parts),now_epoch=111.)
    assert report['windows'][1]['events'][1]['mode']=='OBSERVED_WARM_EXECUTION_ANALYSIS'
    checks.append({'name':'observed_cold_then_warm','status':'PASS'})
    parts=[cold(i) for i in range(3)];parts[1]=graph('Execution','target',KEYS[1])
    rejects('orphan_warm',make(parts),'ORPHAN_WARM_EXECUTION')
    parts=[cold(i) for i in range(3)];parts[1]=graph('Execution','target',KEYS[0])
    rejects('warm_key_for_wrong_module',make(parts),'WARM_COMPUTATION_HASH_MISMATCH')
    parts=[cold(i) for i in range(3)];parts[1]=parts[1].replace('  fixture synchronization','  Graph Hash: '+KEYS[1],1)
    rejects('duplicate_graph_hash_in_cause',make(parts),'DUPLICATE_GRAPH_FIELDS')
    parts=[cold(i) for i in range(3)];parts[1]=parts[1][:-1]
    rejects('truncated_newline',make(parts),'LOG_TRUNCATED')
    parts=[cold(i) for i in range(3)];parts[1]=parts[1].rsplit('Execution Analysis: '+DASH,1)[0]+'\n'
    rejects('truncated_block',make(parts),'BLOCK_INTERLEAVING_OR_TRUNCATION')
    parts=[cold(i) for i in range(3)];parts[1]=parts[1].replace('Compilation Analysis:   Graph Hash:', 'Execution Analysis:   Graph Hash:',1)
    rejects('interleaved_block',make(parts),'BLOCK_INTERLEAVING_OR_TRUNCATION')
    parts=[cold(i) for i in range(3)];parts[1]=parts[1].replace(computation_hash(MODULES[1]),computation_hash(b'wrong-module'))
    rejects('wrong_computation',make(parts),'COMPUTATION_HASH_MISMATCH')
    parts=[cold(i) for i in range(3)];parts[1]=parts[1].replace(computation_hash(MODULES[1]),computation_hash(b'\x0a'+bytes([len(MODULES[1])])+MODULES[1]))
    rejects('whole_envelope_hash',make(parts),'COMPUTATION_HASH_MISMATCH')
    parts=[cold(i) for i in range(3)];parts[1]=parts[1].replace('Execution Analysis:   Graph Hash: '+KEYS[1],'Execution Analysis:   Graph Hash: '+KEYS[0])
    rejects('wrong_cold_execution_key',make(parts),'COLD_EXECUTION_HASH_MISMATCH')
    parts=[cold(i) for i in range(3)];parts[1]=parts[1].replace('Graph Name: target','Graph Name: other')
    rejects('wrong_graph_name',make(parts),'GRAPH_NAME')
    parts=[cold(i) for i in range(3)];parts[1]=graph('Compilation','target',KEYS[1])+post(computation_hash(MODULES[1]))
    rejects('compile_without_execution',make(parts),'INCOMPLETE_WINDOW')
    parts=[cold(i) for i in range(3)];parts[1]+=cold(1)
    rejects('duplicate_compile_same_key',make(parts),'RECOMPILE_AMBIGUOUS_CACHE_KEY')
    parts=[cold(i) for i in range(3)];parts[1]+='unassigned output\n'
    rejects('unknown_log_output',make(parts),'UNKNOWN_OR_INTERLEAVED_LOG_LINE')
    for name,change,label in [
        ('missing_reference',lambda r:r['windows'].pop(),'COMPLETE_PHASE_WINDOWS'),
        ('offset_gap',lambda r:r['windows'][1].update(start=r['windows'][1]['start']+1),'COMPLETE_CONTIGUOUS_OFFSETS'),
        ('no_sync',lambda r:r['windows'][1].update(sync_completed=False),'SYNC_READBACK_REQUIRED'),
        ('no_readback',lambda r:r['windows'][1].update(readback_completed=False),'SYNC_READBACK_REQUIRED'),
        ('wrong_module_seal',lambda r:r['windows'][1].update(module_sha256='0'*64),'MODULE_SEAL'),
        ('wrong_log_seal',lambda r:r.update(raw_log_sha256='0'*64),'LOG_SEAL'),
        ('envelope_format',lambda r:r['windows'][1].update(module_format='HLO_PROTO_ENVELOPE'),'MODULE_FORMAT'),
        ('finish_after_deadline109',lambda r:r.update(deadline_epoch=109.),'ORIGINAL_DEADLINE'),
        ('extended_parent_deadline',lambda r:r.update(deadline_epoch=121.),'ORIGINAL_DEADLINE'),
        ('spmd',lambda r:r['scope'].update(spmd=True),'UNSUPPORTED_SCOPE'),
        ('persistent_cache',lambda r:r['scope'].update(persistent_cache=True),'UNSUPPORTED_SCOPE'),
        ('invalid_nonce',lambda r:r.update(process_nonce='none'),'PROCESS_NONCE'),
        ('overlarge_log',lambda r:r.update(raw_log=b'x'*(MAX_FILE+1)),'LOG_BOUND'),
    ]:
        request=make();change(request);rejects(name,request,label)
    result={'kind':'M6_DEBUG_HASH_V2_ARITY_REV1_CPU_CONTROLS','status':'PASS','checks':checks,'count':len(checks),
        'hash_vectors':vector_rows,'actual_device':'NOT_RUN','loaded_handle_link':'UNKNOWN','m6_status':'NOT_QUALIFIED',
        'generated_fixture_files_retained':0,'compiled_reference_retained':False}
    (ROOT/'controls/results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'PASS','count':len(checks),'output':str(ROOT/'controls/results.json')}))
if __name__=='__main__':run()
