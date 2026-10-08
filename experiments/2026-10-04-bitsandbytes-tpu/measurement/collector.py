"""Stdlib M7 collection protocol. No device import; actual execution needs a separately admitted runner."""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time

HERE=Path(__file__).resolve().parent
PROTOCOL=json.loads((HERE/'protocol.json').read_text())
COMPILE_KEYS=('CompileTime','EagerOpCompileTime')
EXECUTE_KEYS=('ExecuteTime','ExecuteReplicatedTime','EagerOpExecuteTime')

def require(v,m):
    if not v:raise ValueError(m)
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def finite(v):return type(v) in (int,float) and math.isfinite(v)
def validate_contract(v):
    require(v.get('selected_path') in PROTOCOL['comparison'],'SELECTED_PATH_CONTRACT')
    require(v.get('hardware',{}).get('type')=='TPU' and all(v['hardware'].get(k) for k in ('kind','device_binding','allocation_id')),'SAME_IDENTIFIED_TPU_CONTRACT')
    for key in ('m6_acceptance_sha256','source_admission_sha256','input_sha256','runtime_lock_sha256','kernel_sha256','oracle_sha256'):
        require(re.fullmatch('[0-9a-f]{64}',v.get(key,'')) is not None,'SEALED_CONTRACT_'+key)
    require(v['runtime_lock_sha256']==PROTOCOL['runtime_lock_sha256'] and v.get('precision')=='highest','FIXED_RUNTIME_PRECISION')
    require(v.get('case_id') and v.get('shape') and v.get('dtype') in ('float32','bfloat16'),'QUALIFIED_CASE_CONTRACT')
    require(v.get('source_variant') and v.get('kernel_sha256') and v.get('oracle_sha256'),'SOURCE_KERNEL_ORACLE_CONTRACT')
    require(v.get('measurement_boundary')==PROTOCOL['sample_boundary'],'COMPLETE_CALL_BOUNDARY')
    return digest(v)
def metric(data):
    if data is None:return {'total_samples':0,'accumulator_ns':0,'samples_seconds_timestamp_ns_value':[],'source_status':'ABSENT_AFTER_CLEAR'}
    require(isinstance(data,(list,tuple)) and len(data)==3,'METRIC_TRIPLE');count,acc,samples=data
    require(type(count) is int and count>=0 and finite(acc) and acc>=0 and isinstance(samples,(list,tuple)),'METRIC_COUNT_ACCUMULATOR')
    require(all(isinstance(v,(list,tuple)) and len(v)==2 and all(finite(x) for x in v) and v[1]>=0 for v in samples),'METRIC_FINITE_SAMPLES')
    require((count==0 and not samples and acc==0) or (count>0 and 0<len(samples)<=count),'METRIC_SAMPLE_COUNTS')
    return {'total_samples':count,'accumulator_ns':acc,'samples_seconds_timestamp_ns_value':copy.deepcopy(list(samples)),'source_status':'REPORTED'}
def metrics(data):
    require(isinstance(data,dict) and set(data)==set(COMPILE_KEYS+EXECUTE_KEYS),'EXACT_METRIC_KEYS');return {k:metric(v) for k,v in data.items()}
def memory(read):
    try:value=read()
    except Exception as error:return {'status':'UNKNOWN','error':{'type':type(error).__name__,'message':str(error)},'current_bytes':'UNKNOWN','reported_high_water_bytes':'UNKNOWN','target_interval_peak':'UNKNOWN'}
    require(isinstance(value,dict),'MEMORY_INFO_DICT')
    require(set(value)<=set(PROTOCOL['memory_fields']) and 'bytes_used' in value and 'bytes_limit' in value,'MEMORY_FIELDS')
    used,limit=value['bytes_used'],value['bytes_limit'];require(type(used)is int and type(limit)is int and 0<=used<=limit and limit>0,'MEMORY_CURRENT_LIMIT')
    peak=value.get('peak_bytes_used');require(peak is None or (type(peak)is int and used<=peak<=limit),'MEMORY_PEAK_VALIDITY')
    return {'status':'REPORTED','raw':copy.deepcopy(value),'current_bytes':used,'limit_bytes':limit,'reported_high_water_bytes':peak if peak is not None else 'UNKNOWN','reported_high_water_scope':PROTOCOL['allocator_peak_scope'],'target_interval_peak':'UNKNOWN','kernel_vmem':'UNKNOWN'}
def sync_receipt(v):
    require(v=={'api':'_xla_sync_multi','targets':['actual'],'wait':True,'sync_xla_data':True,'wait_device_ops':True},'ACTUAL_SYNC_REQUIRED')

def collect(contract,*,setup,runner,clock_ns=time.perf_counter_ns):
    """runner must audit sealed identity and sync the exact target returned by each call."""
    seal=validate_contract(contract);record={'kind':'R7_RAW_MEASUREMENT','status':'PARTIAL','protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),'contract':copy.deepcopy(contract),'contract_sha256':seal,'samples':[],'warmups':[],'compile_first_call':None,'setup':None,'memory_snapshots':[],'target_interval_peak':'UNKNOWN','compiler_allocation_plan':'SEPARATE_ARTIFACT','m6':'REQUIRES_PRIOR_REVIEWED_ACCEPTANCE','m7':'NOT_QUALIFIED'}
    def audit():require(runner.audit()==contract,'SOURCE_INPUT_RUNTIME_HARDWARE_AUDIT')
    def memory_at(phase):
        start=clock_ns();snapshot=memory(runner.memory_info);finish=clock_ns()
        require(type(start)is int and type(finish)is int and finish>=start>=0,'MEMORY_SNAPSHOT_CLOCK')
        record['memory_snapshots'].append({'phase':phase,'started_ns':start,'finished_ns':finish,'snapshot':snapshot})
    def iteration(phase,index,group=None):
        row={'phase':phase,'index':index,'group':group,'status':'STARTED'}
        if phase=='MEASURED':record['samples'].append(row)
        elif phase=='WARMUP':record['warmups'].append(row)
        else:record['compile_first_call']=row
        audit();runner.drain();runner.clear_metrics();start=clock_ns();require(type(start)is int and start>=0,'CLOCK_START')
        target,event=call();require(event=={'path':contract['selected_path'],'fallback':False,'calls':1},'EXACT_SELECTED_PATH_NO_FALLBACK')
        receipt=runner.sync_actual(target);sync_receipt(receipt);finish=clock_ns();require(type(finish)is int and finish>start,'CLOCK_POSITIVE_INTERVAL')
        observed=metrics(runner.metrics());require(sum(observed[k]['total_samples'] for k in EXECUTE_KEYS)>0,'POSITIVE_ACTUAL_EXECUTION')
        counters=runner.counters();require(isinstance(counters,dict) and all(type(v)is int and v>=0 for v in counters.values()),'FINITE_COUNTERS');require(not any(k.startswith('aten::') and v>0 for k,v in counters.items()),'NO_ATEN_FALLBACK')
        compile_count=sum(observed[k]['total_samples'] for k in COMPILE_KEYS)
        require(phase!='MEASURED' or all(counters.get(k,0)==0 for k in ('CreateCompileHandles','UncachedCompile')),'COMPILATION_COUNTER_IN_MEASURED_SAMPLE')
        row.update(started_ns=start,finished_ns=finish,elapsed_ns=finish-start,elapsed_seconds=(finish-start)*1e-9,sync=receipt,metrics=observed,counters=copy.deepcopy(counters),selected=event)
        # Release the completed output before memory reads or the next call.
        del target
        require(phase!='MEASURED' or compile_count==0,'COMPILATION_IN_MEASURED_SAMPLE');row['status']='COMPLETE'
        return row
    try:
        audit();memory_at('BEFORE_SETUP');runner.drain();start=clock_ns();call=setup();receipt=runner.sync_setup();require(receipt=={'api':'_xla_sync_multi','targets':['canonical_inputs'],'wait':True,'sync_xla_data':True,'wait_device_ops':True},'SETUP_INPUT_SYNC_REQUIRED');finish=clock_ns();require(type(start)is int and type(finish)is int and finish>start,'SETUP_CLOCK_INTERVAL');record['setup']={'started_ns':start,'finished_ns':finish,'elapsed_ns':finish-start,'scope':'PREQUANTIZED_INPUT_TRANSFER_AND_OPERATOR_SETUP_THROUGH_INPUT_SYNC','sync':receipt};memory_at('AFTER_SETUP')
        row=iteration('COLD_FIRST_CALL',0);record['cold_call_wall_scope']='FIRST_CALL_HOST_SETUP_TRACE_COMPILE_AND_EXECUTE_NOT_PURE_COMPILE';record['backend_compile_status']='REPORTED' if sum(row['metrics'][k]['total_samples'] for k in COMPILE_KEYS)>0 else 'NOT_OBSERVED';record['backend_compile_seconds']=sum(row['metrics'][k]['accumulator_ns'] for k in COMPILE_KEYS)*1e-9 if record['backend_compile_status']=='REPORTED' else 'UNKNOWN';record['backend_compile_scope']='PINNED_CLIENT_COMPILE_TIMED_SECTION_NOT_WHOLE_JAX_TRACE';memory_at('AFTER_FIRST_CALL')
        for i in range(PROTOCOL['warmups']):iteration('WARMUP',i)
        memory_at('AFTER_WARMUPS')
        for g in range(PROTOCOL['groups']):
            memory_at('BEFORE_GROUP_'+str(g))
            for i in range(PROTOCOL['iterations_per_group']):iteration('MEASURED',i,g)
            memory_at('AFTER_GROUP_'+str(g))
        audit();record['summary_by_group']=[{'group':g,'count':30,'median_seconds':statistics.median([r['elapsed_seconds'] for r in record['samples'] if r['group']==g]),'mean_seconds':statistics.mean([r['elapsed_seconds'] for r in record['samples'] if r['group']==g])} for g in range(3)];record['status']='COMPLETE'
    except Exception as error:
        record.update(status='ERROR',error={'type':type(error).__name__,'message':str(error)})
        for row in record['samples']+record['warmups']+([record['compile_first_call']] if record['compile_first_call'] else []):
            if row['status']=='STARTED':row['status']='ERROR'
    finally:
        try:audit();record['final_source_audit']='PASS'
        except Exception as error:record.update(status='ERROR',final_source_audit='FAIL',final_source_error={'type':type(error).__name__,'message':str(error)})
    return record


SETUP_SYNC={'api':'_xla_sync_multi','targets':['canonical_inputs'],'wait':True,'sync_xla_data':True,'wait_device_ops':True}
SETUP_SCOPE='PREQUANTIZED_INPUT_TRANSFER_AND_OPERATOR_SETUP_THROUGH_INPUT_SYNC'
COLD_SCOPE='FIRST_CALL_HOST_SETUP_TRACE_COMPILE_AND_EXECUTE_NOT_PURE_COMPILE'
COMPILE_SCOPE='PINNED_CLIENT_COMPILE_TIMED_SECTION_NOT_WHOLE_JAX_TRACE'
MEMORY_STAGES=['BEFORE_SETUP','AFTER_SETUP','AFTER_FIRST_CALL','AFTER_WARMUPS']+[phase+str(g) for g in range(3) for phase in ('BEFORE_GROUP_','AFTER_GROUP_')]

def interval(row,*,allow_zero=False):
    require(type(row.get('started_ns'))is int and type(row.get('finished_ns'))is int and row['started_ns']>=0 and (row['finished_ns']>=row['started_ns'] if allow_zero else row['finished_ns']>row['started_ns']),'RAW_ORDERED_TIME_INTERVAL')
    return row['started_ns'],row['finished_ns']

def recovered_memory(snapshot):
    require(isinstance(snapshot,dict),'RAW_MEMORY_SNAPSHOT_SCHEMA')
    if snapshot.get('status')=='REPORTED':
        require('raw'in snapshot and memory(lambda:snapshot['raw'])==snapshot,'RAW_MEMORY_RECOMPUTATION')
    else:
        require(snapshot.get('status')=='UNKNOWN' and set(snapshot)=={'status','error','current_bytes','reported_high_water_bytes','target_interval_peak'} and isinstance(snapshot.get('error'),dict) and set(snapshot['error'])=={'type','message'} and isinstance(snapshot['error']['type'],str) and bool(snapshot['error']['type']) and isinstance(snapshot['error']['message'],str) and all(snapshot[k]=='UNKNOWN' for k in ('current_bytes','reported_high_water_bytes','target_interval_peak')),'RAW_MEMORY_UNKNOWN_SCHEMA')

def setup_and_memory(record):
    setup=record.get('setup');require(isinstance(setup,dict) and set(setup)=={'started_ns','finished_ns','elapsed_ns','scope','sync'} and setup.get('scope')==SETUP_SCOPE and setup.get('sync')==SETUP_SYNC,'RAW_SETUP_CONTRACT')
    start,finish=interval(setup);require(type(setup.get('elapsed_ns'))is int and setup['elapsed_ns']==finish-start,'RAW_SETUP_DURATION')
    snapshots=record.get('memory_snapshots');require(isinstance(snapshots,list) and [r.get('phase') for r in snapshots]==MEMORY_STAGES,'RAW_MEMORY_STAGE_MATRIX')
    for row in snapshots:
        require(set(row)=={'phase','started_ns','finished_ns','snapshot'},'RAW_MEMORY_STAGE_SCHEMA');interval(row,allow_zero=True);recovered_memory(row['snapshot'])
    # Memory reads and samples have separate intervals, in their recorded order.
    sequence=[snapshots[0],setup,snapshots[1],record['compile_first_call'],snapshots[2],*record['warmups'],snapshots[3]]
    for g in range(3):sequence += [snapshots[4+g*2],*record['samples'][g*30:(g+1)*30],snapshots[5+g*2]]
    previous=0
    for row in sequence:
        start,finish=interval(row,allow_zero='snapshot'in row);require(start>=previous,'RAW_CHRONOLOGICAL_SEQUENCE');previous=finish

def recovered_compile(record):
    cold=record['compile_first_call'];count=sum(cold['metrics'][k]['total_samples']for k in COMPILE_KEYS)
    status='REPORTED'if count>0 else'NOT_OBSERVED';seconds=sum(cold['metrics'][k]['accumulator_ns']for k in COMPILE_KEYS)*1e-9 if count>0 else'UNKNOWN'
    require(record.get('cold_call_wall_scope')==COLD_SCOPE and record.get('backend_compile_scope')==COMPILE_SCOPE and record.get('backend_compile_status')==status and record.get('backend_compile_seconds')==seconds,'RAW_COLD_COMPILE_RECOMPUTATION')

def compare(left,right):
    require(left.get('status')==right.get('status')=='COMPLETE','TWO_COMPLETE_PATHS')
    a,b=copy.deepcopy(left['contract']),copy.deepcopy(right['contract']);paths={a.pop('selected_path'),b.pop('selected_path')};require(paths==set(PROTOCOL['comparison']) and a==b,'SAME_HARDWARE_INPUT_SOURCE_BOUNDARY')
    require(all(len(v['warmups'])==5 and len(v['samples'])==90 and all(r['status']=='COMPLETE' for r in v['samples']) and [g['count'] for g in v['summary_by_group']]==[30,30,30] for v in (left,right)),'FIXED_SAMPLE_MATRIX')
    for record in (left,right):
        require(record.get('kind')=='R7_RAW_MEASUREMENT' and not record.get('error') and not record.get('final_source_error') and record.get('compiler_allocation_plan')=='SEPARATE_ARTIFACT' and record.get('m6')=='REQUIRES_PRIOR_REVIEWED_ACCEPTANCE' and record.get('m7')=='NOT_QUALIFIED','RAW_RECORD_SCOPE')
        require(record.get('contract_sha256')==validate_contract(record['contract']) and record.get('protocol_sha256')==hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),'REVIEWED_CONTRACT_PROTOCOL')
        require([(row['group'],row['index']) for row in record['samples']]==[(g,i) for g in range(3) for i in range(30)],'ORDERED_SAMPLE_MATRIX')
        require(record.get('compile_first_call') and [r.get('index') for r in record['warmups']]==list(range(5)),'COLD_AND_ORDERED_WARMUPS')
        expected=[('COLD_FIRST_CALL',0,None)]+[('WARMUP',i,None)for i in range(5)]+[('MEASURED',i,g)for g in range(3)for i in range(30)]
        for row,(phase,index,group) in zip([record['compile_first_call']]+record['warmups']+record['samples'],expected):
            require(row.get('phase')==phase and type(row.get('index'))is int and row['index']==index and (row.get('group')is None if group is None else type(row.get('group'))is int and row['group']==group),'RAW_ROW_PHASE_INDEX_GROUP')
            require(row.get('status')=='COMPLETE','RAW_ROW_COMPLETE')
            require(set(row['metrics'])==set(COMPILE_KEYS+EXECUTE_KEYS),'RAW_METRIC_KEYS')
            for value in row['metrics'].values():
                canonical=metric(None) if value.get('source_status')=='ABSENT_AFTER_CLEAR' else metric([value['total_samples'],value['accumulator_ns'],value['samples_seconds_timestamp_ns_value']]);require(canonical==value,'RAW_METRIC_VALUES')
            require(type(row.get('started_ns'))is int and type(row.get('finished_ns'))is int and row['finished_ns']>row['started_ns']>=0 and row.get('elapsed_ns')==row['finished_ns']-row['started_ns'] and row.get('elapsed_seconds')==row['elapsed_ns']*1e-9,'RAW_SAMPLE_INTERVAL')
            sync_receipt(row['sync']);require(row['selected']=={'path':record['contract']['selected_path'],'fallback':False,'calls':1},'RAW_SELECTED_PATH')
            require(sum(row['metrics'][k]['total_samples'] for k in EXECUTE_KEYS)>0,'RAW_POSITIVE_EXECUTION')
            require(isinstance(row['counters'],dict) and all(type(v)is int and v>=0 for v in row['counters'].values()) and not any(k.startswith('aten::') and v>0 for k,v in row['counters'].items()),'RAW_FALLBACK_COUNTERS')
            if row['phase']=='MEASURED':require(sum(row['metrics'][k]['total_samples'] for k in COMPILE_KEYS)==0 and all(row['counters'].get(k,0)==0 for k in ('CreateCompileHandles','UncachedCompile')),'RAW_EXECUTION_WITHOUT_COMPILE')
        setup_and_memory(record);recovered_compile(record)
        for g in range(3):
            values=[row['elapsed_seconds'] for row in record['samples'] if row['group']==g];require(record['summary_by_group'][g]=={'group':g,'count':30,'median_seconds':statistics.median(values),'mean_seconds':statistics.mean(values)},'SUMMARY_RAW_RECOMPUTATION')
    require(all(v['final_source_audit']=='PASS' and v['target_interval_peak']=='UNKNOWN' for v in (left,right)),'FINAL_AUDIT_AND_MEMORY_SCOPE')
    return {'status':'COMPARABLE_RAW_RECORDS','scope':PROTOCOL['sample_boundary'],'left_group_medians_seconds':[v['median_seconds'] for v in left['summary_by_group']],'right_group_medians_seconds':[v['median_seconds'] for v in right['summary_by_group']],'speedup_claim':'REQUIRES_INDEPENDENT_RAW_SAMPLE_VERIFICATION','target_interval_peak':'UNKNOWN','compiler_memory':'SEPARATE_ARTIFACT','m7':'NOT_QUALIFIED'}
