"""Deterministic clock, asynchronous execution, memory, and incorrect-record controls. No device imports."""
import argparse
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import collector as C
from xla_runner import XlaRunner


def contract(path):return {'selected_path':path,'hardware':{'type':'TPU','kind':'TPU-v6e-fixture','device_binding':'fixture-allocation-slot-0','allocation_id':'fixture-allocation'},'m6_acceptance_sha256':'a'*64,'source_admission_sha256':'b'*64,'input_sha256':'c'*64,'runtime_lock_sha256':C.PROTOCOL['runtime_lock_sha256'],'precision':'highest','case_id':'qualified-case-fixture','shape':[128,256,512],'dtype':'float32','source_variant':'FIXTURE_ONLY','kernel_sha256':'d'*64,'oracle_sha256':'e'*64,'measurement_boundary':C.PROTOCOL['sample_boundary']}
class Fake:
    def __init__(self,path='PALLAS_CANDIDATE',fault=None):self.contract=contract(path);self.fault=fault;self.now=0;self.executions=0;self.pending=None;self.nmetric=0;self.ncompile=0;self.events=[];self.auditcalls=0;self.memreads=0
    def clock(self):return -1 if self.fault=='clock-negative' else self.now
    def drain(self):C.require(self.pending is None,'FAKE_PENDING_GRAPH');self.events.append('drain')
    def clear_metrics(self):self.nmetric=0;self.ncompile=0;self.events.append('clear')
    def audit(self):
        self.auditcalls+=1;value=copy.deepcopy(self.contract)
        if self.fault=='final-audit' and self.executions==96:value['source_admission_sha256']='f'*64
        return value
    def setup(self):self.now+=50;self.events.append('setup');return self.call
    def sync_setup(self):self.now+=50;self.events.append('setup-sync');return {'api':'_xla_sync_multi','targets':['canonical_inputs'],'wait':True,'sync_xla_data':True,'wait_device_ops':True}
    def call(self):
        self.now+=10;target=object();self.pending=target;self.events.append('call');return target,{'path':self.contract['selected_path'],'fallback':self.fault=='fallback','calls':1}
    def sync_actual(self,target):
        assert target is self.pending;self.events.append('actual-sync')
        self.ncompile=1 if self.executions==0 or (self.fault=='measured-compile' and self.executions==6) else 0
        self.now+=1000 if self.ncompile else 0;self.now+=100;self.executions+=1;self.nmetric=0 if self.fault=='no-execute' else 1;self.pending=None
        return {'api':'_xla_sync_multi','targets':['reference'] if self.fault=='wrong-sync-target' else ['actual'],'wait':self.fault!='async-no-wait','sync_xla_data':True,'wait_device_ops':True}
    def metrics(self):
        v={k:None for k in C.COMPILE_KEYS+C.EXECUTE_KEYS};v['CompileTime']=[1,1000,[[0,1000]]] if self.ncompile else None;v['ExecuteTime']=[self.nmetric,100 if self.nmetric else 0,[[0,float('nan') if self.fault=='nan-metric' else 100]]] if self.nmetric else None;return v
    def counters(self):return {'aten::fixture_fallback':1} if self.fault=='aten-fallback' else {}
    def memory_info(self):
        self.memreads+=1
        if self.fault=='memory-api-absent':raise NotImplementedError('API_ABSENT_FIXTURE')
        return {'bytes_used':100,'bytes_limit':1000,**({} if self.fault=='peak-absent' else {'peak_bytes_used':50 if self.fault=='invalid-peak' else 200})}
    def collect(self):return C.collect(self.contract,setup=self.setup,runner=self,clock_ns=self.clock)

def run(output):
    output=Path(output);output.mkdir(exist_ok=False);rows=[];native=Fake();reference=Fake('SAME_DEVICE_REFERENCE');a,b=native.collect(),reference.collect()
    assert a['status']==b['status']=='COMPLETE' and len(a['warmups'])==5 and len(a['samples'])==90 and native.executions==reference.executions==96
    assert a['setup']['elapsed_ns']==100 and a['compile_first_call']['elapsed_ns']==1110 and math.isclose(a['backend_compile_seconds'],1e-6,rel_tol=1e-14) and all(row['elapsed_ns']==110 for row in a['samples'])
    assert native.events.count('actual-sync')==96 and native.events.count('clear')==96 and all(row['sync']['wait'] for row in a['samples']);C.compare(a,b)
    rows.append({'name':'fixed-five-plus-three-by-thirty-with-cold-compile-separate-and-sync-cost','status':'PASS'})
    for fault,reason in [('wrong-sync-target','ACTUAL_SYNC_REQUIRED'),('async-no-wait','ACTUAL_SYNC_REQUIRED'),('no-execute','POSITIVE_ACTUAL_EXECUTION'),('fallback','EXACT_SELECTED_PATH_NO_FALLBACK'),('aten-fallback','NO_ATEN_FALLBACK'),('measured-compile','COMPILATION_IN_MEASURED_SAMPLE'),('nan-metric','METRIC_FINITE_SAMPLES'),('clock-negative','MEMORY_SNAPSHOT_CLOCK'),('final-audit','SOURCE_INPUT_RUNTIME_HARDWARE_AUDIT'),('invalid-peak','MEMORY_PEAK_VALIDITY')]:
        record=Fake(fault=fault).collect();assert record['status']=='ERROR' and (record.get('error',{}).get('message')==reason or record.get('final_source_error',{}).get('message')==reason),(fault,record);rows.append({'name':fault,'status':'PASS','rejected':reason})
    for fault in ('peak-absent','memory-api-absent'):
        record=Fake(fault=fault).collect();assert record['status']=='COMPLETE' and all(row['snapshot']['reported_high_water_bytes']=='UNKNOWN' for row in record['memory_snapshots']) and record['target_interval_peak']=='UNKNOWN';rows.append({'name':fault+'-explicit-unknown-with-valid-timing','status':'PASS'})
    def reject(name,mutate,reason):
        x,y=copy.deepcopy(a),copy.deepcopy(b);mutate(x,y)
        try:C.compare(x,y);raise AssertionError('FALSE_ACCEPT:'+name)
        except ValueError as e:assert str(e)==reason,(name,str(e),reason);rows.append({'name':name,'status':'PASS','rejected':reason})
    reject('different-allocation-device-binding',lambda x,y:y['contract']['hardware'].__setitem__('device_binding','other'),'SAME_HARDWARE_INPUT_SOURCE_BOUNDARY')
    reject('different-inputs',lambda x,y:y['contract'].__setitem__('input_sha256','f'*64),'SAME_HARDWARE_INPUT_SOURCE_BOUNDARY')
    reject('different-sources',lambda x,y:y['contract'].__setitem__('source_admission_sha256','f'*64),'SAME_HARDWARE_INPUT_SOURCE_BOUNDARY')
    reject('missing-sample',lambda x,y:x['samples'].pop(),'FIXED_SAMPLE_MATRIX')
    reject('wrong-group-index',lambda x,y:x['samples'][0].__setitem__('group',9),'ORDERED_SAMPLE_MATRIX')
    reject('host-submit-only-short-duration',lambda x,y:x['samples'][0].__setitem__('elapsed_ns',10),'RAW_SAMPLE_INTERVAL')
    reject('nonfinite-retained-metric',lambda x,y:x['samples'][0]['metrics']['ExecuteTime']['samples_seconds_timestamp_ns_value'][0].__setitem__(1,float('nan')),'METRIC_FINITE_SAMPLES')
    reject('warmup-sync-not-waited',lambda x,y:x['warmups'][0]['sync'].__setitem__('wait',False),'ACTUAL_SYNC_REQUIRED')
    reject('retained-fallback-counter',lambda x,y:x['samples'][0]['counters'].__setitem__('aten::fallback',1),'RAW_FALLBACK_COUNTERS')
    reject('wrong-reported-median',lambda x,y:x['summary_by_group'][0].__setitem__('median_seconds',1.0),'SUMMARY_RAW_RECOMPUTATION')
    # Use the actual runner implementation to demonstrate submission of the exact
    # output followed by waiting. The injected tensors are source-control doubles.
    device=SimpleNamespace(type='xla');events=[]
    class Tensor:
        def __init__(self):self.device=device
    raw=Tensor();functional=Tensor();input_tensor=Tensor()
    torch=SimpleNamespace(Tensor=Tensor,_is_functional_tensor=lambda v:v is functional,_functionalize_sync=lambda v:events.append('functional-sync'),_from_functional_tensor=lambda v:raw)
    api=SimpleNamespace(_xla_sync_multi=lambda targets,**kw:events.append(('submit',targets,kw)))
    runner=XlaRunner(torch,SimpleNamespace(_XLAC=api),SimpleNamespace(wait_device_ops=lambda:events.append('wait')),None,device,[input_tensor],lambda:contract('PALLAS_CANDIDATE'));receipt=runner.sync_actual(functional)
    assert events==['functional-sync',('submit',[raw],{'devices':[],'wait':True,'sync_xla_data':True}),'wait'];C.sync_receipt(receipt);rows.append({'name':'runner-functional-output-exact-raw-target-submit-then-wait','status':'PASS','scope':'ISOLATED_SOURCE_DOUBLE_NO_TPU'})
    for name,record in [('native-fixture',a),('reference-fixture',b)]:record['scope']='FAKE_CLOCK_SYNC_AND_ALLOCATOR_NOT_A_DEVICE_RESULT';(output/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n')
    result={'status':'PASS','count':len(rows),'controls':rows,'scope':'FAKE_CLOCK_SYNC_ALLOCATOR_AND_SOURCE_RUNNER_NO_TPU','m6':'NOT_QUALIFIED','m7':'NOT_QUALIFIED'};(output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print({'status':'PASS','count':len(rows)})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
