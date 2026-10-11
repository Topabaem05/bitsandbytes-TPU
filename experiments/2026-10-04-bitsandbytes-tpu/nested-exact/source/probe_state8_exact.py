"""Exact nested state8 with fresh owned save and restore children. No cloud calls."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import time
import traceback
import types

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe_nested_exact as E
import exact_admission as A
import record_validation as V
import public_trace as T
import archive_store as Z
import state8_contract as K
import state8_public as S

KIND = 'PRIVATE_NESTED_EXACT_STATE8_PROBE_V1'

def require(value, reason):
    A.require(value, reason)

def fixed_cases(B):
    return K.fixed_cases(B.read(HERE / 'nested-inputs.json'))

def binding(B, args):
    return dict(E.binding(B, args), kind=KIND, state_variant='nested-exact-state8-v1',
                state_probe_sha256=B.sha(__file__), state_public_sha256=B.sha(HERE / 'state8_public.py'),
                nested_oracle_sha256=args.oracle_sha256, case_ids=[c['id'] for c in fixed_cases(B)],
                checkpoint_policy='PLAIN_CPU_FULL_STATE_DICT_EXACT_BYTES',
                restoration='ORIGINAL_TARGET_CONSTRUCTOR_FROM_PREQUANTIZED_STRICT_ASSIGN_FALSE')

def dependencies(B, P, N, args, device=False):
    seal, references, _ = V.verify_cpu(B, P, N, args.oracle, args.oracle_sha256, E.binding(B, args))
    if device:
        gate = A.read(args.cpu_gate)
        require(A.sha(args.cpu_gate) == args.cpu_gate_sha256, 'FRESH_STATE_CPU_GATE_BYTES')
        K.stage_order(accepted79_path=args.accepted79, accepted79_sha256=args.accepted79_sha256,
                      science_manifest_sha256=args.science_manifest_sha256, fresh_cpu_gate=gate,
                      fresh_nested_oracle_sha256=args.oracle_sha256, inputs=B.read(HERE / 'nested-inputs.json'))
        require(gate.get('owned_cpu_pid') == seal['pid'] and gate.get('owned_cpu_pgid') == seal['pgid']
                and gate.get('owned_cpu_token') == seal['process_token'], 'ACTUAL_FRESH_STATE_CPU_LAUNCH')
        for key in ('upstream_wheel_sha256', 'plugin_wheel_sha256', 'cpu_step_result_sha256', 'observation_report_sha256'):
            require(re.fullmatch('[0-9a-f]{64}', gate.get(key, '')) is not None, 'FRESH_STATE_CPU_GATE_' + key)
    return seal, references

def receipt(B, args, root, filename, parent=False, cpu=False):
    root = Path(root); rec = B.read(root / filename)
    require(all(rec.get(k) == v for k,v in binding(B, args).items()), 'STATE_SOURCE_BINDING')
    require(rec.get('status') == 'COMPLETE' and rec.get('source_pre') == rec.get('source_post') == A.INTEGRATION_ADMISSION_SHA,
            'STATE_TERMINAL_SOURCE')
    V.process(rec, group_leader=parent or cpu)
    times = ('started_epoch', 'finished_epoch', 'deadline_epoch', 'started_monotonic', 'finished_monotonic')
    require(all(type(rec.get(k)) in (int,float) and math.isfinite(rec[k]) and rec[k] > 0 for k in times)
            and rec['started_epoch'] <= rec['finished_epoch'] <= rec['deadline_epoch']
            and rec['started_monotonic'] <= rec['finished_monotonic'], 'STATE_FINITE_TIMES')
    if not parent:
        require(rec.get('public_api') == dict(N_PUBLIC(B,args), original_state_method_identity=True), 'STATE_PUBLIC_METHOD_IDENTITY')
        require(rec.get('schemas') == B.read(HERE / 'nine-schemas.json')
                and rec.get('plugin_loaded_code') == {'loaded_plugin_code':'COMPILED_ADMITTED_SOURCE','python_files':11}, 'STATE_NINE_SCHEMAS_PLUGIN')
        if cpu:
            require(rec.get('runtime') == B.load_spec()[0]['runtime']
                    and rec.get('oracle_method') == 'FRESH_NESTED79_DEFAULT_BODY_PLUS_PUBLIC_CPU_STATE', 'STATE_CPU_RUNTIME')
        else:
            require(rec.get('runtime') == B.load_spec()[0]['runtime']
                    and rec.get('device') == {'type':'xla','hardware':'TPU','pjrt':'TPU'}
                    and rec.get('dispatch') == {k:True for k in rec['schemas']}
                    and rec.get('precision') == {'requested':'highest','readback':'highest','set_calls':1,'before_graph':True}, 'STATE_DEVICE_RUNTIME')
    B.check_seal(root, filename, rec)
    return rec

def N_PUBLIC(B,args):
    # Source helper is already admitted by E.sources before this call.
    return E.load(args.nested_reference, 'state_public_expected').PUBLIC

def verify_state_cpu(B, P, N, args):
    _, refs = dependencies(B,P,N,args); root = Path(args.state_oracle)
    require(B.sha(root / 'oracle-seal.json') == args.state_oracle_sha256, 'STATE_ORACLE_SHA')
    rec = receipt(B,args,root,'oracle-seal.json',cpu=True)
    nested = B.read(Path(args.oracle) / 'oracle-seal.json')
    require(rec['pid'] != nested['pid'] and rec['process_token'] != nested['process_token'], 'FRESH_STATE_ORACLE_PROCESS')
    evidence = B.read(root / 'cpu-evidence.json')
    require(evidence == B.read(Path(args.oracle) / 'cpu-evidence.json'), 'STATE_CPU_QUALIFICATION_MATCH')
    required = {'cpu-evidence.json'}; rows = {}; import torch
    for case in fixed_cases(B):
        name = case['id']; stem = 'raw/' + name; required.update(stem+s for s in ('.json','.state.pt'))
        raw = B.read(root/(stem+'.json')); V.complete_case(raw)
        require(raw.get('case_id') == name and raw.get('input_sha256') == B.case_sha(case)
                and raw.get('pid') == rec['pid'] and raw.get('process_token') == rec['process_token'], 'STATE_ORACLE_ROW_IDENTITY')
        N.validate_arrays(B,P,case,raw,False)
        _, state = S.checkpoint(B,N,torch,root/(stem+'.state.pt'),case)
        require(raw.get('state_dict') == state and raw['outputs'] == refs[name]['outputs'], 'STATE_ORACLE_DEFAULT_LINK')
        require(raw.get('quant_state_public_roundtrip') is True and raw.get('packed_serialization_original') is True, 'STATE_ORACLE_PUBLIC_SERIALIZATION')
        metadata = json.loads(bytes(state['weight.quant_state.bitsandbytes__nf4']['values']))
        require(metadata['nested_offset'] == refs[name]['outputs']['offset']['values'][0], 'STATE_OFFSET_ORACLE_LINK')
        for key,dest in [('packed','weight'),('scale_codes','weight.absmax'),('second_scales','weight.nested_absmax'),
                         ('dynamic_map','weight.nested_quant_map'),('nf4_map','weight.quant_map'),('bias','bias')]:
            if key in refs[name]['outputs']:
                require(state[dest] == refs[name]['outputs'][key], 'STATE_ARRAY_ORACLE_LINK')
        rows[name] = raw
    require(set(rec['artifacts']) == required and len(rows) == 8, 'STATE_ORACLE_COMPLETE_MATRIX')
    return rec, rows

def setup(B,P,OldA,N,args,cpu):
    admission,roots = A.admit(args.admission,args.admission_sha256,args.patch_manifest)
    versions=E.exact_runtime(B,not cpu)
    import torch
    require(torch.__version__=='2.9.0+cpu','STATE_LOADED_TORCH');torch.set_num_threads(1)
    C=types.SimpleNamespace(torch=torch,admission=admission,roots=roots,device='cpu',runtime=versions)
    if not cpu:
        import torch_xla as xla
        import torch_xla.backends as backends
        backends.set_mat_mul_precision('highest')
        C.precision={'requested':'highest','readback':backends.get_mat_mul_precision(),'set_calls':1,'before_graph':True}
        require(C.precision['readback']=='highest','STATE_PRECISION')
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as metrics
        C.device=xm.xla_device();require(C.device.type=='xla' and xm.xla_device_hw(C.device)=='TPU','STATE_ACTUAL_TPU')
        C.xla=xla;C.xm=xm;C.metrics=metrics
    import bitsandbytes as bnb
    C.bnb=bnb;C.public=S.state_methods(B,OldA,N,bnb,roots);C.schemas=E.schema_readback(torch)
    C.plugin=A.loaded_plugin_functions(admission,roots['bitsandbytes_tpu'])
    if not cpu:
        C.dispatch={k:torch._C._dispatch_has_kernel_for_dispatch_key(k,'XLA') for k in C.schemas}
        require(all(C.dispatch.values()),'STATE_NINE_DISPATCH')
    require(not torch.__future__.get_overwrite_module_params_on_conversion()
            and not torch.__future__.get_swap_module_params_on_conversion(),'STATE_DEFAULT_CONVERSION_FLAGS')
    return C

def run_phase(B,R,P,OldA,N,args):
    cpu=args.command=='prepare';output=Path(args.output);output.mkdir(parents=True,exist_ok=False);C=None
    rec=dict(binding(B,args),status='PARTIAL',pid=os.getpid(),pgid=os.getpgid(0),parent_pid=os.getppid(),
             process_token=args.process_token,deadline_epoch=args.deadline_epoch,started_epoch=time.time(),started_monotonic=time.monotonic())
    filename='oracle-seal.json' if cpu else 'receipt.json'; failed=False
    try:
        nested,refs=dependencies(B,P,N,args,device=not cpu)
        V.process(rec,group_leader=cpu)
        require(time.time()<args.deadline_epoch,'STATE_DEADLINE')
        review=A.reviewer_dependency(A.read(args.source_review),args.source_review_sha256,record_path=args.source_review)
        require(review['science_map_sha256']==args.science_manifest_sha256,'STATE_SOURCE_REVIEW')
        if not cpu:
            verify_state_cpu(B,P,N,args)
            require(os.getppid()==args.parent_pid and os.getpgid(0)==args.parent_pid and os.getpid()!=args.parent_pid
                    and re.fullmatch('[0-9a-f]{32}',args.parent_process_token),'STATE_FRESH_CHILD_GROUP')
            rec.update(phase=args.phase,parent_process_token=args.parent_process_token,
                       state_oracle_sha256=args.state_oracle_sha256,accepted79_sha256=args.accepted79_sha256)
            if args.phase=='restore':
                require(B.sha(Path(args.saved)/'receipt.json')==args.saved_sha256,'STATE_SAVE_RECEIPT_SHA')
                saved=receipt(B,args,args.saved,'receipt.json')
                require(saved['pid']!=rec['pid'] and saved['process_token']!=rec['process_token'],'STATE_NEW_RESTORE_PROCESS')
                rec.update(saved_sha256=args.saved_sha256,saved_pid=saved['pid'],saved_process_token=saved['process_token'])
            Z.Store(output,create=True)
        C=setup(B,P,OldA,N,args,cpu)
        rec.update(runtime=C.runtime,public_api=C.public,schemas=C.schemas,plugin_loaded_code=C.plugin,source_pre=args.admission_sha256)
        if cpu:
            evidence={'python':__import__('platform').python_version(),'torch':C.torch.__version__,'torch_git':C.torch.version.git_version,
                      'platform':__import__('platform').system(),'machine':__import__('platform').machine(),
                      'capability':C.torch.backends.cpu.get_cpu_capability(),'num_threads':C.torch.get_num_threads(),'build':C.torch.__config__.show(),
                      'xla_or_jax_imported':any(n=='torch_xla' or n.startswith('torch_xla.') or n=='jax' or n.startswith('jax.') for n in sys.modules)}
            V.cpu_evidence(evidence)
            require(evidence==B.read(Path(args.oracle)/'cpu-evidence.json'),'STATE_CPU_QUALIFICATION_MATCH')
            B.write(output/'cpu-evidence.json',evidence)
            rec['oracle_method']='FRESH_NESTED79_DEFAULT_BODY_PLUS_PUBLIC_CPU_STATE'
        else:
            rec.update(device={'type':'xla','hardware':'TPU','pjrt':'TPU'},dispatch=C.dispatch,precision=C.precision,
                       precision_environment=P.precision_environment())
        for case in fixed_cases(B):
            require(time.time()<args.deadline_epoch,'STATE_CASE_DEADLINE')
            stem=output/'raw'/case['id'];stem.parent.mkdir(exist_ok=True)
            raw={'case_id':case['id'],'input_sha256':B.case_sha(case),'pid':rec['pid'],'process_token':rec['process_token'],'status':'ERROR','composite_records':[]}
            mode=None
            try:
                S.state_methods(B,OldA,N,C.bnb,C.roots);A.loaded_plugin_functions(C.admission,C.roots['bitsandbytes_tpu'])
                if cpu:
                    state=S.oracle_state(B,N,case,refs[case['id']],C.bnb,C.torch)
                    module=S.restore(B,N,case,state,C.bnb,C.torch,'cpu')
                    restored={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
                    require(S.records(B,N,state)==S.records(B,N,restored),'STATE_CPU_PUBLIC_ROUNDTRIP')
                    C.torch.save(restored,stem.with_suffix('.state.pt'))
                    raw.update({k:v for k,v in refs[case['id']].items() if k not in ('pid','process_token')})
                    raw.update(pid=rec['pid'],process_token=rec['process_token'],state_dict=S.records(B,N,restored),
                               quant_state_public_roundtrip=S.public_roundtrip(B,N,module.weight.quant_state,C.bnb,'cpu'),
                               packed_serialization_original=True,module_training=True,status='PASS')
                else:
                    C.metrics.clear_all();mode=T.make_mode(C.torch,B,P,C.xla,C.xm,C.metrics,C.roots['bitsandbytes_tpu'],output/'trace'/case['id'],args.deadline_epoch)
                    with mode:
                        if args.phase=='save':
                            module=S.original_module(B,case,C.bnb,C.torch,C.device)
                        else:
                            source=Path(args.saved)/'raw'/(case['id']+'.state.pt')
                            require(B.sha(source)==saved['artifacts']['raw/'+case['id']+'.state.pt']['sha256'],'STATE_SAVED_CHECKPOINT_SHA')
                            state,state_raw=S.checkpoint(B,N,C.torch,source,case)
                            module=S.restore(B,N,case,state,C.bnb,C.torch,C.device)
                            require(S.records(B,N,{k:v.detach().cpu() for k,v in module.state_dict().items()})==state_raw,'STATE_INITIAL_RESTORE_EXACT')
                            stem.with_suffix('.state.pt').write_bytes(source.read_bytes())
                            raw.update(loaded_state_before=state_raw,loaded_checkpoint_sha256=B.sha(source),saved_receipt_sha256=args.saved_sha256,fresh_reconstruction=True)
                        outputs,details=S.compute(B,N,case,module,C.torch,C.bnb,C.device)
                    raw['composite_records']=['%03d'%(i+1) for i in range(len(mode.records))]
                    hlo=C.xla._XLAC._get_xla_tensors_hlo(list(outputs.values()));stem.with_suffix('.hlo.txt').write_text(hlo)
                    raw['hlo_precision']=P.hlo_precision(hlo) if case['dtype']=='float32' and len(case['x_shape'])==2 else None
                    C.metrics.clear_all();C.xla._XLAC._xla_sync_multi(list(outputs.values()),devices=[],wait=True,sync_xla_data=True);C.xm.wait_device_ops()
                    final=T.canonical({'counters':{k:C.metrics.counter_value(k) for k in C.metrics.counter_names()},
                                      'execution_metrics':{k:v for k in B.load_spec()[0]['execution_metrics'] if (v:=C.metrics.metric_data(k)) is not None}})
                    raw['final_output_execution']=final;raw['execution_scope']='DISJOINT_ACTUAL_COMPOSITE_INTERVALS_PLUS_FINAL_OUTPUT_SYNC'
                    raw.update(T.aggregate_execution([{k:r[k] for k in ('counters','execution_metrics')} for r in mode.records if all(k in r for k in ('counters','execution_metrics'))]+[final]))
                    raw['observation_errors']=mode.observation_errors;raw['observation_status']='ERROR' if mode.observation_errors else 'COMPLETE'
                    stem.with_suffix('.metrics.txt').write_text(C.metrics.metrics_report())
                    state={k:v.detach().cpu().clone() for k,v in module.state_dict().items()}
                    raw.update(N.serialize_details(B,details),outputs={k:N.tensor_record(B,v) for k,v in outputs.items()},
                               placements={k:str(v.device) for k,v in outputs.items()},state_dict=S.records(B,N,state),module_training=module.training,
                               quant_state_public_roundtrip=S.public_roundtrip(B,N,module.weight.quant_state,C.bnb,C.device),packed_serialization_original=True,status='PASS')
                    if args.phase=='save':C.torch.save(state,stem.with_suffix('.state.pt'))
                    N.validate_arrays(B,P,case,raw,True);P.validate_execution(B,raw)
                S.validate_state(B,N,case,raw['state_dict'])
            except Exception as error:
                raw=N.case_error_record(raw,error);failed=True
                if mode is not None:raw['composite_records']=['%03d'%(i+1) for i in range(len(mode.records))]
                for suffix in ('.state.pt','.hlo.txt','.metrics.txt') if not cpu else ('.state.pt',):
                    if not stem.with_suffix(suffix).exists():stem.with_suffix(suffix).write_text('NOT_RUN_CASE_ERROR\n')
            B.write(stem.with_suffix('.json'),raw)
            if not cpu:
                store=Z.Store(output)
                for suffix in ('.json','.hlo.txt','.metrics.txt'):store.pack(stem.with_suffix(suffix))
                require(mode is None or not mode.evidence_limit,'EVIDENCE_LIMIT_STOP_AFTER_RETAINED_STATE_ROW')
        if not cpu:rec['lossless_archive']=Z.seal(output)
        rec['status']='FAILED' if cpu and failed else 'COMPLETE'
    except BaseException:
        rec['status']='FAILED';(output/'error.log').write_text(traceback.format_exc())
    finally:
        if C is not None:
            try:
                A.verify_installed(C.admission,C.roots);S.state_methods(B,OldA,N,C.bnb,C.roots);A.loaded_plugin_functions(C.admission,C.roots['bitsandbytes_tpu']);rec['source_post']=args.admission_sha256
            except Exception:
                rec['status']='FAILED';(output/'source-post.error.log').write_text(traceback.format_exc())
        rec.update(finished_epoch=time.time(),finished_monotonic=time.monotonic(),artifacts=B.inventory(output,filename));B.write(output/filename,rec)
    if rec['status']!='COMPLETE':return {'record_validation':'FAIL','numerical_status':'ERROR','m4_status':'NOT_QUALIFIED'},1
    if cpu:
        args.state_oracle=str(output);args.state_oracle_sha256=B.sha(output/filename);verify_state_cpu(B,P,N,args)
    return {'record_validation':'PASS','numerical_status':'ERROR' if failed else 'SEALED_CPU_STATE' if cpu else 'PENDING_PARENT_VERIFY','m4_status':'NOT_QUALIFIED'},2 if failed else 0
def durable_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as file:
        file.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode())
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)

# Unchanged reviewed M3 helper body, from experiments/2026-10-04-bitsandbytes-tpu/cloud/state_coordinator.py
def run_child(argv, *, output, descriptor, parent, parent_path, deadline, env=None, interrupts=None):
    """Launch one direct child in the inherited group; persist identity before waiting."""
    require(math.isfinite(deadline) and deadline <= parent['deadline_epoch'] and time.time() < deadline,
            'STATE_CHILD_DEADLINE')
    streams = []
    process = None
    descriptor.update(argv=list(argv), deadline_epoch=deadline, parent_pid=parent['pid'],
                      parent_process_token=parent['process_token'], started_epoch=time.time(),
                      started_monotonic=time.monotonic(), reaped=False, child_absent=False,
                      cleanup_errors=[], timeout=False, exit_code=None)
    try:
        for filename in ('stdout.raw', 'stderr.raw'):
            streams.append((output / filename).open('wb', buffering=0))
        # A new PID is sufficient. The outer owner must retain the same PGID even
        # if this coordinator receives SIGKILL and cannot execute its finally.
        if interrupts is not None:
            interrupts['launching'] = True
        try:
            process = subprocess.Popen(argv, stdout=streams[0], stderr=streams[1], env=env)
            descriptor.update(pid=process.pid, launched_pid=process.pid, pgid=os.getpgid(process.pid))
            parent['children'].append(descriptor)
            durable_json(parent_path, parent)
        finally:
            if interrupts is not None:
                interrupts['launching'] = False
        if interrupts is not None and interrupts['pending'] is not None:
            raise InterruptedError('STATE_PARENT_SIGNAL:' + str(interrupts['pending']))
        require(process.pid != parent['pid'] and descriptor['pgid'] == parent['pgid'], 'STATE_INHERITED_GROUP')
        try:
            process.wait(timeout=max(0.001, deadline - time.time()))
        except subprocess.TimeoutExpired:
            descriptor['timeout'] = True
            raise TimeoutError('STATE_CHILD_DEADLINE_EXPIRED')
    finally:
        if process is not None:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=2)
                else:
                    process.wait(timeout=2)
                descriptor.update(exit_code=process.returncode, reaped=True)
                try:
                    os.kill(process.pid, 0)
                except ProcessLookupError:
                    descriptor['child_absent'] = True
                require(descriptor['child_absent'], 'STATE_CHILD_STILL_PRESENT')
            except BaseException as error:
                descriptor['cleanup_errors'].append({'type': type(error).__name__, 'message': str(error)})
            descriptor.update(finished_epoch=time.time(), finished_monotonic=time.monotonic())
        for stream in streams:
            try:
                os.fsync(stream.fileno())
                stream.close()
            except Exception as error:
                descriptor['cleanup_errors'].append({'type': type(error).__name__, 'message': str(error)})
        if process is not None:
            durable_json(parent_path, parent)
    require(descriptor['cleanup_errors'] == [] and descriptor['reaped'] and descriptor['child_absent'] and
            not descriptor['timeout'] and descriptor['exit_code'] in (0, 2), 'STATE_CHILD_TERMINAL')
    return descriptor


def verify_phase(B,P,N,args,root,rec,references,phase):
    store=Z.Store(root);store.audit();logical=set();physical={'logical-files.json'};rows={}
    physical.update(r['path'] for r in store.files.values())
    admission=A.admission(args.admission,args.admission_sha256);plugin=Path(args.plugin_source)
    require(not plugin.is_symlink() and all(not p.is_symlink() for p in plugin.rglob('*'))
            and {p.relative_to(plugin).as_posix():B.sha(p) for p in plugin.rglob('*.py')}==admission['plugin_python_files'],'STATE_RECOVERED_PLUGIN_SOURCE')
    import torch
    for case in fixed_cases(B):
        name=case['id'];stem='raw/'+name;names=[stem+s for s in ('.json','.hlo.txt','.metrics.txt')];logical.update(names);physical.add(stem+'.state.pt')
        trace_prefix='trace/'+name+'/';trace_names={n for n in store.files if n.startswith(trace_prefix)};logical.update(trace_names)
        raw=json.loads(store.read(stem+'.json'));rows[name]=raw
        require(raw.get('case_id')==name and raw.get('input_sha256')==B.case_sha(case)
                and raw.get('pid')==rec['pid'] and raw.get('process_token')==rec['process_token'],'STATE_DEVICE_ROW_IDENTITY')
        declared=raw.get('composite_records');require(type(declared) is list and len(declared)==len(set(declared))
                and all(re.fullmatch('[0-9]{3}',n) for n in declared),'STATE_DECLARED_COMPOSITES')
        require({n[len(trace_prefix):].split('/')[0] for n in trace_names}==set(declared),'STATE_TRACE_CASE_MATRIX')
        if raw.get('status')=='ERROR':
            require(all(type(raw.get(k)) is str and raw[k] for k in ('error_type','error_message','traceback')),'STATE_ERROR_RETAINED');continue
        V.complete_case(raw);N.validate_arrays(B,P,case,raw,True);P.validate_execution(B,raw)
        _,state=S.checkpoint(B,N,torch,Path(root)/(stem+'.state.pt'),case)
        require(raw.get('state_dict')==state and raw.get('quant_state_public_roundtrip') is True
                and raw.get('packed_serialization_original') is True and raw.get('module_training') is True,'STATE_PUBLIC_CHECKPOINT_BINDING')
        hlo=store.read(stem+'.hlo.txt').decode();require(hlo.strip() and store.read(stem+'.metrics.txt'),'STATE_ROW_HLO_METRICS')
        if case['dtype']=='float32' and len(case['x_shape'])==2:
            dots=P.hlo_precision(hlo);require(raw.get('hlo_precision')==dots,'STATE_HLO_PRECISION_BINDING')
            f32=[d for d in dots if d['result_dtype']=='f32' and d['operand_dtypes']==['f32','f32']]
            for dot in f32:P.validate_dot_matrix(B,dot)
            require(f32 and all(d['operands']==['highest','highest'] for d in f32)
                    and {tuple(case['x_shape']),tuple(case['x_shape'][:-1]+[case['weight_shape'][0]])}<={tuple(d['result_shape']) for d in f32},'STATE_FORWARD_DX_HIGHEST')
        else:require(raw.get('hlo_precision') is None,'STATE_HLO_OBSERVER_SCOPE')
        intervals=[];traces=[]
        for index in declared:
            selected={n for n in trace_names if n.startswith(trace_prefix+index+'/')}
            with tempfile.TemporaryDirectory(prefix='bnb-state8-call-') as scratch:
                with store.selected(selected,scratch):
                    directory=Path(scratch)/trace_prefix/index
                    observed=B.read(directory/'raw.json')
                    require(observed.get('pid')==rec['pid'] and observed.get('pgid')==rec['pgid'],'STATE_COMPOSITE_PROCESS')
                    expected=E.expected_trace_inputs(case,B.read(HERE/'nested-inputs.json'),references[name],observed['schema_name'])
                    observed,audit=V.composite_record(B,P,directory,expected,admission['plugin_python_files'],V.literals(plugin,len(expected[0]['values'])))
                    output_names=['scale_codes','offset','second_scales'] if observed['schema_name'].endswith('compress_statistics') else ['restored_scales']
                    require(len(observed['outputs'])==len(output_names),'STATE_COMPOSITE_OUTPUT_ARITY')
                    for output_name,got in zip(output_names,observed['outputs']):
                        wanted=references[name]['outputs'][output_name]
                        gate=B.numeric(got['values'],wanted['values'],N.tolerance(B.load_spec()[0],case,output_name))
                        if output_name in N.EXACT and got['bytes_sha256']!=wanted['bytes_sha256']:gate['status']='FAIL'
                        audit['input_gates']['output_'+output_name]=gate['status']
                    traces.append({'schema_name':observed['schema_name'],**audit})
                    intervals.append({k:observed[k] for k in ('counters','execution_metrics')})
        require(raw.get('execution_scope')=='DISJOINT_ACTUAL_COMPOSITE_INTERVALS_PLUS_FINAL_OUTPUT_SYNC'
                and T.aggregate_execution(intervals+[raw['final_output_execution']])=={k:raw[k] for k in ('counters','execution_metrics')},'STATE_EXECUTION_INTERVALS')
        operations=[v['schema_name'] for v in traces]
        require(operations.count('bnb_tpu_exact::compress_statistics')==(1 if phase=='save' else 0)
                and operations.count('bnb_tpu_exact::restore_scales')==(4 if case['dtype']=='float32' else 3),'STATE_PUBLIC_ROUTE_MATRIX')
        raw['verified_composites']=traces
    require(set(store.files)==logical and set(rec['artifacts'])==physical,'STATE_COMPLETE_PHYSICAL_LOGICAL_MATRIX')
    return rows

def verify(B,R,P,OldA,N,args):
    nested,_=dependencies(B,P,N,args,device=True);gold,references=verify_state_cpu(B,P,N,args)
    root=Path(args.actual);parent=receipt(B,args,root,'receipt.json',parent=True)
    require(parent.get('state_oracle_sha256')==args.state_oracle_sha256 and parent.get('accepted79_sha256')==args.accepted79_sha256,'STATE_PARENT_DEPENDENCIES')
    launches=parent.get('children');require(type(launches) is list and [r.get('phase') for r in launches]==['save','restore'],'STATE_CHILD_MATRIX')
    pids={parent['pid'],gold['pid'],nested['pid']};tokens={parent['process_token'],gold['process_token'],nested['process_token']}
    require(len(pids)==len(tokens)==3,'STATE_FRESH_CPU_PARENT_IDENTITIES')
    required=set();phase_rows={};phase_recs={}
    for phase,launch in zip(('save','restore'),launches):
        child=root/phase;rec=receipt(B,args,child,'receipt.json');phase_recs[phase]=rec
        P.validate_environment(B,rec['precision_environment'])
        require(rec.get('phase')==phase and rec['pid'] not in pids and rec['process_token'] not in tokens,'STATE_FRESH_PHASE_IDENTITIES')
        pids.add(rec['pid']);tokens.add(rec['process_token'])
        require(rec['pid']==launch.get('pid')==launch.get('launched_pid') and rec['pgid']==launch.get('pgid')==parent['pid']
                and rec['parent_pid']==launch.get('parent_pid')==parent['pid']
                and rec.get('parent_process_token')==launch.get('parent_process_token')==parent['process_token']
                and rec['process_token']==launch.get('process_token'),'STATE_ACTUAL_LAUNCH_IDENTITY')
        require(launch.get('reaped') is True and launch.get('child_absent') is True and launch.get('cleanup_errors')==[]
                and launch.get('timeout') is False and type(launch.get('exit_code')) is int and launch['exit_code'] in (0,2),'STATE_CHILD_CLOSURE')
        require(launch.get('receipt_sha256')==B.sha(child/'receipt.json') and launch.get('receipt')==phase+'/receipt.json','STATE_CHILD_RECEIPT_LINK')
        require(all(type(launch.get(k)) in (int,float) and math.isfinite(launch[k]) for k in ('started_epoch','finished_epoch','started_monotonic','finished_monotonic','deadline_epoch'))
                and launch['started_epoch']<=rec['started_epoch']<=rec['finished_epoch']<=launch['finished_epoch']<=launch['deadline_epoch']
                and launch['started_monotonic']<=rec['started_monotonic']<=rec['finished_monotonic']<=launch['finished_monotonic']
                and rec['deadline_epoch']==launch['deadline_epoch']<=parent['deadline_epoch'],'STATE_CHILD_TIMELINE')
        require(rec.get('state_oracle_sha256')==args.state_oracle_sha256 and rec.get('accepted79_sha256')==args.accepted79_sha256,'STATE_CHILD_DEPENDENCIES')
        if phase=='restore':
            saved=phase_recs['save'];require(rec.get('saved_sha256')==B.sha(root/'save/receipt.json')
                and rec.get('saved_pid')==saved['pid'] and rec.get('saved_process_token')==saved['process_token'],'STATE_SAVE_PROCESS_LINK')
        phase_rows[phase]=verify_phase(B,P,N,args,child,rec,references,phase)
        required.update(phase+'/'+p for p in set(rec['artifacts'])|{'receipt.json'})
        required.update((phase+'-logs/stdout.raw',phase+'-logs/stderr.raw'))
    require(launches[0]['finished_monotonic']<=launches[1]['started_monotonic']
            and launches[0]['finished_epoch']<=launches[1]['started_epoch'],'STATE_SEQUENTIAL_REAP')
    require(set(parent['artifacts'])==required,'STATE_PARENT_COMPLETE_ARTIFACTS')
    rows=[];profile,_=B.load_spec()
    for case in fixed_cases(B):
        name=case['id'];saved=phase_rows['save'][name];restored=phase_rows['restore'][name];ref=references[name]
        if saved['status']=='ERROR' or restored['status']=='ERROR':rows.append({'case_id':name,'status':'ERROR','gates':{}});continue
        require(B.sha(root/'save/raw'/(name+'.state.pt'))==B.sha(root/'restore/raw'/(name+'.state.pt'))==restored.get('loaded_checkpoint_sha256'),'STATE_EXACT_CHECKPOINT_PAIR')
        require(saved['state_dict']==restored.get('loaded_state_before')==restored['state_dict']
                and restored.get('fresh_reconstruction') is True and restored.get('saved_receipt_sha256')==B.sha(root/'save/receipt.json'),'STATE_EXACT_RESTORATION')
        gates={}
        for phase,raw in (('save',saved),('restore',restored)):
            for key,observed in raw['outputs'].items():
                wanted=ref['outputs'][key];gate=B.numeric(observed['values'],wanted['values'],N.tolerance(profile,case,key))
                if key in N.EXACT and observed['bytes_sha256']!=wanted['bytes_sha256']:gate['status']='FAIL'
                gates[phase+'/'+key]=gate
            for key,observed in raw['state_dict'].items():
                wanted=ref['state_dict'][key];gate=B.numeric(observed['values'],wanted['values'],{'exact':True})
                if observed['bytes_sha256']!=wanted['bytes_sha256']:gate['status']='FAIL'
                gates[phase+'/state/'+key]=gate
            for i,trace in enumerate(raw['verified_composites']):
                for key,value in trace['input_gates'].items():gates[phase+'/trace/'+str(i)+'/'+key]={'status':value}
        for key,observed in restored['outputs'].items():
            gates['roundtrip/'+key]=B.numeric(observed['values'],saved['outputs'][key]['values'],N.tolerance(profile,case,key))
        rows.append({'case_id':name,'status':'PASS' if all(g['status']=='PASS' for g in gates.values()) else 'FAIL','gates':gates})
    return {'record_validation':'PASS','numerical_status':V.numerical_status(rows),'case_count':8,'checkpoint_pairs':sum(r['status']!='ERROR' for r in rows),
            'rows':rows,'source_variant':A.VARIANT,'m4_status':'NOT_QUALIFIED','native_cpu_golden':'NOT_RUN','cuda_golden':'NOT_RUN'}

def execute(B,R,P,OldA,N,args):
    output=Path(args.output);require(os.getpid()==os.getpgid(0) and not output.exists(),'STATE_OWNED_PARENT')
    require(math.isfinite(args.deadline_epoch) and time.time()+30<args.deadline_epoch,'STATE_PARENT_DEADLINE')
    dependencies(B,P,N,args,device=True);verify_state_cpu(B,P,N,args)
    review=A.reviewer_dependency(A.read(args.source_review),args.source_review_sha256,record_path=args.source_review)
    require(review['science_map_sha256']==args.science_manifest_sha256,'STATE_SOURCE_REVIEW')
    admission,roots=A.admit(args.admission,args.admission_sha256,args.patch_manifest);P.validate_environment(B,P.precision_environment())
    output.mkdir(parents=True);parent=dict(binding(B,args),status='PARTIAL',pid=os.getpid(),pgid=os.getpgid(0),parent_pid=os.getppid(),
        process_token=args.process_token,state_oracle_sha256=args.state_oracle_sha256,accepted79_sha256=args.accepted79_sha256,
        source_pre=args.admission_sha256,deadline_epoch=args.deadline_epoch,children=[],started_epoch=time.time(),started_monotonic=time.monotonic())
    V.process(parent);path=output/'receipt.json';durable_json(path,parent)
    previous={s:signal.getsignal(s) for s in (signal.SIGTERM,signal.SIGINT)};interrupts={'launching':False,'pending':None}
    def interrupted(signum,frame):
        interrupts['pending']=signum
        if not interrupts['launching']:raise InterruptedError('STATE_PARENT_SIGNAL:'+str(signum))
    for sig in previous:signal.signal(sig,interrupted)
    try:
        for index,phase in enumerate(('save','restore')):
            remaining=args.deadline_epoch-time.time()-10;require(remaining>0,'STATE_REMAINING_BUDGET')
            deadline=time.time()+remaining/(2-index);token=secrets.token_hex(16)
            cmd=[sys.executable,'-B',str(Path(__file__).resolve()),'device-child']
            for name in COMMON+DEVICE:
                cmd+=['--'+name.replace('_','-'),str(getattr(args,name))]
            cmd+=['--phase',phase,'--output',str(output/phase),'--parent-pid',str(parent['pid']),
                  '--parent-process-token',parent['process_token'],'--process-token',token,'--deadline-epoch',str(deadline)]
            if phase=='restore':cmd+=['--saved',str(output/'save'),'--saved-sha256',B.sha(output/'save/receipt.json')]
            logs=output/(phase+'-logs');logs.mkdir()
            launch=run_child(cmd,output=logs,descriptor={'phase':phase,'process_token':token},parent=parent,parent_path=path,deadline=deadline,
                             env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),interrupts=interrupts)
            launch.update(receipt=phase+'/receipt.json',receipt_sha256=B.sha(output/phase/'receipt.json'));durable_json(path,parent)
        A.verify_installed(admission,roots)
        parent.update(source_post=args.admission_sha256,status='COMPLETE',finished_epoch=time.time(),finished_monotonic=time.monotonic(),artifacts=B.inventory(output,'receipt.json'))
        durable_json(path,parent);args.actual=output
        result=verify(B,R,P,OldA,N,args);return result,0 if result['numerical_status']=='PASS' else 2
    except BaseException as error:
        parent.update(status='FAILED',error={'type':type(error).__name__,'message':str(error)},finished_epoch=time.time(),finished_monotonic=time.monotonic(),artifacts=B.inventory(output,'receipt.json'))
        durable_json(path,parent);raise
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)

COMMON=tuple(E.PINS)+('science_manifest','science_manifest_sha256','admission','admission_sha256','patch_manifest','oracle','oracle_sha256')
DEVICE=('state_oracle','state_oracle_sha256','accepted79','accepted79_sha256','cpu_gate','cpu_gate_sha256','plugin_source','source_review','source_review_sha256')

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('prepare','execute','verify','verify-cpu','device-child'):
        cmd=sub.add_parser(command)
        for name in COMMON:cmd.add_argument('--'+name.replace('_','-'),required=True)
        if command in ('execute','verify','device-child'):
            for name in DEVICE:cmd.add_argument('--'+name.replace('_','-'),required=True)
        if command in ('prepare','execute','device-child'):
            for name in ('output','process_token'):cmd.add_argument('--'+name.replace('_','-'),required=True)
            cmd.add_argument('--deadline-epoch',type=float,required=True)
        if command=='prepare':
            cmd.add_argument('--source-review',required=True);cmd.add_argument('--source-review-sha256',required=True)
        if command=='verify-cpu':
            cmd.add_argument('--state-oracle',required=True);cmd.add_argument('--state-oracle-sha256',required=True)
        if command=='verify':cmd.add_argument('--actual',required=True)
        if command=='device-child':
            cmd.add_argument('--phase',choices=('save','restore'),required=True);cmd.add_argument('--parent-pid',type=int,required=True)
            cmd.add_argument('--parent-process-token',required=True);cmd.add_argument('--saved');cmd.add_argument('--saved-sha256')
    args=parser.parse_args(argv)
    try:
        B,R,P,OldA,N=E.sources(args)
        if args.command=='verify':result=verify(B,R,P,OldA,N,args);code=0 if result['numerical_status']=='PASS' else 2
        elif args.command=='verify-cpu':
            seal,rows=verify_state_cpu(B,P,N,args);result={'record_validation':'PASS','state_oracle_status':'INDEPENDENTLY_VERIFIED','case_count':len(rows),'m4_status':'NOT_QUALIFIED'};code=0
        elif args.command=='execute':result,code=execute(B,R,P,OldA,N,args)
        else:result,code=run_phase(B,R,P,OldA,N,args)
    except Exception as error:
        result={'record_validation':'FAIL','numerical_status':'ERROR','reason':str(error),'m4_status':'NOT_QUALIFIED'};code=1
        if args.command in ('execute','verify') and 'B' in locals():
            try:
                actual=Path(args.actual if args.command=='verify' else args.output)
                observations={phase:E.numerical_observations(B,P,N,args,actual/phase,selected_cases=fixed_cases(B)) for phase in ('save','restore')}
                result.update(numerical_status=V.numerical_status([{'status':r['numerical_status']} for r in observations.values()]),
                              numerical_observations=observations,scope='RETURNED_NUMERICS_ONLY_STATE_AND_DEVICE_NOT_QUALIFIED')
            except Exception:
                pass
    print(json.dumps(result,allow_nan=False));return code

if __name__=='__main__':raise SystemExit(main())
