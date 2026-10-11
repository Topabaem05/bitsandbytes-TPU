"""Independent bounded reader. Local smoke does not establish M5 qualification."""
import argparse,json,math,re,tempfile
from pathlib import Path
import training as R
import admission as A
import math_model as M
import record_validation as V
import archive_store as Z
for _name in ['require','read','sha','digest','input_descriptor','state_hashes','frozen_state','optimizer_descriptor','optimizer','dense_replay','validate_array','generated','ADAPTER_NAMES','PROTOCOL_SHA']:
 globals()[_name]=getattr(M,_name)
methods=R.methods
model=R.model
source_bundle=R.bundle
validate_checkpoint=R.checkpoint_check

def device_records(C,root,rows,receipts,oracle,oracle_sha256,m4_acceptance,m4_acceptance_sha256):
 require(oracle is not None and sha(Path(oracle)/'oracle-seal.json')==oracle_sha256,'FRESH_CPU_DEVICE_ORACLE')
 report=verify(oracle,qualified=True,m4_acceptance=m4_acceptance,m4_acceptance_sha256=m4_acceptance_sha256,_context=C)
 require(report['record_validation']==report['numerical_status']=='PASS','QUALIFIED_CPU_PRIMARY_READER')
 gates=[];plugin=R.ROOT/'integration/plugin/src/bitsandbytes_tpu';literals=V.literals(plugin,1024)
 for phase,receipt in receipts.items():
  require(receipt['device']=={'type':'xla','hardware':'TPU','pjrt':'TPU'}and receipt['precision']=={'requested':'highest','readback':'highest','before_graph':True,'set_calls':1},'ACTUAL_TPU_PRECISION')
  store=Z.Store(root/phase);store.audit();compress=[];restore=[]
  raw_names=sorted(n for n in store.files if n.endswith('/raw.json'))
  require(raw_names,'TYPED_COMPOSITE_GRAPH_EVIDENCE')
  for name in raw_names:
   raw=json.loads(store.read(name));schema=raw['schema_name'];which='compress'if schema.endswith('compress_statistics')else'restore'if schema.endswith('restore_scales')else None
   require(which is not None,'FIXED_M5_COMPOSITE_SCHEMA')
   matched=[]
   for i,ref in enumerate(C.refs):
    inputs=[ref['first'],ref['dynamic']]if which=='compress'else[ref['codes'],ref['second'],ref['dynamic'],ref['offset']]
    expected=[R.T.tensor_record(C.B,t)for t in inputs]
    if raw['inputs']==expected:matched.append((i,expected))
   require(matched,'EXACT_FRESH_REFERENCE_COMPOSITE_INPUTS')
   directory=Path(name).parent;names=[directory.as_posix()+'/'+n for n in ['raw.json','context.hlo.pb','printer.hlo.pb','graph.hlo.txt','context.textproto','metrics.txt']]
   with tempfile.TemporaryDirectory(prefix='bnb-m5-typed-reader-')as d:
    with store.selected(names,d):
     observed,validated=V.composite_record(C.B,C.P,Path(d)/directory,matched[0][1],C.admission['plugin_python_files'],literals)
   require(all(v=='PASS'for v in validated['input_gates'].values()),'EXACT_COMPOSITE_ARGUMENTS')
   if which=='compress':
    candidates=[[R.T.tensor_record(C.B,ref[k])for k in ['codes','offset','second']]for ref in C.refs]
    require(observed['outputs']in candidates,'EXACT_COMPRESSED_NESTED_BYTES');compress.append(name)
   else:
    candidates=[R.T.tensor_record(C.B,C.torch.tensor(ref['rational']['restored'],dtype=C.torch.int64).to(C.torch.int32).view(C.torch.float32))for ref in C.refs]
    require(len(observed['outputs'])==1 and observed['outputs'][0]in candidates,'EXACT_RESTORED1024_BITS');restore.append(name)
  require((len(compress)>=2 if phase!='resume'else not compress)and len(restore)>=2,'FRESH_INITIAL_OR_CHECKPOINT_COMPOSITES')
  for step,raw in rows[phase].items():
   hlo=store.read('trace/step-%02d/training.hlo.txt'%step).decode();require(R.highest_precision(C,hlo)==raw['hlo_precision'],'RECOVERED_TRAINING_HLO_PRECISION')
   C.P.validate_execution(C.B,raw);require(raw['all_trainable_and_output_devices']=='XLA_TPU','ACTUAL_TRAINABLE_PLACEMENT')
   expected=read(Path(oracle)/phase/'steps'/('%02d.json'%step));gates.extend(row_gates(C,raw,expected).values())
 return gates

def row_gates(C,raw,reference):
 result={'output':C.B.numeric(raw['output']['values'],reference['output']['values'],C.tolerance),'loss':C.B.numeric([raw['loss']],[reference['loss']],C.tolerance)}
 for key in ['gradients','adapters_after']:
  for name in ADAPTER_NAMES:result[key+'/'+name]=C.B.numeric(raw[key][name]['values'],reference[key][name]['values'],C.tolerance)
 return result

def verify(actual, closure=None, *, device=False, oracle=None, oracle_sha256=None, m4_acceptance=None, m4_acceptance_sha256=None, qualified=False, _context=None):
    require(not device or qualified,'DEVICE_REQUIRES_QUALIFIED_RUNTIME')
    root = Path(actual)
    C = R.setup(local=True) if _context is None else _context; C.refs=R.references(C)
    methods(C)
    inputs_sha = digest(input_descriptor(C))
    parent_name='receipt.json' if device else 'oracle-seal.json'; parent = read(root / parent_name)
    require(parent['status'] == 'COMPLETE' and not any(k in parent for k in ['error','traceback']) and parent.get('not_run_phases')==[] and parent['protocol_sha256'] == PROTOCOL_SHA and (parent['source_bundle_sha256'] == source_bundle()), 'PARENT_SOURCE_TERMINAL')
    C.B.check_seal(root, parent_name, parent)
    require(parent['pid'] == parent['pgid'] and parent['pid'] > 0, 'PARENT_GROUP')
    launches = parent['children']
    require([row['phase'] for row in launches] == ['baseline', 'save', 'resume'] and len({parent['pid'], *[r['launched_pid'] for r in launches]}) == 4, 'FRESH_PROCESS_SEQUENCE')
    require(all(re.fullmatch('[0-9a-f]{32}',token or '')is not None for token in [parent['process_token'],*[r['process_token']for r in launches]]),'FRESH_PROCESS_TOKEN_FORMAT')
    require(len({parent['process_token'], *[r['process_token'] for r in launches]}) == 4, 'FRESH_TOKEN')
    expected_model = model(C)
    expected_base = state_hashes(frozen_state(expected_model))
    expected_optimizer = optimizer_descriptor(optimizer(C, expected_model))
    rows = {}
    receipts = {}
    dense_rows = dense_replay(C, expected_model)
    for index, launch in enumerate(launches):
        phase = launch['phase']
        receipt = read(root / phase / 'receipt.json')
        receipts[phase] = receipt
        require(launch['reaped'] is True and launch['child_absent'] is True and (launch['exit_code'] in (0, 2)), 'CHILD_TERMINAL')
        require(receipt['pid'] == launch['launched_pid'] and receipt['pgid'] == launch['pgid'] == parent['pgid'] and (receipt['parent_pid'] == parent['pid']) and (receipt['parent_process_token'] == parent['process_token']) and (receipt['process_token'] == launch['process_token']), 'ACTUAL_CHILD_BINDING')
        require(sha(root / phase / 'receipt.json') == launch['receipt_sha256'], 'CHILD_RECEIPT_HASH')
        require(all((type(launch[k]) in (int, float) and math.isfinite(launch[k]) for k in ('started_epoch', 'finished_epoch', 'deadline_epoch', 'started_monotonic', 'finished_monotonic'))) and launch['started_epoch'] <= launch['finished_epoch'] <= launch['deadline_epoch'] <= parent['deadline_epoch'], 'CHILD_DEADLINE')
        if index:
            require(launches[index - 1]['finished_monotonic'] <= launch['started_monotonic'], 'SEQUENTIAL_CHILD_CLOSURE')
        require(receipt['status'] == 'COMPLETE' and not any(k in receipt for k in ['error','traceback']) and (receipt['source_bundle_sha256'] == source_bundle()) and (receipt['protocol_sha256'] == PROTOCOL_SHA) and (receipt['inputs_sha256'] == inputs_sha) and (receipt['public_method_identity'] is True), 'CHILD_SOURCE_INPUTS')
        require(receipt['initial_base'] == expected_base and receipt['optimizer'] == expected_optimizer, 'INITIAL_BASE_OPTIMIZER')
        require(read(root / phase / 'inputs.json') == input_descriptor(C), 'INPUT_TENSOR_SEAL')
        C.B.check_seal(root / phase, 'receipt.json', receipt)
        expected_steps = list(range(11, 21)) if phase == 'resume' else list(range(1, 11 if phase == 'save' else 21))
        require(receipt['step_range'] == [expected_steps[0], expected_steps[-1]] and sorted((p.name for p in (root / phase / 'steps').iterdir())) == [f'{step:02d}.json' for step in expected_steps], 'STEP_SEQUENCE')
        phase_rows = {}
        for step in expected_steps:
            raw = read(root / phase / 'steps' / f'{step:02d}.json')
            require(raw['step'] == step and raw['pid'] == receipt['pid'] and (raw['process_token'] == receipt['process_token']), 'STEP_IDENTITY')
            require(type(raw['loss']) in (float, int) and math.isfinite(raw['loss']) and (raw['loss'] >= 0), 'STEP_LOSS')
            output = validate_array(C, raw['output'], [8, 256])
            loss = C.torch.nn.functional.mse_loss(output, generated(C, 'target', step=step)).item()
            require(abs(raw['loss'] - loss) <= C.tolerance['atol'] + C.tolerance['rtol'] * abs(loss), 'LOSS_OUTPUT_BINDING')
            require(set(raw['gradients']) == set(raw['adapters_after']) == set(ADAPTER_NAMES), 'STEP_ADAPTER_SET')
            for name in ADAPTER_NAMES:
                shape = [4, 256] if name.endswith('.A') else [256, 4]
                validate_array(C, raw['gradients'][name], shape)
                validate_array(C, raw['adapters_after'][name], shape)
            require(raw['base_state'] == expected_base and raw['base_gradients'] == 'ABSENT', 'FROZEN_BASE_STATE')
            require(set(raw['gates']) == {'output', 'loss', *ADAPTER_NAMES, *['updated_' + n for n in ADAPTER_NAMES]}, 'DENSE_GATE_SET')
            phase_rows[step] = raw
        rows[phase] = phase_rows
    checkpoint = C.torch.load(root / 'save/checkpoint.pt', weights_only=True, map_location='cpu')
    validate_checkpoint(C, checkpoint, inputs_sha)
    require(receipts['resume']['checkpoint_sha256'] == sha(root / 'save/checkpoint.pt') and receipts['resume']['restored'] is True and (receipts['resume']['initial_model'] == checkpoint['model_hashes']), 'EXACT_FRESH_RESTORE')
    require(checkpoint['model_hashes'] == receipts['save']['final_model'], 'SAVED_FINAL_MODEL')
    require(all((C.B.tensor_record(checkpoint['model'][name]) == rows['save'][10]['adapters_after'][name] for name in ADAPTER_NAMES)), 'CHECKPOINT_STEP10_ADAPTERS')
    gates = {}
    for phase in ('save', 'resume'):
        for step, raw in rows[phase].items():
            baseline = rows['baseline'][step]
            stage = {}
            for key in ('output',):
                stage[key] = C.B.numeric(raw[key]['values'], baseline[key]['values'], C.tolerance)
            stage['loss'] = C.B.numeric([raw['loss']], [baseline['loss']], C.tolerance)
            for key in ('gradients', 'adapters_after'):
                for name in ADAPTER_NAMES:
                    stage[key + '/' + name] = C.B.numeric(raw[key][name]['values'], baseline[key][name]['values'], C.tolerance)
            gates[phase + '/' + str(step)] = stage
    dense_gates = {}
    for phase, phase_rows in rows.items():
        for step, raw in phase_rows.items():
            reference = dense_rows[step]
            stage = {'output': C.B.numeric(raw['output']['values'], reference['output']['values'], C.tolerance), 'loss': C.B.numeric([raw['loss']], [reference['loss']], C.tolerance)}
            for key in ('gradients', 'adapters_after'):
                for name in ADAPTER_NAMES:
                    stage[key + '/' + name] = C.B.numeric(raw[key][name]['values'], reference[key][name]['values'], C.tolerance)
            dense_gates[phase + '/' + str(step)] = stage
    exact = receipts['baseline']['final_model'] == receipts['resume']['final_model']
    if closure is not None:
        summary = read(closure)
        require(summary['status'] == 'NO_CLEANUP_ERRORS' and summary.get('errors')==[] and summary.get('restoration',{}).get('handlers_restored')is True and summary.get('restoration',{}).get('timer_restored')is True and len(summary['groups']) == 1, 'OUTER_CLOSURE')
        group = summary['groups'][0]
        require(group.get('status')=='CLEANUP_VERIFIED'and group.get('errors')==[] and group['pid'] == parent['pid'] and group['pgid'] == parent['pgid'] and (group['leader_reaped'] is True) and (group['group_absence'] == 'OBSERVED_NO_SUCH_GROUP'), 'OUTER_ACTUAL_PARENT')
    if qualified or device:
        A.m4_dependency(m4_acceptance,m4_acceptance_sha256)
        require(parent.get('m4_acceptance_sha256')==m4_acceptance_sha256,'PARENT_M4_DEPENDENCY')
    require(parent.get('qualified_runtime') is qualified,'QUALIFICATION_LABEL')
    require(parent.get('kind')==('M5_DEVICE_TRAINING_PARENT_V1' if device else 'M5_QUALIFIED_CPU_ORACLE_V1'),'PARENT_KIND')
    require(parent['started_epoch']<=parent['finished_epoch']<=parent['deadline_epoch']<=parent['allocation_epoch']+3600-660,'ORIGINAL_ALLOCATION_BOUND')
    for phase,receipt in receipts.items():
        require(receipt.get('actual_tpu') is device and receipt.get('qualified_runtime') is qualified,'CHILD_RUNTIME_SCOPE')
        require(receipt['source_post']==source_bundle()and receipt['started_epoch']<=receipt['finished_epoch']<=receipt['deadline_epoch'],'CHILD_TERMINAL_SOURCE_TIME')
        require(read(root/phase/'reference-arrays.json')==R.reference_records(C),'REFERENCE_SOURCE_ORDER_BYTES')
        if qualified:
            require(receipt.get('m4_acceptance_sha256')==m4_acceptance_sha256,'CHILD_M4_DEPENDENCY')
            require(receipt.get('plugin_loaded_code')==receipt.get('plugin_loaded_code_post')=={'source_generation':A.GENERATION,'loaded_plugin_code':'COMPILED_ADMITTED_SOURCE','python_files':11},'CHILD_PLUGIN_CODE')
            if not device:V.cpu_evidence(receipt['cpu_evidence'])
        require(all(raw.get('status')=='COMPLETE' and not any(k in raw for k in ['error','traceback']) for raw in rows[phase].values()),'ERROR_BEARING_TRAINING_ROW')
    extra=[]
    if device:extra=device_records(C,root,rows,receipts,oracle,oracle_sha256,m4_acceptance,m4_acceptance_sha256)
    passed = all(g['status']=='PASS'for g in extra) and exact and all((g['status'] == 'PASS' for stage in dense_gates.values() for g in stage.values())) and all((g['status'] == 'PASS' for stage in gates.values() for g in stage.values())) and all((g['status'] == 'PASS' for phase in rows.values() for raw in phase.values() for g in raw['gates'].values()))
    return {'record_validation': 'PASS', 'numerical_status': 'PASS' if passed else 'FAIL', 'm5_status': 'NOT_QUALIFIED', 'qualified_runtime': qualified, 'steps': 20, 'checkpoint_step': 10, 'resumed_steps': 10, 'exact_final_model': exact, 'gates': gates, 'independent_dense_gates': dense_gates, 'source_bundle_sha256': source_bundle(), 'inputs_sha256': inputs_sha, 'tpu_cuda_performance': 'NOT_RUN', 'gate_counts': {'continuation':200,'independent_dense':400,'fresh_cpu_device':len(extra)}}

def main():
 p=argparse.ArgumentParser();p.add_argument('--actual',type=Path,required=True);p.add_argument('--closure',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--qualified',action='store_true');p.add_argument('--device',action='store_true');p.add_argument('--oracle');p.add_argument('--oracle-sha256');p.add_argument('--m4-acceptance');p.add_argument('--m4-acceptance-sha256');args=p.parse_args()
 try:result=verify(args.actual,args.closure,device=args.device,oracle=args.oracle,oracle_sha256=args.oracle_sha256,m4_acceptance=args.m4_acceptance,m4_acceptance_sha256=args.m4_acceptance_sha256,qualified=args.qualified)
 except Exception as e:result={'record_validation':'FAIL','numerical_status':'NOT_EVALUATED','m5_status':'NOT_QUALIFIED','error':{'type':type(e).__name__,'message':str(e)}}
 A.write(args.output,result);print(json.dumps({k:v for k,v in result.items()if k not in ['gates','independent_dense_gates']}));return 0 if result['record_validation']==result['numerical_status']=='PASS'else 2
if __name__=='__main__':raise SystemExit(main())
