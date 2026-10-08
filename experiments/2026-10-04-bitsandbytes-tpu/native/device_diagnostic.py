"""Unexecuted private TPU bridge diagnostic. Requires reviewed owner and fixed installed runtime."""
import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import adapter as D
import protocol as S
import base64
import recorder as E
import verifier as V
PINS_SHA='01563e5ae4d00b6bc6d56bc61847ec6fa5628e5573fc51b21e5ad8d149ca3836'

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def run(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    receipt={'status':'PARTIAL','scope':'UNADOPTED_NON_NESTED_NATIVE_BRIDGE_DIAGNOSTIC','m6_status':'NOT_QUALIFIED','pid':os.getpid(),'pgid':os.getpgid(0),'parent_pid':args.parent_pid,'parent_process_token':args.parent_process_token,'process_token':args.process_token,'deadline_epoch':args.deadline_epoch,'adapter_sha256':D.sha(D.__file__),'diagnostic_sha256':D.sha(__file__),'kernel_sha256':D.sha(args.kernel),'source_admission_sha256':args.admission_sha256,'backward_memory_performance':'NOT_RUN','registration':'NOT_EXECUTED','configuration_binding':'NATIVE_BOUNDARY_DIAGNOSTIC_ONLY','cases':[],'source_variant':'M2_CANONICAL_NON_NESTED_R2_PATCH','m4_source_compatibility':'NOT_QUALIFIED','oracle_sha256':args.oracle_sha256,'expected_sha256':args.expected_sha256}
    write(output/'receipt.json',receipt);B=A=None;admission=roots=None
    try:
        D.require(os.getppid()==args.parent_pid and args.parent_pid==os.getpgid(0),'ACTUAL_INHERITED_OWNER_GROUP')
        D.require(all(re.fullmatch('[0-9a-f]{32}',token) for token in (args.process_token,args.parent_process_token)) and math.isfinite(args.deadline_epoch) and time.time()<args.deadline_epoch,'OWNER_TOKENS_DEADLINE')
        D.require(S.sha(args.expected)==args.expected_sha256,'PRELAUNCH_EXPECTED_HASH')
        expected=S.read(args.expected)
        D.require(expected['kind']=='R6_EXPECTED_PRELAUNCH' and expected['source_variant']==S.spec()['source_variant'] and expected['sources']==S.sources() and expected['protocol_sha256']==S.sha(S.HERE/'protocol.json') and expected['oracle_sha256']==args.oracle_sha256 and expected['source_admission_sha256']==args.admission_sha256,'INDEPENDENT_EXPECTED_PROTOCOL')
        D.require(expected['launch']=={'parent_pid':args.parent_pid,'parent_pgid':os.getpgid(0),'parent_process_token':args.parent_process_token,'process_token':args.process_token,'deadline_epoch':args.deadline_epoch},'EXPECTED_LAUNCH_TOKEN_DEADLINE')
        oracle_seal,oracle_cases=S.oracle(args.oracle,args.oracle_sha256,qualified=True)
        D.require(expected['oracle_artifacts']==oracle_seal['artifacts'] and expected['input_sha256']=={name:row['input_sha256'] for name,row in oracle_cases.items()},'INDEPENDENT_ORACLE_PRELAUNCH')
        D.require(os.environ.get('XLA_DUMP_POST_OPTIMIZATIONS') in (None,'0'),'NO_SEPARATE_DUMP_COMPILATION')
        D.require('jax' not in sys.modules and os.environ.get('JAX_PLATFORMS') in (None,'tpu'),'FRESH_NATIVE_JAX_PROCESS')
        D.require(sys.platform=='linux' and D.sha(args.source_pins)==PINS_SHA,'FIXED_PLATFORM_SOURCE_PINS')
        pins=json.loads(Path(args.source_pins).read_text())['files']
        def check(path,key):D.require(D.sha(path)==pins[key],'PINNED_HELPER_'+key)
        check(args.backend_probe,'experiments/2026-10-04-bitsandbytes-tpu/probe_backend.py');check(args.admission_helper,'experiments/2026-10-04-bitsandbytes-tpu/transfer_admission.py');check(args.precision_probe,'experiments/2026-10-04-bitsandbytes-tpu/probe_precision.py');check(args.kernel,'.work/r6-pallas-preparation/kernel.py')
        B=load(args.backend_probe,'r6_device_B');A=load(args.admission_helper,'r6_device_A');P=load(args.precision_probe,'r6_device_P')
        P.validate_environment(B,P.precision_environment());receipt['precision_environment']=P.precision_environment();receipt['runtime']=B.runtime_check(True)
        D.require(importlib.metadata.version('jax')==importlib.metadata.version('jaxlib')=='0.7.1' and importlib.metadata.version('libtpu')=='0.0.21','FIXED_JAX_LIBTPU')
        admission,roots=A.admit(B,args.admission,args.admission_sha256,args.patch_manifest);receipt['source_pre']=args.admission_sha256
        D.require(admission['bitsandbytes_tpu']['files']==S.spec()['plugin_python_files'],'M2_CANONICAL_NON_NESTED_SOURCE_VARIANT')
        import torch,torch_xla
        import torch_xla.backends as backends
        D.require(torch.__version__=='2.9.0+cpu' and torch_xla.__version__.split('+')[0]=='2.9.0','FIXED_TORCH_XLA')
        backends.set_mat_mul_precision('highest');D.require(backends.get_mat_mul_precision()=='highest','HIGHEST_READBACK');receipt['precision']={'requested':'highest','readback':'highest','set_calls':1,'before_graph':True}
        package=importlib.import_module('bitsandbytes_tpu');C=importlib.import_module('bitsandbytes_tpu.compatibility');R=importlib.import_module('bitsandbytes_tpu.reference');F=importlib.import_module('bitsandbytes_tpu.functionalization')
        for name,module in [('compatibility',C),('reference',R),('functionalization',F)]:check(module.__file__,f'packages/bitsandbytes-tpu/src/bitsandbytes_tpu/{name}.py')
        import bitsandbytes as bnb
        receipt['public_api']=A.public_methods(B,bnb,roots)
        kernel,bridge,native_xla=D.guarded_kernel(args.kernel)
        def entry(activation,packed,scales):return kernel.forward_grouped(activation,packed,scales,interpret=False)
        call,native_calls=E.make_recording_kernel(entry,D.output_spec,bridge,native_xla,torch)
        adapter=D.ForwardAdapter(torch,C,R,call)
        receipt['recorder_sha256']=D.sha(E.__file__);receipt['verifier_sha256']=D.sha(V.__file__)
        wrapped=F.functionalized_kernel(adapter)
        import torch_xla.core.xla_model as xm
        import torch_xla.debug.metrics as metrics
        device=xm.xla_device();D.require(device.type=='xla' and xm.xla_device_hw(device)=='TPU','ACTUAL_TPU');receipt['device']={'type':'xla','hardware':'TPU','pjrt':os.environ.get('PJRT_DEVICE')}
        m,n,k=128,256,512
        codes=((torch.arange(n*k)*7)%16).to(torch.uint8);cpu_packed=((codes[::2]<<4)|codes[1::2]).reshape(-1,1);cpu_scales=(torch.arange(n*k//64)%17).float()/8
        torch_xla._XLAC._set_current_graph_name('r6-transfer-'+args.process_token[:12])
        packed=cpu_packed.to(device);scales=cpu_scales.to(device)
        settings=[('direct-fp32',torch.float32,False,128),('functional-fp32',torch.float32,True,128),('direct-bf16',torch.bfloat16,False,128),('functional-bf16',torch.bfloat16,True,128),('tail-reference-fp32',torch.float32,False,129)]
        for name,dtype,use_wrapper,rows in settings:
            native_before=len(native_calls);prefix=output/'raw'/name;raw={'name':name,'status':'ERROR','pid':os.getpid(),'process_token':args.process_token,'scope':'DIRECT_ADAPTER_OR_FUNCTIONAL_WRAPPER_NO_PUBLIC_REGISTRATION'}
            try:
                D.require(time.time()<args.deadline_epoch,'OWNER_DEADLINE');torch_xla._XLAC._set_current_graph_name('r6-preparation-'+args.process_token[:12]+'-'+name);cpu_a=(((torch.arange(rows*k)%31)-15).float()/32).reshape(rows,k).to(dtype);a=cpu_a.to(device)
                cpu_expected=torch.tensor(oracle_cases[name]['cpu_reference']['values'],dtype=dtype).reshape(rows,n)
                raw['inputs']={key:B.tensor_record(value) for key,value in [('activation',cpu_a),('packed',cpu_packed),('scales',cpu_scales)]};raw['input_sha256']=V.digest(raw['inputs']);D.require(raw['inputs']==oracle_cases[name]['inputs'] and raw['input_sha256']==expected['input_sha256'][name],'SEALED_INPUTS')
                xm.mark_step(wait=True);xm.wait_device_ops()
                graph_name='r6-boundary-'+args.process_token[:12]+'-'+name
                torch_xla._XLAC._set_current_graph_name(graph_name)
                before=len(adapter.events);native_before=len(native_calls)
                actual=(wrapped if use_wrapper else adapter)(a,packed,(n,k),scales,64,'nf4')
                prefix.parent.mkdir(parents=True,exist_ok=True)
                if rows==128:
                    D.require(len(native_calls)==native_before+1,'ONE_ACTUAL_NATIVE_CALL')
                    captured=E.capture_actual(actual,native_calls[-1],torch_xla,xm,metrics,B,P,graph_name)
                    prefix.with_suffix('.hlo.txt').write_text(captured.pop('hlo'))
                    prefix.with_suffix('.hlo.pb').write_bytes(captured.pop('proto'))
                    prefix.with_suffix('.context.textproto').write_text(captured.pop('proto_text'))
                    prefix.with_suffix('.printer.hlo.pb').write_bytes(captured.pop('printer_proto'))
                    if any(not math.isfinite(v) for v in captured['output']['values']):write(prefix.with_suffix('.nonfinite-output.json'),{'forensic_only':True,'shape':captured['output']['shape'],'dtype':captured['output']['dtype'],'values':[v if math.isfinite(v) else repr(v) for v in captured['output']['values']]})
                    V.array(captured['output'])
                    raw.update(captured)
                    raw.update(protocol='R6_NATIVE_BOUNDARY_V1',scope='ACTUAL_NATIVE_BOUNDARY',owner={key:receipt[key] for key in ('pid','pgid','parent_pid','process_token','parent_process_token','deadline_epoch')},case=name,hlo_sha256=D.sha(prefix.with_suffix('.hlo.txt')),selected_path={'path':'PALLAS_CANDIDATE','calls':1,'fallback':False},source_pins={'kernel':D.sha(args.kernel),'bridge':D.BRIDGE_SHA,'reference':D.sha(R.__file__),'backend':D.sha(B.__file__),'precision':D.sha(P.__file__),'adapter':D.sha(D.__file__),'recorder':D.sha(E.__file__),'diagnostic':D.sha(__file__)})
                    prefix.with_suffix('.native-config.json').write_text(raw['native_boundary']['payload'])
                    prefix.with_suffix('.mosaic-body.bin').write_bytes(base64.b64decode(json.loads(raw['native_boundary']['payload'])['custom_call_config']['body'],validate=True))
                else:
                    metrics.clear_all();torch_xla._XLAC._xla_sync_multi([actual],devices=[],wait=True);xm.wait_device_ops()
                    profile,_=B.load_spec();raw['counters']={key:metrics.counter_value(key) for key in metrics.counter_names()};raw['execution_metrics']={key:list(value) for key in profile['execution_metrics'] if (value:=metrics.metric_data(key)) is not None};P.validate_execution(B,raw)
                    raw['output']=B.tensor_record(actual)
                profile,_=B.load_spec();raw['cpu_reference']=B.tensor_record(cpu_expected);raw['selected_events']=adapter.events[before:]
                # Same-device reference happens after the retained actual-only execution interval.
                torch_xla._XLAC._set_current_graph_name('r6-reference-'+args.process_token[:12]+'-'+name)
                same_device=R.gemm_4bit(a,packed,(n,k),scales,64,'nf4')
                torch_xla._XLAC._xla_sync_multi([same_device],devices=[],wait=True);xm.wait_device_ops()
                raw['same_device_reference']=B.tensor_record(same_device);V.array(raw['same_device_reference']);V.array(raw['cpu_reference'])
                tolerance=profile['tolerances']['fp32_forward_gradient' if dtype==torch.float32 else 'bf16_forward_unit_scale'];raw['gates']={key:B.numeric(raw['output']['values'],raw[key]['values'],tolerance) for key in ('same_device_reference','cpu_reference')}
                if rows==128:
                    # Bounded verifier terminal status is separate from the numerical case status below.
                    raw['numerical_case_status']='PASS' if all(g['status']=='PASS' for g in raw['gates'].values()) else 'FAIL'
                    raw['status']='COMPLETE'
                    write(prefix.with_suffix('.boundary-record.json'),raw)
                raw['status']='PASS' if all(gate['status']=='PASS' for gate in raw['gates'].values()) else 'NUMERICAL_FAIL'
            except Exception as error:
                if 'native_calls' in locals() and 'native_before' in locals():write(prefix.with_suffix('.native-failure.json'),[{key:value for key,value in row.items() if key!='tensor_args'} for row in native_calls[native_before:]])
                raw.update(status='ERROR',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc());prefix.parent.mkdir(parents=True,exist_ok=True);prefix.with_suffix('.error.log').write_text(raw['traceback'])
            write(prefix.with_suffix('.json'),raw);receipt['cases'].append({'name':name,'status':raw['status']});write(output/'receipt.json',receipt)
        receipt['status']='COMPLETE';receipt['numerical_status']='PASS' if all(row['status']=='PASS' for row in receipt['cases']) else 'FAIL'
    except Exception as error:
        receipt.update(status='GLOBAL_ERROR',error_type=type(error).__name__,error_message=str(error),traceback=traceback.format_exc());(output/'global.error.log').write_text(receipt['traceback'])
    finally:
        if admission is not None:
            try:A.verify_installed(B,admission,roots);A.public_methods(B,sys.modules['bitsandbytes'],roots);receipt['source_post']=args.admission_sha256
            except Exception as error:receipt.update(status='SOURCE_POST_ERROR',source_post_error=str(error))
        dump_files=[path.relative_to(output).as_posix() for path in sorted((output/'compiler').rglob('*')) if path.is_file()] if (output/'compiler').exists() else []
        receipt['compiler_dumps']={'status':'ARTIFACTS_PRESENT_UNLINKED' if dump_files else 'NO_ARTIFACTS','files':dump_files,'buffer_assignment_files':[p for p in dump_files if 'buffer' in p.lower() and 'assignment' in p.lower()],'executable_link':'UNKNOWN'}
        receipt['artifacts']={path.relative_to(output).as_posix():{'sha256':D.sha(path),'bytes':path.stat().st_size} for path in output.rglob('*') if path.is_file() and path.name!='receipt.json'};write(output/'receipt.json',receipt)
    return 0 if receipt.get('status')=='COMPLETE' and receipt.get('numerical_status')=='PASS' else 2


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('output','kernel','source-pins','backend-probe','admission-helper','precision-probe','admission','admission-sha256','patch-manifest','parent-process-token','process-token','oracle','oracle-sha256','expected','expected-sha256'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--parent-pid',type=int,required=True);parser.add_argument('--deadline-epoch',type=float,required=True)
    return run(parser.parse_args())
if __name__=='__main__':raise SystemExit(main())
