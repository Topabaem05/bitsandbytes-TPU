"""Independent CPU frontend audit. Reads both retained payloads; never executes a TPU backend."""
import base64
import hashlib
import io
import importlib.metadata
import platform
import json
import os
from pathlib import Path
import sys
import time

TCC_SHA='ae78bf3e65571d8c15004e76863747e2b414e7f17073cee151c643f8c7932d37'
HERE=Path(__file__).resolve().parent
MAX=4*1024*1024
def sha(value):return hashlib.sha256(value).hexdigest()
def require(value,message):
    if not value:raise ValueError(message)
def pairs(values):
    result={}
    for key,value in values:
        require(key not in result,'AUDIT_DUPLICATE_JSON_KEY');result[key]=value
    return result
def read_json(text):return json.loads(text,object_pairs_hook=pairs)
def sources():return {name:sha((HERE/name).read_bytes()) for name in ('mosaic_audit.py','mosaic_compatibility.py')}
def runtime():
    require(os.environ.get('JAX_PLATFORMS')=='cpu','AUDIT_CPU_PLATFORM_REQUIRED')
    require(sys.version_info[:3]==(3,12,14),'AUDIT_PYTHON_31214_REQUIRED')
    import jax,jaxlib
    require(jax.__version__==jaxlib.__version__=='0.7.1','AUDIT_EXACT_JAX071_REQUIRED')
    from jax._src import tpu_custom_call as T
    require(sha(Path(T.__file__).read_bytes())==TCC_SHA,'AUDIT_SERIALIZER_SOURCE')
    from jax._src.lib import tpu
    def distribution_record(name):
        d=importlib.metadata.distribution(name);record=next(f for f in d.files if f.name=='RECORD');return sha(Path(d.locate_file(record)).read_bytes())
    extension=Path(jaxlib.__file__).parent/'mlir/_mlir_libs/_tpu_ext.so'
    binary_readback={'mosaic_extension_sha256':sha(extension.read_bytes()),'mosaic_python_sha256':sha(Path(tpu.__file__).read_bytes()),'jax_record_sha256':distribution_record('jax'),'jaxlib_record_sha256':distribution_record('jaxlib'),'platform':sys.platform,'machine':platform.machine(),'python_executable':sys.executable}
    # Backend initialization is unnecessary for MLIR parse/serde. No devices() call.
    return {'python':'.'.join(map(str,sys.version_info[:3])),'jax':jax.__version__,'jaxlib':jaxlib.__version__,'JAX_PLATFORMS':'cpu','serializer_python_sha256':TCC_SHA,'qualification':'EXACT_PINNED_CPU_FRONTEND_ONLY','TPU_backend':'NOT_RUN',**binary_readback}
def body(payload):
    require(type(payload)is str and len(payload.encode())<=MAX,'AUDIT_CONFIG_SIZE')
    config=read_json(payload)
    require(type(config)is dict and type(config.get('custom_call_config'))is dict,'AUDIT_CONFIG_SCHEMA')
    b=base64.b64decode(config['custom_call_config']['body'],validate=True)
    require(0<len(b)<=MAX,'AUDIT_BODY_SIZE');return config,b
def audit(row):
    """Re-derive official downgrade and parse both bodies independently of converter assertions."""
    original=row['original_payload'];converted=row['payload'];old,original_body=body(original);new,converted_body=body(converted)
    require(row['original_payload_sha256']==sha(original.encode()) and row['payload_sha256']==sha(converted.encode()),'AUDIT_RETAINED_PAYLOAD_HASHES')
    a=read_json(original);b=read_json(converted);a['custom_call_config'].pop('body');b['custom_call_config'].pop('body');require(a==b,'AUDIT_NONBODY_CONFIG_UNCHANGED')
    from jax._src.interpreters import mlir
    from jax._src.lib import tpu
    from jaxlib.mlir import ir
    from jaxlib.mlir.passmanager import PassManager
    with mlir.make_ir_context() as ctx,ir.Location.unknown():
        tpu.register_dialect(ctx);ctx.allow_unregistered_dialects=True
        m=ir.Module.parse(original_body);n=ir.Module.parse(converted_body)
        require(ir.IntegerAttr(m.operation.attributes['stable_mosaic.version']).value==8,'AUDIT_ORIGINAL_VERSION8')
        require(ir.IntegerAttr(n.operation.attributes['stable_mosaic.version']).value==7,'AUDIT_CONVERTED_VERSION7')
        PassManager.parse('builtin.module(mosaic-serde{serialize=false})').run(m.operation);require(m.operation.verify(),'AUDIT_ORIGINAL_IR_VERIFY');before=str(m)
        PassManager.parse('builtin.module(mosaic-serde{serialize=false})').run(n.operation);require(n.operation.verify(),'AUDIT_CONVERTED_IR_VERIFY');after=str(n)
        require(before==after,'AUDIT_PARSED_CURRENT_IR_EQUAL')
        PassManager.parse('builtin.module(mosaic-serde{serialize=true target-version=7})').run(m.operation)
        buffer=io.BytesIO();m.operation.write_bytecode(buffer,desired_version=0);reproduced=buffer.getvalue()
        require(reproduced==converted_body,'AUDIT_REPRODUCED_TARGET_BODY_BYTES')
    old['custom_call_config']['body']=base64.b64encode(reproduced).decode();require(json.dumps(old,separators=(',',':'),allow_nan=False)==converted,'AUDIT_REPRODUCED_CONFIG_BYTES')
    derived={'kind':'MOSAIC_SUPPORTED_SERDE_8_TO_7_V1','status':'FRONTEND_CONVERSION_ONLY','source_version':8,'target_version':7,'original_payload_sha256':sha(original.encode()),'original_body_sha256':sha(original_body),'converted_payload_sha256':sha(converted.encode()),'converted_body_sha256':sha(converted_body),'current_ir_sha256':sha(before.encode()),'roundtrip_ir_sha256':sha(after.encode()),'exact_current_ir_equal':True,'jax':'0.7.1','jaxlib':'0.7.1','serializer_python_sha256':TCC_SHA,'converter_sha256':sources()['mosaic_compatibility.py'],'native_backend_acceptance':'NOT_RUN','runtime_changed':False,'executable_link':'UNKNOWN','allocator_peak':'UNKNOWN'}
    require(row['payload_conversion']==derived,'AUDIT_RECOMPUTED_CONVERSION_FIELDS')
    return derived
def run(request):
    started=time.time();deadline=request['deadline_epoch'];require(type(deadline)in(int,float) and started<deadline,'AUDIT_DEADLINE')
    before=sources();require(request['sources']==before,'AUDIT_ADMITTED_SOURCES');profile=runtime();cases={}
    require(type(request['cases'])is dict and ((request.get('phase')=='CONVERSION_REPLAY' and 0<len(request['cases'])<=4) or (request.get('phase')=='CPU_RUNTIME_PREFLIGHT' and request['cases']=={})),'AUDIT_CASE_COUNT')
    for name,row in request['cases'].items():
        require(time.time()<deadline,'AUDIT_DEADLINE');cases[name]=audit(row)
    require(sources()==before,'AUDIT_SOURCE_POST');require(time.time()<deadline,'AUDIT_DEADLINE')
    require(runtime()==profile,'AUDIT_RUNTIME_POST_READBACK')
    return {'kind':'INDEPENDENT_MOSAIC_CPU_FRONTEND_AUDIT','status':'REPRODUCED_EXACT_CONVERSION','pid':os.getpid(),'parent_pid':os.getppid(),'pgid':os.getpgid(0),'deadline_epoch':deadline,'started_epoch':started,'finished_epoch':time.time(),'runtime':profile,'runtime_post':profile,'source_pre':before,'source_post':sources(),'request_sha256':sha(json.dumps(request,sort_keys=True,separators=(',',':'),allow_nan=False).encode()),'cases':cases,'actual_execution':'VERIFIED_SEPARATELY','executable_link':'UNKNOWN','allocator_peak':'UNKNOWN'}
if __name__=='__main__':
    try:
        text=sys.stdin.read(MAX*5+1);require(len(text.encode())<=MAX*5,'AUDIT_REQUEST_SIZE');report=run(read_json(text));code=0
    except Exception as error:report={'status':'REJECTED','type':type(error).__name__,'error':str(error)};code=2
    print(json.dumps(report,sort_keys=True,allow_nan=False));raise SystemExit(code)
