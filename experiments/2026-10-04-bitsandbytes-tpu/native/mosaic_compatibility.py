"""Explicit pinned JAX071 Mosaic8→7 conversion candidate. No installed code changes."""
import base64,hashlib,json
from pathlib import Path
TARGET=7
TCC_SHA='ae78bf3e65571d8c15004e76863747e2b414e7f17073cee151c643f8c7932d37'
MAX_PAYLOAD=4*1024*1024

def sha(b):return hashlib.sha256(b).hexdigest()
def require(v,m):
 if not v:raise ValueError(m)
def versions(jax_version,jaxlib_version):require(jax_version==jaxlib_version=='0.7.1','MOSAIC_CONVERTER_RUNTIME_071_ONLY')

def convert_payload(payload,expected_sha256):
 """Caller retains original payload. Conversion errors propagate before native dispatch."""
 require(type(payload)is str and len(payload.encode())<=MAX_PAYLOAD,'MOSAIC_PAYLOAD_TYPE_SIZE')
 require(sha(payload.encode())==expected_sha256,'MOSAIC_ORIGINAL_PAYLOAD_HASH')
 config=json.loads(payload);require(type(config)is dict and type(config.get('custom_call_config'))is dict,'MOSAIC_CONFIG_SCHEMA')
 nested=config['custom_call_config'];require(type(nested.get('body'))is str,'MOSAIC_BODY_SCHEMA')
 original=base64.b64decode(nested['body'],validate=True);require(original and len(original)<=MAX_PAYLOAD,'MOSAIC_BODY_SIZE')
 # These imports follow the caller's existing client-ownership guard. This module has no import side effect.
 import jax,jaxlib
 versions(jax.__version__,jaxlib.__version__)
 from jax._src import tpu_custom_call as T
 require(sha(Path(T.__file__).read_bytes())==TCC_SHA,'MOSAIC_SERIALIZER_SOURCE_PIN')
 from jax._src.interpreters import mlir
 from jax._src.lib import tpu
 from jaxlib.mlir import ir
 from jaxlib.mlir.passmanager import PassManager
 import io
 with mlir.make_ir_context()as ctx,ir.Location.unknown():
  tpu.register_dialect(ctx);ctx.allow_unregistered_dialects=True
  module=ir.Module.parse(original);require('stable_mosaic.version'in module.operation.attributes,'MOSAIC_SOURCE_VERSION_MISSING')
  source_version=ir.IntegerAttr(module.operation.attributes['stable_mosaic.version']).value
  require(source_version==8,'MOSAIC_SOURCE_VERSION_8_ONLY')
  PassManager.parse('builtin.module(mosaic-serde{serialize=false})').run(module.operation);module.operation.verify();before=str(module)
  # The official versioned pass applies its downgrade rules and rejects unsupported features.
  PassManager.parse('builtin.module(mosaic-serde{serialize=true target-version=7})').run(module.operation)
  require(ir.IntegerAttr(module.operation.attributes['stable_mosaic.version']).value==TARGET,'MOSAIC_TARGET_VERSION_READBACK')
  buffer=io.BytesIO();module.operation.write_bytecode(buffer,desired_version=0);body=buffer.getvalue()
  PassManager.parse('builtin.module(mosaic-serde{serialize=false})').run(module.operation);module.operation.verify();after=str(module)
  # This bounded fixed-kernel candidate requires exact current-IR roundtrip equality.
  # It can conservatively reject other valid migrations, e.g. removal of explicit default attributes.
  require(before==after,'MOSAIC_FIXED_KERNEL_EXACT_IR_ROUNDTRIP')
 new_config=json.loads(payload);new_config['custom_call_config']['body']=base64.b64encode(body).decode()
 candidate=json.dumps(new_config,separators=(',',':'),allow_nan=False)
 old_other=json.loads(payload);new_other=json.loads(candidate)
 old_other['custom_call_config'].pop('body');new_other['custom_call_config'].pop('body');require(old_other==new_other,'MOSAIC_OTHER_CONFIG_UNCHANGED')
 audit={'kind':'MOSAIC_SUPPORTED_SERDE_8_TO_7_V1','status':'FRONTEND_CONVERSION_ONLY','source_version':8,'target_version':TARGET,'original_payload_sha256':expected_sha256,'original_body_sha256':sha(original),'converted_payload_sha256':sha(candidate.encode()),'converted_body_sha256':sha(body),'current_ir_sha256':sha(before.encode()),'roundtrip_ir_sha256':sha(after.encode()),'exact_current_ir_equal':True,'jax':'0.7.1','jaxlib':'0.7.1','serializer_python_sha256':TCC_SHA,'converter_sha256':sha(Path(__file__).read_bytes()),'native_backend_acceptance':'NOT_RUN','runtime_changed':False,'executable_link':'UNKNOWN','allocator_peak':'UNKNOWN'}
 return candidate,audit
