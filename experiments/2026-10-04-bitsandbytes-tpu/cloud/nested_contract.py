"""Exact nested-v1 source contract. Only the explicit nested-79 packet may use it."""
import hashlib
import json
from pathlib import Path
import zipfile

NESTED_BINDINGS = {'nested_probe_sha256':'probe_nested.py','nested_inputs_sha256':'nested-inputs.json',
                   'nested_source_sha256':'nested-source.json','nested_schemas_sha256':'nested-schemas.json'}
NESTED_PROBE_SHA = '9cfbc7c6b9841b365b47ecfd0bf1b184fda754208aec5f214fe3d0cd9198348d'
NESTED_INPUT_SHA = 'b7490f4c622aef793d14bd9a7a38727754cb1fb8e6e1dec7ff91ef8c45fbafbc'
NESTED_SOURCE_SHA = 'c8d7519c92202e0238fbee55a63ffff32860f821b8957c8a135046a4294f13d2'
NESTED_SCHEMA_SHA = 'bd7dd210b2451749556f053b4472d4cbb820320d6608d246153888d0dcda0b71'
NESTED_PLUGIN_MANIFEST_SHA = 'da06577095abc2afe907f64104fe49fe604b1e27f6b777c96714686fbe8689d4'
NESTED_SCOPE = 'FIXED79_NESTED_DEFAULT_BODY_REFERENCE'
NESTED_FILES = {'__init__.py': 'c7ecdd758ed0d50f10e0b8fa3a17900143368ca6799bfa09e11bfe8118ab59db', 'blockwise.py': 'ad8ff158ef4977b47fd07ee0071be347c446abf2fd0c3a16c1852d1a9cde38be', 'compatibility.py': 'a1e3f67950734d3f3a7db6f81f96c7a25617b7bf03d6d48e49c63baf96ff28ed', 'functionalization.py': '0b31b2d86e7dd4aa10629e236fa1c0c18518eb5a3fb565714a9e40f432c2c93e', 'reference.py': '80d8329579c265e0375067285b62c9c2a505c57e2a0ebcfaf56a9d9b8e7f4bcc', 'registration.py': 'fe5b323506cd1bb9b598599ba59f3e948dbad9429fd937d1edea9c193f0617c0'}
NESTED_SCHEMAS = {'quantize_blockwise': 'bitsandbytes::quantize_blockwise(Tensor A, Tensor code, int blocksize) -> (Tensor, Tensor)', 'dequantize_blockwise': 'bitsandbytes::dequantize_blockwise(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype) -> Tensor', 'dequantize_blockwise.out': 'bitsandbytes::dequantize_blockwise.out(Tensor A, Tensor absmax, Tensor code, int blocksize, ScalarType dtype, Tensor($0! -> ) out) -> ()', 'quantize_4bit': 'bitsandbytes::quantize_4bit(Tensor A, int blocksize, str quant_type, ScalarType quant_storage) -> (Tensor, Tensor)', 'dequantize_4bit': 'bitsandbytes::dequantize_4bit(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype) -> Tensor', 'dequantize_4bit.out': 'bitsandbytes::dequantize_4bit.out(Tensor A, Tensor absmax, int blocksize, str quant_type, int[] shape, ScalarType dtype, Tensor($0! -> ) out) -> ()', 'gemm_4bit': 'bitsandbytes::gemm_4bit(Tensor A, Tensor B, int[] shapeB, Tensor absmax, int blocksize, str quant_type, Tensor? bias=None, Tensor? absmax_8bit=None, Tensor? absmax_code=None, Tensor? absmax_offset=None) -> Tensor'}


def require(ok, message):
    if not ok: raise ValueError(message)

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_nested_manifest(manifest):
    pins = (NESTED_PROBE_SHA,NESTED_INPUT_SHA,NESTED_SOURCE_SHA,NESTED_SCHEMA_SHA)
    for (field,name),pin in zip(NESTED_BINDINGS.items(),pins):
        require(manifest.get(field)==pin==manifest['files'].get(name,{}).get('sha256'),'NESTED_SOURCE_BINDING:'+field)
    require(manifest.get('nested_scope')==NESTED_SCOPE and manifest.get('nested_source_variant')=='nested-v1' and
            manifest.get('plugin_source_manifest_sha256')==NESTED_PLUGIN_MANIFEST_SHA and
            manifest['files'].get('plugin/source-manifest.json',{}).get('sha256')==NESTED_PLUGIN_MANIFEST_SHA,
            'NESTED_REVIEWED_SCOPE')

def verify_nested_payload(payload, manifest):
    verify_nested_manifest(manifest)
    plugin=payload/'plugin'; record=json.loads((plugin/'source-manifest.json').read_text())
    require(sha(plugin/'source-manifest.json')==NESTED_PLUGIN_MANIFEST_SHA and
            record.get('source_variant')=='nested-v1' and record.get('installed_python_files')==NESTED_FILES and
            record.get('operator_schemas')==NESTED_SCHEMAS,'NESTED_PLUGIN_MANIFEST')
    observed={p.relative_to(plugin/'src/bitsandbytes_tpu').as_posix():sha(p) for p in (plugin/'src/bitsandbytes_tpu').rglob('*.py')}
    require(observed==NESTED_FILES,'NESTED_PLUGIN_SOURCE')
    for row in record['files']:
        path=plugin/row['path']; require(not path.is_symlink() and path.is_file() and
            sha(path)==row['sha256'] and path.stat().st_size==row['bytes'],'NESTED_PACKAGING_SOURCE')
    admission=json.loads((payload/'source-admission.json').read_text())
    require(admission.get('bitsandbytes_tpu')=={'files':NESTED_FILES,'source_variant':'nested-v1',
            'nested_source_sha256':NESTED_SOURCE_SHA},'NESTED_ADMISSION_VARIANT')
    source=json.loads((payload/'nested-source.json').read_text()); schemas=json.loads((payload/'nested-schemas.json').read_text())
    require(source.get('files')==NESTED_FILES and schemas==NESTED_SCHEMAS,'NESTED_SPEC_MAP')

def verify_nested_wheel(wheel):
    with zipfile.ZipFile(wheel) as archive:
        names=archive.namelist(); require(len(names)==len(set(names)),'NESTED_WHEEL_DUPLICATE')
        files={name.removeprefix('bitsandbytes_tpu/'):hashlib.sha256(archive.read(name)).hexdigest()
               for name in names if name.startswith('bitsandbytes_tpu/') and name.endswith('.py')}
        require(not any(name.startswith('bitsandbytes_tpu/') and name.endswith('.pyc') for name in names)
                and files==NESTED_FILES,'NESTED_WHEEL_SOURCE')
    return files

def nested_cli(payload):
    return ['--backend-probe',payload/'probe_backend.py','--route-probe',payload/'probe_routes.py',
            '--precision-probe',payload/'probe_precision.py','--transfer-admission',payload/'transfer_admission.py',
            '--patch-manifest',payload/'patches/params4bit-xla-v1.json']
