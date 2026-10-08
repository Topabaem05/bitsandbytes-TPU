"""Real pinned CPU serializer identity and version rules; no backend compile or device work."""
import argparse,hashlib,json,os
from pathlib import Path
import mosaic_paths as M
os.environ['JAX_PLATFORMS']='cpu'
def run(args):
    M.admit(args.native);args.output.mkdir(exist_ok=False)
    import jax,jaxlib,mosaic_compatibility as C
    from jax._src import tpu_custom_call as T
    assert jax.__version__==jaxlib.__version__=='0.7.1'
    actual=hashlib.sha256(Path(T.__file__).read_bytes()).hexdigest();assert actual==C.TCC_SHA
    rows=[{'case':'actual-pinned-serializer-source-identity','status':'PASS','serializer_sha256':actual,'jax':'0.7.1','jaxlib':'0.7.1'}]
    data=json.loads((args.fixtures/'printer/metadata.json').read_text())['cases']['f32'];converted,audit=C.convert_payload(data['original'],C.sha(data['original'].encode()));assert converted==data['converted']and audit==data['audit'];rows.append({'case':'exact-supported-official-version8-to7-conversion','status':'PASS'})
    strict=(args.fixtures/'strict-ordering-native-config.json').read_text()
    try:C.convert_payload(strict,C.sha(strict.encode()));raise AssertionError('strict ordering accepted')
    except Exception as error:
        assert 'strict ordering is not set to True'in str(error);rows.append({'case':'valid-unsupported-strict-ordering-rejected','status':'PASS','type':type(error).__name__,'error':str(error)})
    try:C.convert_payload(data['original'],'0'*64);raise AssertionError('wrong independent original hash accepted')
    except ValueError as error:assert str(error)=='MOSAIC_ORIGINAL_PAYLOAD_HASH';rows.append({'case':'wrong-original-payload-hash-rejected','status':'PASS'})
    report={'status':'PASS','count':len(rows),'controls':rows,'scope':'PINNED_JAX071_CPU_SERDE_ONLY','actual_TPU':'NOT_RUN','libtpu_backend':'NOT_RUN','M6':'NOT_QUALIFIED'};(args.output/'results.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PASS','count':len(rows)}))
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in('native','fixtures','output'):p.add_argument('--'+name,type=Path,required=True)
    run(p.parse_args())
