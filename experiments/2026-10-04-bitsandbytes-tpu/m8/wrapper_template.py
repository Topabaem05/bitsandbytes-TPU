"""Private M2 repeat. Submission requires root review of these exact bytes."""
import base64,hashlib,importlib.util,io,json,os,tempfile,zipfile
from pathlib import Path,PurePosixPath
PAYLOAD_B64=__SEALED_PAYLOAD_B64__
BINDING=__SEALED_BINDING__
DEADLINE=__SEALED_DEADLINE__

def entry(*,provider_factory=None,workspace=None,output=None):
    data=base64.b64decode(PAYLOAD_B64,validate=True)
    if hashlib.sha256(data).hexdigest()!=BINDING['packet_sha256']:raise ValueError('EMBEDDED_PACKET_SHA')
    base=Path(workspace) if workspace is not None else Path(tempfile.mkdtemp(prefix='m8-m2-owned-'))
    base.mkdir(parents=True,exist_ok=True)
    if any(base.iterdir()):raise ValueError('FRESH_WORKSPACE')
    (base/'payload.zip').write_bytes(data);payload=base/'payload'
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        rows=z.infolist();names=[r.filename for r in rows]
        if len(names)!=len(set(names)) or len(names)>1000 or sum(r.file_size for r in rows)>50*1024*1024:raise ValueError('PACKET_LIMIT')
        m=json.loads(z.read('manifest.json'))
        if set(names)!=set(m['files'])|{'manifest.json'}:raise ValueError('PACKET_INVENTORY')
        for r in rows:
            p=PurePosixPath(r.filename)
            if p.is_absolute() or any(s in ('','.','..') for s in r.filename.split('/')) or '\\' in r.filename or r.create_system!=3 or r.external_attr!=0o100644<<16:raise ValueError('PACKET_PATH_TYPE')
            raw=z.read(r.filename)
            if r.filename!='manifest.json' and m['files'][r.filename]!={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}:raise ValueError('PACKET_MEMBER_SHA')
            path=payload/r.filename;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    spec=importlib.util.spec_from_file_location('m8_sealed_batch',payload/'batch.py');batch=importlib.util.module_from_spec(spec);spec.loader.exec_module(batch)
    provider=provider_factory(base,m) if provider_factory else None
    return batch.execute(base,Path(output) if output else Path('/kaggle/working'),BINDING,DEADLINE,provider=provider)

if __name__=='__main__':
    result=entry();print(json.dumps({'kind':result['kind'],'status':result['status'],'m8_status':'NOT_QUALIFIED'}))
    raise SystemExit(0 if result['status']=='COMPLETE' else 2)
