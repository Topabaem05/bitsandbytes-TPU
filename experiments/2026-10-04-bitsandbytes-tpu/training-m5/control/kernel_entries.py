"""Canonical, side-effect-free M5 kernel phase bodies and artifact contracts."""
from pathlib import Path

def labels(runtime_wheels):
 return ('browser-marker','readiness','remote-unpack')+(('wheel-cache',)if runtime_wheels else())+('remote-install','remote-cpu','export-cpu','dispose-cpu','remote-device','export-device','dispose-device')
def identities(base,manifest):
 sources={str(Path(base)/name):r['sha256']for name,r in manifest['files'].items()if name.startswith('control/')and name.endswith('.py')}
 modules={Path(name).stem:(str(Path(base)/name),r['sha256'])for name,r in manifest['files'].items()if name.startswith('control/')and name.endswith('.py')and not Path(name).stem.startswith('__')}
 modules['m5_packet_admission']=(str(Path(base)/'payload/source/admission.py'),manifest['files']['source/admission.py']['sha256'])
 return sources,modules

def entry(label,base,packet_sha,birth,manifest,adoption):
 artifact=None;expected={};ready=label not in('browser-marker','readiness')
 if label=='browser-marker':
  body='import hashlib\nfrom pathlib import Path\n'+f'p=Path({adoption["marker_path"]!r});assert p.is_file()and not p.is_symlink()and hashlib.sha256(p.read_bytes()).hexdigest()=={adoption["marker_sha256"]!r}\nprint("EXACT_ROOT_MARKER_MATCH")\n'
 elif label=='readiness':body='from pathlib import Path\n'+f'b=Path({base!r});assert not b.exists()and not b.is_symlink();(b/"control/ownership").mkdir(parents=True)\n'
 elif label=='wheel-cache':body='from pathlib import Path\n'+f'Path({(base+"/wheel-cache")!r}).mkdir()\n'
 elif label.startswith('remote-')and label[7:]in('unpack','install','cpu','device'):
  phase=label[7:];identity={name.removeprefix('control/'):r['sha256']for name,r in manifest['files'].items()if name.startswith('control/')and name.endswith('.py')}
  body='import hashlib,runpy,sys\nfrom pathlib import Path\n'+f'b=Path({base!r}); expected={identity!r}\n'+'for name,digest in expected.items():\n p=b/"control"/name\n if p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest: raise ValueError("CONTROL_SOURCE_BYTES")\n'+f'sys.path.insert(0,str(b/"control"));sys.argv=["remote.py",{phase!r},"--base",str(b),"--packet-sha256",{packet_sha!r},"--allocation-epoch",{str(birth)!r}]\nrunpy.run_path(str(b/"control/remote.py"),run_name="__main__")\n'
  artifact=str(Path(base)/'records'/('allocation.json'if phase=='unpack'else'phase.json'));expected={'packet_sha256':packet_sha,**({'birth_epoch':birth,'original_deadline_epoch':birth+3600}if phase=='unpack'else{'kind':'M5_EXACT1024_STAGE','phase':phase,'allocation_epoch':birth,'status':'TERMINAL_LOCAL_REVIEW_REQUIRED','m5_status':'NOT_QUALIFIED'})}
 elif label.startswith('export-')and label[7:]in('unpack','install','cpu','device'):
  recovery='recovery-'+label[7:];body='import runpy,sys\nfrom pathlib import Path\n'+f'b=Path({base!r});sys.path.insert(0,str(b/"control"));sys.argv=["remote.py","export","--base",str(b),"--allocation-epoch",{str(birth)!r},"--label",{recovery!r}];runpy.run_path(str(b/"control/remote.py"),run_name="__main__")\n';artifact=str(Path(base)/(recovery+'-export.json'))
 elif label.startswith('dispose-')and label[8:]in('unpack','install','cpu','device'):
  recovery='recovery-'+label[8:];body='from pathlib import Path\nimport shutil\n'+f'b=Path({base!r});label={recovery!r}\n'+'(b/(label+".zip")).unlink();shutil.rmtree(b/(label+"-parts"));(b/(label+"-export.json")).unlink()\n'
 else:raise ValueError('UNSUPPORTED_KERNEL_PHASE')
 return {'body':body,'artifact_path':artifact,'artifact_expected':expected,'controls_ready':ready}
