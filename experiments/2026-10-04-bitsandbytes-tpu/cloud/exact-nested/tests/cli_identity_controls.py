"""Offline CLI interpreter-link identity controls. Never execute a CLI command."""
import argparse,copy,hashlib,json,shutil,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(HERE));import cli_transport as T

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(output,actual_python=None,actual_identity=None):
 rows=[]
 def check(name,python,identity,expected=None):
  try:T.verify_cli_identity(python,identity)
  except (ValueError,OSError,RuntimeError) as error:
   if expected is None:raise
   assert any(code in str(error)for code in(expected if isinstance(expected,tuple)else(expected,))),(name,str(error),expected)
   rows.append({'case':name,'status':'PASS','rejected':str(error)})
  else:
   assert expected is None,name
   rows.append({'case':name,'status':'PASS'})
 with tempfile.TemporaryDirectory(prefix='bnb-m4-cli-identity-')as directory:
  root=Path(directory)
  def fixture(name):
   env=root/name;site=env/'lib/python3.12/site-packages';package=site/'colab_cli';dist=site/'google_colab_cli-0.7.4.dist-info';(env/'bin').mkdir(parents=True);package.mkdir(parents=True);dist.mkdir()
   binary=root/(name+'-interpreter');binary.write_bytes(b'admitted executable fixture bytes; never executed')
   python=env/'bin/python';python.symlink_to(binary)
   (env/'pyvenv.cfg').write_text('fixture environment '+name+'\n');(dist/'METADATA').write_text('Name: google-colab-cli\nVersion: 0.7.4\n');(dist/'RECORD').write_text('fixture exact record\n');(package/'cli.py').write_text('# fixture CLI source; never imported\n');(package/'commands').mkdir();(package/'commands/run.py').write_text('# fixture nested source\n')
   identity={'distribution':'google-colab-cli','version':'0.7.4','python_sha256':sha(binary),'venv_config_sha256':sha(env/'pyvenv.cfg'),'metadata_sha256':sha(dist/'METADATA'),'record_sha256':sha(dist/'RECORD'),'files':{p.relative_to(site).as_posix():sha(p)for p in package.rglob('*.py')}}
   return env,python,identity,site,package,dist,binary
  env,python,identity,site,package,dist,binary=fixture('valid')
  check('correct-symlink-interpreter-original-environment',python,identity)
  regular=env/'bin/regular-python';regular.write_bytes(binary.read_bytes());check('alternative-correct-regular-interpreter',regular,identity)
  original_prefix=T.OfficialCLI(python,root/'never-created').prefix;assert original_prefix[0]==str(python) and str(binary)not in original_prefix
  rows.append({'case':'invocation-keeps-original-venv-path','status':'PASS'})
  changed=root/'changed-binary';changed.write_bytes(b'wrong executable');python.unlink();python.symlink_to(changed);check('wrong-resolved-interpreter',python,identity,'CLI_INTERPRETER_OR_DISTRIBUTION_SHA');python.unlink();python.symlink_to(binary)
  wrong=copy.deepcopy(identity);wrong['python_sha256']='0'*64;check('wrong-interpreter-identity-pin',python,wrong,'CLI_INTERPRETER_OR_DISTRIBUTION_SHA')
  for label,path in [('venv-config',env/'pyvenv.cfg'),('metadata',dist/'METADATA'),('wheel-record',dist/'RECORD'),('source-file',package/'cli.py')]:
   raw=path.read_bytes();path.write_bytes(raw+b'# changed\n');check('wrong-'+label,python,identity,'CLI_SOURCE_SHA'if label=='source-file'else'CLI_INTERPRETER_OR_DISTRIBUTION_SHA');path.write_bytes(raw)
  extra=package/'extra.py';extra.write_text('# unexpected source\n');check('extra-cli-source',python,identity,'CLI_SOURCE_INVENTORY');extra.unlink()
  missing=package/'cli.py';raw=missing.read_bytes();missing.unlink();check('missing-cli-source',python,identity,'CLI_SOURCE_INVENTORY');missing.write_bytes(raw)
  foreign=fixture('foreign');check('foreign-environment-even-same-interpreter-bytes',foreign[1],identity,'CLI_INTERPRETER_OR_DISTRIBUTION_SHA')
  for label,path in [('source-file',package/'cli.py'),('source-parent',package/'commands'),('package-root',package),('metadata-parent',dist),('site-parent',env/'lib'),('bin-parent',env/'bin')]:
   saved=root/(label+'-outside')
   path.rename(saved);path.symlink_to(saved,target_is_directory=saved.is_dir());check('symlink-escape-'+label,python,identity,('CLI_ENVIRONMENT_SYMLINK','CLI_SOURCE_INVENTORY'));path.unlink();saved.rename(path)
  alias=root/'env-alias';alias.symlink_to(env,target_is_directory=True);check('symlink-environment-root',alias/'bin/python',identity,'CLI_ENVIRONMENT_SYMLINK')
  python.unlink();python.symlink_to(root/'absent');check('broken-interpreter-link',python,identity,'No such file');python.unlink();python.symlink_to(root,target_is_directory=True);check('interpreter-resolves-directory',python,identity,'REGULAR_FILE');python.unlink();python.symlink_to(binary)
  check('correct-fixture-restored',python,identity)
  files=[p for p in root.rglob('*')if p.is_file()and not p.is_symlink()];disposal={'regular_files_before':len(files),'bytes_before':sum(p.stat().st_size for p in files),'temporary_directory':str(root)}
 assert not root.exists();disposal.update(directory_exists_after=False,fixture_bytes_after=0)
 actual=None
 if actual_python is not None:
  assert actual_identity is not None
  identity=json.loads(actual_identity.read_text());before=sha(actual_identity)
  check('actual-installed-read-only-symlink-cli-identity',actual_python,identity)
  wrong=dict(identity,python_sha256='0'*64);check('actual-installed-wrong-interpreter-pin-rejected',actual_python,wrong,'CLI_INTERPRETER_OR_DISTRIBUTION_SHA')
  assert sha(actual_identity)==before
  actual={'python':str(actual_python),'resolved_python':str(actual_python.resolve()),'identity_sha256':before,'resolved_python_sha256':sha(actual_python.resolve()),'identity_unchanged':True}
 result={'status':'PASS','controls_count':len(rows),'controls':rows,'disposal':disposal,'actual_identity':actual,'cli_transport_sha256':sha(HERE/'cli_transport.py'),'provider_calls':0,'CLI_commands_executed':0,'m4_status':'NOT_QUALIFIED'}
 output.parent.mkdir(parents=True,exist_ok=True)
 with output.open('x')as stream:json.dump(result,stream,indent=2,sort_keys=True);stream.write('\n')
 print(json.dumps({'status':'PASS','controls':len(rows),'fixture_bytes_deleted':disposal['bytes_before'],'provider_calls':0}))

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--actual-python',type=Path);p.add_argument('--actual-identity',type=Path);a=p.parse_args();run(a.output,a.actual_python,a.actual_identity)
