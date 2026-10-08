"""Test complete canonical-map staging and incorrect source rejection in isolated fixtures."""
import argparse,json,shutil,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));import materialize_controls as M
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False);source=M.read(HERE/'source-map.json');adoption=M.read(HERE/'adoption-map.json');root=a.output/'canonical-fixture';root.mkdir();rows=adoption['files']+adoption['exact_prerequisites']
for r in rows:
 q=root/r['target'];q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/r['source'],q)
stage=a.output/'stage';proof=M.materialize(root,HERE/'source-map.json',HERE/'adoption-map.json',stage)
assert set(source)=={p.relative_to(stage).as_posix()for p in stage.rglob('*')if p.is_file()}
for n,r in source.items():assert M.seal(stage/n)==r
bad=root/next(r['target']for r in rows if r['source']=='cloud/remote.py');bad.write_bytes(bad.read_bytes()+b' ')
try:M.materialize(root,HERE/'source-map.json',HERE/'adoption-map.json',a.output/'rejected-stage')
except ValueError as e:assert str(e)=='CANONICAL_SOURCE_SHA:cloud/remote.py'
else:raise AssertionError('FALSE_ACCEPT')
assert not(a.output/'rejected-stage').exists()
M.Path(a.output/'results.json').write_text(json.dumps({'status':'PASS','count':2,'controls':['full_map_canonical_only_stage_exact','changed_canonical_source_rejected_before_write'],'proof':proof,'actual_canonical_adoption':'NOT_RUN','private_prepared_source_dependencies':0},sort_keys=True,indent=2)+'\n');print(json.dumps({'status':'PASS','count':2}))
