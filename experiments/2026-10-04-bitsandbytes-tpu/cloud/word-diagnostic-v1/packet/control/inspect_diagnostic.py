"""Inspect retained diagnostic arrays and exact gzip bodies. Stdlib only."""
import argparse,gzip,hashlib,json,math,zlib
from pathlib import Path
import contract as C
CORE={'torch':'2.9.0+cpu','torch-xla':'2.9.0','libtpu':'0.0.21','jax':'0.7.1','jaxlib':'0.7.1'}
def labels():return ['bitcast-raw-scalar','bitcast-raw-vector','bitcast-functional-scalar','bitcast-functional-vector','amax-1','amax-255','shift-left','shift-right','scalar-compare-select']+[v+'-'+f for v in('original','true_shift','true_shift_amax')for f in('addsub','muldiv','midpoints','zero-normalize','tail-1-255')]
def array(v):
 C.require(set(v)=={'dtype','shape','int_values','hex_words','storage_sha256'},'DIAGNOSTIC_ARRAY_FIELDS');width={'float32':8,'int32':8,'int64':16,'uint8':2,'bool':2}.get(v['dtype']);C.require(width is not None,'DIAGNOSTIC_ARRAY_DTYPE');C.require(all(type(n)is int and n>=0 for n in v['shape']),'DIAGNOSTIC_ARRAY_SHAPE');size=math.prod(v['shape']);C.require(len(v['hex_words'])==len(v['int_values'])==size,'DIAGNOSTIC_ARRAY_EXTENT');raw=b''
 for i,w in zip(v['int_values'],v['hex_words']):
  C.require(type(i)is int and isinstance(w,str)and len(w)==width and w==format(i&((1<<(4*width))-1),'0'+str(width)+'x'),'DIAGNOSTIC_ARRAY_STORAGE_WORD');raw+=int(w,16).to_bytes(width//2,'little')
 C.require(hashlib.sha256(raw).hexdigest()==v['storage_sha256'],'DIAGNOSTIC_ARRAY_STORAGE_SHA')

def inspect(packet,records,*,qualified=True):
 packet=Path(packet);records=Path(records);C.admit(packet);cpu=C.read(records/'cpu/report.json');tpu=C.read(records/'tpu/report.json');manifest=C.sha(packet/'manifest.json')
 C.require(cpu['status']=='CPU_ORACLE_COMPLETE'and cpu['cpu_controls']['status']=='PASS'and tpu['status']=='TPU_DIAGNOSTIC_COMPLETE_WITH_EXACT_COMPARISONS','DIAGNOSTIC_TERMINAL_REPORTS')
 for report,mode in[(cpu,'cpu'),(tpu,'tpu')]:
  C.require(report['mode']==mode and report['m4_status']=='NOT_QUALIFIED'and report['source_manifest_sha256']==manifest and [r['label']for r in report['rows']]==labels(),'DIAGNOSTIC_REPORT_SOURCE_AND_COVERAGE')
  if qualified:
   runtime=report['runtime'];C.require(runtime['versions']==CORE and runtime['python']=='3.12.14'and runtime['system']=='Linux'and runtime['machine']=='x86_64'and runtime['status']==('PINNED_LINUX_CPU_DIAGNOSTIC_ORACLE'if mode=='cpu'else'PINNED_TPU_DIAGNOSTIC_RUNTIME'),'DIAGNOSTIC_RUNTIME_PROFILE')
   if mode=='tpu':C.require(runtime['requested_backend']=='TPU'and runtime['observed_backend']=='TPU'and runtime['devices']and all(d.startswith('TPU:')for d in runtime['devices']),'DIAGNOSTIC_OBSERVED_TPU')
  C.require(C.read(records/mode/'inputs.json')==report['inputs'],'DIAGNOSTIC_PHYSICAL_INPUTS_EQUAL');C.require({p.name for p in(records/mode/'rows').glob('*.json')}=={r['label']+'.json'for r in report['rows']},'DIAGNOSTIC_EXACT_ROW_INVENTORY')
  for v in report['inputs'].values():array(v)
  for row in report['rows']:
   C.require(C.read(records/mode/'rows'/(row['label']+'.json'))==row,'DIAGNOSTIC_PHYSICAL_ROW_EQUAL');C.require(row['status']in('RETURNED','ERROR'),'DIAGNOSTIC_ROW_OUTCOME')
   if mode=='cpu':C.require(row['status']=='RETURNED','CPU_ORACLE_ROW_REQUIRED')
   if row['status']=='ERROR':C.require(isinstance(row.get('error_type'),str)and isinstance(row.get('error_message'),str),'DIAGNOSTIC_ERROR_RETAINED')
   for v in row.get('outputs',{}).values():array(v)
   for v in row.get('parameters',{}).values():array(v)
 C.require(cpu['inputs']==tpu['inputs'],'DIAGNOSTIC_CPU_TPU_INPUT_BYTES');exact=0
 for c,t in zip(cpu['rows'],tpu['rows']):
  if t['status']=='RETURNED':
   C.require(set(c['outputs'])==set(t['outputs']),'DIAGNOSTIC_OUTPUT_INVENTORY');expected={k:t['outputs'][k]==c['outputs'][k]for k in c['outputs']};C.require(t['cpu_exact_comparison']==expected and t['cpu_exact_match']is all(expected.values()),'DIAGNOSTIC_COMPARISONS_RECOMPUTED');exact+=all(expected.values())
 graphs=tpu['physical_graphs'];declared=set();raw_total=0
 for row in graphs:
  rel=C.name(row['path']);C.require(rel.startswith('graphs/')and rel.endswith(('.hlo.txt.gz','.hlo.pb.gz'))and rel not in declared,'DIAGNOSTIC_GRAPH_NAME');declared.add(rel);p=records/'tpu'/rel;raw=p.read_bytes();C.require(len(raw)==row['gzip_bytes']and C.sha(p)==row['gzip_sha256'],'DIAGNOSTIC_GRAPH_GZIP_SHA');decoder=zlib.decompressobj(16+zlib.MAX_WBITS);body=decoder.decompress(raw,8*1024*1024+1);C.require(decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail and len(body)<=8*1024*1024 and len(body)==row['raw_bytes']and hashlib.sha256(body).hexdigest()==row['raw_sha256'],'DIAGNOSTIC_GRAPH_RAW_SHA_SINGLE_MEMBER');raw_total+=len(body)
 C.require(raw_total<=96*1024*1024 and {str(p.relative_to(records/'tpu'))for p in(records/'tpu/graphs').glob('*')}==declared,'DIAGNOSTIC_EXACT_GRAPH_INVENTORY')
 for row in tpu['rows']:
  if row.get('observation_status')=='COMPLETE':C.require({f'graphs/{row["label"]}.hlo.txt.gz',f'graphs/{row["label"]}.hlo.pb.gz'}<=declared,'DIAGNOSTIC_OBSERVED_ROW_HLO')
  elif row['status']=='RETURNED':C.require(row.get('observation_status')=='ERROR'and isinstance(row.get('observation_error',{}).get('message'),str),'DIAGNOSTIC_OBSERVATION_ERROR_RETAINED')
 C.require(tpu['returned']==sum(r['status']=='RETURNED'for r in tpu['rows'])and tpu['errors']==sum(r['status']=='ERROR'for r in tpu['rows'])and tpu['exact_matches']==exact,'DIAGNOSTIC_COUNTS_RECOMPUTED')
 for mode in('cpu','tpu'):
  rec=C.read(records/'steps'/mode/'result.json');C.require(rec['status']=='PASS'and rec['exit_code']==0 and rec['environment']['PJRT_DEVICE']==('TPU'if mode=='tpu'else None),'DIAGNOSTIC_OWNED_MODE_EXECUTION');C.closed(rec['cleanup'])
 return {'kind':'M4_WORD_DIAGNOSTIC_HOST_INSPECTION_V1','status':'COMPLETE_DIAGNOSTIC_EVIDENCE_RECOMPUTED','rows':24,'returned':tpu['returned'],'errors':tpu['errors'],'exact_matches':exact,'graph_objects':len(graphs),'raw_graph_bytes':raw_total,'m4_status':'NOT_QUALIFIED','qualification':'PINNED_LINUX_CPU_AND_TPU_DIAGNOSTIC'if qualified else'SYNTHETIC_FIXTURE_ONLY'}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--packet',type=Path,required=True);p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();C.require(not a.output.exists(),'FRESH_DIAGNOSTIC_INSPECTION');C.write(a.output,inspect(a.packet,a.records))
