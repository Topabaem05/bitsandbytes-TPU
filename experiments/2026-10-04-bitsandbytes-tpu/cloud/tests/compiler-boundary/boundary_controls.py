"""Exact default-dump request, fixed science, and physical entry-boundary controls."""
import argparse,hashlib,json,sys
from pathlib import Path
H=Path(__file__).resolve().parent;sys.path[:0]=[str(H/'compiler-native'),str(H/'cloud')]
import compiler_flags as F,compiler_cloud_contract as C,dump_observation as D
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(exist_ok=False);baseline=C.read(H/'boundary-baseline.json');rows=[];dump='/content/bnb-tpu-first/compiler-flag-preflight-private'
def yes(n):rows.append({'name':n,'status':'PASS'})
def no(n,v):
 try:F.exact_request(v,dump)
 except ValueError as e:assert str(e)=='FLAG_PREFLIGHT_EXACT_REQUEST';yes(n)
 else:raise AssertionError('FALSE_ACCEPT:'+n)
request=F.flags(dump);assert len(request.split())==13;F.exact_request(request,dump);yes('new_exact13_request_accepted')
removed='--xla_dump_hlo_pass_re=pipeline-start|pipeline-end';assert F.FLAGS==[v for v in baseline['old_flags']if v!=removed];yes('only_pipeline_pass_filter_removed')
old=' '.join(['--xla_dump_to='+dump,*baseline['old_flags']]);assert len(old.split())==14;no('old_exact14_request_rejected',old)
for extra in ['--r6_unknown_extra=false','--xla_dump_hlo_pass_re=.*','--xla_dump_hlo_unoptimized_snapshots=false']:no('extra_token_rejected:'+extra,request+' '+extra)
no('missing_token_rejected',' '.join(request.split()[:-1]));no('reordered_token_rejected',' '.join(reversed(request.split())))
for flag in ['--xla_dump_hlo_as_text=true','--xla_dump_hlo_as_proto=true','--xla_dump_module_metadata=true','--xla_dump_max_hlo_modules=32']:assert flag in F.FLAGS
yes('text_proto_module_metadata_and_module32_preserved')
protocol=C.read(H/'compiler-native/protocol.json');assert protocol['case_ids']==['direct-fp32','functional-fp32','direct-bf16','functional-bf16','tail-reference-fp32'];assert protocol['precision']=='highest'
for name,key in [('probe-profile.json','profile'),('protocol.json','protocol')]:
 q=H/'compiler-native'/name;assert {'sha256':C.sha(q),'bytes':q.stat().st_size}==baseline[key]
yes('five_cases_profile_protocol_numerical_criteria_preserved')
policy=C.read(H/'compiler-native/compiler-policy.json');assert policy['limits']==baseline['policy_limits']=={'entries':1024,'file_bytes':16777216,'total_bytes':134217728}
for k in ['minimum_free_bytes','poll_seconds','shutdown_grace','scientific_record_budget_bytes']:assert policy[k]==baseline[k]
yes('all_storage_monitor_closure_limits_unchanged')
d=out/'physical-entries';d.mkdir();identity=(d.stat().st_dev,d.stat().st_ino)
for i in range(1024):(d/str(i)).write_bytes(b'')
r=D.observe(d,identity,policy['limits']);assert r['status']=='WITHIN_OBSERVED_LIMITS'and r['entries']==1024;yes('physical1024_entries_allowed')
(d/'1024').write_bytes(b'');r=D.observe(d,identity,policy['limits']);assert r['status']=='DUMP_ENTRY_LIMIT'and r['entries']==1025;yes('physical1025_entries_rejected')
# A prior generation cannot enter through a freshly supplied outer map.
m=C.read(H/'compiler-native/manifest.json');assert m['generation']!=baseline['old_generation']and C.MANIFEST_SHA!=baseline['old_manifest_sha256'];yes('new_generation_and_manifest_required')
C.write(out/'results.json',{'status':'PASS','count':len(rows),'controls':rows,'actual_TPU':'NOT_RUN','actual_fixed_libtpu_parse':'NOT_RUN','provider_calls':0});print(json.dumps({'status':'PASS','count':len(rows)}))
