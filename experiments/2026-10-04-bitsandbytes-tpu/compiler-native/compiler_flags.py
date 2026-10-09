"""Exact fixed-libTPU dump request and backend-parser preflight admission. No scientific claim."""
import hashlib,math,os,re
from pathlib import Path
LIBTPU_VERSION='0.0.21'
LIBTPU_SHA='cdb7980d4332097b8e16576568e138ef54cf9fb8ed5aae2aeefbd0c5a425e94a'
FLAGS=['--xla_dump_hlo_as_text=true','--xla_dump_hlo_as_proto=true','--xla_dump_module_metadata=true','--xla_dump_hlo_pass_re=pipeline-start|pipeline-end','--xla_dump_include_timestamp=true','--xla_dump_max_hlo_modules=32','--xla_dump_compress_protos=false','--xla_dump_hlo_snapshots=false','--xla_dump_full_hlo_config=false','--xla_dump_large_constants=false','--xla_dump_hlo_as_dot=false','--xla_dump_hlo_as_html=false','--xla_dump_hlo_as_url=false']
TORCH_XLA_INIT_SHA='c3e11a63b8ebe9dcf9ca94d4f7ceebba4983d49e98f57ceddeb79880063eb29c'
LIBTPU_INIT_FLAGS=['--xla_tpu_use_enhanced_launch_barrier=false','--xla_latency_hiding_scheduler_rerun=1','--xla_tpu_prefer_async_allgather_to_allreduce=true','--xla_tpu_enable_flash_attention=false','--xla_enable_async_all_gather=true','--xla_enable_async_collective_permute=true']
VERSIONS={'torch':'2.9.0+cpu','torch-xla':'2.9.0','libtpu':'0.0.21','jax':'0.7.1','jaxlib':'0.7.1'}
def require(v,label):
 if not v:raise ValueError(label)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb')as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def flags(dump_directory):
 p=Path(dump_directory);require(p.is_absolute()and p.resolve()==p.absolute()and not any(c.isspace()for c in str(p)),'FLAG_PREFLIGHT_DUMP_PATH')
 return ' '.join(['--xla_dump_to='+str(p),*FLAGS])
def exact_request(value,dump_directory):
 require(value==flags(dump_directory),'FLAG_PREFLIGHT_EXACT_REQUEST');return value

def admit(record,*,manifest_sha,policy_sha,dump_directory):
 require(record.get('kind')=='R6_LIBTPU_DUMP_FLAG_PREFLIGHT_V1'and record.get('status')=='LIBTPU_FLAGS_RECOGNIZED_BACKEND_INIT_ONLY','FLAG_PREFLIGHT_STATUS')
 require(record.get('manifest_sha256')==manifest_sha and record.get('policy_sha256')==policy_sha,'FLAG_PREFLIGHT_SOURCE_BINDING')
 require(record.get('library_sha256')==LIBTPU_SHA and record.get('versions')==VERSIONS,'FLAG_PREFLIGHT_FIXED_LIBRARY')
 require(record.get('system')=='Linux'and record.get('machine')=='x86_64'and record.get('python')=='3.12.14'and record.get('requested_backend')==record.get('observed_backend')=='TPU','FLAG_PREFLIGHT_RUNTIME')
 require(type(record.get('devices'))is list and record['devices']and all(type(d)is str and d.startswith('TPU:')for d in record['devices']),'FLAG_PREFLIGHT_DEVICES')
 require(record.get('torch_xla_init_sha256')==TORCH_XLA_INIT_SHA and record.get('effective_xla_flags')==flags(dump_directory)+' --xla_cpu_enable_fast_math=false','FLAG_PREFLIGHT_FRONTEND_DEFAULTS')
 require(type(record.get('effective_libtpu_init_args'))is str and record['effective_libtpu_init_args'].split()==LIBTPU_INIT_FLAGS,'FLAG_PREFLIGHT_LIBTPU_DEFAULTS')
 require(re.fullmatch('[0-9a-f]{32}',record.get('process_token','')),'FLAG_PREFLIGHT_TOKEN')
 require(record.get('dump_directory')==str(Path(dump_directory)),'FLAG_PREFLIGHT_DIRECTORY_BINDING');exact_request(record.get('xla_flags'),dump_directory)
 require(record.get('native_case_count')==record.get('tensor_computation_count')==0 and record.get('scope')=='PARSER_AND_BACKEND_INITIALIZATION_ONLY_NOT_NATIVE_SCIENCE','FLAG_PREFLIGHT_SCOPE')
 for k in ('pid','pgid','parent_pid'):require(type(record.get(k))is int and record[k]>1,'FLAG_PREFLIGHT_PROCESS')
 require(record['pid']==record['pgid'],'FLAG_PREFLIGHT_GROUP_LEADER')
 require(all(type(record.get(k))in(int,float)and math.isfinite(record[k])for k in ('started_epoch','finished_epoch'))and record['started_epoch']<=record['finished_epoch'],'FLAG_PREFLIGHT_TIME')
 return record

def owned(record,ownership,result,*,manifest_sha,policy_sha,dump_directory):
 require(str(dump_directory)=='/content/bnb-tpu-first/compiler-flag-preflight-private','FLAG_PREFLIGHT_PRIVATE_SCOPE')
 admit(record,manifest_sha=manifest_sha,policy_sha=policy_sha,dump_directory=dump_directory)
 argv=ownership.get('argv',[]);require(len(argv)==13 and argv[0]=='/content/bnb-tpu-first/venv/bin/python'and argv[1]=='-B'and argv[2]=='/content/bnb-tpu-first/payload/native/compiler_flag_probe.py'and argv[3]=='--output'and argv[4]=='/content/bnb-tpu-first/records/compiler-flag-preflight.json'and argv[5]=='--dump-directory'and argv[6]==str(dump_directory)and argv[7]=='--policy'and argv[8]=='/content/bnb-tpu-first/payload/native/compiler-policy.json'and argv[9]=='--policy-sha256'and argv[10]==policy_sha and argv[11]=='--process-token'and argv[12]==record.get('process_token'),'FLAG_PREFLIGHT_OWNED_ARGV')
 require(record['pid']==ownership.get('pid')==ownership.get('pgid')and record['parent_pid']==ownership.get('owner_pid'),'FLAG_PREFLIGHT_OWNED_IDENTITY')
 cleanup=ownership.get('cleanup',{});require(cleanup.get('status')=='CLEANUP_VERIFIED'and cleanup.get('leader_reaped')is True and cleanup.get('group_absence')=='OBSERVED_NO_SUCH_GROUP'and cleanup.get('errors')==[]and cleanup.get('exit_status')==0,'FLAG_PREFLIGHT_OWNED_CLOSURE')
 require(result.get('status')=='PASS'and result.get('exit_code')==0 and not result.get('error')and result.get('cleanup',{}).get('errors')==[],'FLAG_PREFLIGHT_STEP_PASS')
 require(result.get('flag_dump_scope')=='PRIVATE_PREFLIGHT_ONLY'and result.get('flag_dump_poll_seconds')==.01 and type(result.get('timeout_seconds'))in(int,float)and 0<result['timeout_seconds']<=90,'FLAG_PREFLIGHT_BOUNDED_STAGE')
 terminal=result.get('flag_dump_terminal',{});require(all(type(terminal.get(k))is int and 0<=terminal[k]<=cap for k,cap in (('entries',1024),('total_bytes',134217728),('largest_file_bytes',16777216))),'FLAG_PREFLIGHT_BOUNDED_COUNTS')
 require(result.get('flag_dump_terminal',{}).get('status')=='WITHIN_OBSERVED_LIMITS'and result.get('flag_dump_terminal_after_group_cleanup')is True and type(result.get('flag_dump_observations'))is int and result['flag_dump_observations']>=0 and not result.get('flag_dump_terminal_error'),'FLAG_PREFLIGHT_BOUNDED_DUMP')
 return record
