"""Audit the declared transport record before unchanged scientific recovery."""
import ast,json
from pathlib import Path
from payload_transport import TRANSPORT_MODE,TRANSPORT_MEMBER,PACKET_BYTES,PACKET_SHA,WORK_RESERVE,FETCH_LIMIT,CLEANUP_LIMIT,audit_record,need,digest,packet_url

from build_download_candidate import TRANSPORT_HELPER_SHA,render_wrapper

def literal(source,name):
    matches=[n.value for n in ast.parse(source).body if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id==name for t in n.targets)]
    need(len(matches)==1,'TRANSPORT_LITERAL:'+name)
    return ast.literal_eval(matches[0])

def validate_candidate(candidate,binding):
    candidate=Path(candidate)
    plan=json.loads((candidate/'plan.json').read_text());mode=json.loads((candidate/'transport-plan.json').read_text())
    derived=json.loads((candidate/'derived-maps.json').read_text())
    keys={'mode','commit','url','packet_bytes','packet_sha256','deadline_epoch','binding','work_reserve_seconds',
        'fetch_limit_seconds','cleanup_limit_seconds','scientific_members','transport_member','wrapper_sha256',
        'helper_sha256','original_embedded_wrapper_sha256'}
    need(set(mode)==keys,'TRANSPORT_PLAN_SCHEMA')
    need(derived['transport_mode']==TRANSPORT_MODE and derived['transport_plan_sha256']==digest((candidate/'transport-plan.json').read_bytes()),'TRANSPORT_PLAN_BINDING')
    need(mode['mode']==TRANSPORT_MODE and mode['url']==packet_url(mode['commit']),'TRANSPORT_PLAN_MODE')
    need(mode['packet_bytes']==PACKET_BYTES and mode['packet_sha256']==PACKET_SHA==binding['packet_sha256'],'TRANSPORT_PLAN_PACKET')
    need(mode['work_reserve_seconds']==WORK_RESERVE and mode['fetch_limit_seconds']==FETCH_LIMIT
         and mode['cleanup_limit_seconds']==CLEANUP_LIMIT and mode['transport_member']==TRANSPORT_MEMBER,'TRANSPORT_PLAN_POLICY')
    need(mode['binding']==binding==plan['binding'] and mode['wrapper_sha256']==plan['wrapper_sha256'],'TRANSPORT_WRAPPER_BINDING')
    need(mode['deadline_epoch']==plan['deadline_epoch'],'TRANSPORT_PLAN_DEADLINE')
    source=(candidate/'submission/wrapper.py').read_bytes()
    need(digest(source)==plan['wrapper_sha256'],'TRANSPORT_WRAPPER_SOURCE')
    helper_path=Path(__file__).resolve().parent/'payload_transport.py'
    need(not helper_path.is_symlink() and helper_path.is_file(),'REVIEWED_TRANSPORT_HELPER')
    helper=helper_path.read_bytes()
    need(mode['helper_sha256']==TRANSPORT_HELPER_SHA,'TRANSPORT_HELPER_SOURCE')
    expected=render_wrapper(binding,mode['commit'],plan['deadline_epoch'],mode['scientific_members'],helper=helper)
    need(source==expected,'TRANSPORT_WRAPPER_BODY')
    original=(candidate/'original-embedded-wrapper.py').read_bytes()
    need(digest(original)==mode['original_embedded_wrapper_sha256'],'ORIGINAL_WRAPPER_BINDING')
    need(literal(original,'BINDING')==binding and literal(original,'DEADLINE')==plan['deadline_epoch'],'ORIGINAL_WRAPPER_POLICY')
    import base64
    need(base64.b64decode(literal(original,'PAYLOAD_B64'),validate=True)==(candidate/'payload.zip').read_bytes(),'EXACT_ORIGINAL_EMBEDDED_ZIP')
    return plan,mode

def audit(members,binding,candidate,*,allow_synthetic=False):
    plan,mode=validate_candidate(candidate,binding)
    original=set(mode['scientific_members'])
    need(len(original)==298 and len(mode['scientific_members'])==298 and TRANSPORT_MEMBER not in original,'ORIGINAL_298')
    need(set(members)==set(plan['expected_member_paths'])==original|{TRANSPORT_MEMBER},'EXACT_TRANSPORT_299')
    record=json.loads(members[TRANSPORT_MEMBER]);audit_record(record,mode['commit'],plan['deadline_epoch'],plan['wrapper_sha256'],allow_synthetic=allow_synthetic)
    outer=json.loads(members['records/batch.json']);need(record['parent_pid']==outer['pid'],'TRANSPORT_BATCH_PARENT')
    need(record['finished_epoch']<=outer['phases'][0]['start_epoch'],'TRANSPORT_BEFORE_SCIENCE')
    return {name:data for name,data in members.items() if name!=TRANSPORT_MEMBER}

def recover(members,binding,candidate,output,original_recover,*,allow_synthetic=False):
    scientific=audit(members,binding,candidate,allow_synthetic=allow_synthetic)
    # Only the original 298 scientific members enter the original verifier.
    return original_recover(scientific,binding,candidate,output,allow_synthetic=allow_synthetic)

if __name__=='__main__':
    import argparse,sys
    from build_download_candidate import validate_host
    p=argparse.ArgumentParser()
    for name in ['canonical-host','candidate','evidence','output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();host=validate_host(a.canonical_host);sys.path.insert(0,str(host))
    import batch,runner
    from recovery import recover as original_recover
    plan=batch.read(a.candidate/'plan.json');validate_candidate(a.candidate,plan['binding'])
    manifest=batch.read(a.evidence/'m8-evidence-manifest.json');data=(a.evidence/'m8-evidence.tar').read_bytes()
    members=runner.audit_archive(data,manifest,plan)
    print(json.dumps(recover(members,plan['binding'],a.candidate,a.output,original_recover),sort_keys=True))
