"""Real local OS topology controls; no native science or compiler qualification."""
import argparse
import copy
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent; NATIVE = HERE / 'compiler-native'
sys.path.insert(0, str(NATIVE)); sys.path.insert(0, str(NATIVE / 'ownership'))
import compiler_contract as C
import compiler_verify as V
from cleanup_lifecycle import Ownership
from lifecycle import durable_json

def run(output):
    assert sys.version_info[:3] == (3, 12, 14), 'USE_PINNED_PYTHON_3_12_14'
    output = Path(output).resolve(); output.mkdir(exist_ok=False)
    rows = []
    def process_case(name, mode, seconds=5, supervisor=False, force_parent_kill=False):
        root = output / name; root.mkdir(); actual = root / 'native'
        token = secrets.token_hex(16); deadline = time.time() + seconds
        argv = [sys.executable, '-B', str(HERE / 'fixtures/topology_parent.py'), '--output', str(actual),
                '--mode', mode, '--process-token', token, '--deadline-epoch', str(deadline),
                '--compiler-dumps','request','--compiler-policy-sha256','e'*64]
        if supervisor:
            argv = [sys.executable, '-B', '-c', 'import subprocess,sys;raise SystemExit(subprocess.call(sys.argv[1:]))'] + argv
        owner = Ownership(root)
        with (root / 'stdout.raw').open('wb') as stdout, (root / 'stderr.raw').open('wb') as stderr:
            with owner.guard(seconds + 3):
                child = owner.launch(argv, record=root / 'launch.json', cwd=root, stdout=stdout, stderr=stderr)
                if force_parent_kill:
                    until = time.monotonic() + 1
                    while not (actual/'descendant.json').exists() and time.monotonic() < until:
                        time.sleep(.005)
                    assert (actual/'descendant.json').exists(), 'FORCE_KILL_DESCENDANT_NOT_READY'
                    child.kill()
                child.wait(timeout=seconds + 2)
        launch = C.read(root / 'launch.json')
        assert owner.summary()['errors'] == [] and launch['cleanup']['status'] == 'CLEANUP_VERIFIED', owner.summary()
        assert launch['cleanup']['leader_reaped'] and launch['cleanup']['group_absence'] == 'OBSERVED_NO_SUCH_GROUP'
        return root, actual, launch

    root, actual, outer = process_case('correct-direct-same-pid-exec', 'correct')
    evidence = root / 'native.compiler-private'
    post = V.postclosure(evidence, outer); durable_json(evidence / 'postclosure.json', post)
    parent = C.read(actual / 'parent.json'); native_expected = C.read(actual / 'expected.json')
    receipt = C.read(actual / 'child.json')
    # Match the production coordinator's two fields added after run_child returns.
    parent['children'][0].update(receipt='device/receipt.json', receipt_sha256=C.sha(actual/'child.json'))
    sealed = C.sha(evidence / 'expected.json'); post_sha = C.sha(evidence / 'postclosure.json')
    report = V.capture_verified(evidence, sealed, post_sha, outer, parent, native_expected, receipt)
    assert report['status'] == 'OWNED_BOUNDED_DUMP_CAPTURE_ONLY' and report['selected_executable_link'] == 'UNKNOWN'
    assert receipt['file_limit_soft_hard'] == [8192, 8192]
    rows.append({'name': 'correct-direct-same-pid-exec-real-PPID-PGID-limit', 'status': 'PASS'})

    def incorrect(name, change, reason, reseal_post=False):
        dst = output / name; shutil.copytree(evidence, dst)
        values = [copy.deepcopy(outer), copy.deepcopy(parent), copy.deepcopy(native_expected), copy.deepcopy(receipt)]
        change(dst, *values)
        if reseal_post:
            post_record = C.read(dst / 'postclosure.json'); post_record['evidence_files'] = V.evidence_files(dst)
            durable_json(dst / 'postclosure.json', post_record)
        try:
            V.capture_verified(dst, sealed, C.sha(dst / 'postclosure.json') if reseal_post else post_sha, *values)
        except ValueError as e:
            assert str(e) == reason, (name, str(e), reason)
            rows.append({'name': name, 'status': 'PASS', 'rejected': reason})
        else:
            raise AssertionError('FALSE_ACCEPT:' + name)

    incorrect('forged-supervisor-as-parent', lambda d,o,p,n,r:r.__setitem__('parent_pid', o['pid'] + 1),
              'COMPILER_SAME_PID_DIRECT_EXEC_CHAIN')
    incorrect('changed-post-exec-PID', lambda d,o,p,n,r:r.__setitem__('pid', r['pid'] + 1),
              'COMPILER_SAME_PID_DIRECT_EXEC_CHAIN')
    incorrect('detached-post-exec-group', lambda d,o,p,n,r:r.__setitem__('pgid', r['pid']),
              'COMPILER_SAME_PID_DIRECT_EXEC_CHAIN')
    incorrect('unclosed-outer-group', lambda d,o,p,n,r:o['cleanup'].__setitem__('group_absence','UNVERIFIED'),
              'COMPILER_EXTERNAL_GROUP_CLOSURE')
    incorrect('altered-dump-after-seal', lambda d,o,p,n,r:(d/'raw-private/one.hlo.pb').write_bytes(b'altered'),
              'COMPILER_POSTCLOSURE_DUMP_CONTENT')
    incorrect('late-dump-after-seal', lambda d,o,p,n,r:(d/'raw-private/late').write_bytes(b'late'),
              'COMPILER_POSTCLOSURE_DUMP_CONTENT')
    def late_resealed(d,o,p,n,r):
        (d/'raw-private/late').write_bytes(b'late')
        post_record = V.postclosure(d,o); durable_json(d/'postclosure.json',post_record)
    incorrect('late-dump-with-new-postclosure-seal', late_resealed, 'COMPILER_LATE_DUMP_CHANGE', True)
    def monitor_edit(key,value):
        def edit(d,o,p,n,r):
            m=C.read(d/'monitor.json');m[key]=value;durable_json(d/'monitor.json',m)
        return edit
    incorrect('outer-wrong-policy-admission',lambda d,o,p,n,r:o['argv'].__setitem__(o['argv'].index('--compiler-policy-sha256')+1,'f'*64),'COMPILER_OUTER_POLICY_ADMISSION')
    incorrect('full-source-final-audit-failed',lambda d,o,p,n,r:p.__setitem__('compiler_source_post',None),'COMPILER_FULL_PRE_POST_SOURCE')
    incorrect('monitor-forged-hard-quota',monitor_edit('hard_aggregate_quota','HARD'),
              'COMPILER_AGGREGATE_LIMIT_TRUTH',True)
    incorrect('monitor-zero-observations',monitor_edit('observations',0),'COMPILER_POSITIVE_OBSERVATION',True)
    incorrect('monitor-overflow-forged-complete',monitor_edit('overflow_observation',{'status':'DUMP_TOTAL_SIZE_LIMIT'}),
              'COMPILER_UNTRUNCATED_CHILD_TERMINAL',True)
    incorrect('monitor-final-source-altered',monitor_edit('source_post',{}),'COMPILER_PRE_POST_SOURCE_SEALS',True)
    incorrect('monitor-forged-deadline',monitor_edit('deadline_epoch',0),'COMPILER_FILE_LIMIT_READBACK',True)
    incorrect('monitor-forged-negative-snapshot',monitor_edit('last_observation',{'entries':0,'total_bytes':-1,'largest_file_bytes':0,'status':'WITHIN_OBSERVED_LIMITS'}),'COMPILER_RECOMPUTED_OBSERVATION',True)
    incorrect('monitor-after-deadline',lambda d,o,p,n,r:p['children'][0].__setitem__('deadline_epoch',0),
              'COMPILER_EXACT_NATIVE_CHILD_DESCRIPTOR')
    incorrect('changed-prelaunch-record',lambda d,o,p,n,r:(d/'expected.json').write_text('{}\n'),
              'COMPILER_INDEPENDENT_EXPECTED_HASH')
    incorrect('changed-monitor-after-seal',monitor_edit('observations',1),'COMPILER_CLOSED_RECORD_INVENTORY')

    root2, actual2, outer2 = process_case('real-supervisor-parent-rejected', 'correct', supervisor=True)
    parent2 = C.read(actual2/'parent.json')
    assert parent2['status']=='ERROR' and parent2['error']['message']=='COMPILER_DIRECT_PARENT_LEADER', parent2
    assert parent2['children']==[]
    rows.append({'name':'real-supervisor-parent-rejected-before-launch','status':'PASS'})
    rk, ak, ok = process_case('outer-killed-coordinator-group-closure','ignored-term-descendant',5,force_parent_kill=True)
    pk=C.read(ak/'parent.json'); dk=C.read(ak/'descendant.json')
    assert dk['pgid']==ok['pid'] and pk['children'][0]['pgid']==ok['pid']
    assert pk['children'][0]['reaped'] is False and ok['cleanup']['leader_reaped'] is True
    assert ok['exit_status']==-signal.SIGKILL and ok['cleanup']['group_absence']=='OBSERVED_NO_SUCH_GROUP'
    rows.append({'name':'external-SIGKILL-coordinator-no-finally-all-inherited-descendants-closed','status':'PASS'})
    for name,mode,seconds,reason in [('deadline','timeout',1.1,'STATE_CHILD_DEADLINE_EXPIRED'),
                                    ('total-overflow','total-overflow',5,'DUMP_TOTAL_SIZE_LIMIT'),
                                    ('count-overflow','count-overflow',5,'DUMP_ENTRY_LIMIT'),
                                    ('file-limit','file-overflow',5,'CHILD_FAILED_OR_FILE_LIMIT'),
                                    ('ignored-term-external-kill','ignored-term-descendant',1.5,'STATE_CHILD_DEADLINE_EXPIRED')]:
        rr, aa, oo=process_case(name,mode,seconds)
        mm=C.read(rr/'native.compiler-private/monitor.json'); pp=C.read(aa/'parent.json')
        assert pp['status']=='ERROR' and mm['status']=='BLOCKED' and mm['failure_reason']==reason,(name,mm)
        assert pp['children'][0]['reaped'] and pp['children'][0]['child_absent']
        if mode in ('timeout','ignored-term-descendant','total-overflow','count-overflow'):
            assert mm['receipt_before_term'] is True
            assert mm['shutdown_requested_epoch']<=mm['finished_epoch']
        if mode=='file-overflow':
            assert (rr/'native.compiler-private/raw-private/large').stat().st_size<=8192
        if mode=='ignored-term-descendant':
            descendant=C.read(aa/'descendant.json')
            assert descendant['pgid']==oo['pid']
            assert any(x['signal']==signal.SIGKILL and x['result']=='SENT' for x in oo['cleanup']['signals'])
        rows.append({'name':name+'-blocked-and-external-group-closed','status':'PASS','failure_reason':reason})
    result={'status':'PASS','scope':'REAL_LOCAL_OS_PROCESSES_NO_COMPILER_OR_NATIVE_SCIENCE',
            'count':len(rows),'controls':rows,'selected_executable_link':'UNKNOWN','allocator_peak':'UNKNOWN'}
    durable_json(output/'results.json',result);print(json.dumps({'status':'PASS','count':len(rows)}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();run(a.output)
