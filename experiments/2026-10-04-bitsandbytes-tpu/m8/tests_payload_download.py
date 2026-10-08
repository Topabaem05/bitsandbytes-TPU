"""Offline exact transport and original scientific recovery controls."""
import argparse,ast,copy,hashlib,importlib.util,io,json,os,socket,ssl,subprocess,sys,tarfile,time,unittest,zipfile
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
P=argparse.ArgumentParser();P.add_argument('--canonical-host',type=Path,required=True)
P.add_argument('--project-source',type=Path,required=True);P.add_argument('--base-archive',type=Path,required=True)
P.add_argument('--output',type=Path,required=True);A=P.parse_args();A.output=A.output.resolve();A.output.mkdir(parents=True,exist_ok=False)
HERE=Path(__file__).resolve().parent;HOST=A.canonical_host.resolve();ROOT=A.project_source.resolve()
sys.path[:0]=[str(HERE),str(HOST)];attempts={'network':0,'credentials':0}
def guard(event,args):
    if event in ('socket.connect','socket.getaddrinfo'):
        attempts['network']+=1;raise AssertionError('NETWORK_FORBIDDEN')
    if event=='open' and isinstance(args[0],(str,bytes)) and any(s in os.fsdecode(args[0]) for s in ('/.kaggle/','access_token','credentials.json')):
        attempts['credentials']+=1;raise AssertionError('CREDENTIAL_READ_FORBIDDEN')
sys.addaudithook(guard)
import payload_transport as T
import build_download_candidate as C
import recover_download as V
import batch as B
import shutil
from process_provider import Outcome
import runner as R
import recovery as ORIGINAL

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

deadline=time.time()+3600;commit='a'*40
plan=C.build(A.output/'candidate','fixture-owner','offline-public-zip','Offline Public Zip','e'*32,deadline,
    canonical_host=HOST,project_source=ROOT,base_archive=A.base_archive,commit=commit)
packet=(A.output/'candidate/payload.zip').read_bytes();wrapper=load(A.output/'candidate/submission/wrapper.py','offline_download_wrapper')
mode=json.loads((A.output/'candidate/transport-plan.json').read_text())

class Response:
    def __init__(self,data,*,status=200,url=None,headers=None):
        self.stream=io.BytesIO(data);self.status=status;self.url=url or T.packet_url(commit);self.headers=headers or {};self.closed=False
    def read(self,n):return self.stream.read(n)
    def geturl(self):return self.url
    def close(self):self.closed=True
class Opener:
    def __init__(self,response):self.response=response;self.calls=[]
    def open(self,request,timeout):self.calls.append((request,timeout));return self.response

fixture_path=ROOT/'tests/test_transfer_probe.py'
namespace={'__file__':str(fixture_path),'__name__':'original_transport_science_fixture'}
exec(compile(fixture_path.read_text().split('class TransferControls')[0],str(fixture_path),'exec'),namespace)
make_science=namespace['make_fixture']
node=next(n for n in ast.parse((ROOT/'experiments/2026-10-04-bitsandbytes-tpu/cloud/tests/test_transfer_cloud.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='controls_fixture')
ns={'B':SimpleNamespace(TRANSFER_PATCH_MANIFEST_SHA='e745fbf21aac10ed9118167a131d6505bf6dbe03e1fab0a669c663b26c5a5732')}
exec(compile(ast.Module(body=[node],type_ignores=[]),'original_source_fixture','exec'),ns);source_controls=ns['controls_fixture']
fixtures={'B':B,'sys':sys,'Path':Path,'datetime':datetime,'timezone':timezone,'os':os,'json':json,
          'zipfile':zipfile,'shutil':__import__('shutil'),'make_science':make_science,'source_controls':source_controls,'PLAN_DEADLINE':deadline,'time':time,'Outcome':Outcome,'hashlib':hashlib}
nodes=[n for n in ast.parse((HOST/'tests.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name in ['BatchFixture','ProviderFixture']]
exec(compile(ast.Module(body=nodes,type_ignores=[]),'original_batch_fixture','exec'),fixtures)
BatchFixture=fixtures['BatchFixture'];ProviderFixture=fixtures['ProviderFixture']

class Controls(unittest.TestCase):
    def directory(self):d=A.output/self._testMethodName;d.mkdir();return d
    def test_exact_https_body_and_bounded_metadata(self):
        response=Response(packet,headers={'Content-Length':str(T.PACKET_BYTES)});opener=Opener(response)
        data,http=T.read_https(commit,time.time()+30,opener=opener)
        self.assertEqual(data,packet);self.assertEqual(http,{'status':200,'bytes':T.PACKET_BYTES,'sha256':T.PACKET_SHA})
        self.assertTrue(response.closed);self.assertEqual(len(opener.calls),1)
        request,limit=opener.calls[0];self.assertEqual(request.get_method(),'GET');self.assertEqual(request.full_url,T.packet_url(commit))
        self.assertLessEqual(limit,10);self.assertNotIn('Authorization',request.headers)
    def invalid_body(self,data,code):
        response=Response(data)
        with self.assertRaisesRegex(ValueError,code):T.read_https(commit,time.time()+30,opener=Opener(response))
        self.assertTrue(response.closed)
    def test_truncated_stream_rejected(self):self.invalid_body(packet[:-1],'PAYLOAD_TRUNCATED')
    def test_extra_stream_rejected(self):self.invalid_body(packet+b'x','PAYLOAD_STREAM_CAP')
    def test_wrong_hash_rejected(self):self.invalid_body(bytes([packet[0]^1])+packet[1:],'PAYLOAD_SHA')
    def test_malformed_stream_rejected(self):
        response=Response(packet);response.read=lambda _:None
        with self.assertRaisesRegex(ValueError,'PAYLOAD_STREAM_TYPE'):T.read_https(commit,time.time()+30,opener=Opener(response))
        self.assertTrue(response.closed)
    def test_http_status_and_changed_url_rejected(self):
        for response,code in [(Response(packet,status=302),'PAYLOAD_HTTP_STATUS'),(Response(packet,url='https://evil.invalid/'),'PAYLOAD_URL_CHANGED')]:
            with self.assertRaisesRegex(ValueError,code):T.read_https(commit,time.time()+30,opener=Opener(response))
            self.assertTrue(response.closed)
    def test_wrong_header_rejected(self):
        for headers,code in [({'Content-Length':'983677'},'PAYLOAD_CONTENT_LENGTH'),({'Content-Encoding':'gzip'},'PAYLOAD_CONTENT_ENCODING')]:
            response=Response(packet,headers=headers)
            with self.assertRaisesRegex(ValueError,code):T.read_https(commit,time.time()+30,opener=Opener(response))
            self.assertTrue(response.closed)
    def test_tls_weakening_rejected(self):
        context=ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT);context.check_hostname=False;context.verify_mode=ssl.CERT_NONE
        with patch.object(T.ssl,'create_default_context',return_value=context):
            with self.assertRaisesRegex(ValueError,'TLS_REQUIRED'):T.https_opener()
    def test_redirect_rejected(self):
        with self.assertRaisesRegex(ValueError,'PAYLOAD_REDIRECT_REJECTED'):
            T.RejectRedirect().redirect_request(None,None,307,None,None,'https://evil.invalid')
    def test_arbitrary_reference_rejected(self):
        for value in ['main','A'*40,'a'*39,'a'*40+'/extra','https://evil.invalid/']:
            with self.assertRaisesRegex(ValueError,'FULL_COMMIT_REQUIRED'):T.packet_url(value)
    def setup_child(self,d,mode='good'):
        (d/'fixture.json').write_text(json.dumps({'mode':mode,'packet':str((A.output/'candidate/payload.zip').resolve())}))
        return [sys.executable,'-B',str(HERE/'fake_fetch_child.py')]
    def test_owned_child_success_and_exact_reap(self):
        d=self.directory();data,record=T.fetch_packet(d/'fetch',commit,deadline,worker_command=self.setup_child(d))
        self.assertEqual(data,packet);self.assertTrue(record['launch']['reaped']);self.assertTrue(record['launch']['group_absent'])
        T.audit_record(record,commit,deadline,T.digest((HERE/'payload_transport.py').read_bytes()),allow_synthetic=True)
        self.assertEqual(record['provider_closure'],'UNCONFIRMED')
    def test_owned_timeout_closes_ignored_term(self):
        d=self.directory();narrow=time.time()+T.WORK_RESERVE+T.CLEANUP_LIMIT+0.3
        with self.assertRaises(TimeoutError):T.fetch_packet(d/'fetch',commit,narrow,worker_command=self.setup_child(d,'timeout'))
        record=json.loads((d/'fetch/receipt.json').read_text())
        self.assertEqual(record['status'],'FAIL');self.assertTrue(record['launch']['reaped']);self.assertTrue(record['launch']['group_absent'])
        self.assertLessEqual(record['finished_epoch'],narrow-T.WORK_RESERVE)
    def test_timeout_descendant_group_closes(self):
        d=self.directory();narrow=time.time()+T.WORK_RESERVE+T.CLEANUP_LIMIT+0.3
        with self.assertRaises(TimeoutError):T.fetch_packet(d/'fetch',commit,narrow,worker_command=self.setup_child(d,'descendant'))
        record=json.loads((d/'fetch/receipt.json').read_text());self.assertTrue(record['launch']['reaped']);self.assertTrue(record['launch']['group_absent'])
    def test_no_new_deadline_when_budget_empty(self):
        d=self.directory()
        with self.assertRaisesRegex(ValueError,'NOT_RUN_FETCH_BUDGET'):T.fetch_packet(d/'fetch',commit,time.time()+T.WORK_RESERVE,worker_command=self.setup_child(d))
        self.assertFalse((d/'fetch/packet.zip').exists())
    def test_no_science_before_invalid_admission(self):
        d=self.directory();base=d/'base';base.mkdir();real_fetch=wrapper.fetch_packet
        def fake_fetch(directory,*a,**k):
            self.setup_child(Path(directory).parent,'hash')
            return real_fetch(directory,*a,**k)
        with patch.object(wrapper,'fetch_packet',side_effect=fake_fetch):
            with self.assertRaisesRegex(ValueError,'FETCH_FILE_SHA'):
                wrapper.entry(provider_factory=lambda *_:self.fail('science started'),workspace=base,output=d/'output',fetch_child_command=[sys.executable,'-B',str(HERE/'fake_fetch_child.py')])
        self.assertFalse((base/'payload').exists())
    def complete(self,d,fault=None):
        base=d/'base';base.mkdir();command=self.setup_child(base)
        # The original entry requires an empty workspace. Move the fixture descriptor
        # immediately before the download phase through a bounded test-only seam.
        (base/'fixture.json').unlink();saved={};real_fetch=wrapper.fetch_packet;real_append=wrapper.append_transport
        def fake_fetch(directory,*a,**k):
            (Path(directory).parent/'fixture.json').write_text(json.dumps({'mode':'good','packet':str((A.output/'candidate/payload.zip').resolve())}))
            return real_fetch(directory,*a,**k)
        def observe(output,binding,record,names):
            saved['original_archive']=(Path(output)/'m8-evidence.tar').read_bytes()
            real_append(output,binding,record,names)
        with patch.object(wrapper,'fetch_packet',side_effect=fake_fetch),patch.object(wrapper,'append_transport',side_effect=observe):
            result=wrapper.entry(provider_factory=lambda b,m:BatchFixture(b,m,fault),workspace=base,output=d/'output',fetch_child_command=command)
        self.assertEqual(result['status'],'COMPLETE')
        manifest=json.loads((d/'output/m8-evidence-manifest.json').read_text());data=(d/'output/m8-evidence.tar').read_bytes()
        members=R.audit_archive(data,manifest,plan)
        before={}
        with tarfile.open(fileobj=io.BytesIO(saved['original_archive']),mode='r:') as archive:
            for member in archive:before[member.name]=archive.extractfile(member).read()
        self.assertEqual(len(before),298);self.assertEqual(set(members),set(before)|{T.TRANSPORT_MEMBER})
        self.assertTrue(all(members[name]==raw for name,raw in before.items()))
        return members
    def test_full299_recovery_preserves_original298_and196gates(self):
        d=self.directory();members=self.complete(d)
        result=V.recover(members,plan['binding'],A.output/'candidate',d/'recovered',ORIGINAL.recover,allow_synthetic=True)
        self.assertEqual(result['record_validation'],'PASS');self.assertEqual(result['numerical_status'],'PASS')
        report=json.loads((d/'recovered/recovery.json').read_text())
        self.assertEqual((report['case_count'],report['comparison_count']),(46,196))
        self.assertEqual(report['resource_closure'],'ROOT_OBSERVATION_REQUIRED')
    def test_false_and_missing_transport_closure_rejected(self):
        d=self.directory();members=self.complete(d)
        for key in ['reaped','group_absent']:
            bad=dict(members);record=json.loads(bad[T.TRANSPORT_MEMBER]);record['launch'][key]=False;bad[T.TRANSPORT_MEMBER]=json.dumps(record).encode()
            with self.assertRaisesRegex(ValueError,'TRANSPORT_CLOSURE'):V.audit(bad,plan['binding'],A.output/'candidate',allow_synthetic=True)
        bad=dict(members);record=json.loads(bad[T.TRANSPORT_MEMBER]);record['launch'].pop('reaped');bad[T.TRANSPORT_MEMBER]=json.dumps(record).encode()
        with self.assertRaisesRegex(ValueError,'TRANSPORT_LAUNCH_SCHEMA'):V.audit(bad,plan['binding'],A.output/'candidate',allow_synthetic=True)
    def test_missing_unknown_or_duplicate_members_rejected(self):
        d=self.directory();members=self.complete(d)
        missing=dict(members);missing.pop(T.TRANSPORT_MEMBER)
        with self.assertRaisesRegex(ValueError,'EXACT_TRANSPORT_299'):V.audit(missing,plan['binding'],A.output/'candidate',allow_synthetic=True)
        extra=dict(members);extra['records/unknown.json']=b'{}'
        with self.assertRaisesRegex(ValueError,'EXACT_TRANSPORT_299'):V.audit(extra,plan['binding'],A.output/'candidate',allow_synthetic=True)
        buffer=io.BytesIO();manifest={'format':'m8-evidence-v1','binding':plan['binding'],'members':{n:{'bytes':len(b),'sha256':T.digest(b)} for n,b in members.items()}}
        with tarfile.open(fileobj=buffer,mode='w') as archive:
            for n,b in [*members.items(),(T.TRANSPORT_MEMBER,members[T.TRANSPORT_MEMBER])]:
                member=tarfile.TarInfo(n);member.size=len(b);archive.addfile(member,io.BytesIO(b))
        data=buffer.getvalue();manifest['archive']={'file':'m8-evidence.tar','bytes':len(data),'sha256':T.digest(data)}
        with self.assertRaisesRegex(ValueError,'ARCHIVE_MEMBER'):R.audit_archive(data,manifest,plan)

    def test_original_request_exception_survives_close_failure(self):
        error=TimeoutError('FIXTURE_TIMEOUT');response=Response(packet)
        def read(_):raise error
        def close():raise OSError('FIXTURE_CLOSE')
        response.read=read;response.close=close
        try:T.read_https(commit,time.time()+30,opener=Opener(response))
        except TimeoutError as caught:self.assertIs(caught,error)
        else:self.fail('original exception missing')
        self.assertIn('PAYLOAD_RESPONSE_CLOSE_FAILURE:OSError',error.__notes__)
    def test_original_http_error_closed_without_body_read(self):
        import urllib.error
        response=Response(b'not read');response.read=lambda _:self.fail('error body read')
        error=urllib.error.HTTPError(T.packet_url(commit),400,'PRIVATE',{},response)
        opener=Opener(None)
        def open(*a,**k):raise error
        opener.open=open
        try:T.read_https(commit,time.time()+30,opener=opener)
        except urllib.error.HTTPError as caught:self.assertIs(caught,error)
        else:self.fail('original error missing')
        self.assertTrue(response.closed)
    def test_response_close_failure_cannot_succeed(self):
        response=Response(packet);response.close=lambda:(_ for _ in ()).throw(OSError('FIXTURE_CLOSE'))
        with self.assertRaises(OSError):T.read_https(commit,time.time()+30,opener=Opener(response))
    def test_source_post_audit_rejects_changed_wrapper(self):
        d=self.directory();original=Path.read_bytes;calls=0
        def read(path):
            nonlocal calls
            data=original(path)
            if path==HERE/'payload_transport.py':
                calls+=1
                if calls>=2:return data+b'changed'
            return data
        with patch.object(Path,'read_bytes',read):
            with self.assertRaisesRegex(ValueError,'FETCH_POST_SOURCE_CHANGED'):
                T.fetch_packet(d/'fetch',commit,deadline,worker_command=self.setup_child(d))
        record=json.loads((d/'fetch/receipt.json').read_text());self.assertEqual(record['status'],'FAIL')
        self.assertTrue(record['launch']['reaped']);self.assertTrue(record['launch']['group_absent'])
    def test_source_post_audit_exception_cannot_pass(self):
        d=self.directory();original=Path.read_bytes;calls=0
        def read(path):
            nonlocal calls
            if path==HERE/'payload_transport.py':
                calls+=1
                if calls>=2:raise OSError('FIXTURE_SOURCE_READ')
            return original(path)
        with patch.object(Path,'read_bytes',read):
            with self.assertRaises(OSError):T.fetch_packet(d/'fetch',commit,deadline,worker_command=self.setup_child(d))
        record=json.loads((d/'fetch/receipt.json').read_text());self.assertEqual(record['status'],'FAIL');self.assertIsNone(record['source_post'])
    def test_numerical_failure_preserves_original_196_gates(self):
        d=self.directory();members=self.complete(d,'numeric')
        result=V.recover(members,plan['binding'],A.output/'candidate',d/'recovered',ORIGINAL.recover,allow_synthetic=True)
        self.assertEqual((result['record_validation'],result['numerical_status']),('PASS','FAIL'))
        report=json.loads((d/'recovered/recovery.json').read_text());self.assertEqual((report['case_count'],report['comparison_count']),(46,196))
        self.assertEqual(report['m8_status'],'NOT_QUALIFIED')
    def test_transport_source_and_deadline_and_provider_claim_rejected(self):
        d=self.directory();members=self.complete(d)
        for key,value,code in [('source_post','b'*64,'TRANSPORT_SOURCE'),('url','https://evil.invalid/','TRANSPORT_REF'),
            ('deadline_epoch',deadline+1,'TRANSPORT_BUDGET'),('provider_closure','CLOSED','NO_PROVIDER_CLOSURE_CLAIM'),
            ('body_admitted_epoch',deadline,'TRANSPORT_ADMISSION_BOUND')]:
            bad=dict(members);record=json.loads(bad[T.TRANSPORT_MEMBER]);record[key]=value;bad[T.TRANSPORT_MEMBER]=json.dumps(record).encode()
            with self.assertRaisesRegex(ValueError,code):V.audit(bad,plan['binding'],A.output/'candidate',allow_synthetic=True)
        with self.assertRaisesRegex(ValueError,'TRANSPORT_SYNTHETIC'):V.audit(members,plan['binding'],A.output/'candidate')
    def test_resealed_wrong_commit_plan_rejected_by_wrapper_ast(self):
        d=self.directory();candidate=d/'candidate';shutil.copytree(A.output/'candidate',candidate)
        altered=json.loads((candidate/'transport-plan.json').read_text());altered.update(commit='b'*40,url=T.packet_url('b'*40))
        B.write(candidate/'transport-plan.json',altered);derived=B.read(candidate/'derived-maps.json')
        derived['transport_plan_sha256']=B.sha(candidate/'transport-plan.json');B.write(candidate/'derived-maps.json',derived)
        with self.assertRaisesRegex(ValueError,'TRANSPORT_WRAPPER_BODY'):V.validate_candidate(candidate,plan['binding'])
    def test_changed_missing_symlink_canonical_host_rejected(self):
        d=self.directory();host=d/'host';shutil.copytree(HOST,host)
        target=host/'sdk_worker.py';saved=target.read_bytes();target.write_bytes(saved+b'changed')
        with self.assertRaisesRegex(ValueError,'CANONICAL_HOST_SOURCE_CHANGED:sdk_worker.py'):C.validate_host(host)
        target.unlink()
        with self.assertRaisesRegex(ValueError,'CANONICAL_HOST_SOURCE_CHANGED:sdk_worker.py'):C.validate_host(host)
        target.symlink_to(HOST/'sdk_worker.py')
        with self.assertRaisesRegex(ValueError,'CANONICAL_HOST_SOURCE_CHANGED:sdk_worker.py'):C.validate_host(host)
    def test_final_tar_byte_cap_prevents_export_success(self):
        d=self.directory();members=self.complete(d);out=d/'original-output'
        # Remove the declared extra member by re-exporting the unchanged original records.
        out.mkdir();B.export(d/'base/records',out,plan['binding']);original=(out/'m8-evidence.tar').read_bytes()
        record=json.loads(members[T.TRANSPORT_MEMBER])
        with patch.object(T,'MAX_EXPORT_BYTES',len(original)-1):
            with self.assertRaisesRegex(ValueError,'FINAL_TRANSPORT_TAR_BYTES'):T.append_transport(out,plan['binding'],record,mode['scientific_members'])
        self.assertEqual((out/'m8-evidence.tar').read_bytes(),original)
    def test_full_owner_exact_version_and_transport_recovery(self):
        d=self.directory();self.complete(d);files={p.name:p.read_bytes() for p in (d/'output').iterdir()}
        provider=ProviderFixture(d/'provider',plan,files)
        callback=lambda members,binding:V.recover(members,binding,A.output/'candidate',d/'recovered',ORIGINAL.recover,allow_synthetic=True)
        owner=ORIGINAL.PartialM2Runner(plan,provider,d/'owner',callback,sleep=lambda _:None);result=owner.run()
        self.assertEqual((result['record_validation'],result['numerical_status']),('PASS','PASS'))
        self.assertEqual(provider.calls.count('submit'),1);self.assertEqual(result['identity']['version'],1)
        self.assertEqual(result['closure'],'UNCONFIRMED');self.assertEqual(result['m8_status'],'NOT_QUALIFIED')
        report=json.loads((d/'recovered/recovery.json').read_text());self.assertEqual((report['case_count'],report['comparison_count']),(46,196))

    def test_explicit_owner_uses_original_supervisor_and_sdk(self):
        import supervisor
        entry=load(HERE/'owner_download.py','offline_explicit_download_owner')
        d=self.directory();calls=[]
        def supervise(owner,worker,root,until):
            calls.append((owner,worker,root,until));return {'status':'OWNER_FINISHED'}
        argv=['owner_download.py','execute','--live-provider','--canonical-host',str(HOST),
            '--candidate',str(A.output/'candidate'),'--output',str(d/'owner'),
            '--kaggle-python',str(Path(sys.executable))]
        with patch.object(sys,'argv',argv),patch.object(supervisor,'supervise',side_effect=supervise):
            self.assertEqual(entry.main(),2)
        self.assertEqual(len(calls),1)
        owner,worker,root,until=calls[0]
        self.assertEqual(owner[:5],[sys.executable,'-B',str(HERE/'owner_download.py'),'_worker','--canonical-host'])
        self.assertEqual(worker,[str(Path(sys.executable)),'-B',str(HOST/'sdk_worker.py'),'--live-provider'])
        self.assertEqual(until,deadline)

    def test_complete_renderer_is_deterministic(self):
        helper=(HERE/'payload_transport.py').read_bytes()
        first=C.render_wrapper(plan['binding'],commit,deadline,mode['scientific_members'],helper=helper)
        reversed_binding=dict(reversed(list(plan['binding'].items())))
        self.assertEqual(first,C.render_wrapper(reversed_binding,commit,deadline,mode['scientific_members'],helper=helper))
        self.assertEqual(first,(A.output/'candidate/submission/wrapper.py').read_bytes())
    def test_renderer_rejects_unreviewed_template_helper_and_names(self):
        helper=(HERE/'payload_transport.py').read_bytes()
        for kw,code in [({'template':C.ADMITTED_TEMPLATE+b'changed'},'REVIEWED_CANONICAL_TEMPLATE'),
            ({'helper':helper+b'changed'},'REVIEWED_TRANSPORT_HELPER')]:
            args={'helper':helper,**kw}
            with self.assertRaisesRegex(ValueError,code):C.render_wrapper(plan['binding'],commit,deadline,mode['scientific_members'],**args)
        names=mode['scientific_members'][:];names[0]='records/unknown.json'
        with self.assertRaisesRegex(ValueError,'REVIEWED_SCIENTIFIC_NAMES'):C.render_wrapper(plan['binding'],commit,deadline,names,helper=helper)
    def test_resealed_body_cannot_reach_supervisor(self):
        import supervisor
        d=self.directory();candidate=d/'candidate';shutil.copytree(A.output/'candidate',candidate)
        path=candidate/'submission/wrapper.py';source=path.read_text()
        line="    data,transport_record=fetch_packet(base/'payload-transport',PAYLOAD_COMMIT,DEADLINE,worker_command=fetch_child_command)"
        self.assertEqual(source.count(line),1)
        path.write_text(source.replace(line,"    raise RuntimeError('UNEXECUTED_OFFLINE_BODY_DRIFT')"))
        sha=B.sha(path);altered=B.read(candidate/'plan.json');altered.update(wrapper_sha256=sha,folder=str(candidate/'submission'));B.write(candidate/'plan.json',altered)
        altered_mode=B.read(candidate/'transport-plan.json');altered_mode['wrapper_sha256']=sha;B.write(candidate/'transport-plan.json',altered_mode)
        derived=B.read(candidate/'derived-maps.json');derived.update(wrapper_sha256=sha,transport_plan_sha256=B.sha(candidate/'transport-plan.json'));B.write(candidate/'derived-maps.json',derived)
        with self.assertRaisesRegex(ValueError,'TRANSPORT_WRAPPER_BODY'):V.validate_candidate(candidate,plan['binding'])
        entry=load(HERE/'owner_download.py','offline_body_admission_owner')
        argv=['owner_download.py','execute','--live-provider','--canonical-host',str(HOST),'--candidate',str(candidate),
              '--output',str(d/'owner'),'--kaggle-python',sys.executable]
        with patch.object(sys,'argv',argv),patch.object(supervisor,'supervise') as supervise:
            with self.assertRaisesRegex(ValueError,'TRANSPORT_WRAPPER_BODY'):entry.main()
            supervise.assert_not_called()
        # The altered wrapper is neither imported nor executed.
    def fake_process(self,*,timeout=False):
        def communicate(**_):
            if timeout:raise subprocess.TimeoutExpired('offline_fake_no_process',1)
            return b'',b''
        return SimpleNamespace(pid=424242,returncode=None if timeout else 0,poll=lambda:None if timeout else 0,communicate=communicate)
    def test_cleanup_signal_permission_preserves_timeout_and_final_receipt(self):
        d=self.directory();process=self.fake_process(timeout=True)
        with patch.object(T.subprocess,'Popen',return_value=process),patch.object(T.os,'killpg',side_effect=PermissionError('OFFLINE_EPERM')):
            with self.assertRaisesRegex(TimeoutError,'PAYLOAD_FETCH_TIMEOUT'):T.fetch_packet(d/'fetch',commit,deadline,worker_command=['OFFLINE_NO_PROCESS'])
        receipt=B.read(d/'fetch/receipt.json');self.assertEqual(receipt['status'],'FAIL');self.assertEqual(receipt['error_type'],'TimeoutError')
        self.assertIsNotNone(receipt['source_post']);self.assertIsNotNone(receipt['finished_epoch']);self.assertFalse(receipt['launch']['group_absent'])
        self.assertIn('PermissionError',[r['type'] for r in receipt['cleanup_errors']])
    def test_cleanup_probe_permission_preserves_read_exception_identity(self):
        d=self.directory();process=self.fake_process();error=FileNotFoundError('ORIGINAL_OFFLINE_CHILD_READ');original=Path.read_text
        def read(path,*a,**kw):
            if path.name=='child.json':raise error
            return original(path,*a,**kw)
        with patch.object(T.subprocess,'Popen',return_value=process),patch.object(T.os,'killpg',side_effect=PermissionError('OFFLINE_EPERM')),patch.object(Path,'read_text',read):
            try:T.fetch_packet(d/'fetch',commit,deadline,worker_command=['OFFLINE_NO_PROCESS'])
            except FileNotFoundError as caught:self.assertIs(caught,error)
            else:self.fail('original exception missing')
        receipt=B.read(d/'fetch/receipt.json');self.assertEqual(receipt['status'],'FAIL');self.assertEqual(receipt['error_type'],'FileNotFoundError')
        self.assertIsNotNone(receipt['finished_epoch']);self.assertFalse(receipt['launch']['group_absent']);self.assertTrue(receipt['cleanup_errors'])
    def test_cleanup_internal_exception_retains_failure_receipt(self):
        d=self.directory();process=self.fake_process()
        with patch.object(T.subprocess,'Popen',return_value=process),patch.object(T,'close_owned',side_effect=PermissionError('OFFLINE_INTERNAL')):
            with self.assertRaises(FileNotFoundError):T.fetch_packet(d/'fetch',commit,deadline,worker_command=['OFFLINE_NO_PROCESS'])
        receipt=B.read(d/'fetch/receipt.json');self.assertEqual(receipt['status'],'FAIL');self.assertEqual(receipt['cleanup_errors'],[{'stage':'cleanup_internal','type':'PermissionError'}])
        self.assertFalse(receipt['launch']['reaped']);self.assertFalse(receipt['launch']['group_absent']);self.assertIsNotNone(receipt['source_post'])
    def test_cleanup_uncertainty_after_exact_body_cannot_succeed(self):
        d=self.directory();process=self.fake_process()
        def launch(command,**kw):
            req=B.read(Path(command[command.index('--request')+1]));out=Path(command[-1]);(out/'packet.zip').write_bytes(packet)
            B.write(out/'child.json',{'pid':process.pid,'pgid':process.pid,'parent_pid':os.getpid(),'process_token':req['process_token'],
                'source_sha256':req['source_sha256'],'http':{'status':200,'bytes':T.PACKET_BYTES,'sha256':T.PACKET_SHA}})
            return process
        with patch.object(T.subprocess,'Popen',side_effect=launch),patch.object(T.os,'killpg',side_effect=PermissionError('OFFLINE_EPERM')):
            with self.assertRaisesRegex(RuntimeError,'FETCH_CLEANUP_UNCERTAIN'):T.fetch_packet(d/'fetch',commit,deadline,worker_command=['OFFLINE_NO_PROCESS'])
        receipt=B.read(d/'fetch/receipt.json');self.assertIsNotNone(receipt['body_admitted_epoch']);self.assertEqual(receipt['status'],'FAIL')
        self.assertFalse(receipt['launch']['group_absent'])
    def test_actual_redirect_handler_closes_response_without_read(self):
        from email.message import Message
        response=Response(b'private');response.read=lambda _:self.fail('redirect body read')
        headers=Message();headers['Location']='https://example.invalid/'
        request=T.urllib.request.Request(T.packet_url(commit))
        with self.assertRaisesRegex(ValueError,'PAYLOAD_REDIRECT_REJECTED'):
            T.RejectRedirect().http_error_302(request,response,302,'redirect',headers)
        self.assertTrue(response.closed)
    def test_redirect_close_error_preserves_rejection(self):
        from email.message import Message
        response=Response(b'private');response.close=lambda:(_ for _ in ()).throw(OSError('OFFLINE_CLOSE'))
        headers=Message();headers['Location']='https://example.invalid/'
        request=T.urllib.request.Request(T.packet_url(commit))
        try:T.RejectRedirect().http_error_302(request,response,302,'redirect',headers)
        except ValueError as error:
            self.assertEqual(str(error),'PAYLOAD_REDIRECT_REJECTED');self.assertIn('PAYLOAD_REDIRECT_CLOSE_FAILURE:OSError',error.__notes__)
        else:self.fail('redirect rejection missing')
    def test_cleanup_error_metadata_cannot_pass_record_audit(self):
        d=self.directory();_,record=T.fetch_packet(d/'fetch',commit,deadline,worker_command=self.setup_child(d))
        record['cleanup_errors']=[{'stage':'group_probe','type':'PermissionError'}]
        with self.assertRaisesRegex(ValueError,'TRANSPORT_CLEANUP_ERROR'):
            T.audit_record(record,commit,deadline,T.digest((HERE/'payload_transport.py').read_bytes()),allow_synthetic=True)

result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
report={'status':'PASS' if result.wasSuccessful() else 'FAIL','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
    'network_attempts':attempts['network'],'credential_read_attempts':attempts['credentials'],'provider_calls':0,
    'wrapper_bytes':(A.output/'candidate/submission/wrapper.py').stat().st_size,'packet_bytes':len(packet),'packet_sha256':T.digest(packet),
    'publication_commit':'SYNTHETIC_NOT_PUBLISHED','actual_cpu':'NOT_RUN','actual_tpu':'NOT_RUN','m8_status':'NOT_QUALIFIED'}
(A.output/'controls.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
raise SystemExit(0 if result.wasSuccessful() and not any(attempts.values()) else 1)
