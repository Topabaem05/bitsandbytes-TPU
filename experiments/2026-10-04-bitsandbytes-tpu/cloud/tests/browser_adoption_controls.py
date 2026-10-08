"""Real admitted CLI models/store/client with fake HTTP. No provider or token output."""
import copy,hashlib,io,json,socket,sys,tempfile,time,unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import browser_adoption as A
import official_cli_no_oauth as W
import colab_cli.common as common
from colab_cli.client import Client,Prod
from colab_cli.state import StateStore,SessionState
from google.auth.transport.requests import AuthorizedSession
from allocation_transport_controls import FakeCredentials,FakeAdapter


def data():
 now=time.time()
 return {'format':'bnb-tpu.browser-adoption.v1','status':'ROOT_CREATED_BROWSER_RUNTIME_READY','endpoint':'OFFLINE_ENDPOINT',
  'session':'offline-browser','hardware':'V6E1','variant':'TPU','authuser':'0','allocation_epoch':now-100,'observed_epoch':now-1,
  'packet_sha256':'1'*64,'driver_sha256':'2'*64,'cli_identity_sha256':'3'*64,
  'marker_path':'/tmp/bnb-tpu-root-browser-'+('4'*32)+'.json','marker_sha256':'5'*64}


def assignment():return {'endpoint':'OFFLINE_ENDPOINT','accelerator':'V6E1','variant':2,'machineShape':0,
 'runtimeProxyInfo':{'token':'OFFLINE_SECRET_NOT_FOR_OUTPUT','tokenExpiresInSeconds':300,'url':'https://offline.invalid/'}}

class Controls(unittest.TestCase):
 def run_case(self,fault=None,remove=False,hardware="V6E1"):
  with tempfile.TemporaryDirectory() as directory,ExitStack() as stack:
   base=Path(directory);r=data();a=assignment();r['hardware']=hardware;a['accelerator']=hardware;items=[a]
   if fault=='wrong-endpoint':a['endpoint']='UNOWNED_ENDPOINT'
   elif fault=='extra-runtime':items.append(copy.deepcopy(a))
   elif fault=='missing-runtime':items=[]
   elif fault=='wrong-hardware':a['accelerator']='V5E1' if hardware=='V6E1' else 'V6E1'
   elif fault=='unlisted-hardware':r['hardware']='V4'
   elif fault=='unlisted-gpu':r['hardware']='T4';r['variant']='GPU'
   elif fault=='expired-proxy':a['runtimeProxyInfo']['tokenExpiresInSeconds']=0
   elif fault=='wrong-account':r['authuser']='1'
   elif fault=='stale-record':r['observed_epoch']-=200
   elif fault=='expired-record':r['allocation_epoch']-=3600
   p=base/'adoption.json';p.write_text(json.dumps(r));expected=A.sha(p)
   if fault=='wrong-record-sha':expected='0'*64
   http=AuthorizedSession(FakeCredentials());adapter=FakeAdapter([(200,{'assignments':items}),(200,{'assignments':items})]);http.mount('https://',adapter)
   state=common.State();state._client=Client(Prod(),http);state._store=StateStore(str(base/'sessions.json'))
   if fault=='local-nonempty':state.store.add(SessionState(name='unowned',endpoint='UNOWNED_ENDPOINT',token='NOTOUTPUT',url='https://offline.invalid/'))
   stdout=io.StringIO();stderr=io.StringIO()
   stack.enter_context(patch.object(common,'state',state));stack.enter_context(patch.object(W,'official_main',side_effect=AssertionError('NO_OFFICIAL_ASSIGNMENT_COMMAND')))
   stack.enter_context(patch.object(sys,'stdout',stdout));stack.enter_context(patch.object(sys,'stderr',stderr))
   stack.enter_context(patch.object(socket,'create_connection',side_effect=AssertionError('NO_NETWORK')))
   if fault:
    with self.assertRaises(ValueError):W.main([A.FLAG,str(p),expected])
    self.assertIsNone(state.store.get(r['session']))
   else:
    W.main([A.FLAG,str(p),expected]);registered=state.store.get(r['session']);self.assertEqual(registered.endpoint,r['endpoint'])
    self.assertEqual(state.get_session(r['session']).token,'OFFLINE_SECRET_NOT_FOR_OUTPUT')
    if remove:
     W.main([A.REMOVE_FLAG,str(p),expected]);self.assertIsNone(state.store.get(r['session']))
   self.assertTrue(all(request.method=='GET' and request.url.endswith('/tun/m/assignments?authuser=0') for request,kwargs in adapter.calls))
   self.assertNotIn('OFFLINE_SECRET',stdout.getvalue()+stderr.getvalue())
   self.assertFalse(any(request.method=='POST' for request,kwargs in adapter.calls));http.close()
 def test_register_official_api_no_post(self):self.run_case()
 def test_remove_provisional_local_only(self):self.run_case(remove=True)
 def test_v5_register_official_api_no_post(self):self.run_case(hardware='V5E1')
 def test_v5_remove_provisional_local_only(self):self.run_case(remove=True,hardware='V5E1')
 def test_v5_wrong_hardware(self):self.run_case('wrong-hardware',hardware='V5E1')
 def test_unlisted_tpu(self):self.run_case('unlisted-hardware')
 def test_unlisted_gpu(self):self.run_case('unlisted-gpu')
 def test_wrong_endpoint(self):self.run_case('wrong-endpoint')
 def test_extra_runtime(self):self.run_case('extra-runtime')
 def test_missing_runtime(self):self.run_case('missing-runtime')
 def test_wrong_hardware(self):self.run_case('wrong-hardware')
 def test_expired_proxy(self):self.run_case('expired-proxy')
 def test_wrong_account_profile(self):self.run_case('wrong-account')
 def test_stale_record(self):self.run_case('stale-record')
 def test_expired_original_lifecycle(self):self.run_case('expired-record')
 def test_wrong_source_record_sha(self):self.run_case('wrong-record-sha')
 def test_local_nonempty(self):self.run_case('local-nonempty')

if __name__=='__main__':unittest.main()
