"""Offline controls: real official Client and real AuthorizedSession, fake HTTP only."""
import builtins
from contextlib import ExitStack, contextmanager
import inspect
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import socket
import sys
import tempfile
import traceback
import unittest
from unittest.mock import patch
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from google.auth.credentials import Credentials
from google.auth.transport.requests import AuthorizedSession
from requests import Response
from requests.adapters import HTTPAdapter
from requests.exceptions import ReadTimeout, HTTPError
import colab_cli.common as common
import colab_cli.cli as cli
from colab_cli.client import Client, Prod, Variant, Accelerator, PostAssignmentResponse, ColabRequestError
from colab_cli.history import HistoryLogger
from colab_cli.state import StateStore, SessionState
import allocation_transport as A
import official_cli_no_oauth as W

URL = 'https://colab.research.google.com/tun/m/assign?nbh=fixture&variant=TPU&accelerator=V6E1'
GET = {'acc':'V6E1','nbh':'fixture','token':'OFFLINE_XSRF','variant':'TPU'}
POST = {'accelerator':'V6E1','endpoint':'OFFLINE_ENDPOINT','variant':2,
        'runtimeProxyInfo':{'token':'OFFLINE_TOKEN','tokenExpiresInSeconds':300,'url':'https://offline.invalid/'}}

class FakeCredentials(Credentials):
    def __init__(self):
        super().__init__(); self.token = 'OFFLINE_CREDENTIAL'; self.refreshes = 0
    def refresh(self, request):
        self.refreshes += 1; self.token = 'OFFLINE_REFRESHED'

class FakeAdapter(HTTPAdapter):
    def __init__(self, replies=()):
        super().__init__(max_retries=0); self.replies = list(replies); self.calls = []; self.closed_count = 0
    def send(self, request, **kwargs):
        self.calls.append((request, kwargs))
        item = self.replies.pop(0) if self.replies else (200, POST if request.method == 'POST' else GET)
        if isinstance(item, BaseException): raise item
        status, body = item
        response = Response(); response.status_code = status; response.reason = 'OFFLINE_REASON'
        response.request = request; response.url = request.url
        response._content = json.dumps(body).encode()
        if status in (307,308): response.headers['Location'] = URL
        return response
    def close(self): self.closed_count += 1

def fixture(replies=(), get_replies=()):
    credentials = FakeCredentials(); original = AuthorizedSession(credentials)
    getter = FakeAdapter(get_replies); original.mount('https://', getter)
    transport = A.allocation_session(original)
    poster = FakeAdapter(replies); transport.allocation.mount('https://', poster)
    return transport, getter, poster, credentials

def assign(transport):
    return Client(Prod(), transport).assign(uuid.UUID(int=1), Variant.TPU, Accelerator.V6E1, None)

@contextmanager
def factory_fixture(replies=(), close_failure=False):
    state = common.State(); transports = []; original_constructor = A.allocation_session
    def capture(original):
        transport = original_constructor(original); poster = FakeAdapter(replies)
        transport.allocation.mount('https://', poster)
        if close_failure:
            transport.allocation.close=lambda: (_ for _ in ()).throw(ValueError('OFFLINE_CLOSE'))
        transports.append((transport, poster)); return transport
    original = AuthorizedSession(FakeCredentials()); getter = FakeAdapter(); original.mount('https://', getter)
    calls = []
    def get_credentials(config, provider): calls.append((config, provider)); return original
    with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
        state._store = StateStore(str(Path(directory)/'sessions.json'))
        state._history = HistoryLogger(str(Path(directory)/'history'))
        stack.enter_context(patch.object(common, 'state', state)); stack.enter_context(patch.object(cli, 'state', state))
        stack.enter_context(patch.object(common, 'get_credentials', get_credentials))
        stack.enter_context(patch.object(A, 'allocation_session', capture))
        stack.enter_context(patch.object(cli, 'setup_logging')); stack.enter_context(patch.object(cli.auto_update,'run_background_check'))
        stack.enter_context(patch.object(sys,'stdout',io.StringIO())); stack.enter_context(patch.object(sys,'stderr',io.StringIO()))
        yield state, transports, getter, calls

class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        project = next(parent for parent in Path(__file__).resolve().parents
                       if (parent/'packages/bitsandbytes-tpu/pyproject.toml').is_file())
        identity = json.loads((project/'.work/task2-cloud/cli-identity.json').read_text())
        site = Path(common.__file__).resolve().parents[1]
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        assert importlib.metadata.version('google-colab-cli') == identity['version'] == '0.7.4'
        assert all(digest(site/name) == sha for name,sha in identity['files'].items())
        assert digest(site/'google_colab_cli-0.7.4.dist-info/METADATA') == identity['metadata_sha256']
        assert digest(site/'google_colab_cli-0.7.4.dist-info/RECORD') == identity['record_sha256']
        review = project/'.work/allocation-transport-controls'
        review.mkdir(parents=True, exist_ok=True)
        (review/'official-source-readback.json').write_text(json.dumps({
            'status':'PASS','distribution':'google-colab-cli','version':'0.7.4',
            'python_files_checked':len(identity['files']),'metadata_match':True,'record_match':True,
            'account_requests':'NOT_RUN'},indent=2)+'\n')
    def setUp(self):
        self.network = patch.object(socket.socket,'connect',side_effect=AssertionError('NETWORK_FORBIDDEN')); self.network.start()
    def tearDown(self): self.network.stop()

    def test_correct_real_client_and_preserved_request(self):
        t,g,p,c = fixture(); t.original.headers['X-Fixture'] = 'retained'; t.original.cookies.set('fixture','retained')
        result = assign(t)
        self.assertIsInstance(result,PostAssignmentResponse); self.assertEqual(result.endpoint,'OFFLINE_ENDPOINT')
        self.assertEqual(g.calls[0][1]['timeout'],120); self.assertEqual(p.calls[0][1]['timeout'],(10,300))
        request, policy = p.calls[0]; self.assertIs(policy['verify'],True)
        self.assertEqual(request.headers['X-Fixture'],'retained'); self.assertIn('fixture=retained',request.headers['Cookie'])
        self.assertEqual(request.headers['X-Goog-Colab-Token'],'OFFLINE_XSRF')
        self.assertEqual(request.headers['X-Colab-Client-Agent'],'colab-cli'); self.assertIsNone(request.body)
        self.assertIn('authuser=0',request.url); self.assertEqual(c.refreshes,0); t.close()
        self.assertEqual(g.closed_count,1); self.assertEqual(p.closed_count,1)

    def test_finite_public_call_budget(self):
        t,g,p,c = fixture(); original = t.allocation.request
        with patch.object(t.allocation,'request',wraps=original) as request:
            assign(t); self.assertEqual(request.call_args.kwargs['max_allowed_time'],310)
            self.assertIs(request.call_args.kwargs['allow_redirects'],False)
        self.assertEqual(t.allocation.get_adapter(URL).max_retries.total,0); t.close()

    def test_timeout_original_identity_no_retry(self):
        error = ReadTimeout('OFFLINE_TIMEOUT'); t,g,p,c = fixture([error])
        try: assign(t)
        except ReadTimeout as caught:
            self.assertIs(caught,error); self.assertIn('send', ''.join(traceback.format_tb(caught.__traceback__)))
        else: self.fail('timeout missing')
        with self.assertRaisesRegex(RuntimeError,'ALREADY_CONSUMED'): assign(t)
        self.assertEqual(len(p.calls),1); self.assertEqual(c.refreshes,0); t.close()

    def test_401_no_post_auth_replay(self):
        t,g,p,c = fixture([(401,{'error':'OFFLINE_DENIED'})])
        with self.assertRaises(ColabRequestError) as caught: assign(t)
        self.assertEqual(caught.exception.response.status_code,401); self.assertEqual(len(p.calls),1)
        self.assertEqual(c.refreshes,0); t.close()

    def test_default_get_refresh_behavior_preserved(self):
        t,g,p,c = fixture(get_replies=[(401,{}),(200,GET)])
        assign(t); self.assertEqual(len(g.calls),2); self.assertEqual(c.refreshes,1); self.assertEqual(len(p.calls),1); t.close()

    def test_redirect_rejected_without_replay(self):
        for status in (307,308):
            with self.subTest(status=status):
                t,g,p,c = fixture([(status,{})])
                with self.assertRaisesRegex(HTTPError,'REDIRECT_FORBIDDEN') as caught: assign(t)
                self.assertEqual(caught.exception.response.status_code,status); self.assertEqual(len(p.calls),1); t.close()

    def test_non_numeric_http_error_preserved(self):
        t,g,p,c = fixture([(500,{'error':'OFFLINE_BODY'})])
        with self.assertRaises(ColabRequestError) as caught: assign(t)
        self.assertEqual(caught.exception.response.reason,'OFFLINE_REASON')
        self.assertIn('OFFLINE_BODY',caught.exception.response_body); self.assertIsNotNone(caught.exception.request); t.close()

    def test_non_target_delegation(self):
        for method,url in [('GET',URL),('POST',URL.replace('colab.research.google.com','other.invalid')),
                           ('POST',URL.replace('/assign','/other')),('PUT',URL)]:
            with self.subTest(method=method,url=url):
                t,g,p,c = fixture(); t.request(method,url,params={'authuser':'0'})
                self.assertEqual(len(g.calls),1); self.assertEqual(g.calls[0][1]['timeout'],120)
                self.assertEqual(len(p.calls),0); self.assertEqual(t.post_count,0); t.close()

    def test_policy_and_tls_rejected_before_consumption(self):
        for key,value in [('timeout',300),('max_allowed_time',310),('allow_redirects',False),
                          ('verify',False),('verify',None),('verify','custom-ca.pem')]:
            with self.subTest(key=key,value=value):
                t,g,p,c = fixture()
                with self.assertRaises(ValueError): t.request('POST',URL,params={'authuser':'0'},**{key:value})
                self.assertEqual(t.post_count,0); self.assertEqual(len(p.calls),0); t.close()
        for attr in ('original','allocation'):
            t,g,p,c = fixture(); getattr(t,attr).verify=False
            with self.assertRaises(ValueError): t.request('POST',URL,params={'authuser':'0'})
            self.assertEqual(t.post_count,0); t.close()

    def test_profile_rejected_before_consumption(self):
        for url,params in [(URL.replace('V6E1','V5E1'),{'authuser':'0'}),(URL+'&shape=hm',{'authuser':'0'}),
                           (URL,{'authuser':'1'}),(URL+'#fragment',{'authuser':'0'})]:
            t,g,p,c = fixture()
            with self.assertRaises(ValueError): t.request('POST',url,params=params)
            self.assertEqual(t.post_count,0); self.assertEqual(len(p.calls),0); t.close()

    def test_optin_exact_argv(self):
        good = [A.FLAG,'new','-s','offline-fixture','--tpu','v6e1']
        self.assertEqual(A.validate_opt_in(good),good[1:])
        for bad in [good+[A.FLAG],good[1:]+[A.FLAG],[A.FLAG,'sessions'],good+['--high-mem'],
                    [A.FLAG,'new','--session','offline-fixture','--tpu','v6e1'],
                    [A.FLAG,'new','-s','offline-fixture','--tpu','v5e1']]:
            with self.assertRaises(ValueError): A.validate_opt_in(bad)

    def test_real_callback_new_and_state_store(self):
        official = common.Client
        with factory_fixture() as (state,transports,getter,calls):
            with A.adapted_client_factory(common):
                cli.app(args=['--config','offline-config','--client-oauth-config','offline-oauth','--auth','adc',
                              'new','-s','offline-fixture','--tpu','v6e1'],standalone_mode=False)
                stored = state.store.get('offline-fixture'); self.assertIsInstance(stored,SessionState)
                self.assertEqual((stored.endpoint,stored.accelerator,stored.variant,stored.machine_shape),
                                 ('OFFLINE_ENDPOINT','V6E1','TPU','STANDARD'))
                self.assertEqual((stored.token,stored.url),('OFFLINE_TOKEN','https://offline.invalid/'))
                self.assertEqual(calls,[('offline-oauth',common.AuthProvider.ADC)])
                self.assertEqual(state.config_path,'offline-config'); self.assertIsInstance(state.client,Client)
            self.assertIs(common.Client,official); self.assertTrue(transports[0][0].closed)
        defaults = inspect.signature(cli.callback).parameters
        self.assertIs(defaults['auth'].default,common.AuthProvider.OAUTH2); self.assertIsNone(defaults['config'].default)

    def test_factory_restored_after_request_error(self):
        official = common.Client; error=ReadTimeout('OFFLINE_TIMEOUT')
        with factory_fixture([error]) as (state,transports,getter,calls):
            with self.assertRaises(ReadTimeout) as caught:
                with A.adapted_client_factory(common): assign(state.client.session)
            self.assertIs(caught.exception,error); self.assertIs(common.Client,official); self.assertTrue(transports[0][0].closed)

    def test_cleanup_failure_keeps_original_traceback(self):
        official = common.Client; error=ReadTimeout('OFFLINE_TIMEOUT')
        with factory_fixture([error]) as (state,transports,getter,calls):
            retained_traceback = None; serialized_exception = None
            try:
                with A.adapted_client_factory(common):
                    client=state.client; client.session.allocation.close=lambda: (_ for _ in ()).throw(ValueError('OFFLINE_CLOSE'))
                    assign(client.session)
            except ReadTimeout as caught:
                self.assertIs(caught,error); retained_traceback = ''.join(traceback.format_tb(caught.__traceback__))
                serialized_exception = ''.join(traceback.format_exception(caught))
            else: self.fail('timeout missing')
            self.assertIs(common.Client,official)
            self.assertEqual(error.allocation_cleanup_errors,[{'type':'ValueError','stage':'close_session'}])
            self.assertIn('send',retained_traceback)
            self.assertIn('ReadTimeout: OFFLINE_TIMEOUT',serialized_exception)
            self.assertIn('ALLOCATION_TRANSPORT_CLEANUP_FAILED [{"type":"ValueError","stage":"close_session"}]',serialized_exception)
            self.assertNotIn('ValueError: OFFLINE_CLOSE',serialized_exception)
            self.assertEqual(getter.closed_count,1)

    def test_cleanup_failure_without_request_error(self):
        official = common.Client
        with factory_fixture() as (state,transports,getter,calls):
            with self.assertRaisesRegex(RuntimeError,'TRANSPORT_CLEANUP_FAILED'):
                with A.adapted_client_factory(common):
                    client=state.client; client.session.allocation.close=lambda: (_ for _ in ()).throw(ValueError('OFFLINE_CLOSE'))
                    assign(client.session)
            self.assertIs(common.Client,official); self.assertEqual(getter.closed_count,1)

    def test_wrong_environment_factory_rejected_and_closed(self):
        official = common.Client
        with factory_fixture() as (state,transports,getter,calls):
            original=AuthorizedSession(FakeCredentials()); adapter=FakeAdapter(); original.mount('https://',adapter)
            env=Prod(); env.domain='https://other.invalid'
            with A.adapted_client_factory(common):
                with self.assertRaisesRegex(ValueError,'ENVIRONMENT_PROFILE'): common.Client(env,original)
            self.assertIs(common.Client,official); self.assertEqual(adapter.closed_count,1); self.assertEqual(transports,[])

    def test_factory_rejects_used_client(self):
        with factory_fixture() as (state,transports,getter,calls):
            state._client = object()
            with self.assertRaisesRegex(RuntimeError,'FRESH_OFFICIAL_CLIENT'):
                with A.adapted_client_factory(common): pass
            self.assertEqual(transports,[])

    def test_default_wrapper_no_transport_import_or_factory(self):
        official = common.Client; real_import=builtins.__import__; imported=[]
        def watch(name,*args,**kwargs):
            imported.append(name); return real_import(name,*args,**kwargs)
        with patch.object(W,'official_main',return_value='OFFLINE_DEFAULT') as main, patch('builtins.__import__',side_effect=watch), \
             patch.object(sys,'stdout',io.StringIO()),patch.object(sys,'stderr',io.StringIO()),patch.object(sys,'argv',['fixture']):
            self.assertEqual(W.main(['sessions']),'OFFLINE_DEFAULT'); self.assertIs(common.Client,official)
            self.assertNotIn('allocation_transport',imported); self.assertEqual(sys.argv,['colab','sessions'])
            main.assert_called_once_with()

    def test_optin_wrapper_real_cli_and_restoration(self):
        official = common.Client
        with factory_fixture() as (state,transports,getter,calls), patch.object(sys,'argv',['fixture']), \
             patch.object(W,'official_main',side_effect=lambda:cli.app(args=sys.argv[1:],standalone_mode=False)):
            W.main([A.FLAG,'new','-s','offline-fixture','--tpu','v6e1'])
            self.assertIs(common.Client,official); self.assertEqual(len(transports),1)
            self.assertEqual(calls[0][1],common.AuthProvider.OAUTH2)
            self.assertEqual(calls[0][0],inspect.signature(cli.callback).parameters['client_oauth_config'].default)
            self.assertIsNotNone(state.store.get('offline-fixture'))

    def test_real_standalone_wrapper_cleanup_failure_is_nonzero(self):
        official = common.Client
        with factory_fixture(close_failure=True) as (state,transports,getter,calls), patch.object(sys,'argv',['fixture']):
            # W.official_main is the real installed main(); its app() is standalone.
            self.assertIs(W.official_main,cli.main)
            with self.assertRaisesRegex(RuntimeError,'TRANSPORT_CLEANUP_FAILED') as caught:
                W.main([A.FLAG,'new','-s','offline-fixture','--tpu','v6e1'])
            self.assertEqual(caught.exception.allocation_cleanup_errors,[{'type':'ValueError','stage':'close_session'}])
            self.assertIn('"stage":"close_session"',''.join(traceback.format_exception(caught.exception)))
            self.assertIs(common.Client,official); self.assertEqual(getter.closed_count,1)
            self.assertEqual(len(transports[0][1].calls),1); self.assertIsNotNone(state.store.get('offline-fixture'))

    def test_real_standalone_wrapper_clean_zero_exit_preserved(self):
        official = common.Client
        with factory_fixture() as (state,transports,getter,calls), patch.object(sys,'argv',['fixture']):
            with self.assertRaises(SystemExit) as caught:
                W.main([A.FLAG,'new','-s','offline-fixture','--tpu','v6e1'])
            self.assertEqual(caught.exception.code,0); self.assertIs(common.Client,official)
            self.assertTrue(transports[0][0].closed); self.assertEqual(getter.closed_count,1)
            self.assertEqual(len(transports[0][1].calls),1); self.assertIsNotNone(state.store.get('offline-fixture'))

    def test_real_standalone_wrapper_nonzero_exit_preserved(self):
        official = common.Client
        with factory_fixture([(400,{'error':'OFFLINE_DENIED'})],close_failure=True) as (state,transports,getter,calls), \
             patch.object(sys,'argv',['fixture']):
            with self.assertRaises(SystemExit) as caught:
                W.main([A.FLAG,'new','-s','offline-fixture','--tpu','v6e1'])
            self.assertEqual(caught.exception.code,1)
            self.assertEqual(caught.exception.allocation_cleanup_errors,[{'type':'ValueError','stage':'close_session'}])
            self.assertIs(common.Client,official); self.assertEqual(getter.closed_count,1)
            self.assertEqual(len(transports[0][1].calls),1); self.assertIsNone(state.store.get('offline-fixture'))

    def test_explicit_systemexit_none_zero_cleanup_failure_promoted(self):
        official = common.Client
        for code in (None,0):
            with self.subTest(code=code), factory_fixture(close_failure=True) as (state,transports,getter,calls):
                original_exit = SystemExit(code)
                with self.assertRaisesRegex(RuntimeError,'TRANSPORT_CLEANUP_FAILED') as caught:
                    with A.adapted_client_factory(common):
                        assign(state.client.session); raise original_exit
                self.assertIsNot(caught.exception,original_exit)
                self.assertEqual(caught.exception.allocation_cleanup_errors,[{'type':'ValueError','stage':'close_session'}])
                self.assertIs(common.Client,official); self.assertEqual(getter.closed_count,1)

    def test_explicit_nonzero_systemexit_cleanup_failure_identity_preserved(self):
        official = common.Client
        for code in (7,'OFFLINE_EXIT',0.0):
            with self.subTest(code=code), factory_fixture(close_failure=True) as (state,transports,getter,calls):
                original_exit = SystemExit(code)
                with self.assertRaises(SystemExit) as caught:
                    with A.adapted_client_factory(common):
                        assign(state.client.session); raise original_exit
                self.assertIs(caught.exception,original_exit); self.assertEqual(caught.exception.code,code)
                self.assertEqual(caught.exception.allocation_cleanup_errors,[{'type':'ValueError','stage':'close_session'}])
                self.assertIs(common.Client,official); self.assertEqual(getter.closed_count,1)

    def test_explicit_systemexit_clean_identity_preserved(self):
        official = common.Client
        for code in (None,0,7):
            with self.subTest(code=code), factory_fixture() as (state,transports,getter,calls):
                original_exit = SystemExit(code)
                with self.assertRaises(SystemExit) as caught:
                    with A.adapted_client_factory(common):
                        assign(state.client.session); raise original_exit
                self.assertIs(caught.exception,original_exit); self.assertEqual(caught.exception.code,code)
                self.assertIs(common.Client,official); self.assertEqual(getter.closed_count,1)
                self.assertEqual(transports[0][0].cleanup_errors,[])

if __name__ == '__main__': unittest.main(verbosity=2)
