"""Explicit, private adaptation of one official CLI allocation transport.

This module never obtains credentials or performs a request on import.
Only the opt-in wrapper may install its factory for one CLI invocation.
"""
from contextlib import contextmanager
import copy
import json
import re
import sys
import threading
from urllib.parse import parse_qs, urlsplit

FLAG = '--allocation-transport-v1'
CONNECT_SECONDS = 10
READ_SECONDS = 300
CALL_SECONDS = 310


def validate_opt_in(argv):
    """Accept only the existing owner command with one leading opt-in flag."""
    if (len(argv) != 6 or argv.count(FLAG) != 1 or argv[:3] != [FLAG, 'new', '-s']
            or argv[4:] != ['--tpu', 'v6e1']
            or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,127}', argv[3])):
        raise ValueError('EXACT_ALLOCATION_OPT_IN_REQUIRED')
    return argv[1:]


class AllocationSession:
    """Keep the original GET transport; consume at most one assignment POST."""
    def __init__(self, original, allocation):
        self.original = original
        self.allocation = allocation
        self._lock = threading.Lock()
        self.post_count = 0
        self.closed = False
        self.cleanup_errors = []

    def request(self, method, url, **kwargs):
        parsed = urlsplit(url)
        target = (method == 'POST' and parsed.scheme == 'https'
                  and parsed.hostname == 'colab.research.google.com'
                  and parsed.path == '/tun/m/assign')
        if not target:
            return self.original.request(method, url, **kwargs)
        if parsed.username is not None or parsed.password is not None or parsed.port not in (None,443) or parsed.fragment:
            raise ValueError('ALLOCATION_URL_STRUCTURE')
        query = parse_qs(parsed.query, keep_blank_values=True)
        if (set(query) != {'nbh','variant','accelerator'} or len(query['nbh']) != 1
                or not query['nbh'][0] or query['variant'] != ['TPU']
                or query['accelerator'] != ['V6E1']):
            raise ValueError('ALLOCATION_REQUEST_PROFILE')
        if kwargs.get('params') != {'authuser': '0'}:
            raise ValueError('ALLOCATION_ACCOUNT_PARAMETER')
        if kwargs.get('verify', True) is not True or self.original.verify is not True or self.allocation.verify is not True:
            raise ValueError('ALLOCATION_TLS_VERIFICATION_REQUIRED')
        if any(key in kwargs for key in ('timeout','max_allowed_time','allow_redirects')):
            raise ValueError('ALLOCATION_REQUEST_POLICY_ALREADY_SET')
        with self._lock:
            if self.closed or self.post_count != 0:
                raise RuntimeError('ALLOCATION_POST_ALREADY_CONSUMED')
            self.post_count = 1
        # Preserve GET-established request state without touching the original session.
        self.allocation.headers = self.original.headers.copy()
        self.allocation.cookies = copy.copy(self.original.cookies)
        self.allocation.proxies = self.original.proxies.copy()
        self.allocation.cert = self.original.cert
        self.allocation.trust_env = self.original.trust_env
        adapted = dict(kwargs, timeout=(CONNECT_SECONDS, READ_SECONDS),
                       max_allowed_time=CALL_SECONDS, allow_redirects=False, verify=True)
        response = self.allocation.request(method, url, **adapted)
        if 300 <= response.status_code < 400:
            from requests.exceptions import HTTPError
            raise HTTPError('ALLOCATION_REDIRECT_FORBIDDEN', request=response.request, response=response)
        return response

    def close(self):
        with self._lock:
            if self.closed: return
            self.closed = True
        errors = []
        for session in (self.allocation,self.original):
            try: session.close()
            except Exception as error: errors.append(error)
        self.cleanup_errors = [{'type':type(error).__name__,'stage':'close_session'} for error in errors]
        if errors: raise RuntimeError('ALLOCATION_SESSION_CLOSE_FAILED') from errors[0]


def allocation_session(original):
    """Use documented public constructors. Default GET authentication is untouched."""
    from google.auth.transport.requests import AuthorizedSession
    from requests.adapters import HTTPAdapter
    if not isinstance(original, AuthorizedSession):
        raise TypeError('ORIGINAL_AUTHORIZED_SESSION_REQUIRED')
    if original.verify is not True:
        raise ValueError('ALLOCATION_TLS_VERIFICATION_REQUIRED')
    allocation = AuthorizedSession(original.credentials, max_refresh_attempts=0)
    try:
        allocation.mount('https://', HTTPAdapter(max_retries=0))
        return AllocationSession(original,allocation)
    except BaseException:
        allocation.close()
        raise


@contextmanager
def adapted_client_factory(common):
    """Replace one factory reference, never official classes/functions or timeout globals."""
    from colab_cli.client import Client, Prod
    official_factory = common.Client
    if official_factory is not Client or common.state._client is not None:
        raise RuntimeError('FRESH_OFFICIAL_CLIENT_REQUIRED')
    sessions = []
    def factory(env, original, logger=None):
        if type(env) is not Prod or env.domain != 'https://colab.research.google.com' or env.api != 'https://colab.pa.googleapis.com':
            original.close()
            raise ValueError('ALLOCATION_ENVIRONMENT_PROFILE')
        try:
            transport = allocation_session(original)
        except BaseException:
            original.close()
            raise
        sessions.append(transport)
        return Client(env,transport,logger=logger)
    common.Client = factory
    try:
        yield sessions
    finally:
        original_error = sys.exc_info()[1]
        common.Client = official_factory
        errors = []
        for transport in sessions:
            try: transport.close()
            except Exception as error: errors.append(error)
        if errors:
            details = [detail for transport in sessions for detail in transport.cleanup_errors]
            successful_exit = (isinstance(original_error, SystemExit)
                               and (original_error.code is None
                                    or isinstance(original_error.code, int) and original_error.code == 0))
            if original_error is not None and not successful_exit:
                original_error.allocation_cleanup_errors = details
                original_error.add_note('ALLOCATION_TRANSPORT_CLEANUP_FAILED ' + json.dumps(details, separators=(',', ':')))
            else:
                cleanup_error = RuntimeError('ALLOCATION_TRANSPORT_CLEANUP_FAILED')
                cleanup_error.allocation_cleanup_errors = details
                cleanup_error.add_note('ALLOCATION_TRANSPORT_CLEANUP_FAILED ' + json.dumps(details, separators=(',', ':')))
                raise cleanup_error from errors[0]
