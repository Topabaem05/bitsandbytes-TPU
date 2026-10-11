"""Observe original dispatcher calls and admitted Python frames. No method replacement."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import struct
import time
import traceback
import graph_witness as G
import archive_store as Z

def require(value, message):
    if not value:
        raise ValueError(message)

def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))

def retained_error(value):
    """Preserve invalid numeric diagnostics as text, only after status ERROR."""
    require(value.get('status') == 'ERROR', 'ERROR_RECORD_STATUS')
    paths = []
    def clean(item, path):
        if isinstance(item, float) and not math.isfinite(item):
            paths.append(path)
            return 'INVALID_NONFINITE:' + repr(item)
        if isinstance(item, dict): return {k: clean(v, path + '/' + str(k)) for k, v in item.items()}
        if isinstance(item, (list, tuple)): return [clean(v, path + '/' + str(i)) for i, v in enumerate(item)]
        return item
    result = clean(value, '')
    if paths: result['invalid_nonfinite_paths'] = paths
    return canonical(result)

def aggregate_execution(evidence):
    """Sum retained, disjoint actual intervals; never invent an execution sample."""
    counters, execution_metrics = {}, {}
    for item in evidence:
        for key, value in item['counters'].items():
            require(type(value) is int and value >= 0, 'AGGREGATE_COUNTER')
            counters[key] = counters.get(key, 0) + value
        for key, value in item['execution_metrics'].items():
            count, total, samples = value
            old = execution_metrics.setdefault(key, [0, 0, []])
            old[0] += count; old[1] += total; old[2].extend(samples)
    return canonical({'counters': counters, 'execution_metrics': execution_metrics})

def tensor_metadata(torch, tensor):
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype).removeprefix('torch.'),
            'device': str(tensor.device), 'native_functional': bool(torch._is_functional_tensor(tensor))}

def flat_tensors(torch, value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for v in value for t in flat_tensors(torch, v)]
    if isinstance(value, dict):
        return [t for v in value.values() for t in flat_tensors(torch, v)]
    return []

def tensor_record(B, tensor):
    record = B.tensor_record(tensor)
    formats = {'float32': 'f', 'int32': 'i', 'uint8': 'B'}
    require(record['dtype'] in formats, 'COMPOSITE_CAPTURE_DTYPE')
    data = b''.join(struct.pack('<' + formats[record['dtype']], value) for value in record['values'])
    record['bytes_sha256'] = hashlib.sha256(data).hexdigest()
    return record


class SourceFrames:
    """Source-bound function entry/return metadata, including native wrappers."""
    def __init__(self, torch, plugin_root):
        self.torch = torch; self.root = Path(plugin_root).resolve()
        self.events = []; self.errors = []; self.active = {}; self.previous = None
    def callback(self, frame, event, arg):
        if event not in ('call', 'return'):
            return
        path = Path(frame.f_code.co_filename).resolve()
        if path.parent != self.root or path.name not in ('exact.py', 'functionalization.py'):
            return
        name = frame.f_code.co_name
        if name not in ('bitcast', 'compress_statistics', 'restore_scales', 'wrapped'):
            return
        try:
            if event == 'call':
                tensors = flat_tensors(self.torch, {k: v for k, v in frame.f_locals.items()
                                                   if k in ('value', 'first', 'dynamic', 'codes', 'second', 'offset', 'args', 'kwargs')})
                rec = {'source': path.name, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                       'function': name, 'line': frame.f_code.co_firstlineno,
                       'inputs': [tensor_metadata(self.torch, t) for t in tensors], 'status': 'CALL'}
                self.active[id(frame)] = rec; self.events.append(rec)
            elif id(frame) in self.active:
                rec = self.active.pop(id(frame))
                rec.update(status='RETURN', outputs=[tensor_metadata(self.torch, t) for t in flat_tensors(self.torch, arg)])
        except Exception as error:
            self.errors.append({'type': type(error).__name__, 'message': str(error)})
    def __enter__(self):
        require(sys.getprofile() is None, 'EXISTING_PROFILE_OBSERVER_UNSUPPORTED')
        self.previous = sys.getprofile(); sys.setprofile(self.callback)
        return self
    def __exit__(self, *_):
        sys.setprofile(self.previous)

def tls(torch):
    return {'include': str(torch._C._dispatch_tls_local_include_set()),
            'exclude': str(torch._C._dispatch_tls_local_exclude_set())}

def validate_frames(events, opname, plugin_map):
    require(type(events) is list and events, 'FUNCTIONALIZATION_FRAMES_REQUIRED')
    for event in events:
        require(event.get('source') in ('exact.py', 'functionalization.py')
                and event.get('source_sha256') == plugin_map[event['source']]
                and event.get('status') == 'RETURN' and type(event.get('line')) is int,
                'FUNCTIONALIZATION_SOURCE_RETURN')
    names = [v['function'] for v in events]
    body = opname.split('::')[-1]
    require(body in ('compress_statistics', 'restore_scales') and body in names and 'wrapped' in names,
            'PUBLIC_COMPOSITE_SOURCE_FRAMES')
    body_events = [v for v in events if v['function'] == body]
    require(len(body_events) == 1 and body_events[0]['inputs']
            and all(v['native_functional'] is True for v in body_events[0]['inputs']),
            'NATIVE_FUNCTIONAL_BODY_INPUTS')
    wrappers = [v for v in events if v['function'] == 'wrapped']
    require(len(wrappers) == 1 and wrappers[0]['inputs']
            and all(v['native_functional'] is False for v in wrappers[0]['inputs'])
            and wrappers[0]['outputs'] and all(v['native_functional'] is False for v in wrappers[0]['outputs']),
            'SINGLE_NATIVE_WRAPPER_BOUNDARY')
    require(set(names) == {body, 'wrapped', 'bitcast'}
            and wrappers[0]['source'] == 'functionalization.py' and body_events[0]['source'] == 'exact.py',
            'EXACT_OBSERVER_FRAME_INVENTORY')
    bitcasts = [v for v in events if v['function'] == 'bitcast']
    require(len(bitcasts) == 4 and all(v['source'] == 'exact.py' and len(v['inputs']) == len(v['outputs']) == 1
            and v['inputs'][0]['native_functional'] and v['outputs'][0]['native_functional']
            and (v['inputs'][0]['dtype'], v['outputs'][0]['dtype']) in
            (('float32', 'int32'), ('int32', 'float32')) for v in bitcasts), 'ACTUAL_NATIVE_BITCAST_BOUNDARY')
    return {'original_composite_body': 'OBSERVED', 'single_native_wrapper': 'OBSERVED',
            'functional_bitcast_input_output': 'OBSERVED', 'evidence': 'SOURCE_BOUND_PYTHON_CALL_RETURN_METADATA'}

def make_mode(torch, B, P, xla, xm, metrics, plugin_root, output, deadline_epoch):
    from torch.utils._python_dispatch import TorchDispatchMode
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    class OriginalCallMode(TorchDispatchMode):
        def __init__(self):
            super().__init__(); self.records = []; self.index = 0; self.observation_errors = []; self.evidence_limit = False
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}; name = func._schema.name
            if name not in ('bnb_tpu_exact::compress_statistics', 'bnb_tpu_exact::restore_scales'):
                return func(*args, **kwargs)
            if self.evidence_limit:
                return func(*args, **kwargs)
            require(time.time() < deadline_epoch, 'COMPOSITE_OWNER_DEADLINE')
            self.index += 1; directory = output / ('%03d' % self.index); directory.mkdir()
            raw = {'schema_name': name, 'status': 'PARTIAL', 'pid': os.getpid(), 'pgid': os.getpgid(0),
                   'tensor_order': 'ORIGINAL_POSITIONAL_ARGUMENTS', 'method_replacement': False,
                   'calls': 0, 'graph_scope': 'PRE_OPTIMIZATION_COMPOSITE_OUTPUT'}
            self.records.append(raw); before = tls(torch); result = None; returned = False
            inputs = list(args)
            try:
                require(not kwargs and all(isinstance(t, torch.Tensor) and t.device.type == 'xla'
                        and not torch._is_functional_tensor(t) for t in inputs), 'RAW_XLA_POSITIONAL_INPUTS')
                # Materialize actual inputs only. This isolates each selected composite graph.
                xla._XLAC._xla_sync_multi(inputs, devices=[], wait=True, sync_xla_data=True); xm.wait_device_ops()
                input_arrays = [tensor_record(B, t) for t in inputs]
                metrics.clear_all(); started = time.monotonic()
                with SourceFrames(torch, plugin_root) as frames:
                    raw['calls'] += 1
                    result = func(*args, **kwargs)
                    returned = True
                outputs = list(result) if isinstance(result, tuple) else [result]
                require(outputs and all(isinstance(t, torch.Tensor) and t.device.type == 'xla'
                        and not torch._is_functional_tensor(t) for t in outputs), 'RAW_XLA_COMPOSITE_OUTPUTS')
                ctx = xla._XLAC.lowering.LoweringContext('NestedExactComposite'); ctx.build(outputs)
                proto = bytes(ctx.hlo()); printer_proto = bytes(xla._XLAC._get_xla_tensors_hlo_proto(outputs))
                hlo = xla._XLAC._get_xla_tensors_hlo(outputs)
                (directory / 'context.hlo.pb').write_bytes(proto)
                (directory / 'printer.hlo.pb').write_bytes(printer_proto)
                (directory / 'graph.hlo.txt').write_text(hlo)
                (directory / 'context.textproto').write_text(ctx.hlo_text())
                ids = [ctx.tensor_parameter_id(t) for t in inputs]
                mapping = ctx.device_parameter_id_tensor_mapping()
                raw.update(inputs=input_arrays, input_parameter_ids=ids, frames=frames.events,
                           frame_errors=frames.errors, profile_restored=sys.getprofile() is frames.previous,
                           tls_before=before, tls_after=tls(torch),
                           parameter_mapping_api='LoweringContext.device_parameter_id_tensor_mapping+tensor_parameter_id')
                xla._XLAC._xla_sync_multi(outputs, devices=[], wait=True, sync_xla_data=True); xm.wait_device_ops()
                evidence = canonical({'counters': {key: metrics.counter_value(key) for key in metrics.counter_names()},
                                      'execution_metrics': {key: value for key in B.load_spec()[0]['execution_metrics']
                                                            if (value := metrics.metric_data(key)) is not None}})
                raw.update(evidence, execution={'api': '_xla_sync_multi', 'targets': 'ACTUAL_COMPOSITE_OUTPUTS_ONLY',
                           'wait': True, 'wait_device_ops': True, 'started_monotonic': started,
                           'finished_monotonic': time.monotonic(), 'metrics_before_host_extraction': True})
                (directory / 'metrics.txt').write_text(metrics.metrics_report())
                raw['parameters'] = {str(k): tensor_record(B, t) for k, t in mapping.items()}
                raw['outputs'] = [tensor_record(B, t) for t in outputs]
                # Fail at the affected call, retaining actual values and graph before any role rejection.
                import record_validation as V
                roles, actual_argument_gates = V.frame_parameter_roles(raw, input_arrays, V.literals(plugin_root, len(input_arrays[0]['values'])))
                require(all(value == 'PASS' for value in actual_argument_gates.values()), 'ACTUAL_ARGUMENT_GRAPH_VALUE_BINDING')
                raw['live_parameter_roles'] = roles
                raw['live_functionalization'] = validate_frames(raw['frames'], name,
                    {file: hashlib.sha256((Path(plugin_root) / file).read_bytes()).hexdigest()
                     for file in ('exact.py', 'functionalization.py')})
                raw['graph_witness'] = G.bind(proto, printer_proto, hlo, raw['parameters'],
                                             [G.array_signature(t) for t in raw['outputs']])
                P.validate_execution(B, raw)
                require(not frames.errors and not frames.active and raw['profile_restored']
                        and raw['tls_before'] == raw['tls_after'], 'FUNCTIONALIZATION_OBSERVER_RESTORATION')
                raw['status'] = 'OBSERVED'
                return result
            except Exception as error:
                raw.update(status='ERROR', error_type=type(error).__name__, error_message=str(error),
                           traceback=traceback.format_exc())
                raw['numerical_returned'] = returned
                if returned:
                    self.observation_errors.append({'record': '%03d' % self.index, 'reason': str(error)})
                    return result
                raise
            finally:
                raw.setdefault('tls_after', tls(torch))
                B.write(directory / 'raw.json', retained_error(raw) if raw['status'] == 'ERROR' else canonical(raw))
                try:
                    Z.pack_directory(output.parent.parent, directory)
                except Exception as error:
                    self.observation_errors.append({'record': '%03d' % self.index, 'reason': str(error)})
                    self.evidence_limit = 'EVIDENCE_LIMIT' in str(error)
                    if not returned:
                        raise
    return OriginalCallMode()
