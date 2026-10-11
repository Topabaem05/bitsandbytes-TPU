"""Frozen M5 model choices, generators and independent dense reference functions."""
import ast,copy,hashlib,importlib.util,json,math,types
from pathlib import Path
PROTOCOL = {'format': 'PRIVATE_QLORA_CPU_V1', 'layers': 2, 'width': 256, 'rank': 4, 'alpha': 4, 'adapter_scale': 1, 'batch': 8, 'dtype': 'float32', 'loss': 'mean_squared_error', 'optimizer': {'name': 'SGD', 'lr': 0.01, 'momentum': 0, 'dampening': 0, 'weight_decay': 0, 'nesterov': False, 'maximize': False, 'foreach': None, 'differentiable': False, 'fused': None}, 'steps': 20, 'checkpoint_after_update': 10, 'resume_steps': [11, 20], 'composition': 'base0(x)+B0(A0(x)), then base1(h)+B1(A1(h)); no activation/dropout', 'quantization': {'blocksize': 64, 'quant_type': 'nf4', 'quant_storage': 'uint8', 'compress_statistics': True}, 'base': 'weights,bias,and_all_nested_state_frozen', 'reference': 'PINNED_DEFAULT_PYTHON_BYTE_BODY_PLUS_INDEPENDENT_DENSE_FP32_AUTOGRAD', 'qualified': False}
ADAPTER_NAMES = ['layers.0.A', 'layers.0.B', 'layers.1.A', 'layers.1.B']
GENERATOR = {'base_weight': '((i*17+layer*31)%257-128)/4096', 'base_bias': '((i+layer*3)%17-8)/128', 'adapter_A': '((i*7+layer*3)%17-8)/256', 'adapter_B': 'zero', 'step_x': '((i*13+step*7)%41-20)/32', 'step_target': '((i*11+step*5)%31-15)/64', 'rng': 'NONE'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')

def read(path):
    return json.loads(Path(path).read_text())

def require(value, name):
    if not value:
        raise ValueError(name)

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def tensor_hash(tensor):
    tensor = tensor.detach().cpu().contiguous()
    raw = bytes(tensor.reshape(-1).view(dtype=__import__('torch').uint8).tolist())
    return {'shape': list(tensor.shape), 'dtype': str(tensor.dtype).removeprefix('torch.'), 'sha256': hashlib.sha256(raw).hexdigest()}

def state_hashes(state):
    return {key: tensor_hash(value) for key, value in sorted(state.items())}

def generated(C, kind, layer=0, step=0):
    t = C.torch
    if kind == 'weight':
        return (((t.arange(256 * 256) * 17 + layer * 31) % 257 - 128).float() / 4096).reshape(256, 256)
    if kind == 'bias':
        return ((t.arange(256) + layer * 3) % 17 - 8).float() / 128
    if kind == 'A':
        return (((t.arange(4 * 256) * 7 + layer * 3) % 17 - 8).float() / 256).reshape(4, 256)
    if kind == 'B':
        return t.zeros(256, 4)
    if kind == 'x':
        return (((t.arange(8 * 256) * 13 + step * 7) % 41 - 20).float() / 32).reshape(8, 256)
    if kind == 'target':
        return (((t.arange(8 * 256) * 11 + step * 5) % 31 - 15).float() / 64).reshape(8, 256)
    raise ValueError(kind)

def input_descriptor(C):
    return {'generator': GENERATOR, 'base': [{key: tensor_hash(generated(C, key, layer)) for key in ('weight', 'bias', 'A', 'B')} for layer in range(2)], 'steps': {str(step): {key: tensor_hash(generated(C, key, step=step)) for key in ('x', 'target')} for step in range(1, 21)}}

def default_oracle(C):
    path = C.source / 'bitsandbytes/backends/default/ops.py'
    require(sha(path) == 'e39afd16dca6e6f34a14305ade0ba4acbb49dc2c7a9baf937341d7e33b682819', 'DEFAULT_BYTE_SOURCE')
    namespace = dict(vars(__import__('bitsandbytes.backends.default.ops', fromlist=['ops'])))
    tree = ast.parse(path.read_text())

    def extract(name=None, op=None):
        nodes = [v for v in tree.body if isinstance(v, ast.FunctionDef) and (v.name == name or (op and any((isinstance(d, ast.Call) and d.args and isinstance(d.args[0], ast.Constant) and (d.args[0].value == 'bitsandbytes::' + op) for d in v.decorator_list))))]
        require(len(nodes) == 1, 'DEFAULT_FUNCTION')
        node = copy.deepcopy(nodes[0])
        node.decorator_list = []
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), namespace)
        return namespace[node.name]
    extract(name='_get_4bit_quantize_bounds')
    return (extract(op='quantize_4bit'), extract(op='quantize_blockwise'), extract(name='_dequantize_blockwise_compute'), extract(name='_dequantize_4bit_compute'))

def forward(network, x):
    for layer in network.layers:
        x = layer(x)
    return x

def adapter_parameters(network):
    named = dict(network.named_parameters())
    require([name for name, p in named.items() if p.requires_grad] == ADAPTER_NAMES, 'ADAPTER_MAPPING')
    return [named[name] for name in ADAPTER_NAMES]

def frozen_state(network):
    return {key: value for key, value in network.state_dict().items() if '.base.' in key}

def assert_frozen(C, network, expected):
    require(state_hashes(frozen_state(network)) == expected, 'FROZEN_BASE_STATE')
    for layer in network.layers:
        base = layer.base
        s = base.weight.quant_state
        require(type(base) is C.bnb.nn.Linear4bit and type(base.weight) is C.bnb.nn.Params4bit and (type(s) is C.bnb.functional.QuantState) and (type(s.state2) is C.bnb.functional.QuantState) and s.nested and (base.quant_state is s) and (base.weight.module is base), 'PUBLIC_NESTED_ALIASES')
        require(not base.weight.requires_grad and (not base.bias.requires_grad) and (base.weight.grad is None) and (base.bias.grad is None), 'FROZEN_BASE_GRADIENT')

def optimizer(C, network):
    return C.torch.optim.SGD(adapter_parameters(network), lr=0.01, momentum=0, weight_decay=0)

def optimizer_descriptor(opt):
    state = opt.state_dict()
    require(len(state['param_groups']) == 1 and state['param_groups'][0]['params'] == [0, 1, 2, 3] and (state['state'] == {}), 'SGD_ADAPTER_MAPPING')
    group = state['param_groups'][0]
    require({key: group[key] for key in PROTOCOL['optimizer'] if key != 'name'} == {key: value for key, value in PROTOCOL['optimizer'].items() if key != 'name'}, 'OPTIMIZER_CONFIG')
    return {'class': 'torch.optim.SGD', 'adapter_names': ADAPTER_NAMES, 'state': state}

def compare(C, actual, expected):
    require(actual.shape == expected.shape and actual.dtype == expected.dtype, 'COMPARISON_SHAPE')
    return C.B.numeric(actual.detach().reshape(-1).tolist(), expected.detach().reshape(-1).tolist(), C.tolerance)

def validate_array(C, value, shape):
    require(type(value) is dict and value.get('shape') == shape and (value.get('dtype') == 'float32'), 'STEP_ARRAY_SHAPE')
    require(type(value.get('values')) is list and len(value['values']) == math.prod(shape) and all((type(v) in (int, float) and math.isfinite(v) for v in value['values'])), 'STEP_ARRAY_VALUES')
    return C.torch.tensor(value['values'], dtype=C.torch.float32).reshape(shape)

def dense_replay(C, expected_model):
    """Independent public dense autograd replay from admitted default byte bodies."""
    t = C.torch
    oq4, oq8, od8, od4 = default_oracle(C)
    weights = []
    for index, layer in enumerate(expected_model.layers):
        packed, scales = oq4(generated(C, 'weight', index), 64, 'nf4', t.uint8)
        offset = scales.mean()
        state = layer.base.weight.quant_state
        codes, second = oq8(scales - offset, state.state2.code, 256)
        require(t.equal(packed, layer.base.weight.data) and t.equal(codes, state.absmax) and t.equal(second, state.state2.absmax) and t.equal(offset, state.offset), 'VERIFIER_DEFAULT_BYTE_REFERENCE')
        weights.append(od4(packed.reshape(-1), od8(codes, second, state.state2.code, 256, t.float32) + offset, state.code, 64, (256, 256), t.float32))
    adapters = [t.nn.Parameter(generated(C, key, index)) for index in range(2) for key in ('A', 'B')]
    opt = t.optim.SGD(adapters, lr=0.01)
    result = {}
    for step in range(1, 21):
        opt.zero_grad(set_to_none=True)
        value = generated(C, 'x', step=step)
        for index in range(2):
            a, b = adapters[index * 2:index * 2 + 2]
            value = t.nn.functional.linear(value, weights[index], generated(C, 'bias', index)) + value @ a.t() @ b.t()
        loss = t.nn.functional.mse_loss(value, generated(C, 'target', step=step))
        loss.backward()
        record = {'loss': loss.item(), 'output': C.B.tensor_record(value), 'gradients': {name: C.B.tensor_record(p.grad) for name, p in zip(ADAPTER_NAMES, adapters)}}
        opt.step()
        record['adapters_after'] = {name: C.B.tensor_record(p) for name, p in zip(ADAPTER_NAMES, adapters)}
        result[step] = record
    return result
PROTOCOL_SHA=digest(PROTOCOL)
