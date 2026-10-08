"""Minimal injected Torch/XLA 2.9 runner. Caller must admit runtime/client/source/ownership before use."""
from collector import COMPILE_KEYS,EXECUTE_KEYS,require


class XlaRunner:
    def __init__(self,torch,xla,xm,metrics,device,canonical_inputs,audit):
        self.torch,self.xla,self.xm,self.metric_api=torch,xla,xm,metrics
        self.device,self.inputs,self._audit=device,tuple(canonical_inputs),audit
        require(self.inputs and all(isinstance(v,torch.Tensor) and v.device==device and v.device.type=='xla' for v in self.inputs),'ADMITTED_CANONICAL_XLA_INPUTS')
    def audit(self):return self._audit()
    def drain(self):self.xm.wait_device_ops()
    def clear_metrics(self):self.metric_api.clear_all()
    def _sync(self,targets):
        raw=[]
        for value in targets:
            require(isinstance(value,self.torch.Tensor) and value.device==self.device and value.device.type=='xla','SAME_DEVICE_ACTUAL_TENSOR')
            if self.torch._is_functional_tensor(value):
                self.torch._functionalize_sync(value);value=self.torch._from_functional_tensor(value)
            raw.append(value)
        self.xla._XLAC._xla_sync_multi(raw,devices=[],wait=True,sync_xla_data=True)
        self.xm.wait_device_ops()
    def sync_setup(self):
        self._sync(self.inputs);return {'api':'_xla_sync_multi','targets':['canonical_inputs'],'wait':True,'sync_xla_data':True,'wait_device_ops':True}
    def sync_actual(self,target):
        self._sync([target]);return {'api':'_xla_sync_multi','targets':['actual'],'wait':True,'sync_xla_data':True,'wait_device_ops':True}
    def metrics(self):return {name:self.metric_api.metric_data(name) for name in COMPILE_KEYS+EXECUTE_KEYS}
    def counters(self):return {name:self.metric_api.counter_value(name) for name in self.metric_api.counter_names()}
    def memory_info(self):
        getter=getattr(self.xm,'get_memory_info',None)
        if not callable(getter):raise NotImplementedError('PINNED_MEMORY_API_UNAVAILABLE')
        return getter(self.device)
