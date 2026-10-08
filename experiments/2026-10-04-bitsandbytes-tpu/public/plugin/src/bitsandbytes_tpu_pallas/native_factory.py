"""Admitted lazy trace and serde conversion. No installed API replacement."""
import hashlib
from pathlib import Path
from . import adapter as D
from . import mosaic_compatibility as C

def digest(value):return hashlib.sha256(value.encode()).hexdigest()

def make_call(kernel,output_shape_dtype_fn,bridge,xla,torch,calls):
    D.require(D.sha(bridge.__file__)==D.BRIDGE_SHA,'PINNED_BRIDGE_SOURCE')
    def call(*args):
        D.require(len(args)==3 and all(isinstance(v,torch.Tensor) for v in args),'THREE_POSITIONAL_TENSORS')
        payload,tensor_args=bridge.trace_pallas(kernel,*args,static_argnums=None,static_argnames=None)
        D.require(len(tensor_args)==3 and all(a is b for a,b in zip(args,tensor_args)),'ORIGINAL_OPERAND_OBJECT_ORDER')
        spec=output_shape_dtype_fn(*args);D.require(isinstance(spec,list) and len(spec)==1,'SINGLE_OUTPUT_SPEC')
        shapes=[s for s,_ in spec];dtypes=[d for _,d in spec]
        row={'status':'TRACE_RETURNED','native_api':'_xla_tpu_custom_call','calls':0,
             'original_payload':payload,'original_payload_sha256':digest(payload),
             'metadata':[{'shape':list(v.shape),'dtype':str(v.dtype).removeprefix('torch.')}for v in tensor_args],
             'operand_object_ids':[id(v)for v in tensor_args],
             'output_shapes':[list(s)for s in shapes],'output_dtypes':[str(d).removeprefix('torch.')for d in dtypes],
             'factory_sha256':D.sha(__file__),'converter_sha256':D.sha(C.__file__)}
        calls.append(row)
        try:
            payload,audit=C.convert_payload(payload,digest(payload))
            row.update(status='STARTED',calls=1,payload=payload,payload_sha256=digest(payload),payload_conversion=audit)
        except Exception as error:
            row.update(status='CONVERSION_FAILED',conversion_error={'type':type(error).__name__,'message':str(error)});raise
        outputs=xla._XLAC._xla_tpu_custom_call(tensor_args,payload,shapes,dtypes)
        D.require(len(outputs)==1,'SINGLE_NATIVE_OUTPUT');row['status']='RETURNED'
        return outputs[0]
    return call
