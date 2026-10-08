"""Unadopted direct native recorder. Mirrors the pinned factory algorithm without replacing any installed method."""
import hashlib
import json
from pathlib import Path
import time

import verifier as V
import graph_binding as G
import mosaic_compatibility as C

BRIDGE_SHA='c347f8fcb4844fa8849109680ad81b94e221a2c8b011462553b48ec9f67e6338'

def make_recording_kernel(kernel,output_shape_dtype_fn,bridge,xla,torch):
    V.require(hashlib.sha256(Path(bridge.__file__).read_bytes()).hexdigest()==BRIDGE_SHA,'PINNED_BRIDGE_SOURCE')
    calls=[]
    def call(*args):
        V.require(len(args)==3 and all(isinstance(value,torch.Tensor) for value in args),'THREE_POSITIONAL_TENSORS')
        payload,tensor_args=bridge.trace_pallas(kernel,*args,static_argnums=None,static_argnames=None)
        V.require(len(tensor_args)==3 and all(a is b for a,b in zip(args,tensor_args)),'ORIGINAL_OPERAND_OBJECT_ORDER')
        output_spec=output_shape_dtype_fn(*args)
        V.require(isinstance(output_spec,list) and len(output_spec)==1,'SINGLE_OUTPUT_SPEC')
        shapes=[shape for shape,_ in output_spec];dtypes=[dtype for _,dtype in output_spec]
        row={'status':'STARTED','native_api':'_xla_tpu_custom_call','calls':1,'payload':payload,'payload_sha256':V.digest(payload),'metadata':[{'shape':list(v.shape),'dtype':str(v.dtype).removeprefix('torch.')} for v in tensor_args],'tensor_args':tuple(tensor_args),'output_shapes':[list(v) for v in shapes],'output_dtypes':[str(v).removeprefix('torch.') for v in dtypes]}
        row.update(status='TRACE_RETURNED',calls=0,original_payload=payload,original_payload_sha256=V.digest(payload))
        calls.append(row)
        try:
            payload,audit=C.convert_payload(payload,hashlib.sha256(payload.encode()).hexdigest())
            row.update(payload=payload,payload_sha256=V.digest(payload),payload_conversion=audit,status='STARTED',calls=1)
        except Exception as error:
            row.update(status='CONVERSION_FAILED',conversion_error={'type':type(error).__name__,'message':str(error)})
            raise
        # Exact captured payload and original ordered tensors go directly to the actual native binding.
        outputs=xla._XLAC._xla_tpu_custom_call(tensor_args,payload,shapes,dtypes)
        V.require(len(outputs)==1,'SINGLE_NATIVE_OUTPUT');row['status']='RETURNED'
        return outputs[0]
    return call,calls


def execution_evidence(metrics,B,P):
    """Validate exactly the JSON evidence retained for recovered inspection."""
    profile,_=B.load_spec()
    evidence={'counters':{key:metrics.counter_value(key) for key in metrics.counter_names()},'execution_metrics':{key:value for key in profile['execution_metrics'] if (value:=metrics.metric_data(key)) is not None}}
    evidence=json.loads(json.dumps(evidence,allow_nan=False))
    P.validate_execution(B,evidence)
    return evidence

def capture_actual(actual,row,xla,xm,metrics,B,P,graph_name):
    """Capture one returned actual graph. No same-device reference runs in its measured interval."""
    V.require(row['status']=='RETURNED','NATIVE_RETURNED')
    context=xla._XLAC.lowering.LoweringContext(graph_name);context.build([actual])
    proto_text=context.hlo_text();proto=bytes(context.hlo())
    # The pinned context text API emits protobuf text. Use the tensor HLO printer
    # for HLO syntax, and bind its proto to the mapped context before sync.
    hlo=xla._XLAC._get_xla_tensors_hlo([actual]);printer_proto=bytes(xla._XLAC._get_xla_tensors_hlo_proto([actual]))
    parameters={str(key):B.tensor_record(value) for key,value in context.parameter_id_tensor_mapping().items()}
    graph_binding=G.bind(proto,printer_proto,hlo,parameters,row['payload'])
    graph_hash=xla._XLAC._get_graph_hash([actual]).hex()
    # GetGraphHash uses force_ltc_data=false; sync uses the default config. This is a pre-sync ID, not an executable ID.
    metrics.clear_all();started=time.monotonic()
    xla._XLAC._xla_sync_multi([actual],devices=[],wait=True,sync_xla_data=True)
    xm.wait_device_ops();finished=time.monotonic()
    evidence=execution_evidence(metrics,B,P)
    execution={'api':'_xla_sync_multi','targets':['actual'],'wait':True,'wait_device_ops':True,'metrics_after_sync_before_output':True,'other_execution_in_interval':False,'graph_hash':graph_hash,'graph_hash_binding':'PRE_SYNC_IDENTIFIER_NOT_EXECUTABLE_LINK','started_monotonic':started,'finished_monotonic':finished}
    # Device-to-host tensor records are outside the execution evidence interval.
    output=B.tensor_record(actual)
    operands=[B.tensor_record(value) for value in row['tensor_args']]
    boundary={key:value for key,value in row.items() if key not in ('metadata','tensor_args')}
    boundary.update(recorder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),graph_hash=graph_hash,operands=[{'shape':v['shape'],'dtype':v['dtype'],'value_sha256':V.digest(v)} for v in operands])
    return {'hlo':hlo,'proto':proto,'proto_text':proto_text,'printer_proto':printer_proto,'graph_binding':graph_binding,'parameters':parameters,'output':output,'native_boundary':boundary,'execution':execution,**evidence}
