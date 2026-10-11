"""Verify downloaded complete bodies or retained excerpts. Never accesses network."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RANGES={
 'hlo_schema':[(401,428),(553,566),(751,757)],
 'computation_client':[(114,121)],
 'xla_computation':[(28,31)],
 'xla_data':[(27,60),(126,155),(326,377)],
 'workspace':[(45,63)],
 'debug_util':[(245,310),(379,425),(432,459),(486,519)],
 'pjrt_client':[(643,671)],
 'graph_executor':[(1106,1128),(1148,1168),(1230,1282),(1431,1454),(1488,1497),(1581,1604)],
 'hash':[(10,60),(80,85)],
 'hash_header':[(21,33),(148,155)],
 'int128':[(103,107),(157,163)],
 'xla_util':[(120,138)],
 'bindings':[(3438,3451)],
 'registry':[(136,147)],
 'c_api':[(657,675),(1450,1461),(1516,1548),(1655,1692)],
}
TOKENS={
 'computation_client':['XLA_ASSIGN_OR_THROW(program_shape_, computation_.GetProgramShape());'],
 'xla_computation':['return ProgramShape::FromProto(proto_.host_program_shape());'],
 'xla_data':['TUPLE = 13;','repeated ShapeProto parameters = 1;','ShapeProto result = 2;','repeated ShapeProto tuple_shapes = 4;'],
 'hlo_schema':['message HloModuleProto {','string name = 1;','int64 id = 5;','HloModuleProto hlo_module = 1;'],
 'workspace':["xla_hash = '92f7b5952dd585c5be17c9a5caad27407005b513'"],
 'debug_util':['torch::lazy::HashToString(graph_hash)','GetDeterministicSerializedModuleProto(','torch::lazy::Hash(serialized_computation)'],
 'pjrt_client':['client_->CompileAndLoad(instance.computation,','executable->GetHloModules()','xla::XlaComputation(hlo_modules[0]->ToProto())'],
 'graph_executor':['/*graph_hash=*/coll.hash','computation.GetProgramShape()','XlaHelpers::WrapXlaComputation','DebugUtil::post_compilation_analysis(computations[0])','GetComputationCache()->Add(coll.hash, cached_computation)','client->ExecuteComputation(*async->cached_computation->computation,'],
 'hash':['0xc6a4a7935bd1e995','0xc2b2ae3d27d4eb4f','std::memcpy(&v, *data, sizeof(v))',"std::setfill('0') << std::setw(16)"],
 'hash_header':['class TORCH_API hash_t : public c10::uint128','return DataHash(value.data(), value.size());'],
 'int128':['uint64_t lo_;','uint64_t hi_;',': lo_(bottom), hi_(top)'],
 'xla_util':['const xla::HloModuleProto& hlo_proto','SerializeToBufferDeterministic(hlo_proto, serialized.data(),','return serialized;'],
 'bindings':['std::string bin((const char*)&hash, sizeof(hash))','return py::bytes(bin);'],
 'registry':['env::kEnvTpuLibraryPath','pjrt::LoadPjrtPlugin("tpu", tpu_library_path)'],
 'c_api':['PJRT_LoadedExecutable* executable;  // out','PJRT_Executable* executable;','PJRT_Executable_OptimizedProgram','PJRT_LoadedExecutable_Execute'],
}
def sha(b):return hashlib.sha256(b).hexdigest()
def run():
 sources=json.loads((ROOT/'sources.json').read_text());results=[]
 for name,ranges in RANGES.items():
  source=ROOT/(name+'.source'); info=sources[name]
  excerpt_path=ROOT/(name+'.excerpt.json')
  if source.exists():
   b=source.read_bytes();assert len(b)==info['bytes'] and sha(b)==info['sha256'],name+' full body seal'
   text=b.decode();assert all(t in text for t in TOKENS[name]),name+' required source expression'
   lines=text.splitlines(keepends=True);sections=[]
   for start,end in ranges:
    body=''.join(lines[start-1:end]);sections.append({'start':start,'end':end,'text':body,'sha256':sha(body.encode())})
   excerpt_path.write_text(json.dumps({'source':info,'sections':sections},indent=2)+'\n')
   level='COMPLETE_BODY_SHA256_AND_EXACT_EXPRESSIONS_VERIFIED'
  else:
   excerpt=json.loads(excerpt_path.read_text());assert excerpt['source']==info
   for section in excerpt['sections']:assert sha(section['text'].encode())==section['sha256']
   level='RETAINED_EXCERPT_HASH_VERIFIED_FULL_BODY_PREVIOUSLY_CHECKED'
  results.append({'source':name,'status':'PASS','verification':level,'complete_body_sha256':info['sha256']})
 return {'kind':'M6_PINNED_SOURCE_PROOF_V2','status':'PASS','sources':results,'libtpu_internal_revision':'UNKNOWN','actual_runtime_behavior':'NOT_RUN'}
if __name__=='__main__':
 report=run();print(json.dumps(report,indent=2))
