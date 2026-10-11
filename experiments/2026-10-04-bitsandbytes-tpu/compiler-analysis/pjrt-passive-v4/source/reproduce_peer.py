"""Repeat the peer counterexample with exact frozen v3 fake inputs and v4 code."""
import hashlib,json,os,shutil,sys,tempfile,urllib.request
from pathlib import Path
sys.dont_write_bytecode=True;os.environ['PYTHONDONTWRITEBYTECODE']='1';ROOT=Path(__file__).resolve().parents[1];FIXTURE=ROOT/'proof/repro-inputs';sys.path.insert(0,str(ROOT/'source'))
import run_controls as R,verify as V
out=Path(sys.argv[1])
with tempfile.TemporaryDirectory(prefix='m6-pjrt-interleaving-',dir='/tmp')as td:
 b=Path(td);body=urllib.request.urlopen(R.HEADER_URL,timeout=30).read();assert hashlib.sha256(body).hexdigest()==R.HEADER_SHA;(b/'pjrt_c_api.h').write_bytes(body)
 provider=(FIXTURE/'fake_provider.cpp').read_text();needle='  static const char bytes[] =';injection='''  if (mode == 9) {
    blocked = 1;
    while (!release_flag.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    return &error;
  }
''';assert provider.count(needle)==1;provider=provider.replace(needle,injection+needle);(b/'private_fake_provider.cpp').write_text(provider)
 harness=(FIXTURE/'harness.cpp').read_text();needle='      if (scenario == "overlap") {';injection='''      if (scenario == "optimized_interleaving") {
        auto blocked = symbol<int (*)()>(vendor, "FakeBlocked");
        auto release = symbol<void (*)()>(vendor, "FakeRelease");
        PJRT_Program out = {};
        out.struct_size = PJRT_Program_STRUCT_SIZE;
        out.code = nullptr;
        PJRT_Executable_OptimizedProgram_Args oa = {
            PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE, nullptr,
            executable, &out};
        mode(9);
        expect(&oa, sizeof oa);
        ++expected_calls;
        std::thread t([&] {
          assert(api->PJRT_Executable_OptimizedProgram(&oa) == error());
        });
        while (!blocked())
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        exec();
        assert(finalize() == 1);
        release();
        t.join();
      } else if (scenario == "overlap") {''';assert harness.count(needle)==1;harness=harness.replace(needle,injection);(b/'private_harness.cpp').write_text(harness)
 common=[shutil.which('clang++'),'-std=c++17','-Wall','-Wextra','-Werror','-pthread','-I',str(b),'-I',str(ROOT/'source')];shim=b/'recorder.dylib';fake=b/'fake.dylib';exe=b/'harness'
 R.run(common+['-dynamiclib','-fPIC',ROOT/'source/recorder.cpp','-o',shim]);R.run(common+['-dynamiclib','-fPIC',b/'private_fake_provider.cpp','-o',fake]);R.run(common+[b/'private_harness.cpp','-o',exe]);capture=b/'capture';execution=R.run([exe,'optimized_interleaving',fake,shim,capture]);V.seal(capture,0);report=V.verify(capture)
 evidence={'kind':'REAL_COMPILED_PRIVATE_PROVIDER_INTERLEAVING_COUNTEREXAMPLE','candidate_recorder_sha256':hashlib.sha256((ROOT/'source/recorder.cpp').read_bytes()).hexdigest(),'candidate_verifier_sha256':hashlib.sha256((ROOT/'source/verify.py').read_bytes()).hexdigest(),'private_provider_sha256':hashlib.sha256(provider.encode()).hexdigest(),'private_harness_sha256':hashlib.sha256(harness.encode()).hexdigest(),'stdout':execution.stdout,'journal':(capture/'journal.jsonl').read_text(),'footer':json.loads((capture/'footer.json').read_text()),'verifier':report,'bulk_bytes_deleted':sum(p.stat().st_size for p in b.rglob('*')if p.is_file()),'provider_calls':0,'TPU':'NOT_RUN','actual_linux':'NOT_RUN','source_modified':False}
 # Baseline remains passing with the same compiled private provider/harness.
 cold=b/'cold';correct=R.run([exe,'cold',fake,shim,cold]);V.seal(cold,0);evidence['correct_cold_status']=V.verify(cold)['status'];assert evidence['correct_cold_status']=='PASS'
 assert report['status']=='UNKNOWN';evidence['expected_stale_payload_claim']='UNKNOWN';evidence['regression_status']='PASS'
out.write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n');print(json.dumps({'correct_cold':'PASS','interleaving_actual':report['status'],'expected':'UNKNOWN','bulk_deleted':not Path(td).exists()}))
