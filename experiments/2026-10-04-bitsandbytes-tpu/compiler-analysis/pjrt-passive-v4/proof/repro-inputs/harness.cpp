#include "pjrt_c_api.h"
#include "recorder.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <dlfcn.h>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <time.h>
#include <unistd.h>
#include <vector>
uint64_t now() {
  struct timespec t{};
  if (clock_gettime(CLOCK_MONOTONIC, &t) != 0)
    return UINT64_MAX;
  return uint64_t(t.tv_sec) * 1000000000 + uint64_t(t.tv_nsec);
}
template <class F> F symbol(void *p, const char *n) {
  auto f = reinterpret_cast<F>(dlsym(p, n));
  assert(f);
  return f;
}
int main(int argc, char **argv) {
  assert(argc == 5);
  std::string scenario = argv[1];
  void *vendor = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  assert(vendor);
  auto setup = symbol<void (*)(int, size_t)>(vendor, "FakeSetup");
  auto expect = symbol<void (*)(void *, size_t)>(vendor, "FakeExpect");
  auto select = symbol<void (*)(int, int)>(vendor, "FakeSelect");
  auto calls = symbol<int (*)()>(vendor, "FakeCalls");
  auto args_ok = symbol<int (*)()>(vendor, "FakeArgsOK");
  auto destroyed = symbol<int (*)()>(vendor, "FakeErrorsDestroyed");
  auto error = symbol<PJRT_Error *(*)()>(vendor, "FakeError");
  auto getvendor = symbol<const PJRT_Api *(*)()>(vendor, "GetPjrtApi");
  auto mode = symbol<void (*)(int)>(vendor, "FakeMode");
  int m = scenario == "compile_error"    ? 1
          : scenario == "execute_error"  ? 2
          : scenario == "changed_length" ? 3
          : scenario == "changed_format" ? 4
          : scenario == "oversize_query" ? 5
          : (scenario == "inflight" || scenario == "overlap" ||
             scenario == "overlap_destroy" || scenario == "deadline_inflight")
              ? 6
          : scenario == "changed_pointer" ? 7
          : scenario == "unsupported_abi" ? 8
                                          : 0;
  size_t short_size = scenario == "short_api"
                          ? PJRT_STRUCT_SIZE(PJRT_Api, PJRT_Client_Compile)
                          : 0;
  setup(m, short_size);
  void *shim = dlopen(argv[3], RTLD_NOW | RTLD_LOCAL);
  assert(shim);
  auto config = symbol<decltype(&M6TraceConfigure)>(shim, "M6TraceConfigure");
  auto finalize = symbol<decltype(&M6TraceFinalize)>(shim, "M6TraceFinalize");
  auto phase = symbol<decltype(&M6TraceSetPhase)>(shim, "M6TraceSetPhase");
  auto get = symbol<const PJRT_Api *(*)()>(shim, "GetPjrtApi");
  size_t max_entries = scenario == "entry_quota"      ? 4
                       : scenario == "entry_boundary" ? 6
                                                      : 1024,
         total = scenario == "byte_quota"      ? 3 * 1024 * 1024 + 5
                 : scenario == "byte_boundary" ? 3 * 1024 * 1024 + 25
                                               : 128 * 1024 * 1024;
  assert(config(scenario == "self_recursion" ? argv[3] : argv[2], argv[4],
                "fixture", "fixture-recorder",
                scenario == "expired"             ? now() - 1
                : scenario == "deadline_inflight" ? now() + 100000000
                                                  : now() + 30ULL * 1000000000,
                max_entries, total, 16 * 1024 * 1024) == 0);
  if (scenario == "self_recursion") {
    assert(get() == nullptr);
    puts("PASS self_recursion");
    return 0;
  }
  const PJRT_Api *api = get();
  assert(api);
  const PJRT_Api *original = getvendor();
  assert(api != original);
  assert(api->struct_size == original->struct_size);
  // Verify every byte outside hook slots, including unknown tail bytes.
  std::vector<unsigned char> expected(
      reinterpret_cast<const unsigned char *>(original),
      reinterpret_cast<const unsigned char *>(original) +
          original->struct_size);
#define SLOT(name)                                                             \
  if (original->struct_size >= PJRT_STRUCT_SIZE(PJRT_Api, name))               \
  memcpy(expected.data() + offsetof(PJRT_Api, name),                           \
         reinterpret_cast<const char *>(api) + offsetof(PJRT_Api, name),       \
         sizeof api->name)
  SLOT(PJRT_Client_Compile);
  SLOT(PJRT_LoadedExecutable_GetExecutable);
  SLOT(PJRT_Executable_OptimizedProgram);
  SLOT(PJRT_LoadedExecutable_Execute);
  SLOT(PJRT_LoadedExecutable_Destroy);
  SLOT(PJRT_Executable_Destroy);
  SLOT(PJRT_LoadedExecutable_Delete);
  SLOT(PJRT_Executable_DeserializeAndLoad);
#undef SLOT
  assert(memcmp(api, expected.data(), expected.size()) == 0);
  assert(get() == api);
  assert(symbol<int (*)()>(vendor, "FakeGetCalls")() == 2);
  assert(phase("fixture-phase") == 0);
  int expected_calls = 0;
  auto invoke = [&](auto fn, auto *args) {
    size_t size = args ? args->struct_size : 0;
    expect(args, size);
    ++expected_calls;
    return fn(args);
  };
  std::string submitted("module\0whole", 12);
  PJRT_Program p = {PJRT_Program_STRUCT_SIZE, nullptr, submitted.data(),
                    submitted.size(),         "mlir",  4};
  PJRT_Client_Compile_Args ca = {};
  ca.struct_size = PJRT_Client_Compile_Args_STRUCT_SIZE;
  ca.program = &p;
  ca.executable = reinterpret_cast<PJRT_LoadedExecutable *>(0xDEAD);
  if (scenario == "short_args") {
    size_t tiny = sizeof(size_t);
    auto a = reinterpret_cast<PJRT_Client_Compile_Args *>(&tiny);
    assert(invoke(api->PJRT_Client_Compile, a) == error());
    assert(invoke(api->PJRT_Client_Compile,
                  static_cast<PJRT_Client_Compile_Args *>(nullptr)) == error());
  } else {
    if (scenario == "write_failure")
      assert(chmod(argv[4], 0500) == 0);
    PJRT_Error *ce = invoke(api->PJRT_Client_Compile, &ca);
    if (scenario == "write_failure")
      assert(chmod(argv[4], 0700) == 0);
    assert(ce == (m == 1 ? error() : nullptr));
    assert(ca.executable ==
           symbol<PJRT_LoadedExecutable *(*)()>(vendor, "FakeCurrentLoaded")());
    if (scenario == "short_api" || scenario == "compile_error") {
    } else {
      PJRT_LoadedExecutable *loaded = ca.executable;
      auto getexe = [&](PJRT_LoadedExecutable *l) {
        PJRT_LoadedExecutable_GetExecutable_Args ga = {
            PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE, nullptr, l,
            nullptr};
        assert(invoke(api->PJRT_LoadedExecutable_GetExecutable, &ga) ==
               nullptr);
        assert(ga.executable == symbol<PJRT_Executable *(*)()>(
                                    vendor, "FakeCurrentExecutable")());
        return ga.executable;
      };
      auto optimized = [&](PJRT_Executable *exe) {
        PJRT_Program
            out; // Deliberately leave query output fields uninitialized.
        out.struct_size = PJRT_Program_STRUCT_SIZE;
        out.extension_start = nullptr;
        out.code = nullptr;
        PJRT_Executable_OptimizedProgram_Args oa = {
            PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE, nullptr, exe,
            &out};
        assert(invoke(api->PJRT_Executable_OptimizedProgram, &oa) == nullptr);
        if (out.code_size > 16 * 1024 * 1024)
          return;
        std::vector<char> bytes(out.code_size +
                                (scenario == "capacity_larger" ? 7 : 0));
        out.code = bytes.data();
        if (scenario == "capacity_larger")
          out.code_size = bytes.size();
        assert(invoke(api->PJRT_Executable_OptimizedProgram, &oa) == nullptr);
        assert(bytes[0] == '\x0a');
        if (scenario == "changed_pointer")
          assert(out.code == reinterpret_cast<char *>(0xBAD));
      };
      PJRT_Executable *executable = getexe(loaded);
      if (scenario == "short_program") {
        size_t tiny = sizeof(size_t);
        PJRT_Executable_OptimizedProgram_Args oa = {
            PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE, nullptr,
            executable, reinterpret_cast<PJRT_Program *>(&tiny)};
        assert(invoke(api->PJRT_Executable_OptimizedProgram, &oa) == error());
      } else if (scenario == "unknown_fill") {
        char b[9];
        PJRT_Program out = {PJRT_Program_STRUCT_SIZE, nullptr, b, 9,
                            "hlo_with_config",        15};
        PJRT_Executable_OptimizedProgram_Args oa = {
            PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE, nullptr,
            executable, &out};
        assert(invoke(api->PJRT_Executable_OptimizedProgram, &oa) == nullptr);
      } else
        optimized(executable);
      if (scenario == "stale_optimized") {
        mode(4);
        optimized(executable);
      }
      if (scenario == "multiple_executables") {
        select(0, 1);
        optimized(getexe(loaded));
      }
      if (scenario == "wrong_loaded") {
        select(1, 1);
        assert(invoke(api->PJRT_Client_Compile, &ca) == nullptr);
        loaded = ca.executable;
      }
      if (scenario == "wrong_executable") {
        select(1, 1);
        assert(invoke(api->PJRT_Client_Compile, &ca) == nullptr);
        loaded = ca.executable;
        getexe(loaded);
        optimized(executable);
      }
      if (scenario == "deserialize") {
        select(1, 1);
        PJRT_Executable_DeserializeAndLoad_Args da = {};
        da.struct_size = PJRT_Executable_DeserializeAndLoad_Args_STRUCT_SIZE;
        da.serialized_executable = "serialized";
        da.serialized_executable_size = 10;
        assert(invoke(api->PJRT_Executable_DeserializeAndLoad, &da) == nullptr);
        loaded = da.loaded_executable;
        optimized(getexe(loaded));
      }
      if (scenario == "duplicate_loaded") {
        assert(invoke(api->PJRT_Client_Compile, &ca) == nullptr);
      }
      if (scenario == "duplicate_executable") {
        getexe(loaded);
      }
      if (scenario == "destroy_reuse") {
        PJRT_Executable_Destroy_Args ed = {
            PJRT_Executable_Destroy_Args_STRUCT_SIZE, nullptr, executable};
        assert(invoke(api->PJRT_Executable_Destroy, &ed) == nullptr);
        PJRT_LoadedExecutable_Destroy_Args ld = {
            PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE, nullptr, loaded};
        assert(invoke(api->PJRT_LoadedExecutable_Destroy, &ld) == nullptr);
        assert(invoke(api->PJRT_Client_Compile, &ca) == nullptr);
        loaded = ca.executable;
        executable = getexe(loaded);
        optimized(executable);
      }
      if (scenario == "delete_before") {
        PJRT_LoadedExecutable_Delete_Args da = {
            PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE, nullptr, loaded};
        assert(invoke(api->PJRT_LoadedExecutable_Delete, &da) == nullptr);
      }
      PJRT_Buffer *buffer = nullptr;
      PJRT_Buffer **outputs = &buffer;
      PJRT_Event *event = nullptr;
      PJRT_LoadedExecutable_Execute_Args ea = {};
      ea.struct_size = PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE;
      ea.executable = loaded;
      ea.num_devices = 1;
      ea.output_lists = &outputs;
      ea.device_complete_events = &event;
      auto exec = [&] {
        PJRT_Error *e = invoke(api->PJRT_LoadedExecutable_Execute, &ea);
        assert(e == (m == 2 ? error() : nullptr));
        assert(event == reinterpret_cast<PJRT_Event *>(0x1234));
        assert(buffer == reinterpret_cast<PJRT_Buffer *>(0x5678));
      };
      if (scenario == "overlap") {
        auto blocked = symbol<int (*)()>(vendor, "FakeBlocked");
        auto release = symbol<void (*)()>(vendor, "FakeRelease");
        ea.output_lists = nullptr;
        ea.device_complete_events = nullptr;
        expect(&ea, sizeof ea);
        int baseline = expected_calls;
        expected_calls += 2;
        std::thread t1([&] {
          assert(api->PJRT_LoadedExecutable_Execute(&ea) == nullptr);
        });
        while (!blocked())
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        std::thread t2([&] {
          assert(api->PJRT_LoadedExecutable_Execute(&ea) == nullptr);
        });
        while (calls() < baseline + 2)
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        assert(finalize() == 1);
        release();
        t1.join();
        t2.join();
      } else if (scenario == "overlap_destroy") {
        auto blocked = symbol<int (*)()>(vendor, "FakeBlocked");
        auto release = symbol<void (*)()>(vendor, "FakeRelease");
        std::thread t(exec);
        while (!blocked())
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        PJRT_LoadedExecutable_Destroy_Args da = {
            PJRT_LoadedExecutable_Destroy_Args_STRUCT_SIZE, nullptr, loaded};
        assert(invoke(api->PJRT_LoadedExecutable_Destroy, &da) == nullptr);
        assert(finalize() == 1);
        release();
        t.join();
      } else if (scenario == "inflight" || scenario == "deadline_inflight") {
        auto blocked = symbol<int (*)()>(vendor, "FakeBlocked");
        auto release = symbol<void (*)()>(vendor, "FakeRelease");
        std::thread t(exec);
        while (!blocked())
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        assert(finalize() == 1);
        if (scenario == "deadline_inflight")
          std::this_thread::sleep_for(std::chrono::milliseconds(150));
        release();
        t.join();
      } else
        exec();
      if (scenario == "warm_reuse")
        exec();
      if (scenario == "delete_after") {
        PJRT_LoadedExecutable_Delete_Args da = {
            PJRT_LoadedExecutable_Delete_Args_STRUCT_SIZE, nullptr, loaded};
        assert(invoke(api->PJRT_LoadedExecutable_Delete, &da) == nullptr);
      }
      if (scenario == "expired_inflight") {
        mode(6);
        auto blocked = symbol<int (*)()>(vendor, "FakeBlocked");
        auto release = symbol<void (*)()>(vendor, "FakeRelease");
        std::thread t(exec);
        while (!blocked())
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
        release();
        t.join();
      }
    }
  }
  assert(calls() == expected_calls);
  assert(args_ok());
  assert(destroyed() == 0);
  int f = finalize();
  assert(f == 0 || f == -1);
  assert(finalize() == f);
  printf("PASS %s calls=%d finalize=%d\n", scenario.c_str(), expected_calls, f);
}
