#include "pjrt_c_api.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <cstring>
#include <thread>
#include <vector>
struct PJRT_LoadedExecutable {
  int index;
};
struct PJRT_Executable {
  int index;
};
struct PJRT_Error {
  int code;
};
namespace {
PJRT_LoadedExecutable ls[4] = {{0}, {1}, {2}, {3}};
PJRT_Executable es[4] = {{0}, {1}, {2}, {3}};
PJRT_Error error{7};
std::atomic<int> api_gets{0};
std::atomic<int> calls{0}, errors_destroyed{0}, blocked{0}, release_flag{0};
int selected_l = 0, selected_e = 0, mode = 0;
void *expected = nullptr;
std::vector<unsigned char> snapshot;
bool args_ok = true;
PJRT_Api api{};
std::vector<unsigned char> table;
void enter(void *p) {
  ++calls;
  if (p != expected)
    args_ok = false;
  if (p && !snapshot.empty() && memcmp(p, snapshot.data(), snapshot.size()))
    args_ok = false;
}
void error_destroy(PJRT_Error_Destroy_Args *a) {
  ++errors_destroyed;
  assert(a->error == &error);
}
PJRT_Error *initialize(PJRT_Plugin_Initialize_Args *a) {
  enter(a);
  return nullptr;
}
PJRT_Error *compile(PJRT_Client_Compile_Args *a) {
  enter(a);
  if (!a || a->struct_size < PJRT_Client_Compile_Args_STRUCT_SIZE)
    return &error;
  a->executable = &ls[selected_l];
  return mode == 1 ? &error : nullptr;
}
PJRT_Error *getexe(PJRT_LoadedExecutable_GetExecutable_Args *a) {
  enter(a);
  if (!a ||
      a->struct_size < PJRT_LoadedExecutable_GetExecutable_Args_STRUCT_SIZE)
    return &error;
  a->executable = &es[selected_e];
  return nullptr;
}
PJRT_Error *optimized(PJRT_Executable_OptimizedProgram_Args *a) {
  enter(a);
  if (!a ||
      a->struct_size < PJRT_Executable_OptimizedProgram_Args_STRUCT_SIZE ||
      !a->program || a->program->struct_size < PJRT_Program_STRUCT_SIZE)
    return &error;
  static const char bytes[] = {'\x0a', '\x03', 'a', 'b', 'c',
                               '\x12', '\x02', 'x', 'y'};
  static const char format[] = "hlo_with_config";
  PJRT_Program *p = a->program;
  p->format = format;
  p->format_size = sizeof(format) - 1;
  if (!p->code) {
    p->code_size = mode == 5 ? 17 * 1024 * 1024 : sizeof(bytes);
    return nullptr;
  }
  if (p->code_size < sizeof(bytes))
    return &error;
  memcpy(p->code, bytes, sizeof(bytes));
  if (mode == 7)
    p->code = reinterpret_cast<char *>(0xBAD);
  if (mode == 3)
    p->code_size = sizeof(bytes) + 1;
  if (mode == 4) {
    p->format = "mlir";
    p->format_size = 4;
  }
  return nullptr;
}
PJRT_Error *execute(PJRT_LoadedExecutable_Execute_Args *a) {
  enter(a);
  if (!a || a->struct_size < PJRT_LoadedExecutable_Execute_Args_STRUCT_SIZE)
    return &error;
  if (mode == 6) {
    blocked = 1;
    while (!release_flag.load())
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  // Sentinel outputs must pass through without recorder changes.
  if (a->device_complete_events)
    a->device_complete_events[0] = reinterpret_cast<PJRT_Event *>(0x1234);
  if (a->output_lists)
    a->output_lists[0][0] = reinterpret_cast<PJRT_Buffer *>(0x5678);
  return mode == 2 ? &error : nullptr;
}
PJRT_Error *ld(PJRT_LoadedExecutable_Destroy_Args *a) {
  enter(a);
  return nullptr;
}
PJRT_Error *ed(PJRT_Executable_Destroy_Args *a) {
  enter(a);
  return nullptr;
}
PJRT_Error *del(PJRT_LoadedExecutable_Delete_Args *a) {
  enter(a);
  return nullptr;
}
PJRT_Error *deserialize(PJRT_Executable_DeserializeAndLoad_Args *a) {
  enter(a);
  if (!a ||
      a->struct_size < PJRT_STRUCT_SIZE(PJRT_Executable_DeserializeAndLoad_Args,
                                        loaded_executable))
    return &error;
  a->loaded_executable = &ls[selected_l];
  return nullptr;
}
} // namespace
extern "C" void FakeSetup(int m, size_t bytes) {
  mode = m;
  api = {};
  api.struct_size = bytes ? bytes : sizeof(PJRT_Api) + 32;
  api.pjrt_api_version = {PJRT_Api_Version_STRUCT_SIZE, nullptr,
                          m == 8 ? 1 : PJRT_API_MAJOR, PJRT_API_MINOR};
  api.extension_start = reinterpret_cast<PJRT_Extension_Base *>(0x99);
  api.PJRT_Error_Destroy = error_destroy;
  api.PJRT_Plugin_Initialize = initialize;
  api.PJRT_Client_Compile = compile;
  api.PJRT_LoadedExecutable_GetExecutable = getexe;
  api.PJRT_Executable_OptimizedProgram = optimized;
  api.PJRT_LoadedExecutable_Execute = execute;
  api.PJRT_LoadedExecutable_Destroy = ld;
  api.PJRT_Executable_Destroy = ed;
  api.PJRT_LoadedExecutable_Delete = del;
  api.PJRT_Executable_DeserializeAndLoad = deserialize;
  table.resize(api.struct_size);
  memset(table.data(), 0xA7, table.size());
  memcpy(table.data(), &api,
         table.size() < sizeof api ? table.size() : sizeof api);
}
extern "C" const PJRT_Api *GetPjrtApi() {
  ++api_gets;
  return reinterpret_cast<const PJRT_Api *>(table.data());
}
extern "C" void FakeExpect(void *p, size_t bytes) {
  expected = p;
  snapshot.clear();
  if (p && bytes) {
    auto b = static_cast<unsigned char *>(p);
    snapshot.assign(b, b + bytes);
  }
}
extern "C" void FakeSelect(int l, int e) {
  selected_l = l;
  selected_e = e;
}
extern "C" int FakeCalls() { return calls.load(); }
extern "C" int FakeArgsOK() { return args_ok; }
extern "C" int FakeErrorsDestroyed() { return errors_destroyed.load(); }
extern "C" PJRT_Error *FakeError() { return &error; }
extern "C" int FakeBlocked() { return blocked.load(); }
extern "C" void FakeRelease() { release_flag = 1; }
extern "C" void FakeMode(int m) { mode = m; }
extern "C" PJRT_LoadedExecutable *FakeCurrentLoaded() {
  return &ls[selected_l];
}
extern "C" PJRT_Executable *FakeCurrentExecutable() { return &es[selected_e]; }

extern "C" int FakeGetCalls() { return api_gets.load(); }
