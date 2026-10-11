#include "recorder.h"
#include "pjrt_c_api.h"
#include "sha256.h"
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#include <vector>

extern "C" const PJRT_Api *GetPjrtApi();
namespace {
constexpr size_t MiB = 1024 * 1024, HardEntries = 1024, HardTotal = 128 * MiB,
                 HardFile = 16 * MiB;
constexpr size_t MetadataReserve = 3 * MiB, JournalLimit = MiB;
uint64_t now() {
  struct timespec t{};
  if (clock_gettime(CLOCK_MONOTONIC, &t) != 0)
    return UINT64_MAX;
  return uint64_t(t.tv_sec) * 1000000000 + uint64_t(t.tv_nsec);
}
std::string q(const std::string &s) {
  std::string o = "\"";
  for (unsigned char c : s) {
    if (c == '"' || c == '\\') {
      o += '\\';
      o += char(c);
    } else if (c >= 32 && c < 127)
      o += char(c);
    else {
      char b[7];
      snprintf(b, 7, "\\u%04x", c);
      o += b;
    }
  }
  return o + '"';
}
bool label(const char *s) {
  if (!s || !*s || strlen(s) > 64)
    return false;
  for (const char *p = s; *p; ++p)
    if (!((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
          (*p >= '0' && *p <= '9') || *p == '_' || *p == '-'))
      return false;
  return true;
}
struct Handle {
  std::string id, origin, replaced_alive;
  uint64_t born = 0;
  bool alive = true, deleted = false, ambiguous = false;
  size_t busy = 0;
  std::shared_ptr<Handle> parent;
  size_t query_size = 0;
  std::string query_format;
  uint64_t query_seq = 0, optimized_epoch = 0;
  bool query_valid = false;
};
struct Ref {
  std::string file, sha;
  size_t bytes = 0;
  bool ok = false;
  std::string json() const {
    return ok ? "{\"file\":" + q(file) + ",\"bytes\":" + std::to_string(bytes) +
                    ",\"sha256\":" + q(sha) + "}"
              : "null";
  }
};
struct State {
  std::mutex mu;
  bool configured = false, initialized = false, failed = false,
       finalized = false;
  std::string failure;
  std::string provider, out, nonce, recorder_id, phase = "UNSET", provider_sha;
  uint64_t deadline = 0, seq = 0, calls = 0;
  size_t max_entries = 0, max_total = 0, max_file = 0, payload_bytes = 0,
         files = 3, records = 0, journal_bytes = 0;
  int dir = -1, journal = -1;
  void *vendor = nullptr;
  const PJRT_Api *original = nullptr;
  void *copied = nullptr;
  std::atomic<size_t> inflight{0};
  std::map<void *, std::shared_ptr<Handle>> loaded, executable;
  std::vector<Ref> inventory;
};
State &s() {
  static State *p = new State;
  return *p;
}
void fail(const char *code) {
  auto &a = s();
  a.failed = true;
  if (a.failure.empty())
    a.failure = code;
}
bool live_capture() {
  auto &a = s();
  if (!a.initialized || a.finalized || a.failed)
    return false;
  if (now() >= a.deadline) {
    fail("DEADLINE");
    return false;
  }
  return true;
}
bool writeall(int fd, const std::string &b) {
  size_t n = 0;
  while (n < b.size()) {
    ssize_t k = write(fd, b.data() + n, b.size() - n);
    if (k < 0 && errno == EINTR)
      continue;
    if (k <= 0)
      return false;
    n += size_t(k);
  }
  return true;
}
bool journal(const std::string &b) {
  auto &a = s();
  if (a.journal < 0)
    return false;
  if (b.size() + 1 > JournalLimit - a.journal_bytes ||
      a.records >= HardEntries) {
    fail("JOURNAL_QUOTA");
    return false;
  }
  if (!writeall(a.journal, b + "\n")) {
    fail("JOURNAL_WRITE");
    return false;
  }
  a.journal_bytes += b.size() + 1;
  ++a.records;
  return true;
}
Ref payload(const char *data, size_t n, const std::string &kind, uint64_t seq) {
  auto &a = s();
  Ref r;
  if (!live_capture())
    return r;
  if ((n && !data) || n > a.max_file ||
      n > a.max_total - MetadataReserve - a.payload_bytes ||
      a.files >= a.max_entries) {
    fail("PAYLOAD_QUOTA_OR_NULL");
    return r;
  }
  // Reserve quota before creating or writing the file. Failed reservations stay
  // charged.
  a.payload_bytes += n;
  ++a.files;
  r.file = std::to_string(seq) + "-" + kind + ".bin";
  r.bytes = n;
  int fd = openat(a.dir, r.file.c_str(),
                  O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
  if (fd < 0) {
    fail("PAYLOAD_OPEN");
    return r;
  }
  bool ok = writeall(fd, std::string(data ? data : "", n));
  if (fsync(fd) != 0)
    ok = false;
  if (close(fd) != 0)
    ok = false;
  if (!ok) {
    fail("PAYLOAD_WRITE");
    return r;
  }
  r.sha = digest(data ? data : "", n);
  r.ok = true;
  a.inventory.push_back(r);
  return r;
}
template <class T> bool has(const T *p, size_t end) {
  return p && p->struct_size >= end;
}
#define HAS(p, T, f) has(p, PJRT_STRUCT_SIZE(T, f))
std::string fmt(const PJRT_Program *p) {
  if (!p->format || p->format_size > 64)
    return {};
  return std::string(p->format, p->format_size);
}
std::string id(const std::shared_ptr<Handle> &h) { return h ? h->id : ""; }
std::string creation(const std::shared_ptr<Handle> &h) {
  return ",\"created_valid\":" +
         std::string(h && !h->ambiguous ? "true" : "false") +
         ",\"replaced_alive\":" + q(h ? h->replaced_alive : "");
}
std::shared_ptr<Handle> lookup(void *p, bool l) {
  auto &m = l ? s().loaded : s().executable;
  auto it = m.find(p);
  return it == m.end() ? nullptr : it->second;
}
std::shared_ptr<Handle> create(void *p, bool l, uint64_t seq,
                               const char *origin,
                               std::shared_ptr<Handle> parent = {}) {
  if (!p)
    return {};
  auto &a = s();
  auto &m = l ? a.loaded : a.executable;
  auto old = lookup(p, l);
  auto h = std::make_shared<Handle>();
  h->id = a.nonce + (l ? "-L-" : "-E-") + std::to_string(seq);
  h->born = seq;
  h->origin = origin;
  h->parent = parent;
  if (old && old->alive) {
    old->ambiguous = true;
    h->ambiguous = true;
    h->replaced_alive = old->id;
  }
  if (parent && (!parent->alive || parent->deleted || parent->ambiguous))
    h->ambiguous = true;
  m[p] = h;
  return h;
}
struct Call {
  const char *op;
  uint64_t pre = 0, post = 0, t0 = 0, t1 = 0, number = 0;
  bool active = false, pre_valid = false;
  std::string phase;
  std::shared_ptr<Handle> h;
  std::string extra;
  uint64_t saved_query_seq = 0;
  size_t saved_query_size = 0;
  std::string saved_query_format;
  Call(const char *o, void *p = nullptr, bool l = true) : op(o) {
    s().inflight.fetch_add(1);
    try {
      std::lock_guard<std::mutex> lock(s().mu);
      if (!live_capture())
        return;
      active = true;
      t0 = now();
      pre = ++s().seq;
      number = ++s().calls;
      phase = s().phase;
      if (p) {
        h = lookup(p, l);
        if (h) {
          if (h->busy)
            h->ambiguous = true;
          ++h->busy;
          pre_valid = h->alive && !h->deleted && !h->ambiguous;
          if (strcmp(op, "optimized") == 0) {
            // BEGIN supersedes prior evidence even if arguments or the API fail.
            // Snapshot and consume the preceding completed query atomically.
            if (h->query_valid) {
              saved_query_seq = h->query_seq;
              saved_query_size = h->query_size;
              saved_query_format = h->query_format;
            }
            h->query_valid = false;
            h->optimized_epoch = pre;
          }
        }
      }
    } catch (...) {
      observer_fail();
    }
  }
  static void observer_fail() noexcept {
    try {
      std::lock_guard<std::mutex> lock(s().mu);
      fail("OBSERVER_EXCEPTION");
    } catch (...) {
    }
  }
  template <class F> void before(F f) noexcept {
    if (!active)
      return;
    try {
      std::lock_guard<std::mutex> lock(s().mu);
      f();
    } catch (...) {
      observer_fail();
    }
  }
  template <class F> void finish(PJRT_Error *err, F f) noexcept {
    if (active)
      try {
        std::lock_guard<std::mutex> lock(s().mu);
        t1 = now();
        post = ++s().seq;
        if (now() >= s().deadline)
          fail("DEADLINE");
        f();
        bool valid = pre_valid && h && h->alive && !h->deleted && !h->ambiguous;
        if (h && h->busy)
          --h->busy;
        journal("{\"type\":\"call\",\"seq\":" + std::to_string(post) +
                ",\"pre_seq\":" + std::to_string(pre) +
                ",\"call\":" + std::to_string(number) + ",\"op\":" + q(op) +
                ",\"pre_ns\":" + std::to_string(t0) +
                ",\"post_ns\":" + std::to_string(t1) +
                ",\"phase\":" + q(phase) + ",\"handle\":" + q(id(h)) +
                ",\"pre_valid\":" + (pre_valid ? "true" : "false") +
                ",\"post_valid\":" + (valid ? "true" : "false") +
                ",\"error_returned\":" + (err ? "true" : "false") + extra +
                "}");
      } catch (...) {
        observer_fail();
      }
    s().inflight.fetch_sub(1);
    active = false;
  }
  ~Call() {
    if (active) {
      if (h)
        try {
          std::lock_guard<std::mutex> lock(s().mu);
          if (h->busy)
            --h->busy;
          fail("UNFINISHED_OBSERVER");
        } catch (...) {
        }
      s().inflight.fetch_sub(1);
    }
  }
};
PJRT_Error *compile(PJRT_Client_Compile_Args *a) {
  Call c("compile");
  Ref code, format;
  c.before([&] {
    if (HAS(a, PJRT_Client_Compile_Args, executable) &&
        HAS(a->program, PJRT_Program, format_size)) {
      auto p = a->program;
      code = payload(p->code, p->code_size, "submitted", c.pre);
      format = payload(p->format, p->format_size, "submit-format", c.pre);
    } else
      c.extra += ",\"capture_failure\":\"SHORT_ARGS\"";
  });
  PJRT_Error *e = s().original->PJRT_Client_Compile(a);
  c.finish(e, [&] {
    std::shared_ptr<Handle> h;
    if (!e && HAS(a, PJRT_Client_Compile_Args, executable))
      h = create(a->executable, true, c.post, "compile");
    c.extra += ",\"created\":" + q(id(h)) + creation(h) +
               ",\"origin\":\"compile\",\"submitted\":" + code.json() +
               ",\"format_ref\":" + format.json();
  });
  return e;
}
PJRT_Error *deserialize(PJRT_Executable_DeserializeAndLoad_Args *a) {
  Call c("deserialize");
  Ref code;
  c.before([&] {
    if (HAS(a, PJRT_Executable_DeserializeAndLoad_Args, loaded_executable))
      code = payload(a->serialized_executable, a->serialized_executable_size,
                     "serialized", c.pre);
  });
  PJRT_Error *e = s().original->PJRT_Executable_DeserializeAndLoad(a);
  c.finish(e, [&] {
    std::shared_ptr<Handle> h;
    if (!e &&
        HAS(a, PJRT_Executable_DeserializeAndLoad_Args, loaded_executable))
      h = create(a->loaded_executable, true, c.post, "deserialize");
    c.extra += ",\"created\":" + q(id(h)) + creation(h) +
               ",\"origin\":\"deserialize\",\"serialized\":" + code.json();
  });
  return e;
}
PJRT_Error *getexe(PJRT_LoadedExecutable_GetExecutable_Args *a) {
  void *p = HAS(a, PJRT_LoadedExecutable_GetExecutable_Args, loaded_executable)
                ? a->loaded_executable
                : nullptr;
  Call c("get_executable", p);
  PJRT_Error *e = s().original->PJRT_LoadedExecutable_GetExecutable(a);
  c.finish(e, [&] {
    std::shared_ptr<Handle> h;
    if (!e && HAS(a, PJRT_LoadedExecutable_GetExecutable_Args, executable))
      h = create(a->executable, false, c.post, "get_executable", c.h);
    if (h && (!c.pre_valid || !c.h))
      h->ambiguous = true;
    c.extra +=
        ",\"created\":" + q(id(h)) + creation(h) + ",\"parent\":" + q(id(c.h));
  });
  return e;
}
PJRT_Error *optimized(PJRT_Executable_OptimizedProgram_Args *a) {
  void *p = HAS(a, PJRT_Executable_OptimizedProgram_Args, executable)
                ? a->executable
                : nullptr;
  Call c("optimized", p, false);
  PJRT_Program *program = nullptr;
  char *input_code = nullptr;
  size_t capacity = 0;
  bool query = false, valid_args = false;
  uint64_t query_seq = 0;
  size_t query_size = 0;
  std::string query_format;
  c.before([&] {
    valid_args = HAS(a, PJRT_Executable_OptimizedProgram_Args, program) &&
                 HAS(a->program, PJRT_Program, format_size);
    if (!valid_args)
      return;
    program = a->program;
    input_code = program->code;
    query = input_code == nullptr;
    // Never read code_size or format fields before the first size query.
    if (!query) {
      capacity = program->code_size;
      query_seq = c.saved_query_seq;
      query_size = c.saved_query_size;
      query_format = c.saved_query_format;
    }
  });
  PJRT_Error *e = s().original->PJRT_Executable_OptimizedProgram(a);
  c.finish(e, [&] {
    Ref code;
    std::string format, failure;
    size_t returned = 0;
    if (!valid_args)
      failure = "SHORT_ARGS";
    else if (!e && a->program == program) {
      returned = program->code_size;
      format = fmt(program);
      if (query) {
        if (c.h && c.pre_valid && c.h->alive && !c.h->ambiguous &&
            c.h->optimized_epoch == c.pre && format.size() && returned <= s().max_file) {
          c.h->query_valid = true;
          c.h->query_size = returned;
          c.h->query_format = format;
          c.h->query_seq = c.post;
        } else
          failure = "INVALID_QUERY";
      } else if (!query_seq || query_size > capacity || returned != capacity ||
                 format != query_format || program->code != input_code ||
                 !c.h || !c.pre_valid || !c.h->alive || c.h->ambiguous ||
                 c.h->optimized_epoch != c.pre)
        failure = "UNKNOWN_FILL_LENGTH_OR_LINEAGE";
      else
        code = payload(input_code, query_size, "optimized", c.pre);
    } else if (!e)
      failure = "PROGRAM_POINTER_CHANGED";
    c.extra += ",\"optimized_epoch\":" + std::to_string(c.pre) +
               ",\"mode\":" + q(query ? "query" : "fill") +
               ",\"query_seq\":" + std::to_string(query_seq) +
               ",\"query_size\":" + std::to_string(query_size) +
               ",\"capacity\":" + std::to_string(capacity) +
               ",\"returned_size\":" + std::to_string(returned) +
               ",\"format\":" + q(format) + ",\"payload\":" + code.json() +
               ",\"capture_failure\":" + q(failure);
  });
  return e;
}
PJRT_Error *execute(PJRT_LoadedExecutable_Execute_Args *a) {
  void *p = HAS(a, PJRT_LoadedExecutable_Execute_Args, executable)
                ? a->executable
                : nullptr;
  Call c("execute", p);
  PJRT_Error *e = s().original->PJRT_LoadedExecutable_Execute(a);
  c.finish(e, [] {});
  return e;
}
template <class T, class F>
PJRT_Error *lifecycle(T *a, F f, const char *op, bool l) {
  void *p = HAS(a, T, executable) ? a->executable : nullptr;
  Call c(op, p, l);
  PJRT_Error *e = f(a);
  c.finish(e, [&] {
    if (!e && c.h) {
      if (strcmp(op, "loaded_delete") == 0)
        c.h->deleted = true;
      else {
        c.h->alive = false;
        c.h->query_valid = false;
      }
    }
  });
  return e;
}
PJRT_Error *ld(PJRT_LoadedExecutable_Destroy_Args *a) {
  return lifecycle(a, s().original->PJRT_LoadedExecutable_Destroy,
                   "loaded_destroy", true);
}
PJRT_Error *ed(PJRT_Executable_Destroy_Args *a) {
  return lifecycle(a, s().original->PJRT_Executable_Destroy,
                   "executable_destroy", false);
}
PJRT_Error *del(PJRT_LoadedExecutable_Delete_Args *a) {
  return lifecycle(a, s().original->PJRT_LoadedExecutable_Delete,
                   "loaded_delete", true);
}
std::string hashfile(const char *path) {
  int fd = open(path, O_RDONLY | O_NOFOLLOW);
  if (fd < 0)
    return {};
  Sha256 h;
  char b[65536];
  ssize_t n;
  while ((n = read(fd, b, sizeof b)) > 0)
    h.add(b, size_t(n));
  bool ok = n == 0;
  close(fd);
  return ok ? h.end() : "";
}
} // namespace
extern "C" int M6TraceConfigure(const char *provider, const char *output,
                                const char *nonce, const char *rid,
                                uint64_t deadline, size_t entries, size_t total,
                                size_t file) {
  try {
    auto &a = s();
    std::lock_guard<std::mutex> lock(a.mu);
    if (a.configured || !provider || *provider != '/' || !output ||
        *output != '/' || !label(nonce) || !label(rid) || entries < 3 ||
        entries > HardEntries || total < MetadataReserve || total > HardTotal ||
        file < MiB || file > HardFile)
      return -1;
    a.provider = provider;
    a.out = output;
    a.nonce = nonce;
    a.recorder_id = rid;
    a.deadline = deadline;
    a.max_entries = entries;
    a.max_total = total;
    a.max_file = file;
    a.configured = true;
    return 0;
  } catch (...) {
    return -1;
  }
}
extern "C" int M6TraceSetPhase(const char *phase) {
  try {
    std::lock_guard<std::mutex> lock(s().mu);
    if (!label(phase) || s().finalized)
      return -1;
    s().phase = phase;
    return 0;
  } catch (...) {
    return -1;
  }
}
extern "C" const PJRT_Api *GetPjrtApi() {
  try {
    static std::mutex init_mu;
    std::lock_guard<std::mutex> init_lock(init_mu);
    auto &a = s();
    std::unique_lock<std::mutex> lock(a.mu);
    if (a.initialized)
      return a.copied ? static_cast<PJRT_Api *>(a.copied) : a.original;
    if (!a.configured)
      return nullptr;
    if (a.failed)
      return a.copied ? static_cast<PJRT_Api *>(a.copied) : a.original;
    Dl_info self{};
    struct stat st{}, me{};
    if (!dladdr(reinterpret_cast<void *>(&GetPjrtApi), &self) ||
        stat(self.dli_fname, &me) || stat(a.provider.c_str(), &st) ||
        (st.st_dev == me.st_dev && st.st_ino == me.st_ino)) {
      fail("PROVIDER_RECURSION_OR_STAT");
      return nullptr;
    }
    lock.unlock();
    void *vendor = dlopen(a.provider.c_str(), RTLD_NOW | RTLD_LOCAL);
    lock.lock();
    a.vendor = vendor;
    if (!a.vendor) {
      fail("PROVIDER_LOAD");
      return nullptr;
    }
    auto get =
        reinterpret_cast<const PJRT_Api *(*)()>(dlsym(a.vendor, "GetPjrtApi"));
    Dl_info origin{};
    struct stat origin_stat{};
    if (!get || get == &GetPjrtApi ||
        !dladdr(reinterpret_cast<void *>(get), &origin) ||
        stat(origin.dli_fname, &origin_stat) ||
        origin_stat.st_dev != st.st_dev || origin_stat.st_ino != st.st_ino) {
      fail("PROVIDER_RECURSION_OR_SYMBOL_ORIGIN");
      return nullptr;
    }
    // The provider handle remains open until process exit.
    lock.unlock();
    const PJRT_Api *original = get();
    lock.lock();
    a.original = original;
    if (!a.original || a.original->struct_size < sizeof(size_t)) {
      fail("PROVIDER_ABI");
      return nullptr;
    }
    size_t n = a.original->struct_size;
    a.copied = malloc(n);
    if (!a.copied) {
      fail("API_ALLOCATION");
      a.initialized = true;
      return a.original;
    }
    memcpy(a.copied, a.original, n);
    auto api = static_cast<PJRT_Api *>(a.copied);
    bool compatible =
        n >= PJRT_STRUCT_SIZE(PJRT_Api, pjrt_api_version) &&
        a.original->pjrt_api_version.struct_size >=
            PJRT_Api_Version_STRUCT_SIZE &&
        a.original->pjrt_api_version.major_version == PJRT_API_MAJOR;
    if (!compatible)
      fail("UNSUPPORTED_ABI_MAJOR_OR_VERSION");
#define HOOK(slot, fn)                                                         \
  if (compatible && n >= PJRT_STRUCT_SIZE(PJRT_Api, slot) && a.original->slot) \
  api->slot = &fn
    HOOK(PJRT_Client_Compile, compile);
    HOOK(PJRT_LoadedExecutable_GetExecutable, getexe);
    HOOK(PJRT_Executable_OptimizedProgram, optimized);
    HOOK(PJRT_LoadedExecutable_Execute, execute);
    HOOK(PJRT_LoadedExecutable_Destroy, ld);
    HOOK(PJRT_Executable_Destroy, ed);
    HOOK(PJRT_LoadedExecutable_Delete, del);
    HOOK(PJRT_Executable_DeserializeAndLoad, deserialize);
#undef HOOK
    if (mkdir(a.out.c_str(), 0700) != 0) {
      fail("OUTPUT_CREATE");
      a.initialized = true;
      return api;
    }
    a.dir = open(a.out.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW);
    if (a.dir < 0) {
      fail("OUTPUT_OPEN");
      a.initialized = true;
      return api;
    }
    a.journal = openat(a.dir, "journal.jsonl",
                       O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (a.journal < 0) {
      fail("JOURNAL_OPEN");
      a.initialized = true;
      return api;
    }
    a.provider_sha = hashfile(a.provider.c_str());
    if (a.provider_sha.empty())
      fail("PROVIDER_HASH");
    a.initialized = true;
    std::string version = "null";
    if (n >= PJRT_STRUCT_SIZE(PJRT_Api, pjrt_api_version))
      version =
          "{\"major\":" + std::to_string(api->pjrt_api_version.major_version) +
          ",\"minor\":" + std::to_string(api->pjrt_api_version.minor_version) +
          "}";
    journal("{\"type\":\"header\",\"schema\":\"M6_DIRECT_PJRT_TRACE_V4\","
            "\"nonce\":" +
            q(a.nonce) + ",\"recorder_id\":" + q(a.recorder_id) +
            ",\"provider_sha256\":" + q(a.provider_sha) +
            ",\"api_struct_size\":" + std::to_string(n) + ",\"api_version\":" +
            version + ",\"deadline_ns\":" + std::to_string(a.deadline) +
            ",\"limits\":{\"entries\":" + std::to_string(a.max_entries) +
            ",\"total\":" + std::to_string(a.max_total) +
            ",\"file\":" + std::to_string(a.max_file) +
            ",\"metadata_reserve\":" + std::to_string(MetadataReserve) +
            "},\"claim\":\"DIRECT_LOADED_HANDLE_EXECUTE_CALL\",\"extra_"
            "observer_probes\":false,\"clock\":\"CLOCK_MONOTONIC\"}");
    return api;
  } catch (...) {
    try {
      auto &a = s();
      std::lock_guard<std::mutex> lock(a.mu);
      fail("OBSERVER_INITIALIZATION_EXCEPTION");
      a.initialized = a.original != nullptr;
      return a.copied ? static_cast<PJRT_Api *>(a.copied) : a.original;
    } catch (...) {
      return nullptr;
    }
  }
}
extern "C" int M6TraceFinalize() {
  try {
    auto &a = s();
    std::lock_guard<std::mutex> lock(a.mu);
    if (a.inflight.load() != 0)
      return 1;
    if (a.finalized)
      return a.failed ? -1 : 0;
    a.finalized = true;
    if (!a.initialized || a.journal < 0) {
      fail("NO_JOURNAL");
      return -1;
    }
    std::string j =
        "{\"state\":\"API_FINALIZED\",\"inflight\":0,\"last_seq\":" +
        std::to_string(a.seq) +
        ",\"capture_status\":" + q(a.failed ? "UNKNOWN" : "OK") +
        ",\"failure\":" + q(a.failure) +
        ",\"payload_reserved_bytes\":" + std::to_string(a.payload_bytes) +
        ",\"entries_reserved\":" + std::to_string(a.files) + ",\"inventory\":[";
    for (size_t i = 0; i < a.inventory.size(); ++i) {
      if (i)
        j += ',';
      j += a.inventory[i].json();
    }
    j += "]}\n";
    if (j.size() > MiB || j.size() > a.max_file) {
      fail("FOOTER_QUOTA");
      return -1;
    }
    if (fsync(a.journal) != 0) {
      fail("JOURNAL_SYNC");
      return -1;
    }
    if (close(a.journal) != 0) {
      a.journal = -1;
      fail("JOURNAL_CLOSE");
      return -1;
    }
    a.journal = -1;
    int fd = openat(a.dir, "footer.json",
                    O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) {
      fail("FOOTER_OPEN");
      return -1;
    }
    bool ok = writeall(fd, j);
    if (fsync(fd) != 0)
      ok = false;
    if (close(fd) != 0)
      ok = false;
    if (!ok) {
      fail("FOOTER_WRITE");
      return -1;
    }
    return a.failed ? -1 : 0;
  } catch (...) {
    return -1;
  }
}
