#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Call once before GetPjrtApi. Paths must be absolute. Output must not exist.
// Deadline uses CLOCK_MONOTONIC nanoseconds. Limits cannot exceed hard bounds.
// Minimum limits are three entries, 3 MiB total, and 1 MiB per file.
int M6TraceConfigure(const char *provider, const char *output,
                     const char *nonce, const char *recorder_id,
                     uint64_t deadline_ns, size_t max_entries,
                     size_t max_total_bytes, size_t max_file_bytes);
// Phase is a bounded public label, not arbitrary user data.
int M6TraceSetPhase(const char *phase);
// Returns 1 while a wrapped API call is active. Returns -1 on capture failure.
// A return of zero means API_FINALIZED. Host readback must establish CLOSED.
int M6TraceFinalize(void);
#ifdef __cplusplus
}
#endif
