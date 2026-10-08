# R2: Bounded allocation transport

Two CLI assignment requests exceeded the original 120-second read limit.
The later web diagnostic obtained a V6E1 runtime between the 175-second and 255-second observations.
The next CLI experiment uses a separate transport with a longer, bounded wait.
This change does not establish the cause of the previous failures.

The wrapper requires the explicit `--allocation-transport-v1` flag and the exact V6E1 allocation command.
The transport permits one assignment POST to the fixed Colab endpoint.
It rejects a second assignment POST and redirects.
It disables automatic repeats of that POST.
The original session handles other requests and retains its authentication behavior.
The wrapper continues to prohibit a new OAuth flow.
The installed CLI source remains unchanged.

| Limit | Seconds |
| --- | --- |
| Connection | 10 |
| Assignment read | 300 |
| Transport call | 310 |
| Allocation command | 360 |
| Complete lifecycle | 3,600 |

The transport evaluates its call limit after a request returns.
The outer command owner supplies the hard process deadline.
The existing lifecycle owner retains responsibility for records and resource closure.

The wrapper restores its temporary factory reference and closes both sessions.
A session closure failure cannot produce a successful command result.
The reviewer found this error in the initial candidate through the actual standalone CLI path.
The correction preserves an original request failure or unsuccessful exit and records the additional closure failure.
The regression test rejects the earlier candidate.

Validation: The reviewer repeated 25 transport controls and 191 cloud controls.
All passed without provider requests.
The controls include correct requests, incorrect profiles, request errors, and actual standalone CLI exits with simulated responses.
The installed CLI identity and the complete packet passed separate inspections.

The new packet contains 37 members.
Only the owner, wrapper, and added transport differ from the earlier packet files.
The other 33 files match exactly, including all scientific inputs, source, runtime requirements, and numerical gates.
Refer to [the preparation record](../../experiments/2026-10-04-bitsandbytes-tpu/results/allocation-transport-preparation.json).

The device experiment remains required.
This preparation does not qualify R2, API42, or M3.
Accepted milestones remain **1 of 8**.
