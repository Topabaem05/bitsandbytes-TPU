# R2: Web allocation diagnostic

The Colab web interface obtained a V6E1 runtime on 2026-10-08.
The reviewer requested this runtime once after two CLI assignment timeouts.
At 175 seconds, the interface still showed allocation in progress.
At 255 seconds, it showed a connected TPU runtime.
These observations give an interval, not an exact connection time.

The reviewer executed no notebook cells and uploaded no scientific payload.
The runtime package versions were not examined.
The reviewer stopped the exact browser runtime after the diagnostic.
The browser session dialog was empty.
Independent CLI queries also showed an empty server list and zero active usage.

The reviewer inspected the installed CLI source and the worker's isolated transport controls.
The CLI identity and hardware arguments matched the earlier successful precision run.
CLI version `0.7.4` supplies no supported setting for the assignment read timeout.
Its transport uses a 120-second read limit.
The outer 180-second command limit does not extend that read limit.

The browser result supports a separate experiment with a longer, bounded assignment wait.
It does not establish the cause of the earlier CLI failures.
The reviewer did not compare the browser and CLI account identities.
The CLI cannot directly adopt or stop an unknown browser runtime through its public session commands.

Review an explicit allocation transport before another scientific run.
Restrict the proposed 300-second read limit to one assignment POST.
Keep the command limit at 360 seconds and the complete lifecycle limit at 3,600 seconds.
Retain the exact scientific source, runtime, inputs, numerical gates, and resource closure requirements.

R2, API42, and M3 remain unqualified.
Accepted milestones remain **1 of 8**.
Refer to [the diagnostic record](../../experiments/2026-10-04-bitsandbytes-tpu/results/web-allocation-diagnostic.json).
