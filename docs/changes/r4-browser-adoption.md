# Reuse of the browser runtime

Date: 2026-10-08.

Two CLI allocation requests returned `Bad Gateway` before payload execution.
The reviewer then connected a Colab V6E1 runtime through the browser.
The browser executed one marker cell successfully.
That cell did not install packages or execute numerical cases.

The new mode uses the official CLI session APIs to register that existing runtime.
It does not request another allocation.
The root record binds the exact endpoint, hardware, source packet, driver, CLI identity, and original allocation time.
The CLI must return the exact browser marker before source upload or package installation.
An incorrect marker removes only the provisional local registration.
It does not authorize termination of the remote runtime.

After successful identification, the owner retains the original result retrieval and exact-session termination procedures.
An additional session causes rejection.
The owner uses the original 3,600-second limit from the browser connection request.
The host guard permits at most 60 additional seconds for emergency closure.
An expired deadline prevents launch.

## Inspection

The reviewer repeated 12 official CLI controls and 32 owner, bootstrap, and host controls.
All 44 controls passed.
Incorrect endpoint, additional session, expired record, missing marker, and closure failure controls passed.
Numerical failure and error records remained unsuccessful results.
The controls used isolated service fixtures and did not execute TPU calculations.

The reviewer also inspected all 46 packet members and the installed CLI identity.
The scientific sources, plugin, runtime lock, admission record, inputs, and tolerances remain byte-identical to the corrected nested packet.
The preparation record supplies the exact packet size and SHA-256.

## Browser observation

The browser connection request started at 12:58:19.612 UTC.
The page still showed allocation after 338.157 seconds.
The page showed a connected runtime after a later reload.
The exact allocation duration is unknown.
The diagnostic exceeded its 360-second observation target.
This record preserves that limit violation.
The runtime deadline remains 13:58:19.612 UTC.

M4 remains unqualified until the actual nested and saved-state records pass review.
The [preparation record](../../experiments/2026-10-04-bitsandbytes-tpu/results/nested-browser-adoption-preparation.json) identifies the inspected source and test artifacts.
