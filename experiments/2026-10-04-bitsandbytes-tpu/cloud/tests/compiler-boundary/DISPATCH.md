# Actual compiler experiment recipe

The reviewer owns actual dispatch and provider actions.
This document does not authorize an allocation.

1. Adopt the reviewed source, policy, manifest, cloud pins, and replacement maps.
2. Build a fresh compiler packet with the pinned upstream checkout and exact accepted M2 dependency.
3. Inspect every packet member and all six accepted native dependency bindings.
4. Create a new exact root dispatch gate for this packet and generation.
5. Keep the fixed runtime, hardware admission, precision, marker, and original provider budget.
6. Generate and recover fresh qualified Linux CPU oracles for all five cases.
7. Verify those oracles independently before the device phase.
8. Run the separate fixed-library backend flag preflight with the exact 13-token request.
9. Require successful parser admission and process closure before the scientific parent starts.
10. Run all five unchanged cases through the bounded compiler parent.
11. Keep the 1,024-entry, 128 MiB total, and 16 MiB per-file limits.
12. Recover complete scientific, CPU, and compiler archives, including failed records.
13. Verify process closure, an empty provider session list, zero active usage, and temporary tab closure.
14. Apply the independent compiler reader to the recovered raw records.

If required metadata or stage references are absent, record an unsupported or failed analysis.
Do not infer allocation sizes, selected executable identity, or an internal revision from missing records.
Do not change reader gates to obtain success.
The additive compiler allocation audit remains a separate root-reviewed path.
Its success cannot establish loaded executable identity or allocator peak.
A parser pass cannot establish compiler memory acceptance.
A partial numerical pass cannot qualify M6.

Do not reuse the prior failed dump inventory or any prior generation's CPU gate.
Do not reuse a prior exact root dispatch gate.

## Exact local packet and owner commands

After canonical adoption, use the portable reconstruction and packet build in
[REPRODUCE.md](REPRODUCE.md). The resulting `packet1.json` binds the payload hash.
The packet builder also accepts this direct command. Set each variable to an
existing, reviewed path; create a fresh `PACKET` directory through the builder.
The M2 dependency is the exact mapped `m2-dependency.json`.

```sh
"$HOST_PYTHON" -B "$REPO/experiments/2026-10-04-bitsandbytes-tpu/cloud/build_compiler_packet.py" --upstream "$UPSTREAM" --out "$PACKET" --candidate "$REPO/experiments/2026-10-04-bitsandbytes-tpu/compiler-native" --candidate-sha256 b1383b3112eebf9ed77c1edca60fa0985a41192f7304a35c6861a860c0bca0c8 --m2-dependency "$STAGE/m2-dependency.json" --m2-dependency-sha256 "$M2_DEPENDENCY_SHA256"
"$HOST_PYTHON" -B "$REPO/experiments/2026-10-04-bitsandbytes-tpu/cloud/owner.py" --packet "$PACKET" --packet-sha256 "$PAYLOAD_SHA256" --root-acceptance "$ROOT_GATE" --root-acceptance-sha256 "$ROOT_GATE_SHA256" --cli-python "$CLI_PYTHON" --cli-identity "$CLI_IDENTITY" --mosaic-audit-python "$JAX_PYTHON" --output "$ACTUAL_OUTPUT" --preflight-only
```

The second command checks local packet bytes and CLI identity. It makes no API call.
For the actual experiment, the reviewer must issue the same owner command with
`--preflight-only` removed. Use the fresh root gate bound to the payload hash and
new generation. The existing owner enforces the original 3,600-second lifecycle,
fresh CPU gate, separate backend flag preflight, remote monitor, and closure.
If the reviewer adopts an existing browser allocation, add its newly reviewed
`--browser-adoption` and `--browser-adoption-sha256` arguments. Keep its original deadline.
This worker did not run the actual command or create those external records.
