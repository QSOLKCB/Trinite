# Preserved v2 robustness repeat: incomplete device lane

The [reports-only fixture](../fixtures/music-robustness-result-v1/README.md)
preserves a completed native CPU repeat plus an unsupported device-freeze
receipt under the [v2 robustness contract](MUSIC-ROBUSTNESS-V2.md). The overall
requested repeat is incomplete, not a negative GPU learning result.

```sh
python -I -B scripts/check_music_robustness_result.py
PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests/music_robustness_result_v1 -v
PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 python -O -B -m unittest discover -s tests/music_robustness_result_v1 -v
```

All 800 CPU journal records and the complete summary match the original
[preserved v1 result](MUSIC-PLACEMENT-RESULT.md). This is exact retained-output
agreement on the same exposed families, not independent execution replay or new
scientific confirmation. Four CPU cells completed; no new device pair or
allocator-admission receipt is available. Missing device outcomes are not zero
accuracy and cannot form a timing comparison. The original successful device
execution and its measurements remain separate historical evidence.

The standard-library checker authenticates immutable originals and full scoring
closure before imports, validates v2 CPU links/scores and hard-pinned v1 journal
parity, and checks the standalone unsupported stderr's schema. It cannot bind
that stderr to an independently retained device request: none was published in
the supplied handoff. No specific driver/hardware root cause is established.

The interpreter, standard library and verifier are trusted. Use `-I -B`; the
launcher cannot undo startup/site code already executed without those flags.
Unsafe paths, import shadows, changed scorer sources, altered/resealed artifacts,
missing rows and historical comparison mutations are rejected. Verification is
read-only and uses no model/backend. It is not full custody, a repeat of original
audit/inference/optimization, hardware/timing attestation, validation of the new
allocator policy or proof of no other attempts.

Affected TRI-I01/02/03/05/06/07/08/11/12/13/14: additive evidence preservation only.
No historical source/fixture refresh, automatic retry, GPU repair, new experiment,
release promotion or broader curriculum change is included.
