# Preserved PR98 placement and CPU/CUDA development result

The [reports-only fixture](../fixtures/music-placement-result-v1/README.md)
preserves a mixed/negative placement result and a descriptive matched CPU/CUDA
comparison from the frozen [placement](MUSIC-PLACEMENT.md) and
[device](MUSIC-DEVICE-TRIAL.md) protocols. It adds no training or evaluation run.
All prior sources and evidence remain unchanged.

```sh
python -I -B scripts/check_music_placement_result.py
PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 python -B -m unittest discover -s tests/music_placement_result_v1 -v
PYTHONPATH=src:scripts PYTHONDONTWRITEBYTECODE=1 python -O -B -m unittest discover -s tests/music_placement_result_v1 -v
```

The isolated checker uses standard-library authentication and original pinned
corpus/scoring code only. The interpreter, its standard library and this verifier
are trusted; `-I -B` cannot undo Python startup/site code already run without those
flags. It rejects fixture tampering/resealing, unsafe paths, import shadows and
changed scoring sources before scoring. Verification is read-only, not a cached
claim, inference/optimizer replay, numerical probe replay, or hardware/timing
attestation. No raw requests or tensors are available in this public subset.

CPU canonical development stays 16/40 at seed 17 and rises 15→16/40 at seed 29;
all complete quartets and canonical rest-only strata remain zero. Both layouts
already fit in the control, limiting what the placement intervention shows.
The separate same-wheel device pair has 16/40 CPU versus 15/40 CUDA canonical,
116/120 identical development outputs, and longer retained CUDA update/cold-worker
times. These are one fixed-order small-workload measurements, not a general GPU
performance conclusion. See the fixture for exact denominators and custody limits.

Affected TRI-I01/02/03/05/06/07/08/11/12/13/14: additive evidence preservation only.
The confirmed allocator/failure-seal robustness issues require a separate reviewed
prospective implementation; this preservation neither patches frozen producer
sources nor authorizes another attempt.
