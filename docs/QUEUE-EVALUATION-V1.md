# Queue prediction/evaluation v1

Status: checkout-only engineering infrastructure, 1 November 2026. No model is
invoked or trained. This is a prerequisite to later within-domain experiments,
not completion of the cross-environment transfer gate.

## One fixed task, two independently retained anchors

The target is `forecast.dropped`: arrivals rejected during the unobserved future
suffix of one [queue-v1](QUEUE-SIM-V1.md) episode. Predictions are integers from
0 through `32 * horizon`, inclusive, or explicit abstentions. This is a count,
not a probability, reward, rank or confidence estimate. The small Rust evaluator
uses the unchanged queue crate by local path; it does not duplicate dynamics.

1. **Prepare**, privileged: validate a bounded queue request through the fixed
   Rust `observe` operation, retain the exact private request, observation,
   process evidence, runtime source inventory and executable digest. Export only
   observation and public protocol. Retain the SHA-256 of the private preparation
   manifest separately from the mutable bundle, before accepting a prediction.
2. **Commit**, public-only: validate the observation, fixed protocol and prediction;
   create an exclusive bundle and return its manifest SHA-256. This operation has
   no private request argument and never runs the evaluator. Independently retain
   this second digest before scoring.
3. **Score**, privileged: require both caller-retained digests. Validate the public
   commitment and prediction before opening private input. Check the anchored
   private request/source/binary, replay the exact observation, then execute the
   authoritative queue target. Record absolute error and exact match for numeric
   predictions. Abstention records the actual target and status but **no numeric
   error, predicted value or exact-match field**.
4. **Verify**, privileged: repeat those checks and executions, and compare exact
   result/process bytes. It does not trust a recorded target or score. Artifact
   verification is read-only, but it executes the fixed local evaluator.

The public protocol is pinned in the prediction commitment. Neither public
manifest nor observation digest incorporates private request bytes. The private
preparation digest, request, process evidence and scored outcome belong to the
privileged operator: **do not give them to the learner**. The public packet is
exactly `observation.json` and `protocol.json`. A result is not a learner packet.

Two different future suffixes can produce identical public observations. The
future target is therefore not generally identifiable from one observation;
matching an exposed test target demonstrates accounting, not predictive skill.
The tests substitute such a suffix and reseal local metadata: the independently
retained preparation anchor still rejects it. Re-reading either expected digest
from a mutable manifest at verification time defeats that protection.

## Running the engineering example

Requires a POSIX checkout, Python 3 standard library and Rust 1.90.0. Build the
fixed executable (no configurable runner, shell command or plugin registry):

```sh
(cd external/queue-evaluator-v1 && cargo build --offline --locked --release)
work=$(mktemp -d)
cat > "$work/request.json" <<'JSON'
{"schema":"trinite.queue-request.v1","capacity":2,"initial_queue":0,"arrivals":[1,4,3],"service":[1,0,1],"outages":[false,false,false],"observe_ticks":1,"observed":[true,true,true,true,true,true]}
JSON
# Keep this variable and the private directory out of the learner context.
prepare_anchor=$(python3 -I -B scripts/queue_evaluation_v1.py prepare \
  --request "$work/request.json" --out "$work/private" --public-out "$work/public")
# This public-only example abstains; it neither reads nor guesses a hidden target.
python3 -I -B - "$work" <<'PY'
import hashlib, json, pathlib, sys
w = pathlib.Path(sys.argv[1])
p = {'schema': 'trinite.queue-prediction.v1', 'status': 'abstain',
     'observation_sha256': hashlib.sha256((w/'public/observation.json').read_bytes()).hexdigest()}
(w/'prediction.json').write_text(json.dumps(p) + '\n', encoding='ascii')
PY
commit_anchor=$(python3 -I -B scripts/queue_evaluation_v1.py commit \
  --public "$work/public" --prediction "$work/prediction.json" --out "$work/commit")
python3 -I -B scripts/queue_evaluation_v1.py score \
  --prepared "$work/private" --commit "$work/commit" \
  --prepare-sha "$prepare_anchor" --commit-sha "$commit_anchor" --out "$work/result"
python3 -I -B scripts/queue_evaluation_v1.py verify \
  --prepared "$work/private" --commit "$work/commit" \
  --prepare-sha "$prepare_anchor" --commit-sha "$commit_anchor" --result "$work/result"
```

A numeric prediction instead has `"status":"prediction","value":5`; all other
prediction fields are identical. Unknown/duplicate fields, boolean-as-integer,
floats, non-finite numbers, excessive nesting/size and out-of-range values are
rejected. The queue request retains its existing strict grammar and 16 KiB limit;
controller JSON is bounded to 256 KiB and integer magnitudes to signed 64-bit.

## Process and artifact boundaries

Each fixed runner process has a five-second deadline and independently bounded
64 KiB stdout/stderr, checked while streaming, not after unbounded buffering.
The controller uses no shell, kills the process group on failure and waits for
the child. Process evidence retains bounded base64 stdout/stderr, return code
and failure category. Nonzero exits, timeouts, floods and malformed output never
become successful metrics. Invalid CLI use exits 2.

All output directories must be fresh, mutually disjoint and outside the checkout;
parents must already exist. Exclusive creation prevents overwriting historical
or concurrent attempts. Symlink components, overlapping inputs/outputs and
hardlinked artifact inputs are rejected. Cargo's executable hardlinks are allowed
for binary hashing. Exact bundle inventories reject extra, missing or partial
files. Failures after destination creation retain a `failure.json` and available
partial evidence; these are not successful bundles. Pre-creation failures cannot
create a receipt. A failed `verify` writes nothing. Preserve partial directories
and choose new output paths for retries. Use `-I -B` as shown to avoid environment
imports and Python bytecode writes.

These checks are **not an OS sandbox**, malicious-owner filesystem-race defense,
or cryptographic proof of chronology/operator honesty. The same OS user can
inspect hidden files, replace programs or fabricate an entire history and new
anchors. Use an independently retained anchor and separately enforced access
boundaries for an eventual learner. Source and binary digests identify bytes;
they do **not attest that this binary was built from those sources**. The manifest
records Python/platform identity and the declared Rust version, not a measured
compiler or complete reproducible-build attestation. The declared source inventory
is not a complete compiler-input closure: Cargo configuration, build environment
and compiler inputs outside that explicit inventory are not attested. Replay evidence is local
process evidence, not third-party certification. Verification requires unchanged
runtime sources and binary; rebuilds may legitimately invalidate a prior bundle.

## Conformance, not learning evidence

Run the new crate's offline locked fmt/clippy/debug/release checks, build the old
queue release executable for byte parity, then:

```sh
python3 -I -B -m unittest discover -s tests/queue_evaluation_v1 -v
python3 -I -B -O -m unittest discover -s tests/queue_evaluation_v1 -v
```

Tests cover exact/wrong/abstained predictions, both anchors, tampered artifacts
and source identity, missing/extra files, same-prefix/different-future substitution,
private-read ordering, strict JSON, exclusive/symlink/overlap paths, concurrent
creation, bounded failed processes, malformed target output and a separate FIFO
oracle. These are exposed synthetic engineering examples, not held-out data,
model evaluation, model improvement, transfer evidence or authorization to train.
All seven prior modules, native learning paths and historical fixtures stay unchanged.
