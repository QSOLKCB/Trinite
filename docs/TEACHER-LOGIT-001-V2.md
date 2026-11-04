# TT-003-LOGIT-001 v2 — frozen full-vocabulary logit audit

This separately versioned CPU-only, inference-only diagnostic uses the unchanged
90,624-parameter dense seed-17 teacher and original safe reader. It targets the
original TT-003 checkpoint at source 6f2ec05b407a3bafe7a5ed947fd9caadc4be65f7,
and the TT-003-DIAG-001 implementation at b3dd408a51e1cbc9b3280a47352ddb09a71b434a.
The historical request is c68c774a6cafafe76bf49ff43ce4e4be6d6c5ee1a3d93d233e0b39eadc3a0616;
its result manifest is 18522822e789682ac07e2390792c931e5559fa66f7bba7d9413b28bd0ffecd1f.
All existing historical checkpoint/model/source pins remain mandatory. No update,
backward pass, GPU, hidden evaluation, upload or checkpoint writing is permitted.

## Frozen observations

The request binds the complete historical source/import closure plus every new
runtime module, this protocol, plan, regression suite and workflow; it records the
Git commit and rejects uncommitted closure bytes. It binds the exact TRAIN row
order, actual 128-visit schedule, prior predictions and canonical position-index
SHA-256. No target run is authorized by CI or a software double.

Order is stage initial/final, then probe gold-prefix/greedy-prefix, then original
TRAIN order, then increasing answer position. Gold-prefix observes the complete
prompt plus each correct answer prefix separately: 2 positions per scalar check
and 5 per four-residue vector. Each stage has 3,402 positions; total 6,804.
Greedy-prefix uses only the historical prompt and already-generated tokens, never
target length or answer as a decoding hint. Each native argmax must equal the
saved token. No call follows EOS. Expected observed inventories, checked from
saved raw predictions, are initial 4,860 and final 2,025; total 6,885. Combined
inventory is 13,689 distinct positions. No cache, batch substitution or extra
sampling is used. A full prefix is forwarded separately at each position.

Gold target probabilities/NLL are teacher-forced measurements. Greedy target
fields are the original answer token at the same position, where one exists;
after a wrong prefix they are **positional references**, not a claim about the
correct conditional continuation. Positions beyond the original terminal EOS
have null targets. Terminal EOS and premature EOS at residue positions must
remain separate. H3 comparisons are paired only at positions present in both
probes; early stopping makes this a selected subset, never a full continuation
comparison. Identical prefixes should have identical raw vectors.

## Exact representation and arithmetic

`logits.f32le` contains exactly 13,689 × 259 × 4 = 14,181,804 bytes: headerless,
row-major, little-endian IEEE-754 float32, token IDs 0 through 258. No score is
modified or quantized. `index.jsonl` is canonical sorted-key compact UTF-8 JSON,
LF terminated, and identifies every vector, byte offset, row/family/task/exposure,
model, probe, position, full prefix, phase, target and historical greedy token.
`metrics.jsonl` joins by record identity. All finite derived scalars use Python
`float.hex()` (binary64); decimal displays in research reports are rounded only
for presentation. Raw binary32 values convert exactly into binary64. NaN,
infinities, wrong sizes, extra/missing/duplicate records fail closed.

Temperature is exactly one. With m=max(z), a=log(fsum(exp(z-m))), log_p[t] is
(z[t]-m)-a and p[t]=exp(log_p[t]) across **all 259 tokens**. This is a stable
logsumexp calculation; logarithms are computed before exponentiation, so tiny
probabilities may underflow to zero without losing finite log probabilities/NLL.
Normalization absolute error must be <=1e-12. EOS probability must agree with a
stable logistic transform of z[257]-LSE(other 258 scores) with absolute and
relative tolerances 1e-12. Where nonsaturated, a separately summed probability
ratio supplies another check. Saturation is recorded, not repaired.

EOS/correct and EOS/best-residue margins are pairwise differences, hence log
probability ratios, **not binary EOS/rest log-odds**. The best residue includes
48,49,50 even on scalar tasks. An optional EOS probability normalized over
48,49,50,257 is explicitly named restricted_four_token_eos_probability. It must
never be presented as full-vocabulary probability. Raw score magnitudes are not
absolute confidence and are not intrinsically comparable between contexts.

Rank is one-based descending score with lower token ID first on ties, matching
native argmax's first-maximum behavior. Residue-only rank and best-non-EOS status
separate weak residue preference from EOS dominance without changing inference.
Common-offset invariance is tested in binary64 without requantizing scores;
finite arithmetic cannot preserve distinctions if an enormous added constant
rounds them away. Tests use exactly representable offsets through 10,000 and
1e-12 tolerances. No claim of invariance to information-destroying rounding.
Summary quantiles use nearest rank at 5%, 50%, 95%; means use math.fsum.

Reference definitions: [PyTorch 2.8 logsumexp](https://docs.pytorch.org/docs/2.8/generated/torch.logsumexp.html),
[LogSoftmax](https://docs.pytorch.org/docs/2.8/generated/torch.nn.LogSoftmax.html),
and [argmax ties](https://docs.pytorch.org/docs/2.8/generated/torch.argmax.html).
These references supply terminology, not imported code or training data.

## Execution, verification and failure policy

Use the existing pinned CPU FP32 PyTorch 2.8.0+cpu environment, one intra-op
thread, strict deterministic algorithms, original environment equality and
unchanged model identities before/after inference. The native reader may restore
optimizer slots to check checkpoint integrity; zero optimizer steps execute.
Each audit/replay has a 1,800-second cooperative forward deadline; a local outer
1,800-second whole-command supervisor is recommended. Result bound is 128 MiB.
No retry, relaxed tolerance, reduced inventory or replacement success manifest.
Failures retain partial files and failure.json (or supervisor stderr/status if
killed). Model identity checks are mandatory after a complete pass.

`python scripts/run_teacher_logit.py freeze --dataset DATA --teacher-run RUN
--previous-result DIAG --out REQUEST` emits the independently retained request
identity. `verify-request` uses those three inputs plus `--request REQUEST
--identity SHA256`. `audit` adds `--out RESULT`. `verify` and `replay-forward`
use `--result RESULT --result-identity SHA256`, externally retained from audit.
All output directories must be new and outside checkout/protected evidence.

Recorded-evidence verify recomputes every numerical metric and aggregate from
all 259 stored scores, plus canonical index/source/input identities. It does
not run a neural model. Separate replay-forward restores the original models,
reruns every full prefix and requires **exact FP32 byte equality**, including
negative zero. No tolerance fallback is allowed. Replay is not optimizer replay.

## Claims and rights

Report H1 EOS dominance, H2 residue preference excluding EOS, H3 prefix effects,
and H4 exposure comparisons as descriptive TRAIN observations. Separate scalar
labels, vector positions, circuit/task/cohort crosses and terminal EOS. A high
model probability is not calibrated correctness/safety or a causal explanation.
Unvisited TRAIN is not a scientific holdout. No generalization or successful
teacher-learning claim follows from this audit. Future interventions require a
new frozen protocol and explicit authorization; no TT-004 activity is enabled.

Source remains MPL-2.0; original dataset/weight rights remain separate. The owner
permits this logit-observation payload in the local handoff; it is not model
parameters or optimizer tensors and grants no weight distribution right.
Tests and exact-head review are required before reporting verified results.
Affected TRI-I01/I02/I03/I05/I07/I08/I10/I11/I12/I13/I14. Historical contracts and
sources are unchanged; this new v2 request/evidence schema is separate.
