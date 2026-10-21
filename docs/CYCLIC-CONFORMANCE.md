# Exact cyclic-angle conformance v1

This is a standalone **CPU software-conformance** reference and finite checker,
not a Trinite model component, admitted training curriculum, geometric reasoning
result or physical claim. NEXUS, integral-geometry learning and all historical
sources/evidence are unchanged. Geometry loss remains zero.

## Mathematics and interfaces

`CyclicElement(k, n)` is an additive element of Z/nZ with positive integer n and
canonical integer 0 <= k < n. Ticks/moduli reject floats, strings and booleans;
notation factories accept only integers or exact `Fraction` values. Normalization
handles negative and arbitrarily large integer ticks. Changing n changes the
group, not merely its notation. A 60-tick circle has no privileged geometry.

Addition requires the same modulus. `inverse()` means additive inverse; `neg()`
is an alias. Order is n/gcd(n,k), including order 1 for identity and the trivial
group. The checker independently computes order by a bounded integer counter,
without calling the class addition or gcd. The public diagnostic
`order_bruteforce()` does reuse class addition and costs time linear in the order;
it is not the checker's independent oracle.

Exact projection from tick k on n to m exists precisely when k*m is divisible by
n. It preserves the rational turn, has an exact reverse on its image and preserves
element order. Every tick embeds angle-preservingly iff n divides m. Partial
shared-angle subgroups also exist when neither divides the other: 12/60 = 20/100,
whereas a fifth-turn has no tick on the 61-grid. No rounding fallback exists.

Shortest signed displacement has absolute value at most n/2. At an even-grid
half turn, the declared policy selects positive/negative or rejects (default).
Invalid policy names are rejected even on inputs without a tie. The Z4-to-60
mapping 0/15/30/45 is exact for a single register; it is not proof that the whole
coupled NEXUS lattice is a rotation model.

Degrees and rational turns are exact representations. DMS output is canonical
nonnegative [0,360) degrees with minutes/seconds below 60. The inherited DMS input
interface is explicitly **normalizing**: nonnegative overflow minutes/seconds
are allowed, sign is carried on integer degrees, and negative subdegree values
cannot be expressed by that interface. Use `from_degrees(Fraction(-1,2), n)` or
`from_turn` for such values; no negative-zero convention is implied.

The sexagesimal codec is for nonnegative **integer** positional values, not an
arbitrary fractional numeral language. Encoder output is canonical, including
zero `(0,)`. Decoder accepts finite lists/tuples and leading zeros; re-encoding
normalizes them. Thus decode(encode(x)) = x, but encode(decode(digits)) equals the
canonical spelling, not every accepted spelling. Public reference operations
have no hard whole-process resource guarantee. Ordinary attribute writes,
deletes and reinitialization are rejected; this is value semantics, not a Python
security boundary against deliberate `object.__setattr__` or hostile subclasses.

## Input provenance and data scope

The user supplied a Google AI Mode discussion/sketch, an initial Fable 5.1 module,
and a corrected Fable module. [Input provenance](../fixtures/cyclic-conformance-v1/input-provenance.json)
records their exact SHA-256s, sizes, origin and limits. Original files remain
unchanged outside the public repository; their digests document lineage, **not**
a claim of independently retrievable public donor closure. No full chat is
published, no separately verified external license/author declaration is asserted,
and external claims of 31 passing tests are not imported as our results.

The repository module and split regression tests explicitly adapt that corrected
input. Changes include finite decoder-container validation, accurate displacement
range documentation, separation of tests/runtime and a new bounded independent
counter oracle with source-bound records. Repository adaptations follow its
software license; software rights do not imply native data admission.

Every generated row carries source ID, source-closure generator revision,
no-RNG seed/procedure, exact formal specification, minimal JSON prompt, answer,
family/split procedure, transformation lineage, verifier outcome and request
identity. All rows are **conformance-only**, with no train/validation/test split.
Answers are recomputed from exact formal mathematics, not taken from teacher
prose. This narrow generated-fact software test is not a policy change to DATA,
a native admission result or permission to train on these exposed examples.

## Run and acceptance

Python 3.11+ standard library only. The pure reference is portable. The receipt
checker and its contract suite require Linux for `RUSAGE_SELF` KiB resource units.

```sh
PYTHONPATH=scripts python -B -m unittest discover -s tests/cyclic_conformance_v1 -v
PYTHONPATH=scripts python -O -B -m unittest discover -s tests/cyclic_conformance_v1 -v
mkdir /fresh/cyclic-evidence
python -I -S scripts/check_cyclic_conformance.py freeze --request /fresh/cyclic-evidence/request.json
# Supply the independently retained printed SHA-256:
python -I -S scripts/check_cyclic_conformance.py check --request /fresh/cyclic-evidence/request.json --request-identity SHA256 --output-dir /fresh/cyclic-evidence/run
```

The checker requires isolated/no-site startup and captures project source bytes
rather than using project bytecode caches. The fixed request binds runtime,
launcher, tests, source-provenance record, documentation and workflow. A changed source
or case set requires a new request. Only the exact regenerated bounded workload
is accepted: moduli at most 360, fewer than 2,000 cases, request/rows capped at
8 MiB. Unit tests exhaust small-modulus axioms, sample larger associativity with
a fixed seed, and test integer/rational projection and rejected contracts. They
are finite implementation checks, not general formal proofs.

`check` creates an exclusive attempt directory, retains request and partial
JSONL rows, and publishes one atomic/no-clobber `result.json` combining result and
controller status. Failed checks have status failed; missing result is incomplete,
never success. A final storage failure can retain rows without a result, and no
claim promises a new durable receipt during total storage failure. Reuse of an
existing attempt is rejected; retain failures and use a distinct new directory.
Elapsed time and whole-process peak RSS are observational, not model benchmarks.
Replay uses a fresh directory and compares exact request/rows bytes, not resource
or platform observations.

CI runs normal and -O tests and fresh replay on Python 3.11–3.13. Its uploaded
attempt records expire after 14 days; they are not permanent repository evidence.
The source/provenance record is tracked, but no learning outcome is claimed or
archived by this PR. A separate reviewed protocol is required before training.

Affected requirements: TRI-I01/I03/I05/I07/I08/I09/I11/I12/I13/I14, with the narrow
software-conformance scopes above. No later research gate is unlocked.
