# Symbolic music curriculum preparation v1

**Engineering preparation, not a learning result.** This CPU/standard-library
increment implements an original, small monophonic timing corpus. It does not
train or call a target model, introduce an arithmetic bridge, import E8 geometry,
change native data admission, or clear any learning, Phase 5 or release gate.
Music comes first; whether a later music curriculum helps numerical learning is
an untested hypothesis requiring its own frozen matched experiment.

## Exact symbolic contract

`scripts/symbolic_music_v1/core.py` owns immutable `Meter`, `Event`, `Score`, validation,
JSON-value conversion, exact timeline and symbolic annotations. Durations are
positive **whole-note fractions**, not seconds or floating-point labels. For
example `[3, 8]` is a dotted quarter; `[1, 16]` is a sixteenth. Reduced integer pairs
are required on input. Python callers must supply `Fraction`, never a float.

- Supported notated meters: **3/4, 4/4, 5/8, 7/8, 8/16**. The signature remains
  explicit: 8/16 is not silently relabeled as 2/4 even though bar lengths agree.
- Explicit positive accent groups must sum to the numerator, in denominator
  units. 5/8 `(2,3)` differs from `(3,2)`; 7/8 supports `(2,2,3)`, `(3,2,2)` and
  `(2,3,2)`. A group start is an accent position, not necessarily a note attack.
- Bars contain ordered notes (`pitch` 0–127) or rests (`pitch: null`). Every bar
  fills its exact signature. An event cannot spill across a bar: use matching
  `tie_out`/`tie_in` segments, including across boundaries. Ties require the same
  pitch and immediate adjacency; rest ties, unmatched and dangling ties fail.
- Tempo is integer **quarter notes per minute**, 30–300, independently of meter
  denominator and grouping. Changing tempo never changes notation labels.
- Bounds: 1–8 bars, 1–64 events/bar, positive durations at most one whole note,
  denominator at most 64. Only the five initial meters are accepted; extension
  needs separately reviewed coverage, not silent best-effort coercion.

`labels()` returns attack count (excluding tied continuations), notated rest
count (not rest runs), tie-edge count, absolute bar boundaries (including final
end), accent-group onsets (excluding final end), and total notated duration.
`oracle.py` independently checks annotations using integer ticks. No annotation
is measured from synthesized sound.

## Small deterministic corpus and leakage boundary

`corpus.py` enumerates **160 scores**: eight parent rhythmic rest-mask families,
ten meter/grouping settings and two variants. Each score has two explicit bars.
The base pulse is an eighth in /4 meters, otherwise the denominator unit.
Positions 1–3 can be rests; the first and last pulses remain notes. The second
variant coalesces equal adjacent symbols, splits the first note with a tie,
adds a bar-crossing tie, transposes pitch and changes tempo. These are correlated
variants, not independent examples or an isolated causal ablation.

Whole parent families are SHA-256 ranked under fixed schema/seed 0: four train,
two validation and two test (**80/40/40 rows**). All meter, grouping, notation,
pitch and tempo variants of a parent remain in its split. Exact input duplicates
are rejected. The rest-mask parent is deliberately coarser than the rendered
score; no per-row random split. Families still share a construction grammar and
many short motifs; this is not a claim of universal musical independence.

The task is `annotate-rhythm-v1`: explicit notation is input, rhythm annotations
are target. It is **not** meter inference, melody prediction, audio recognition,
or a native model-ready training protocol. Some labels follow directly from
notated meter/group fields; those are legitimate task operands, not a hidden
answer benchmark. The full structured scores may exceed native token/context
budgets. No tokenizer or truncation adapter is added.

Bundle `*.inputs.jsonl` contains only `{task, score}`; IDs, source hashes,
families, split tags, lineage and targets live separately in `*.audit.jsonl`.
Consumers must not tokenize audit files, manifests or receipts. Merely keeping
files separate is not a training firewall: a future adapter must enforce its own
train-only allowlist and frozen exposure controls. All current rows, including
software test splits, are public/development-exposed and **not fresh confirmatory
heldouts**. A future learning study needs fresh families and separate review.

## Source-specific rights and receipts

This is a bounded original formal generator commissioned by Trent Slade /
QSOL-IMC, not a music collection. It consumes no composition, sheet music,
recording, teacher response, donor score, preset, sample or external asset.
`admission.json` records exact source hashes, authorship, generation and checking
procedure, formal-fact rights basis, reviewer/date, scope and limitations.
Its outcome is **engineering-conformance-only**, not blanket legal certification,
blanket synthetic-data admission, or permission to feed native training.
Existing [DATA](DATA.md) policy and historical admissions remain unchanged.

The bundle binds the complete symbolic runtime/launcher and this contract via
SHA-256, plus hashes of input-only, audit and admission bytes. Rendering is not
part of symbolic truth and is absent from that source closure. Verification
requires the independently retained manifest SHA-256, regenerates all expected
bytes from current source and rejects altered labels, source, splits, inventory,
symlinks or extra audio. It is read-only exact source/corpus replay, **not** closed
PROVENANCE custody, external authorship attestation or optimizer replay. Retain
failed bundles; generate only into a fresh destination, never reseal history.

## Usage (checkout root, Python 3.11+)

```sh
PYTHONPATH=scripts python -B -m unittest discover -s tests/music_v1 -v
PYTHONPATH=scripts python -O -B -m unittest discover -s tests/music_v1 -v
# The parent output directory must already exist; the destination must not.
PYTHONPATH=scripts python scripts/run_symbolic_music.py generate /tmp/music-v1
# Retain the printed SHA-256 independently, then supply it verbatim:
PYTHONPATH=scripts python scripts/run_symbolic_music.py verify /tmp/music-v1 \
  --manifest-sha256 <retained-sha256>
# Optional, separate audition. Never write audio inside the symbolic bundle.
PYTHONPATH=scripts python scripts/run_symbolic_music.py render /tmp/music-example.wav --row 1
```

`render.py` is an opt-in standard-library sine-wave audition: mono 16-bit PCM,
8–48 kHz, absolute rational sample boundaries rounded down, at most 64 seconds
(about 6 MiB PCM). Tied chains use one continuous oscillator/envelope; rests are
silent. Accent positions are **not synthesized as dynamic accents** in this
minimal renderer. Floating synthesis is reproducible only within a specified
Python/libm environment, not a cross-platform audio-byte identity guarantee.
Audio neither supplies labels nor alters corpus generation or split assignment.
There is no root rendered audio, audio fixture or automatic playback.

## Selective donor influence, not wholesale import

Inspected immutable donor contracts:

| Donor | Pin | Borrowed architectural idea / excluded limitation |
| --- | --- | --- |
| [SONIFICATION](https://github.com/QSOLKCB/SONIFICATION/tree/27cfea99a3bab445c74b1aba5c38993fe931f568) | `27cfea99a3bab445c74b1aba5c38993fe931f568` | `src/d4-triality-algebra-v2-artifacts.mjs`: symbolic export and source/manifest receipts, no root rendered audio. Its Python lab float beat representation is not exact truth here. |
| [E8_MUSIC](https://github.com/QSOLKCB/E8_MUSIC/tree/d8e5983d84af03f03a969abe3356dcf80c0e0e97) | `d8e5983d84af03f03a969abe3356dcf80c0e0e97` | `js/audio-engine.js`: separate rendering. Fixed sixteenth steps/four-step accents and E8Core dependency are not imported. |
| [SPECTRAL](https://github.com/QSOLKCB/SPECTRAL/tree/2aa091c7e0eacd925fe5a44583de9e87c3ae7570) | `2aa091c7e0eacd925fe5a44583de9e87c3ae7570` | RB-420 deterministic fixed-frame render separate from live audio; its 16-step drum lane and FNV receipt are not arbitrary-meter notation or cryptographic custody. |

No donor source or assets are copied. SONIFICATION and E8_MUSIC are MPL-2.0;
SPECTRAL/RB-420 is MIT. Attribution: Trent Slade / QSOL-IMC. This original code
uses Trinite's MPL-2.0. Future copying must retain applicable source/notices and
license obligations; ownership of these repositories does not establish that all
third-party rights are waived. No donor demonstrates music-to-arithmetic transfer.

## Compatibility and verification scope

Affected requirements: **TRI-I01, I03, I05, I07–I09, I11–I14**. The isolated
`scripts/symbolic_music_v1` checkout-only package imports only the standard library.
It stays outside `src/trinite` because historical receipts recursively inventory
that native source tree; adding even unrelated modules there would break original
source closure. No installed native API is extended. Model/tokenizer/updater,
existing admission/scorers, geometry, all historical fixtures and producer source
commitments remain unchanged. E8 is wholly excluded from this baseline, not an
uncontrolled optional input. Any later geometry comparison or numerical bridge
needs a separate versioned proposal and matched controls.

Tests cover exact parser/notation rejection, hand-worked and integer-tick oracle
parity, all requested meters, grouped variants, ties, tempo separation, duplicate
and split tampering, source/input binding, read-only replay, exclusive outputs,
and renderer isolation. The dedicated CI repeats them on Python 3.11–3.13 with
normal and optimized interpreters. These are software-conformance checks only;
no target model, research training, success claim or new empirical gate is run.
