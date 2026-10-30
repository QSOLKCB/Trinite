# Retained PR102 task audit and limited CUDA availability observations

**DERIVED preservation note.** The four JSON/exit artifacts are unchanged owner
handoff bytes. This is not a new training result. The earlier successful
CPU/GPU comparison and every historical source and fixture remain unchanged.

The model-free audit contains 3,520 retained output occurrences in 88 labeled
sections, joined from 29 historical journals. Reused seeds, prompts, meters and
views are not independent subjects. All 16 canonical development sections have
zero correct rest-only answers out of ten: nine raw `8` answers and one `6`, each
with EOS, versus gold `7`. Training rest positions are 2/6 and tie starts 0/4;
development uses rests 0/7 and tie starts 1/5. These descriptive coverage gaps
and errors do not establish a causal neural mechanism.

The separate CUDA receipt reports Torch `2.8.0+cu128`, compiled CUDA `12.8`,
`available: false`, `device_count: 0`, and a stderr-observed flag from a child
that exited zero. The fixed NVIDIA utility was **not invoked**: its executable
trust gate returned `unsafe_executable`, with no process return code. That
category can reflect ownership, writable paths, executable type or filesystem
errors; this receipt does not identify which check failed. It is not a malware,
broken-driver or hardware-fault verdict. Both launcher exit files contain zero,
which means reports were emitted, not that CUDA works or a workload is admitted.
Root cause and exact historical host/container/namespace equivalence are unknown.

The read-only preservation checker authenticates bytes and historical sources,
recomputes the complete task audit through the unchanged authoritative checker,
and checks the diagnostic's canonical identity, source hashes and fixed retained
schema/statuses. It never imports or invokes the CUDA diagnostic, Torch, a model,
a checkpoint, training or inference. It does not attest the original invocation,
interpreter path, hardware state, timing, absence of other attempts or full custody.

`source-SHA256SUMS` is the unchanged handoff manifest. Its owner-derived REPORT.md
and HANDOFF.md entries are deliberately omitted from this public subset; their
checksums are retained, not presented as local files. Private raw stderr,
environment and command/source-verification records remain owner-local. The new
provenance file distinguishes those owner assertions from checker-established facts.

See [the preservation contract](../../docs/MUSIC-TASK-AUDIT-RESULT.md).
