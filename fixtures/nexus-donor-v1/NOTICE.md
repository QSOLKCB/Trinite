# NEXUS donor attribution and scope

NEXUS contributors / QSOLKCB authored the upstream software. Source files here
retain their original MPL-2.0 notices and exact bytes. The complete license is
[upstream/LICENSE](upstream/LICENSE). Trinite's reference and adapters are
MPL-2.0 adaptations commissioned by Trent Slade / QSOL-IMC.

- Runtime release v1.0.0: commit
  `24c7b5bb8b158d84e08b10ccc8fdd16b6889b7e4`.
- Separate Lean formal package 0.1.0: commit
  `a9486c34eebd395c8060f8b2b03885eff10115c6`, Lean 4.19.0.
- Origin: https://github.com/QSOLKCB/NEXUS
- Per-file revision, Git blob identity and SHA-256: [source-lock.json](source-lock.json).

Only the declared dependency subset is retained, not a source distribution of
all NEXUS features. The build copies verified bytes into a new attempt directory.
It removes exactly one `pub mod atlas;` line from ve/mod.rs to exclude the
unrelated atlas dependencies, and appends the separately identified test adapter
to lattice.rs. No donor numerical body is changed. The adapter sets private
registers to valid supplied states and calls donor methods. This is not the
unmodified donor CLI or public arbitrary-state API. Every compiled source byte
is identified in the attempt receipt. Lean source files remain unmodified.

Software licensing and formal-fact data assessment are separate. The generated
integer facts and minimal JSON prompts contain no copied donor prose, benchmark
text or model answers. The policy admits these only as fully visible software
conformance items; it does not admit a native training curriculum. No legal
certification, quantum/compression claim or end-to-end Rust refinement follows.
The Zenodo deposit is not required by this checker and is not independently
verified here. Placeholder upstream citation metadata is deliberately not copied.
