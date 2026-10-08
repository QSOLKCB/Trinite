# qBraid agent stress-test handoff

Status: initial pre-launch plan requested by Trent. No qBraid run, account,
agent authorization, quantum execution or independent replication is claimed.
Update this handoff when a stable Trinite candidate and runnable release
artifacts exist. The current model remains a classical CPU research decoder;
ternary and four-state weights do not require quantum hardware.

## Scope and entry conditions

Read [INSTRUCTIONS.md](INSTRUCTIONS.md), [INVARIANTS.md](INVARIANTS.md),
[DATA.md](DATA.md) and the frozen protocol for the proposed run. Use a disposable
checkout pinned to an immutable candidate commit, the repository's hash-locked
CPU environment, one CPU thread and deterministic algorithms. Record the actual
OS, Python, packages and CPU. Strict replay may reject a different environment;
report that rejection rather than relaxing checks or replacing historical bytes.

qBraid Lab provides notebook/terminal and coding-agent interfaces; choose the
available agent and CPU instance explicitly using the then-current
[Lab documentation](https://qbraid.com/lab) and
[CLI guide](https://docs.qbraid.com/v2/cli/user-guide/overview).
Record agent name/version or model identifier, prompt, tool permissions, session
identifier, runtime image and resource/time budget. Do not assume free compute,
a particular available agent, or compatibility of a quantum runtime API with
Trinite. Platform installation belongs outside Trinite's numerical core.

## Initial jobs

| Job | Expected evidence |
| --- | --- |
| Conformance | Root, upstream, CPU, comparison and foundations suites; full logs, counts and exit status, including failures |
| Integrity | Corrupted/missing/extra evidence, stale source identities, wrong request/environment, changed admission fields; explicit rejection before accepting a result |
| Safe restoration | Traversal, duplicate names, case-folded file/ancestor collisions in both orders, oversized objects, bad digests and existing destination; no partial publication |
| Replay | Interrupted/resumed checkpoints for every supported lane; metadata/lane/config and optimizer mutations rejected; exact replay only within its declared environment |
| Resource boundaries | Worker timeouts, token/context/tensor/archive bounds, failed aggregation and upload/retention behavior; retain original outputs and a failed/blocked outcome |
| Model assessment | Separately frozen tasks, prompt-only generation, complete answer/EOS scoring, training-only baselines and family/exposure controls; no tuning against held-out labels |
| Export / edge, once implemented | Invalid importer inputs, tensor/logit/token/capture parity and measured CPU memory/time; identify each device and packed format |

Start with offline software checks. Full training, broader stress budgets,
provider jobs or public reports need an explicitly selected run protocol and
budget. The existing foundations learning gate is blocked; an agent must not
unlock test scoring by editing the decision, changing thresholds or ignoring a
failed dense task. Training-only development belongs to a new frozen protocol.

## Review packet and acceptance

Retain candidate/source and dependency identities, exact commands and prompts,
resource limits, start/end times, stdout/stderr, exit/timeout records, requests,
checkpoints, predictions, summaries, closed PROVENANCE bundles and current
verifier reports where the corresponding stage produces them. Include a
bounded inventory with hashes and an accessible archive location. Verify the
archive after retrieval. A provider job identifier alone is not retained evidence.

Keep failures and original protocol/results immutable. Proposed fixes go into a
separate reviewed patch and new source-bound request. Stress runs do not admit
new data, publish weights, issue release tags, merge changes or send reports to
others automatically. Label synthetic fault injection separately from observed
model behavior. The owner reviews unresolved findings and the implementation's
scoped formal-verification report before launch; green checks do not establish
reasoning ability, universal hardware support or release readiness.
