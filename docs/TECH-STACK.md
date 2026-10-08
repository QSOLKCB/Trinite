# Tech stack

Status: selected design; exact package versions and a compatible lock are a Phase 1 implementation gate.

| Concern | Choice | Reason / boundary |
| --- | --- | --- |
| Primary language | Python 3.11 reference environment | One language for model, experiments, inspection, and evidence integration |
| Model and native training | PyTorch, explicit eager modules | Visible forward/gradient path; CPU reference and separate optional GPU lane |
| Numerical analysis | NumPy when vector analysis needs it | Detached arrays; no silent dtype conversion |
| Tokenization | Small in-project byte tokenizer | Fully auditable mapping and no inherited tokenizer corpus |
| Tensor artifacts | safetensors plus versioned JSON manifests | Explicit tensor inventory; no arbitrary pickle execution for model loading |
| Tests and CLI | unittest and argparse initially | Standard-library harness and entrypoints with minimal dependencies |
| Geometry reference | Pinned QSOL-GEO-REASON adapter or frozen cross-check fixtures | Reuse measurement definitions after compatibility tests |
| Observation | Pinned PROVENANCE adapter | Independent run/artifact lineage; preserve its canonicalization rules |
| Automation | GitHub Actions for CPU CI and manually dispatched training | Workflow orchestration is distinct from provisioned compute |
| Optional accelerated training | Unsloth only after compatibility and parity checks | Preserve the idea's training direction without making core semantics depend on an unverified custom-model backend |
| Deployment optimization | Defer a native CPU kernel until measured need | PyTorch CPU establishes the reference; packed storage alone does not establish speed |

Python/PyTorch is the first implementation stack. A second runtime language is deliberately undecided until export and profiling establish what it must implement. A later bitnet.cpp experiment or small native kernel must preserve the model/tokenizer/quantizer contracts and pass parity checks; compatibility is not assumed.

## Unsloth decision

Unsloth's published requirements and architecture support are backend-specific. The first Trinite model changes linear-layer semantics and trains from scratch; this plan has not verified support for that exact model. Native PyTorch is therefore the authoritative first lane. Keep Unsloth optional and separately pinned. If a compatible path is demonstrated, compare outputs, gradients, export codes, replay scope, and measured cost before adopting it.

A fine-tuned upstream pretrained model may be an explicitly labelled comparator. It cannot become the native Trinite artifact under the strict training-data and random-initialization contracts. No GAN or teacher/student pipeline is selected. Architecture choice alone does not make training deterministic.

## Sources reviewed for this design

| Source | What it informs |
| --- | --- |
| [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html) | Reproducibility scope and controls; cross-platform exact replay is not guaranteed |
| [Deterministic algorithms](https://docs.pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html) | Explicit failure on unsupported deterministic operations |
| [Unsloth requirements](https://unsloth.ai/docs/get-started/fine-tuning-for-beginners/unsloth-requirements) | Optional backend provisioning; recheck at integration time |
| [BitNet b1.58 paper, arXiv:2402.17764](https://arxiv.org/abs/2402.17764) | Ternary-weight prior art; no transfer of its performance claims to Trinite |
| [CPU BitNet infrastructure, arXiv:2410.16144](https://arxiv.org/abs/2410.16144) | Specialized CPU kernels as a later candidate, subject to compatibility |
| [QSOL-GEO-REASON at reviewed revision](https://github.com/QSOLKCB/QSOL-GEO-REASON/tree/e770f585bf3136b47a657772156116d5f02b1e77) | Geometry contracts and evidence limits |

Sources are prior art and engineering references, not Trinite results. API/version claims must be rechecked when the first dependency lock is produced. No dependency is installed or introduced by this PR.
