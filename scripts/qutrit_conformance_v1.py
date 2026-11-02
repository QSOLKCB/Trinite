"""Bounded public GF(3) tasks; no model, decoder, or optimizer imports."""
from pathlib import Path
import base64
import json
import subprocess

import queue_evaluation_v1 as gate
from trinite.tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / 'external/qutrit-conformance-v1'
BINARY = CRATE / 'target/release/trinite-qutrit-conformance-v1'
RECORD = CRATE / 'target/qutrit-build.json'
CIRCUITS = ('S1', 'S2', 'W', 'S1W', 'WS1', 'S2W', 'WS2')
TASKS = ('apply', 'inverse', 'check')
PROMPT_BYTE_CAP = 120
QEC_SHA = '27bbc798c4e38138ca5b0fff70b15c3271a1917d'
DONOR_SHA = 'a48048c1097f4d1b37a55eca9a1796a823f0c124'


def trits(word):
    gate.require(type(word) is str and len(word) == 4 and all(c in '012' for c in word), 'four canonical GF(3) residues required')
    return tuple(map(int, word))


def validate(value, schema='trinite.qutrit-conformance-observation.v1'):
    gate.keys(value, ('schema', 'word', 'circuit', 'task', 'candidate'))
    gate.require(value['schema'] == schema, 'unknown schema')
    trits(value['word'])
    gate.require(type(value['circuit']) is str and value['circuit'] in CIRCUITS, 'unsupported circuit')
    gate.require(type(value['task']) is str and value['task'] in TASKS, 'unsupported task')
    if value['task'] == 'check':
        trits(value['candidate'])
    else:
        gate.require(value['candidate'] == '' and type(value['candidate']) is str, 'candidate only permitted for check')
    return value


def public(word, circuit='S1', task='apply', candidate=''):
    return validate(dict(schema='trinite.qutrit-conformance-observation.v1', word=word,
                         circuit=circuit, task=task, candidate=candidate))


def family(circuit):
    gate.require(type(circuit) is str and circuit in CIRCUITS, 'unsupported circuit')
    # Entire operator structures, including weights, operands, inverse/check views.
    return 'sum' if circuit in ('S1', 'S2') else 'swap' if circuit == 'W' else 'mixed'


def split(circuit):
    return {'sum': 'train', 'swap': 'validation', 'mixed': 'test'}[family(circuit)]


def permutation(circuit):
    """Computational basis ket permutation, independent of Pauli update equations."""
    gates = ('W', circuit[1:]) if circuit.startswith('W') and circuit != 'W' else ((circuit[:2], 'W') if circuit.endswith('W') and circuit != 'W' else (circuit,))
    result = []
    for first in range(3):
        for second in range(3):
            a, b = first, second
            for item in gates:
                if item == 'W':
                    a, b = b, a
                else:
                    b = (b + int(item[1]) * a) % 3
            result.append((a, b))
    return result


def expected(observation):
    value = validate(observation)
    x0, x1, z0, z1 = trits(value['word'])
    basis = [(a,b) for a in range(3) for b in range(3)]
    perm = permutation(value['circuit'])
    if value['task'] == 'inverse':
        perm = [basis[perm.index(point)] for point in basis]
    # U P U^-1: recover translation and phase on each of nine basis kets.
    image, phases = [], []
    for point in basis:
        a,b = basis[perm.index(point)]
        image.append(perm[basis.index(((a+x0)%3, (b+x1)%3))])
        phases.append((z0*a + z1*b) % 3)
    dx,dy = image[0]
    zz0,zz1 = phases[3], phases[1]
    gate.require(all(image[i] == ((a+dx)%3,(b+dy)%3) and phases[i] == (zz0*a+zz1*b)%3
                     for i,(a,b) in enumerate(basis)), 'non-Pauli conjugation')
    answer = ''.join(map(str,(dx,dy,zz0,zz1)))
    return str(int(value['candidate'] == answer)) if value['task'] == 'check' else answer


def prompt(observation):
    value = validate(observation)
    text = json.dumps(['Q1', value['word'], value['circuit'], value['task'], value['candidate']], separators=(',', ':')) + '\n='
    gate.require(len(text.encode('ascii')) <= PROMPT_BYTE_CAP, 'prompt budget exceeded')
    ByteTokenizer().encode(text)
    return text


def decode_prompt(text):
    gate.require(type(text) is str and text.isascii() and text.endswith('\n='), 'invalid carrier')
    carrier = gate.decode(text[:-2].encode('ascii'))
    gate.require(type(carrier) is list and len(carrier) == 5 and carrier[0] == 'Q1', 'unknown carrier')
    value = public(*carrier[1:])
    gate.require(prompt(value) == text, 'noncanonical carrier')
    return value


def public_only(observation):
    return list(ByteTokenizer().encode(prompt(observation)).input_ids[:-1])


def ledger(observation):
    text = prompt(observation)
    size = 1 if observation['task'] == 'check' else 4
    encoded = ByteTokenizer().encode(text + '0'*size, answer_start=len(text))
    prefix = len(public_only(observation))
    emitted = size+1
    return dict(prompt_ascii_bytes=len(text), prompt_tokens_including_bos=prefix,
                answer_bytes=size, emitted_tokens_including_eos=emitted,
                complete_context_tokens=len(encoded.input_ids), supervised_tokens=sum(encoded.loss_mask),
                full_prefix_forward_positions=sum(prefix+k for k in range(emitted)),
                context_limit=256, reserved_context_cap=126, prompt_byte_cap=PROMPT_BYTE_CAP,
                truncated=False, optimizer_updates=0)


def sources():
    paths = [Path(__file__).resolve(), ROOT/'scripts/run_qutrit_conformance_v1.py', ROOT/'src/trinite/__init__.py', ROOT/'src/trinite/tokenizer.py', ROOT/'src/trinite/contracts.py',
             ROOT/'scripts/queue_evaluation_v1.py', ROOT/'docs/QUTRIT-CONFORMANCE-V1.md', ROOT/'AGENTS.md']
    paths += [p for p in CRATE.rglob('*') if p.is_file() and 'target' not in p.relative_to(CRATE).parts]
    return {str(p.relative_to(ROOT)): gate.digest(gate.read(p, 4*1024*1024)) for p in sorted(paths)}


def identity():
    return dict(sources=sources(), binary_sha256=gate.digest(gate.read(BINARY, 64*1024*1024, allow_links=True)))


def build():
    version = subprocess.run(['rustc', '--version'], capture_output=True, check=True, timeout=10).stdout
    gate.require(version.startswith(b'rustc 1.90.0 '), 'Rust 1.90.0 required')
    before = sources()
    subprocess.run(['cargo', 'build', '--release', '--offline', '--locked'], cwd=CRATE, check=True, timeout=120)
    gate.require(sources() == before, 'source changed during build')
    gate.safe_path(RECORD)
    # Disposable local binding, never a historical scientific receipt.
    RECORD.write_bytes(gate.encode(identity()))


def native(operation, observation):
    validate(observation)
    gate.require(operation in ('observe', 'evaluate-privileged'), 'unknown operation')
    before = identity()
    gate.require(gate.decode(gate.read(RECORD)) == before, 'stale build')
    request = dict(observation, schema='trinite.qutrit-conformance-request.v1')
    output, evidence = gate.bounded_process([str(BINARY), operation], gate.encode(request), timeout=5, cap=8192)
    gate.require(not base64.b64decode(evidence['stderr_b64']), 'unexpected diagnostics')
    gate.require(identity() == before, 'source/binary changed during call')
    return gate.decode(output)


def admit_train(observation):
    """Reject structural nontrain families before invoking the privileged runner."""
    value = validate(observation)
    gate.require(split(value['circuit']) == 'train', 'nontrain structural family')
    gate.require(native('observe', value) == value, 'public mismatch')
    outcome = native('evaluate-privileged', value)
    gate.keys(outcome, ('schema', 'answer'))
    gate.require(outcome == dict(schema='trinite.qutrit-conformance-outcome.v1', answer=expected(value)), 'oracle mismatch')
    text, answer = prompt(value), outcome['answer']
    enc = ByteTokenizer().encode(text+answer, answer_start=len(text))
    return dict(observation=value, answer=answer, encoding=enc.to_dict(), ledger=ledger(value),
                provenance=dict(family=family(value['circuit']), split='train', sources=sources(),
                                source_id='commissioned-gf3-conjugation-v1', qec_sha=QEC_SHA, donor_sha=DONOR_SHA,
                                exposure='public engineering conformance, not fresh held-out data',
                                rights='commissioned finite arithmetic facts; no corpus or teacher',
                                attribution='QSOLKCB/QEC conventions; hmx21/css-decoupling reference (MIT)',
                                training_authorized=False))
