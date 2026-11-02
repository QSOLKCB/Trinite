"""OMI-only engineering admission. No model imports, optimizer or experiment runner."""
from pathlib import Path
import base64
import hashlib
import json
import subprocess

import queue_evaluation_v1 as gate
from trinite.tokenizer import ByteTokenizer

ROOT = Path(__file__).resolve().parents[1]
CRATE = ROOT / 'external/omi-v1'
BINARY = CRATE / 'target/release/trinite-omi-v1'
RECORD = CRATE / 'target/omi-build.json'
TASKS = ('next', 'inverse', 'forward', 'backward', 'period')
PROMPT_BYTE_CAP = 120  # Includes compact JSON and separator, excludes BOS.
DONOR = '9d228aed9791dccac6e62c9dbc55738cc26ac679'


def validate(value, schema='trinite.omi-observation.v1'):
    gate.keys(value, ('schema', 'word', 'task', 'horizon'))
    gate.require(value['schema'] == schema, 'unknown OMI schema')
    gate.integer(value['word'], 0, 65535)
    gate.integer(value['horizon'], 1, 7)
    gate.require(type(value['task']) is str and value['task'] in TASKS, 'unknown task')
    gate.require(value['task'] in ('forward', 'backward') or value['horizon'] == 1,
                 'single-step/period horizon must be one')
    return value


def arithmetic_delta(word):
    """Independent bit-parity arithmetic; no bitwise shifts/XOR or Rust execution."""
    gate.integer(word, 0, 65535)
    rotations = [(word * 2**n) % 65536 + word // 2**(16-n) for n in (1, 3, 14)]
    return sum((sum(v // 2**b % 2 for v in (*rotations, 0x5A3C)) % 2) * 2**b
               for b in range(16))


def orbit(word):
    gate.integer(word, 0, 65535)
    states, current = [], word
    for _ in range(8):
        states.append(current)
        current = arithmetic_delta(current)
        if current == word:
            return tuple(states)
    raise gate.GateError('OMI orbit did not close')


def family(word):
    return min(orbit(word))


def split(word):
    # All task/horizon forms and inverse views of the same orbit stay together.
    bucket = hashlib.sha256(f'omi-v1:{family(word):04X}'.encode('ascii')).digest()[0] % 10
    return 'validation' if bucket == 8 else 'test' if bucket == 9 else 'train'


def public(word, task='next', horizon=1):
    return validate(dict(schema='trinite.omi-observation.v1', word=word, task=task, horizon=horizon))


def prompt(observation):
    value = validate(observation)
    carrier = ['O1', f"{value['word']:04X}", value['task'], value['horizon']]
    text = json.dumps(carrier, separators=(',', ':'), ensure_ascii=True) + '\n='
    gate.require(len(text.encode('ascii')) <= PROMPT_BYTE_CAP, 'prompt budget exceeded')
    ByteTokenizer().encode(text)
    return text


def decode_prompt(text):
    gate.require(type(text) is str and text.isascii() and text.endswith('\n='), 'invalid carrier')
    carrier = gate.decode(text[:-2].encode('ascii'))
    gate.require(type(carrier) is list and len(carrier) == 4 and carrier[0] == 'O1', 'unknown carrier')
    word = carrier[1]
    gate.require(type(word) is str and len(word) == 4 and all(c in '0123456789ABCDEF' for c in word), 'invalid hex word')
    value = public(int(word, 16), carrier[2], carrier[3])
    gate.require(prompt(value) == text, 'noncanonical carrier')
    return value


def public_only(observation):
    return list(ByteTokenizer().encode(prompt(observation)).input_ids[:-1])


def ledger(observation):
    text = prompt(observation)
    answer_bytes = 1 if observation['task'] == 'period' else 4
    encoded = ByteTokenizer().encode(text + '0' * answer_bytes, answer_start=len(text))
    prefix = len(ByteTokenizer().encode(text).input_ids) - 1
    emitted = answer_bytes + 1
    return dict(prompt_ascii_bytes=len(text), prompt_tokens_including_bos=prefix,
                answer_bytes=answer_bytes, emitted_tokens_including_eos=emitted,
                complete_context_tokens=len(encoded.input_ids), supervised_tokens=sum(encoded.loss_mask),
                full_prefix_forward_positions=sum(prefix + k for k in range(emitted)),
                context_limit=256, prompt_byte_cap=PROMPT_BYTE_CAP, reserved_context_cap=126,
                truncated=False, optimizer_updates=0)


def expected(observation):
    value = validate(observation)
    states = orbit(value['word'])
    if value['task'] == 'period':
        return str(len(states))
    n = -value['horizon'] if value['task'] in ('inverse', 'backward') else value['horizon']
    return f'{states[n % len(states)]:04X}'


def sources():
    paths = [Path(__file__).resolve(), ROOT/'scripts/run_omi_module_v1.py', ROOT/'src/trinite/__init__.py', ROOT/'src/trinite/tokenizer.py', ROOT/'src/trinite/contracts.py',
             ROOT/'scripts/queue_evaluation_v1.py', ROOT/'docs/OMI-MODULE-V1.md', ROOT/'AGENTS.md']
    paths += [p for p in CRATE.rglob('*') if p.is_file() and 'target' not in p.relative_to(CRATE).parts]
    return {str(p.relative_to(ROOT)): gate.digest(gate.read(p, 4*1024*1024)) for p in sorted(paths)}


def identity():
    return dict(sources=sources(), binary_sha256=gate.digest(gate.read(BINARY, 64*1024*1024, allow_links=True)))


def build():
    version = subprocess.run(['rustc', '--version'], cwd=CRATE, capture_output=True, check=True, timeout=10).stdout
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
    request = dict(observation, schema='trinite.omi-request.v1')
    output, evidence = gate.bounded_process([str(BINARY), operation], gate.encode(request), timeout=5, cap=8192)
    gate.require(not base64.b64decode(evidence['stderr_b64']), 'unexpected diagnostics')
    gate.require(identity() == before, 'source/binary changed during call')
    return gate.decode(output)


def admit_train(observation):
    """Train-only API. Reject nontrain orbit BEFORE any privileged native call."""
    validate(observation)
    gate.require(split(observation['word']) == 'train', 'nontrain orbit cannot export target')
    gate.require(native('observe', observation) == observation, 'native public mismatch')
    outcome = native('evaluate-privileged', observation)
    gate.keys(outcome, ('schema', 'answer'))
    gate.require(outcome == dict(schema='trinite.omi-outcome.v1', answer=expected(observation)), 'oracle mismatch')
    text, answer = prompt(observation), outcome['answer']
    enc = ByteTokenizer().encode(text+answer, answer_start=len(text))
    return dict(observation=observation, answer=answer, encoding=enc.to_dict(), ledger=ledger(observation),
                provenance=dict(family=f'{family(observation["word"]):04X}', split='train', sources=sources(),
                                exposure='public engineering conformance, not fresh held-out data',
                                rights='commissioned finite arithmetic facts; no corpus or teacher',
                                attribution='Brian Thorne ARCH OMI; VANILLAJS encoding', donor_sha=DONOR,
                                training_authorized=False))
