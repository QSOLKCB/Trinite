"""Strict bounded, offline Bonsai SSE capture profile (not an answer scorer).

Grammar: content/initial-role frames*, optional usage-only frame, exactly one
terminal choice-0 frame, optional usage-only frame, [DONE]. Usage is allowed only
in the terminal frame or immediately adjacent to it; repeated usage must agree.
A role can occur only in the first choice frame. Empty content chunks are valid;
null content is valid only on that initial assistant-role frame. Terminal delta
may contain final content, but no later choice frame is accepted. The terminal
reason is stop or length; length is truncation metadata, never correctness.

SSE framing accepts LF, CRLF or CR, one or more data lines per event joined with
LF, and requires blank-line JSON event termination. Final [DONE] may end with
one newline at EOF, as in the retained historical capture, or a blank line.
This deliberately narrow capture
profile rejects comments, event/id/retry fields, empty events and any bytes after
the terminating [DONE] event. JSON metadata is allowlisted; reasoning, tools,
errors, unknown fields, duplicate keys and nonfinite numbers fail closed.
UTF-8 content is concatenated verbatim, without stripping, JSON extraction,
reasoning-tag removal or answer repair. Input is the complete retained stream;
the caller must independently retain its original bytes and source identity.
The stream limit is 1 MiB and reconstructed content limit is 64 KiB; overflow
fails rather than returning a truncated prefix. These are retention limits,
not the separate answer acceptance/scoring limit.
"""
import json
import math
import re

MAX_STREAM_BYTES = 1024 * 1024
MAX_CONTENT_BYTES = 64 * 1024


class SSEError(ValueError):
    """The retained stream does not satisfy this capture profile."""


def _fail(message):
    raise SSEError(message)


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            _fail('duplicate JSON key: ' + key)
        result[key] = value
    return result


def _constant(value):
    _fail('nonfinite JSON constant: ' + value)


def _integer(value, label):
    if type(value) is not int or value < 0:
        _fail(label + ' must be an exact nonnegative integer')


def _usage(value):
    required = {'completion_tokens', 'prompt_tokens', 'total_tokens'}
    if type(value) is not dict or not required <= value.keys() or value.keys() - required - {'prompt_tokens_details'}:
        _fail('unsupported usage shape')
    result = {key: value[key] for key in sorted(required)}
    for key, count in result.items():
        _integer(count, key)
    if result['total_tokens'] != result['completion_tokens'] + result['prompt_tokens']:
        _fail('inconsistent usage total')
    if 'prompt_tokens_details' in value:
        details = value['prompt_tokens_details']
        if type(details) is not dict or set(details) != {'cached_tokens'}:
            _fail('unsupported prompt token details')
        _integer(details['cached_tokens'], 'cached_tokens')
        if details['cached_tokens'] > result['prompt_tokens']:
            _fail('cached tokens exceed prompt tokens')
    return result


def _metadata(frame):
    allowed = {'choices', 'usage', 'id', 'model', 'object', 'created', 'system_fingerprint', 'timings'}
    if type(frame) is not dict or frame.keys() - allowed or 'choices' not in frame:
        _fail('unsupported frame fields')
    for key in ('id', 'model', 'system_fingerprint'):
        if key in frame and type(frame[key]) is not str:
            _fail('invalid ' + key)
    if 'object' in frame and frame['object'] != 'chat.completion.chunk':
        _fail('invalid chunk object')
    if 'created' in frame:
        _integer(frame['created'], 'created')
    if 'timings' in frame:
        timings = frame['timings']
        counts = {'cache_n', 'prompt_n', 'predicted_n'}
        durations = {'prompt_ms', 'prompt_per_token_ms', 'prompt_per_second',
                     'predicted_ms', 'predicted_per_token_ms', 'predicted_per_second'}
        if type(timings) is not dict or timings.keys() - counts - durations:
            _fail('unsupported timings')
        for key, value in timings.items():
            if key in counts:
                _integer(value, key)
            elif (type(value) not in (int, float) or value < 0
                  or (type(value) is float and not math.isfinite(value))):
                _fail('invalid timing')


def parse_sse(raw: bytes) -> dict:
    """Return exact content bytes, terminal/truncation metadata and optional usage.

    Missing usage is None with an explicit reason, never a fabricated zero.
    No I/O, runtime loading, model calls or target-answer scoring occurs here.
    """
    if type(raw) is not bytes or not raw or len(raw) > MAX_STREAM_BYTES:
        _fail('expected nonempty complete bytes within stream bound')
    try:
        text = raw.decode('utf-8', errors='strict')
    except UnicodeError as exc:
        raise SSEError('invalid UTF-8 stream') from exc
    lines = re.split(r'\r\n|\r|\n', text)
    if lines[-2:] == ['data: [DONE]', '']:
        lines.append('')  # Explicit archived DONE-at-EOF grammar, not content repair.
    if len(lines) < 3 or lines[-2:] != ['', '']:
        _fail('unterminated SSE event')
    frames = []
    data = []
    for line in lines[:-1]:
        if line == '':
            if not data:
                _fail('empty SSE event')
            frames.append('\n'.join(data))
            data = []
        elif line == 'data' or line.startswith('data:'):
            value = '' if line == 'data' else line[5:]
            data.append(value[1:] if value.startswith(' ') else value)
        else:
            _fail('unsupported SSE field')
    content = []
    content_size = 0
    finish = None
    usage = None
    usage_full = None
    pending_usage = False
    choices_seen = 0
    done = False
    for position, payload in enumerate(frames):
        if done:
            _fail('frame after DONE')
        if payload == '[DONE]':
            if finish is None:
                _fail('DONE without terminal choice')
            done = True
            continue
        try:
            frame = json.loads(payload, object_pairs_hook=_object, parse_constant=_constant)
        except (ValueError, RecursionError) as exc:
            raise SSEError('malformed JSON frame: ' + str(exc)) from exc
        _metadata(frame)
        choices = frame['choices']
        if type(choices) is not list or len(choices) > 1:
            _fail('expected at most one choice')
        frame_usage = frame.get('usage')
        if frame_usage is not None:
            parsed_usage = _usage(frame_usage)
            if usage_full is not None and usage_full != frame_usage:
                _fail('conflicting duplicate usage')
            usage, usage_full = parsed_usage, frame_usage
        if not choices:
            if frame_usage is None:
                _fail('empty choices without usage')
            if pending_usage or (finish is not None and position + 1 < len(frames) and frames[position + 1] != '[DONE]'):
                _fail('usage-only frame must be adjacent to terminal')
            pending_usage = finish is None
            continue
        if finish is not None:
            _fail('choice after terminal')
        choice = choices[0]
        if type(choice) is not dict or set(choice) != {'index', 'delta', 'finish_reason'}:
            _fail('unsupported choice fields')
        if type(choice['index']) is not int or choice['index'] != 0:
            _fail('expected choice index zero')
        reason = choice['finish_reason']
        if reason is not None and (type(reason) is not str or reason not in ('stop', 'length')):
            _fail('unsupported terminal reason')
        if pending_usage and reason is None:
            _fail('usage-only frame must immediately precede terminal')
        if frame_usage is not None and reason is None:
            _fail('usage on nonterminal content frame')
        delta = choice['delta']
        if type(delta) is not dict or delta.keys() - {'role', 'content'}:
            _fail('unsupported delta; reasoning/tools are not content')
        if 'role' in delta and (choices_seen != 0 or delta['role'] != 'assistant'):
            _fail('assistant role allowed only on initial choice')
        if 'content' in delta:
            value = delta['content']
            if value is None and choices_seen == 0 and delta.get('role') == 'assistant':
                pass
            elif type(value) is not str:
                _fail('content must be a string')
            else:
                try:
                    encoded = value.encode('utf-8', errors='strict')
                    content_size += len(encoded)
                    if content_size > MAX_CONTENT_BYTES:
                        _fail('content exceeds retained byte bound')
                    content.append(encoded)
                except UnicodeError as exc:
                    raise SSEError('invalid Unicode content') from exc
        choices_seen += 1
        if reason is not None:
            finish = reason
            pending_usage = False
    if not done:
        _fail('missing DONE')
    return {'content': b''.join(content), 'finish_reason': finish,
            'truncated': finish == 'length', 'usage': usage,
            'usage_unavailable_reason': 'runtime did not report usage' if usage is None else None}
