"""Offline strict SSE regression tests; archived runtime bytes are data only."""
import copy
import hashlib
import json
from pathlib import Path
import types
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
m = types.ModuleType('tiny_sse_v2_tested')
source = ROOT / 'scripts/tiny_sse_v2.py'
exec(compile(source.read_bytes(), str(source), 'exec'), m.__dict__)
DONE = b'data: [DONE]\n\n'
USAGE = {'completion_tokens': 2, 'prompt_tokens': 3, 'total_tokens': 5}


def choice(content=None, reason=None, delta=None, **extra):
    return {'choices': [{'index': 0, 'delta': ({'content': content} if delta is None else delta),
                         'finish_reason': reason}], **extra}


def event(value):
    return b'data: ' + json.dumps(value, ensure_ascii=True).encode() + b'\n\n'


def stream(*frames):
    return b''.join(event(frame) for frame in frames) + DONE


def terminal(reason='stop', **extra):
    return choice(reason=reason, delta={}, **extra)


class StrictSSETests(unittest.TestCase):
    def reject(self, raw):
        with self.assertRaises(m.SSEError):
            m.parse_sse(raw)

    def test_content_exact_and_empty_chunks(self):
        text = ' \n```json\n{"answer":1}\n```\n</think>\n☃\t'
        frames = [choice(delta={'role': 'assistant', 'content': None}),
                  choice(''), choice(text[:8]), choice(text[8:]), terminal()]
        result = m.parse_sse(stream(*frames))
        self.assertEqual(result['content'], text.encode())
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertFalse(result['truncated'])
        self.assertIsNone(result['usage'])
        self.assertEqual(result['usage_unavailable_reason'], 'runtime did not report usage')

    def test_length_is_metadata_not_correctness(self):
        result = m.parse_sse(stream(choice('{"answer":3}', reason='length')))
        self.assertEqual(result['content'], b'{"answer":3}')
        self.assertEqual(result['finish_reason'], 'length')
        self.assertTrue(result['truncated'])
        self.assertNotIn('correct', result)

    def test_line_endings_and_multiline_data(self):
        raw = stream(choice('a\rb\nc'), terminal())
        for ending in (b'\n', b'\r', b'\r\n'):
            self.assertEqual(m.parse_sse(raw.replace(b'\n', ending))['content'], b'a\rb\nc')
        self.assertEqual(m.parse_sse(raw[:-1])['content'], b'a\rb\nc')
        raw = event(terminal()).replace(b'"choices": ', b'"choices":\ndata: ')
        self.assertEqual(m.parse_sse(raw + DONE)['content'], b'')

    def test_usage_adjacent_or_on_terminal(self):
        usage_frame = {'choices': [], 'usage': USAGE}
        for frames in ([choice('x'), usage_frame, terminal()],
                       [choice('x'), terminal(), usage_frame],
                       [choice('x'), terminal(usage=USAGE)],
                       [usage_frame, terminal(usage=USAGE), usage_frame]):
            result = m.parse_sse(stream(*frames))
            self.assertEqual(result['usage'], USAGE)
            self.assertIsNone(result['usage_unavailable_reason'])
        zero = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
        self.assertEqual(m.parse_sse(stream(terminal(usage=zero)))['usage'], zero)

    def test_done_and_terminal_are_both_required(self):
        for raw in (DONE, stream(choice('x')), event(terminal()), b'',
                    stream(terminal(), terminal()), stream(terminal(), terminal('length')),
                    stream(terminal(), choice('x')), stream(terminal(), choice('')),
                    stream(terminal()) + DONE, stream(terminal()) + event(choice('x')),
                    stream(terminal()) + b'\n', stream(terminal()) + b': heartbeat\n\n'):
            with self.subTest(raw=raw):
                self.reject(raw)

    def test_usage_position_and_conflicting_duplicates(self):
        u = {'choices': [], 'usage': USAGE}
        other = {'choices': [], 'usage': {**USAGE, 'completion_tokens': 3, 'total_tokens': 6}}
        for frames in ([u, choice('x'), terminal()], [u, u, terminal()],
                       [terminal(), u, u], [choice('x', usage=USAGE), terminal()],
                       [u, terminal(), other], [{'choices': []}, terminal()]):
            with self.subTest(frames=frames):
                self.reject(stream(*frames))

    def test_usage_types_and_total(self):
        for key in USAGE:
            for value in (True, False, -1, 2.0, '2', None, [], {}):
                with self.subTest(key=key, value=value):
                    self.reject(stream(terminal(usage={**USAGE, key: value})))
        for usage in ([], {}, {'completion_tokens': 2}, {**USAGE, 'total_tokens': 7},
                      {**USAGE, 'extra': 0}, {**USAGE, 'prompt_tokens_details': {'cached_tokens': True}},
                      {**USAGE, 'prompt_tokens_details': {'cached_tokens': 4}}):
            self.reject(stream(terminal(usage=usage)))

    def test_choice_and_delta_types(self):
        base = terminal()
        variants = [{'choices': None}, {'choices': {}}, {'choices': [None]},
                    {'choices': [base['choices'][0]] * 2}]
        for field, values in {'index': [True, 1, '0', 0.0, None],
                              'delta': [None, [], ''],
                              'finish_reason': [True, 0, [], {}, 'error', 'tool_calls', '']}.items():
            for value in values:
                bad = copy.deepcopy(base)
                bad['choices'][0][field] = value
                variants.append(bad)
        for bad in variants:
            self.reject(stream(bad))
        for value in (None, 0, True, [], {}):
            self.reject(stream(choice(value), terminal()))
        self.reject(stream(choice('a'), choice(delta={'role': 'assistant'}), terminal()))
        self.reject(stream(choice(delta={'role': 'user', 'content': ''}), terminal()))

    def test_errors_reasoning_tools_and_unknown_fields(self):
        for key in ('reasoning', 'reasoning_content', 'tool_calls', 'function_call', 'error', 'refusal'):
            self.reject(stream(choice(delta={key: None}), terminal()))
            self.reject(stream(terminal(**{key: None})))
        self.reject(stream({'error': 'upstream error'}))
        self.reject(stream(terminal(object='chat.completion')))
        self.reject(stream(terminal(created=True)))
        self.reject(stream(terminal(model=[])))
        self.reject(stream(terminal(timings={'predicted_ms': float('inf')})))

    def test_malformed_json_unicode_and_framing(self):
        for raw in (b'data: {\n\n', b'data: []\n\n', b'data: null\n\n',
                    b'data: {"choices":[],"choices":[]}\n\n',
                    b'data: {"choices":[],"usage":NaN}\n\n',
                    b'data: \xff\n\n', b'\xef\xbb\xbf' + stream(terminal()),
                    b'event: message\n' + stream(terminal()),
                    b'\n' + stream(terminal()), stream(terminal())[:-2]):
            self.reject(raw)
        self.reject(stream(choice('\ud800'), terminal()))
        self.reject(b'data: ' + b'[' * 2000 + b'0' + b']' * 2000 + b'\n\n' + DONE)

    def test_byte_limits(self):
        self.reject('not bytes')
        self.reject(bytearray(stream(terminal())))
        self.reject(b'x' * (m.MAX_STREAM_BYTES + 1))
        self.assertEqual(len(m.parse_sse(stream(choice('x' * m.MAX_CONTENT_BYTES), terminal()))['content']),
                         m.MAX_CONTENT_BYTES)
        self.reject(stream(choice('x' * m.MAX_CONTENT_BYTES), choice('x'), terminal()))
        self.reject(stream(choice('☃' * (m.MAX_CONTENT_BYTES // 3 + 1)), terminal()))

    def test_archived_smoke_exact_bytes_terminal_and_tokens(self):
        # Pinned historical objects read as bytes, never importing adapter/runtime.
        digest = '059b3f8d0d4a2fb147c115c44802df83b2b3ad01a3a1e4dadb7facee1354fb89'
        with zipfile.ZipFile(ROOT / 'fixtures/tiny-preflight-local-20261021/objects.zip') as archive:
            raw = archive.read('objects/' + digest + '.bin')
        self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
        self.assertEqual(len(raw), 11158)
        result = m.parse_sse(raw)
        self.assertEqual(result['content'], b'{\n  "target": 1234\n}\n</think>\n\n{\n  "target": 1234\n}')
        self.assertEqual(len(result['content']), 51)
        self.assertEqual(hashlib.sha256(result['content']).hexdigest(),
                         '9888dc64fde9e017e6ab591d624c0b2d9919098c87ffa6bd1d6625b662c3c5bf')
        self.assertEqual(result['finish_reason'], 'stop')
        self.assertFalse(result['truncated'])
        self.assertEqual(result['usage'], {'completion_tokens': 30, 'prompt_tokens': 54, 'total_tokens': 84})
        scorer = types.ModuleType('tiny_v1_scorer_tested')
        path = ROOT / 'scripts/tiny_output_v1.py'
        scorer.__file__ = str(path)
        exec(compile(path.read_bytes(), str(path), 'exec'), scorer.__dict__)
        with self.assertRaises(scorer.ContractError):
            scorer.score_payload('bonsai2-2b', result['content'])


if __name__ == '__main__':
    unittest.main()
