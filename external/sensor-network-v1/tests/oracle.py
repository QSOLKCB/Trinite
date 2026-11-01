"""Independent standard-library synthetic sensing oracle; not learning evidence."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

BINARY = str(Path(sys.argv.pop(1)).resolve())
EXAMPLE = json.loads((Path(__file__).parents[1] / 'example-request.json').read_text())


def invoke(request, mode='observe'):
    data = request if isinstance(request, bytes) else json.dumps(request, separators=(',', ':')).encode()
    return subprocess.run([BINARY, mode], input=data, capture_output=True, timeout=5)


def result(request, mode='observe'):
    p = invoke(request, mode)
    if p.returncode != 0 or p.stderr:
        raise AssertionError((p.returncode, p.stderr))
    return json.loads(p.stdout)


def reference(r):
    sensors = sorted(r['sensors'], key=lambda s: s['id'])
    topology = [{'id': s['id'], 'signal': s['signal']} for s in sensors]
    public_rows, private_rows = [], []
    # Independent per-channel schedule materialization, then transpose to ticks.
    channels = []
    for s in sensors:
        channel = []
        for t, row in enumerate(r['latent']):
            ideal = row[s['signal']]
            mode, value = s['fault'][t], s['fault_value'][t]
            response = (ideal, ideal + value, value)[mode]
            channel.append({'id': s['id'], 'ideal': ideal, 'fault': mode,
                            'fault_value': value, 'response': response,
                            'noise': s['noise'][t], 'measured': response + s['noise'][t]})
        channels.append(channel)
    for t, readings in enumerate(zip(*channels)):
        private_rows.append({'tick': t, 'latent': r['latent'][t], 'readings': list(readings)})
        if t < r['observe_ticks']:
            mask = [s['observed'][t] for s in sensors]
            public_rows.append({'tick': t, 'values': [v['measured'] if known else 0 for v, known in zip(readings, mask)], 'observed': mask})
    common = {'signal_count': len(r['latent'][0]), 'sensors': topology, 'observe_ticks': r['observe_ticks']}
    return ({'schema': 'trinite.sensor-network-observation.v1', **common,
             'horizon': len(r['latent'])-r['observe_ticks'], 'rows': public_rows},
            {'schema': 'trinite.sensor-network-ground-truth.v1', 'privileged': True, **common,
             'steps': len(r['latent']), 'rows': private_rows})


class Oracle(unittest.TestCase):
    def check(self, r):
        expected, truth = reference(r)
        self.assertEqual(result(r), expected)
        self.assertEqual(result(r, 'simulate-privileged'), truth)

    def test_independent_schedules_and_cutoffs(self):
        for family in range(48):
            r = copy.deepcopy(EXAMPLE)
            steps = 2 + family % 13
            r['latent'] = [[((t*7+family)%31)-15, ((t*t+family)%27)-13] for t in range(steps)]
            r['observe_ticks'] = 1 + family % (steps-1)
            for j, s in enumerate(r['sensors']):
                s['noise'] = [((t+j+family)%9)-4 for t in range(steps)]
                s['fault'] = [(t+j+family)%3 for t in range(steps)]
                s['fault_value'] = [0 if f == 0 else ((t+family)%21)-10 for t, f in enumerate(s['fault'])]
                s['observed'] = [(t+j+family)%4 != 0 for t in range(steps)]
            self.check(r)

    def test_all_prefix_masks(self):
        r = copy.deepcopy(EXAMPLE)
        truth = result(r, 'simulate-privileged')
        for mask in itertools.product([False, True], repeat=4):
            for j in range(2):
                r['sensors'][j]['observed'][:2] = mask[j*2:j*2+2]
            self.check(r)
            self.assertEqual(result(r, 'simulate-privileged'), truth)

    def test_no_future_leak(self):
        r = copy.deepcopy(EXAMPLE)
        before = invoke(r).stdout
        for t in range(r['observe_ticks'], len(r['latent'])):
            r['latent'][t] = [-1000, 1000]
            for s in r['sensors']:
                s['noise'][t] = 100
                s['fault'][t] = 2
                s['fault_value'][t] = -1000
                s['observed'][t] = False
        self.assertEqual(invoke(r).stdout, before)

    def test_zero_missing_and_fault_labels(self):
        r = copy.deepcopy(EXAMPLE)
        r['sensors'] = r['sensors'][:1]
        r['latent'] = [[0]]*4
        s = r['sensors'][0]
        s.update(noise=[0]*4, fault=[0]*4, fault_value=[0]*4, observed=[True]*4)
        public = result(r)
        self.assertEqual(public['rows'][0], {'tick': 0, 'values': [0], 'observed': [True]})
        s['observed'][0] = False
        self.assertEqual(result(r)['rows'][0], {'tick': 0, 'values': [0], 'observed': [False]})
        s['observed'][0] = True
        s['fault'] = [2]*4  # A stuck-at-zero fault is observationally identical here.
        self.assertEqual(public, result(r))

    def test_bias_latent_cancellation(self):
        r = copy.deepcopy(EXAMPLE)
        for s in r['sensors']:
            s['fault'] = [0]*4
            s['fault_value'] = [0]*4
        before = invoke(r).stdout
        r['latent'] = [[x-10 for x in row] for row in r['latent']]
        for s in r['sensors']:
            s['fault'] = [1]*4
            s['fault_value'] = [10]*4
        self.assertEqual(invoke(r).stdout, before)

    def test_canonical_ids_and_shared_signal(self):
        r = copy.deepcopy(EXAMPLE)
        r['sensors'][1]['signal'] = 0
        self.check(r)
        before = invoke(r).stdout
        r['sensors'].reverse()
        self.assertEqual(invoke(r).stdout, before)

    def test_bounds(self):
        r = {'schema': EXAMPLE['schema'], 'latent': [[1000]*4 for _ in range(64)], 'observe_ticks': 63,
             'sensors': [{'id': i+1, 'signal': i%4, 'noise': [100]*64, 'fault': [1]*64,
                          'fault_value': [1000]*64, 'observed': [True]*64} for i in range(8)]}
        self.check(r)
        self.assertTrue(all(v == 2100 for row in result(r)['rows'] for v in row['values']))

    def test_invalid_inputs(self):
        variants = []
        for change in [lambda r: r.update(observe_ticks=0), lambda r: r.update(latent=[]),
                       lambda r: r['sensors'][0].update(signal=9), lambda r: r['sensors'][0].update(id=0),
                       lambda r: r['sensors'][1].update(id=1), lambda r: r['sensors'][0].update(observed=[1]*4),
                       lambda r: r['sensors'][0].update(noise=[101]*4), lambda r: r['sensors'][0].update(fault=[3]*4),
                       lambda r: r['sensors'][0].update(fault_value=[1]*4), lambda r: r.update(seed=0)]:
            r = copy.deepcopy(EXAMPLE); change(r); variants.append(r)
        variants += [b'', b'null', b'{"x":1,"x":2}', b'1e999', b' '*16385,
                     b'[[[[[[[[[[0]]]]]]]]]]', b'{} trailing']
        for r in variants:
            for mode in ['observe', 'simulate-privileged']:
                p = invoke(r, mode)
                self.assertEqual(p.returncode, 2)
                self.assertEqual(p.stdout, b'')
                self.assertTrue(p.stderr)

    def test_public_allowlist(self):
        r = result(EXAMPLE)
        self.assertEqual(set(r), {'schema','signal_count','sensors','observe_ticks','horizon','rows'})
        for row in r['rows']:
            self.assertEqual(set(row), {'tick','values','observed'})
        for s in r['sensors']:
            self.assertEqual(set(s), {'id','signal'})

    def test_replay(self):
        for mode in ['observe','simulate-privileged']:
            self.assertEqual(invoke(EXAMPLE,mode).stdout, invoke(EXAMPLE,mode).stdout)

if __name__ == '__main__':
    unittest.main()
