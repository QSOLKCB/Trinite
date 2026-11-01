"""Independent note-grid materialization oracle; exposed engineering cases only."""
import copy
import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

BINARY = str(Path(sys.argv.pop(1)).resolve())
EXAMPLE = json.loads((Path(__file__).parents[1] / 'example-request.json').read_text())
REST = {'sounding': False, 'pitch': 0, 'velocity': 0, 'onset': False}


def invoke(r, mode='observe'):
    data = r if isinstance(r, bytes) else json.dumps(r, separators=(',', ':')).encode()
    return subprocess.run([BINARY, mode], input=data, capture_output=True, timeout=5)


def result(r, mode='observe'):
    p = invoke(r, mode)
    if p.returncode or p.stderr:
        raise AssertionError((p.returncode, p.stderr))
    return json.loads(p.stdout)


def reference(r):
    # Materialize events into complete channel timelines, unlike Rust's tick scan.
    n, steps = len(r['channels']), r['steps']
    ideal = [[dict(REST) for _ in range(steps)] for _ in range(n)]
    transported = [[dict(REST) for _ in range(steps+8)] for _ in range(n)]
    for e in r['events']:
        c = e['channel']
        for t in range(e['start'], e['end']):
            ideal[c][t] = {'sounding': True, 'pitch': e['pitch'], 'velocity': e['velocity'], 'onset': t == e['start']}
            if e['delivered']:
                shifted = t + r['channels'][c]['delay']
                transported[c][shifted] = {'sounding': True, 'pitch': e['pitch']+r['channels'][c]['transpose'], 'velocity': e['velocity'], 'onset': t == e['start']}
    censored = sum(cell['sounding'] for timeline in transported for cell in timeline[steps:])
    public, private = [], []
    for t in range(steps):
        pub, priv = [], []
        for c, config in enumerate(r['channels']):
            actual = dict(REST) if config['muted'][t] else transported[c][t]
            priv.append({'channel': c, 'ideal': ideal[c][t], 'transported': actual, 'muted': config['muted'][t]})
            pub.append({'channel': c, 'observed': config['observed'][t], 'note': actual if config['observed'][t] else dict(REST)})
        private.append({'tick': t, 'channels': priv})
        if t < r['observe_ticks']:
            public.append({'tick': t, 'channels': pub})
    return ({'schema': 'trinite.music-events-observation.v1', 'channel_count': n, 'observe_ticks': r['observe_ticks'], 'horizon': steps-r['observe_ticks'], 'rows': public},
            {'schema': 'trinite.music-events-ground-truth.v1', 'privileged': True, 'steps': steps, 'observe_ticks': r['observe_ticks'], 'channels': [{'channel': i, 'delay': c['delay'], 'transpose': c['transpose']} for i,c in enumerate(r['channels'])], 'events': sorted(r['events'], key=lambda e:(e['channel'],e['start'])), 'censored_transport_ticks': censored, 'rows': private})


class Oracle(unittest.TestCase):
    def check(self, r):
        public, private = reference(r)
        self.assertEqual(result(r), public)
        self.assertEqual(result(r, 'simulate-privileged'), private)

    def test_original_synthetic_schedules(self):
        for family in range(48):
            steps = 4+family%12
            r = {'schema': EXAMPLE['schema'], 'steps': steps, 'observe_ticks': 1+family%(steps-1), 'channels': [], 'events': []}
            for c in range(1+family%4):
                r['channels'].append({'delay': (family+c)%9, 'transpose': (family+c)%25-12, 'muted': [(t+c+family)%5==0 for t in range(steps)], 'observed': [(t+c+family)%3!=0 for t in range(steps)]})
                for start in range(c%2, steps, 3):
                    r['events'].append({'channel': c, 'start': start, 'end': min(start+2,steps), 'pitch': 48+(family+c+start)%24, 'velocity': 1+(family+start)%127, 'delivered': (start+c+family)%4!=0})
            self.check(r)

    def test_all_prefix_masks(self):
        r=copy.deepcopy(EXAMPLE); truth=result(r,'simulate-privileged')
        for mask in itertools.product([False,True],repeat=6):
            for c in range(2): r['channels'][c]['observed'][:3]=mask[c*3:c*3+3]
            self.check(r); self.assertEqual(result(r,'simulate-privileged'),truth)

    def test_future_prefix_isolation(self):
        r=copy.deepcopy(EXAMPLE); before=result(r)
        r['events'].append({'channel':0,'start':4,'end':6,'pitch':50,'velocity':30,'delivered':True})
        for c in r['channels']:
            c['muted'][3:]=[True]*3;c['observed'][3:]=[False]*3
        self.assertEqual(result(r),before)

    def test_loss_mute_rest_equivalence(self):
        r=copy.deepcopy(EXAMPLE);r['events'][0]['delivered']=False
        before=result(r);r['events']=[]
        self.assertEqual(result(r),before)
        r=copy.deepcopy(EXAMPLE);r['channels'][0]['muted']=[True]*6
        self.assertEqual(result(r),before)

    def test_event_order_and_replay(self):
        r=copy.deepcopy(EXAMPLE); a=invoke(r).stdout;b=invoke(r,'simulate-privileged').stdout
        r['events'].reverse()
        self.assertEqual(invoke(r).stdout,a);self.assertEqual(invoke(r,'simulate-privileged').stdout,b)

    def test_note_zero_rest_and_adjacent_onsets(self):
        r=copy.deepcopy(EXAMPLE);r['channels'][0]['delay']=0;r['channels'][0]['transpose']=0
        r['events']=[{'channel':0,'start':t,'end':t+1,'pitch':0,'velocity':1,'delivered':True} for t in range(2)]
        self.check(r); rows=result(r)['rows']
        self.assertTrue(rows[0]['channels'][0]['note']['sounding']);self.assertTrue(rows[1]['channels'][0]['note']['onset'])
        self.assertFalse(rows[2]['channels'][0]['note']['sounding'])

    def test_bounds_and_censoring(self):
        r={'schema':EXAMPLE['schema'],'steps':128,'observe_ticks':127,'channels':[{'delay':8,'transpose':12,'muted':[False]*128,'observed':[True]*128} for _ in range(4)],'events':[{'channel':c,'start':0,'end':128,'pitch':115,'velocity':127,'delivered':True} for c in range(4)]}
        self.check(r);self.assertEqual(result(r,'simulate-privileged')['censored_transport_ticks'],32)
        r['events']=[];self.check(r)
        r['events']=[{'channel':0,'start':t,'end':t+1,'pitch':0,'velocity':1,'delivered':True} for t in range(128)]
        # This legal structural maximum is still subject to the independent byte cap.
        r['channels']=r['channels'][:1];r['channels'][0]['transpose']=0
        self.check(r)

    def test_rejected_requests(self):
        for change in [lambda r:r.update(steps=1),lambda r:r.update(observe_ticks=6),lambda r:r.update(hidden=0),lambda r:r['events'].append(dict(r['events'][0])),lambda r:r['events'][0].update(pitch=127),lambda r:r['events'][0].update(delivered=1),lambda r:r['channels'][0].update(delay=-1),lambda r:r['channels'][0].update(muted=[]),lambda r:r['events'][0].update(start=2)]:
            r=copy.deepcopy(EXAMPLE);change(r)
            for mode in ['observe','simulate-privileged']:
                p=invoke(r,mode);self.assertEqual(p.returncode,2);self.assertFalse(p.stdout)
        for data in [b'{}',b'[]',b' '*16385,b'{"schema":1,"schema":2}',b'{"a":1e999}',b'{"a":null}',b'{"a":-0}',b'{"a":01}',b'{"a":"\\u0061"}',b'{"a":true}x']:
            p=invoke(data);self.assertEqual(p.returncode,2);self.assertFalse(p.stdout)

    def test_public_allowlist(self):
        p=result(EXAMPLE)
        self.assertEqual(set(p),{'schema','channel_count','observe_ticks','horizon','rows'})
        for row in p['rows']:
            self.assertEqual(set(row),{'tick','channels'})
            for c in row['channels']:
                self.assertEqual(set(c),{'channel','observed','note'})
                self.assertEqual(set(c['note']),set(REST))

    def test_non_utf8_argument_is_rejected_without_echo(self):
        if sys.platform != 'win32':
            p=subprocess.run([BINARY.encode(),b'private-\xff'],capture_output=True,timeout=5)
            self.assertEqual(p.returncode,2);self.assertFalse(p.stdout);self.assertNotIn(b'private-',p.stderr)

if __name__=='__main__': unittest.main()
