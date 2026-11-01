"""Independent finite-field oracle and real two-process wire conformance.
Classical privileged error enumeration lives only in this test, never in decode.
Exposed engineering cases are not held-out model data or an empirical error rate.
"""
import itertools
import json
from pathlib import Path
import subprocess
import sys
import unittest

CODE = 'cyclic-5-qutrit-v1'
ROWS = ((1,0,0,1,0,0,1,1,0,0), (0,1,0,0,1,0,0,1,1,0),
        (1,0,1,0,0,0,0,0,1,1), (0,1,0,1,0,1,0,0,0,1))
ZERO = (0,)*10

def syndrome(e):
    return tuple(sum(g[i]*e[i+5]-g[i+5]*e[i] for i in range(5)) % 3 for g in ROWS)
def weight(e):
    return sum(bool(e[i] or e[i+5]) for i in range(5))
def balanced(n):
    return (0,1,-1)[n % 3]
def request(s=(0,0,0,0), mask=(True,)*4):
    return dict(schema='trinite.qutrit-syndrome.v1',code=CODE,
                syndrome=[balanced(v) if k else 0 for v,k in zip(s,mask)],observed=list(mask))
def sensor_request(s=(0,)*4,noise=(0,)*4,mask=(True,)*4):
    return dict(schema='trinite.qutrit-sensor-request.v1',code=CODE,
                true_syndrome=list(map(balanced,s)),additive_noise=list(map(balanced,noise)),observed=list(mask))
def span():
    return {tuple(sum(c*g[i] for c,g in zip(coeff,ROWS))%3 for i in range(10))
            for coeff in itertools.product(range(3),repeat=4)}

class Oracle(unittest.TestCase):
    decoder = sensor = None
    def invoke(self,which,req=None,raw=None,mode=None):
        if raw is None: raw=json.dumps(req,separators=(',',':')).encode()
        return subprocess.run([getattr(self,which),mode or ('decode' if which=='decoder' else 'observe')],
                              input=raw,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=5,check=False)
    def output(self,which,req):
        out=self.invoke(which,req)
        self.assertEqual(out.returncode,0,out.stderr);self.assertEqual(out.stderr,b'')
        self.assertEqual(out.stdout.count(b'\n'),1)
        def unique(pairs):
            d={}
            for k,v in pairs:self.assertNotIn(k,d);d[k]=v
            return d
        return json.loads(out.stdout,object_pairs_hook=unique)
    def decode(self,s,mask=(True,)*4):
        out=self.output('decoder',request(s,mask))
        self.assertEqual(set(out),{'schema','code','syndrome','observed','status','search_assumption','correction_x','correction_z'})
        self.assertEqual(out['schema'],'trinite.qutrit-decoding.v1');self.assertEqual(out['code'],CODE)
        self.assertEqual(out['search_assumption'],'at_most_one_data_qudit_pauli')
        self.assertEqual(out['syndrome'],request(s,mask)['syndrome']);self.assertEqual(out['observed'],list(mask))
        for v in out['syndrome']+out['correction_x']+out['correction_z']:
            self.assertIs(type(v),int);self.assertIn(v,[-1,0,1])
        return out
    def test_all_errors_independent_stabilizer_and_distance_oracle(self):
        stabilizers=span();self.assertEqual(len(stabilizers),81)
        self.assertTrue(all(syndrome(e)==(0,)*4 for e in stabilizers))
        table={}
        # Enumerate all Pauli vectors independently, selecting weight <= 1.
        for e in itertools.product(range(3),repeat=10):
            if weight(e)<=1:
                s=syndrome(e);self.assertNotIn(s,table);table[s]=tuple((-n)%3 for n in e)
        self.assertEqual(len(table),41)
        outputs={s:self.decode(s) for s in itertools.product(range(3),repeat=4)}
        for s,out in outputs.items():
            if s in table:
                self.assertEqual(out['status'],'candidate')
                self.assertEqual(out['correction_x'],list(map(balanced,table[s][:5])))
                self.assertEqual(out['correction_z'],list(map(balanced,table[s][5:])))
            else:
                self.assertEqual(out['status'],'outside_bounded_table')
                self.assertEqual(out['correction_x'],[]);self.assertEqual(out['correction_z'],[])
        normalizer=logical=accepted=corrected=0;distance=6
        for e in itertools.product(range(3),repeat=10):
            s=syndrome(e)
            if s==(0,)*4:
                normalizer+=1
                if e not in stabilizers:logical+=1;distance=min(distance,weight(e))
            out=outputs[s]
            if out['status']=='candidate':
                accepted+=1;c=out['correction_x']+out['correction_z']
                residual=tuple((a+b)%3 for a,b in zip(e,c))
                good=residual in stabilizers;corrected+=good
                if weight(e)<=1:self.assertTrue(good)
        self.assertEqual((normalizer,logical,distance),(729,648,3))
        self.assertEqual((accepted,corrected),(29889,3321))
    def test_masks_and_canonical_missing(self):
        for s in itertools.product(range(3),repeat=4):
            for mask in itertools.product([False,True],repeat=4):
                if all(mask):continue
                out=self.decode(s,mask)
                self.assertEqual(out['status'],'incomplete_observation')
                self.assertEqual(out['correction_x'],[]);self.assertEqual(out['correction_z'],[])
    def test_sensor_all_noise_and_masks_interoperate(self):
        true=(1,2,0,1)
        for noise in itertools.product(range(3),repeat=4):
            for mask in itertools.product([False,True],repeat=4):
                req=sensor_request(true,noise,mask)
                observation=self.output('sensor',req)
                expected=request(tuple((a+b)%3 for a,b in zip(true,noise)),mask)
                self.assertEqual(observation,expected)
                decoded=self.output('decoder',observation)
                self.assertEqual(decoded['syndrome'],expected['syndrome'])
                if not all(mask):self.assertEqual(decoded['status'],'incomplete_observation')
    def test_real_process_pipe(self):
        source=subprocess.Popen([self.sensor,'observe'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        target=subprocess.Popen([self.decoder,'decode'],stdin=source.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        source.stdout.close()
        source.stdin.write(json.dumps(sensor_request()).encode());source.stdin.close()
        output,error=target.communicate(timeout=5)
        self.assertEqual(source.wait(timeout=5),0);self.assertEqual(source.stderr.read(),b'');source.stderr.close()
        self.assertEqual(target.returncode,0);self.assertEqual(error,b'')
        self.assertEqual(json.loads(output)['status'],'candidate')
    def test_noisy_candidate_can_be_wrong(self):
        # True error identity; measurement corruption imitates one nonzero X error.
        e=(1,0,0,0,0,0,0,0,0,0)
        measured=self.output('sensor',sensor_request(noise=syndrome(e)))
        out=self.output('decoder',measured);self.assertEqual(out['status'],'candidate')
        residual=tuple(n%3 for n in out['correction_x']+out['correction_z'])
        self.assertNotIn(residual,span());self.assertNotEqual(residual,ZERO)
    def test_zero_syndrome_logical_is_not_success(self):
        stabilizers=span()
        logical=next(e for e in itertools.product(range(3),repeat=10) if syndrome(e)==(0,)*4 and e not in stabilizers)
        out=self.decode(syndrome(logical));self.assertEqual(out['status'],'candidate')
        self.assertEqual(out['correction_x']+out['correction_z'],[0]*10)
        self.assertNotIn(logical,stabilizers)
    def test_private_decompositions_same_observation(self):
        a=sensor_request((1,)*4,(2,)*4);b=sensor_request()
        self.assertEqual(self.invoke('sensor',a).stdout,self.invoke('sensor',b).stdout)
        a['observed']=[False]*4;b['observed']=[False]*4;a['additive_noise']=[1]*4
        self.assertEqual(self.invoke('sensor',a).stdout,self.invoke('sensor',b).stdout)
    def reject(self,which,req=None,raw=None,mode=None):
        out=self.invoke(which,req,raw,mode);self.assertEqual(out.returncode,2,(which,req,raw,out.stderr))
        self.assertEqual(out.stdout,b'');self.assertTrue(out.stderr)
    def test_request_and_hidden_field_rejections(self):
        for which,req in [('decoder',request()),('sensor',sensor_request())]:
            for key in req:
                bad=dict(req);del bad[key];self.reject(which,bad)
            for key in ['error','state','true_error','success','seed','path']:
                bad=dict(req);bad[key]=[0];self.reject(which,bad)
            for key in ['schema','code']:
                bad=dict(req);bad[key]='wrong';self.reject(which,bad)
            arrays=['syndrome'] if which=='decoder' else ['true_syndrome','additive_noise']
            for key in arrays:
                for value in [[],[0]*3,[0]*5,[2]*4,[-2]*4,[True]*4,[0.0]*4]:
                    bad=dict(req);bad[key]=value;self.reject(which,bad)
            for value in [[],[True]*3,[1]*4]:
                bad=dict(req);bad['observed']=value;self.reject(which,bad)
        bad=request();bad['syndrome'][0]=1;bad['observed'][0]=False;self.reject('decoder',bad)
        # Ideal truth and corruption must never be accepted as decoder input.
        self.reject('decoder',sensor_request())
    def test_strict_wire_and_operations(self):
        for which,req in [('decoder',request()),('sensor',sensor_request())]:
            raw=json.dumps(req,separators=(',',':')).encode()
            for bad in [b'',b'{}',b'[]',b'null',raw+b'{}',b' '*16385,b'['*10+b'0'+b']'*10,
                        raw.replace(b'"code":',b'"code":"x","code":'),
                        raw.replace(b'"code"',b'"co\\u0064e"'),raw.replace(b'[0,',b'[-0,'),
                        raw.replace(b'[0,',b'[01,'),raw.replace(b'[0,',b'[9223372036854775808,')]:
                self.reject(which,raw=bad)
            for mode in ['invalid','simulate-privileged','']:self.reject(which,req,mode=mode or 'invalid')
    def test_exact_replay_and_field_order(self):
        for which,req in [('decoder',request()),('sensor',sensor_request())]:
            a=self.invoke(which,req);b=self.invoke(which,dict(reversed(list(req.items()))))
            self.assertEqual(a.stdout,b.stdout);self.assertEqual(a.returncode,0)

if __name__=='__main__':
    if len(sys.argv)!=3:raise SystemExit('usage: oracle.py DECODER_BINARY SENSOR_BINARY')
    Oracle.decoder=str(Path(sys.argv[1]).resolve(strict=True));Oracle.sensor=str(Path(sys.argv[2]).resolve(strict=True))
    sys.argv=sys.argv[:1];unittest.main(verbosity=2)
