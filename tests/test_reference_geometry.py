import copy
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, identity, json_bytes
from trinite.reference_geometry import encode_vector, decode_vector, cka, distances, cosine_distances
from trinite.reference_inputs import probes, anchors, modelfile, settings
from trinite.reference_evidence import retain, verify_retained, publish, verify

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reference_oracle',ROOT/'scripts/reference_math_oracle.py')
oracle = importlib.util.module_from_spec(spec); spec.loader.exec_module(oracle)


class ReferenceGeometryTests(unittest.TestCase):
    def test_exact_vector_bytes_and_rejections(self):
        values = [0.0,-0.0,1.25,-2.0]; record = encode_vector(values)
        self.assertEqual(decode_vector(record),values)
        for key,value in [('width',3),('bytes_identity','sha256:'+'0'*64),('float_hex',['nan']*4)]:
            changed = {**record,key:value}
            with self.assertRaises(ContractError): decode_vector(changed)
        with self.assertRaises(ContractError): encode_vector([float('inf')])

    def test_unequal_width_geometry_oracle_and_invariances(self):
        x = [[1.0,2.0],[2.0,-1.0],[-2.0,3.0],[0.0,1.0]]
        y = [[2*a+5,-2*b-3,0.0] for a,b in x]
        value = cka(x,y); self.assertAlmostEqual(value,1.0)
        self.assertTrue(oracle.verify(x,y,value,distances(x)))
        with self.assertRaises(ValueError): oracle.verify(x,y,0.5,distances(x))
        changed = distances(x); changed[0][1] += 0.01
        with self.assertRaises(ValueError): oracle.verify(x,y,value,changed)
        zero = [[0.0,0.0]]*4
        self.assertIsNone(cka(x,zero)); self.assertIsNone(cosine_distances(zero)[0][1])
        self.assertIsNone(oracle.cka_squared(x,zero))
        with self.assertRaises(ContractError): cka(x,y[:2])

    def test_selection_is_admitted_visible_paired_and_stable(self):
        rows,admission = probes(); self.assertEqual(len(rows),16)
        self.assertEqual((rows,admission),probes())
        self.assertTrue(all(r['split']=='train' and r['source_ids'] for r in rows))
        for a,b in zip(rows[::2],rows[1::2]):
            self.assertEqual(a['semantic_identity'],b['semantic_identity'])
            self.assertEqual((a['carrier'],b['carrier']),('infix','fields'))
        self.assertEqual(anchors('abcde'),[2,3,4,5])
        with self.assertRaises(ContractError): anchors('é')

    def test_modelfile_is_closed_profile(self):
        raw = ('FROM ./model\nSYSTEM '+settings()['system']+'\n').encode()
        self.assertEqual(modelfile(raw),settings()['system'])
        for changed in (raw+b'PARAMETER temperature 1\n',raw.replace(b'./model',b'other'),b'FROM ./model\n'):
            with self.assertRaises(ContractError): modelfile(changed)

    def test_missing_changed_receipt_and_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); inputs={'request':b'input'}; outputs={'capture':b'output'}
            retain(path,inputs,outputs,'test.reference')
            receipt = (path/'verification.json').read_bytes()
            (path/'verification.json').unlink()
            with self.assertRaises(ContractError): verify_retained(path,inputs,outputs)
            (path/'verification.json').write_bytes(receipt+b' ')
            with self.assertRaises(ContractError): verify_retained(path,inputs,outputs)
            (path/'verification.json').write_bytes(receipt)
            verify_retained(path,inputs,outputs)
            artifact = next((path/'provenance/artifacts/sha256').iterdir())
            artifact.write_bytes(b'tampered')
            with self.assertRaises(ContractError): verify_retained(path,inputs,outputs)

    def test_complete_simulation_evidence_and_derived_receipt(self):
        # Closed orchestration conformance only; these vectors are synthetic.
        selected,admission = probes()
        request = {'schema':'simulation-only','probes':selected,'admission':admission,'protocol':settings()}
        request_raw = json_bytes(request)
        with tempfile.TemporaryDirectory() as tmp, patch('trinite.reference_evidence.validate',return_value=(request,{})):
            path=Path(tmp); (path/'request.json').write_bytes(request_raw)
            (path/'Modelfile').write_bytes(b'simulation-only');(path/'locations.json').write_bytes(b'{}\n')
            for condition in ('native','stock','modelfile'):
                cell=path/condition;cell.mkdir();rows=[]
                for i,probe in enumerate(selected):
                    states=[]
                    for end in anchors(probe['prompt']):
                        encoding={'rendered':probe['prompt'][:end],'input_ids':[1]*end,
                            'offsets':[[k,k+1] for k in range(end)],'user_span':[0,end],'selected_position':end-1}
                        vector=encode_vector([float(i+1),float(end),float(i%3)])
                        states.append({'endpoint':end,'encoding':encoding,'vectors':{'block.0':vector,'final':vector}})
                    rows.append({'example_identity':probe['example_identity'],'anchors':states,
                        'generation_input':encoding,'output':{'token_ids':[],'text':probe['answer'],
                            'utf8_hex':probe['answer'].encode().hex(),'stop_reason':'eos'},'exact_visible_answer':True})
                raw=json_bytes({'schema':'trinite.reference-capture.v1','condition':condition,
                    'request_identity':identity(request_raw),'model':{},'rng_identity':identity(b'simulation'),
                    'layers':{},'rows':rows})
                (cell/'capture.json').write_bytes(raw)
                (cell/'replay.json').write_bytes(json_bytes({'capture_identity':identity(raw),'fresh_process_exact_match':True}))
                (cell/'worker.log').write_bytes(b'simulation');(cell/'replay.log').write_bytes(b'simulation')
            publish(path,oracle);self.assertTrue(verify(path,oracle)['verified'])
            (path/'comparison/verification.json').unlink()
            with self.assertRaises(ContractError):verify(path,oracle)


if __name__ == '__main__': unittest.main()
