"""End-to-end retrospective analysis using the real published native activation packet."""
import ast
import io
import zipfile
import copy
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from trinite.contracts import ContractError, identity, json_bytes, parse_json
from trinite.lattice_observation import experiment as e

ROOT = Path(__file__).resolve().parents[1]


class ActivationLatticeTests(unittest.TestCase):
    def test_lossless_index_resolution_and_object_integrity(self):
        request = json_bytes({'schema':'example-request'})
        capture = json_bytes({'schema':'example-capture'})
        def make_archive(changes=None):
            objects = {identity(raw):raw for raw in (request,capture)}
            files = {
                'run/request.json':{'identity':identity(request),'bytes':len(request)},
                'run/native/capture.json':{'identity':identity(capture),'bytes':len(capture)},
            }
            index = {'schema':'trinite.lossless-review-archive.v1','files':files}
            if changes is not None: changes(index,objects)
            raw = io.BytesIO()
            with zipfile.ZipFile(raw, 'w') as z:
                z.writestr('restore-index.json',json_bytes(index))
                for key,payload in objects.items():
                    z.writestr('objects/sha256/'+key[7:],payload)
            raw.seek(0)
            return raw
        with zipfile.ZipFile(make_archive()) as z:
            self.assertEqual(e.extract_native_packet(z),(request,capture))
        def wrong_digest(index,objects):
            objects[identity(capture)] = capture+b'changed'
        with zipfile.ZipFile(make_archive(wrong_digest)) as z:
            with self.assertRaises(ContractError):e.extract_native_packet(z)
        def missing_capture(index,objects):
            index['files'].pop('run/native/capture.json')
        with zipfile.ZipFile(make_archive(missing_capture)) as z:
            with self.assertRaises(ContractError):e.extract_native_packet(z)
        def changed_object_size(index,objects):
            index['files']['run/native/capture.json']['bytes'] += 1
        with zipfile.ZipFile(make_archive(changed_object_size)) as z:
            with self.assertRaises(ContractError):e.extract_native_packet(z)

    def test_frozen_protocol_and_actual_historical_native_capture_audit(self):
        rules = e.protocol()
        self.assertEqual(rules['anchor_depth'], 1)
        self.assertEqual(rules['original_reference_request_identity'], identity(e.inputs()[2]))
        req, capture, raw_request, raw_capture = e.inputs()
        self.assertEqual(len(capture['rows']), 16)
        self.assertEqual(identity(raw_request), rules['original_reference_request_identity'])
        self.assertEqual(capture['request_identity'], identity(raw_request))
        self.assertEqual(capture['condition'], 'native')

    def test_all_controls_match_anchor_and_edge_budgets(self):
        records = e.conditions()
        self.assertEqual(tuple(records), tuple(e.protocol()['controls']))
        anchor_count = 20
        edge_count = len(records['menger'][1])
        self.assertGreater(edge_count, 0)
        self.assertEqual(len(set(tuple(x) for x in records['menger'][1])), edge_count)
        for name, (coords, edges) in records.items():
            with self.subTest(name=name):
                self.assertEqual(len(coords), anchor_count)
                self.assertEqual(len(set(coords)), anchor_count)
                self.assertEqual(len(edges), edge_count)
                self.assertTrue(all(type(a) is int and type(b) is int and 0 <= a < b < 20 for a, b in edges))
        self.assertEqual(records['shuffled-topology'][0], records['menger'][0])
        self.assertNotEqual(records['shuffled-topology'][1], records['menger'][1])
        self.assertNotEqual(records['random-anchors'][0], records['menger'][0])

    def test_real_packet_analysis_and_independent_read_only_regeneration(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp) / 'request.json'
            run = Path(tmp) / 'analysis.json'
            frozen = e.freeze(request)
            before = request.read_bytes()
            r = e.run(request, frozen['request_identity'], run)
            generated = run.read_bytes()
            outcome = parse_json(generated, canonical=True)
            self.assertEqual(outcome['schema'], 'trinite.activation-lattice-analysis.v1')
            self.assertEqual(outcome['research_class'], 'exploratory-retrospective-training-exposed')
            self.assertFalse(outcome['training_or_fitting_of_model'])
            self.assertFalse(outcome['held_out_learning_adequacy'])
            self.assertFalse(outcome['phase5_ready'])
            self.assertEqual(set(outcome['layers']), {'block.0', 'final'})
            for layer in outcome['layers'].values():
                self.assertEqual(len(layer['capture_vector_identities']), 16)
                self.assertEqual(set(layer['arms']), set(e.protocol()['controls']))
                for arm in layer['arms'].values():
                    self.assertEqual(arm['anchor_count'], 20)
                self.assertTrue(arm['anchor_identity'].startswith('sha256:'))
                    self.assertEqual(arm['evaluation_count'], 8)
                    self.assertEqual(sum(arm['graph_hops']), arm['graph_hops_sum'])
                    self.assertEqual(arm['unique_anchors'] + arm['anchor_collisions'], 8)
            self.assertEqual(e.verify(request, frozen['request_identity'], run, r['artifact_identity'])['outcome'], 'passed')
            self.assertEqual(before, request.read_bytes())
            self.assertEqual(generated, run.read_bytes())
            with self.assertRaises(FileExistsError):
                e.run(request, frozen['request_identity'], run)
            with self.assertRaises(FileExistsError):
                e.freeze(request)

    def test_external_identity_tampering_cannot_be_blest(self):
        with tempfile.TemporaryDirectory() as tmp:
            req, output = Path(tmp)/'request.json', Path(tmp)/'analysis.json'
            frozen = e.freeze(req)['request_identity']
            recorded = e.run(req, frozen, output)['artifact_identity']
            changed = parse_json(output.read_bytes(), canonical=True)
            changed['phase5_ready'] = True
            output.write_bytes(json_bytes(changed))
            with self.assertRaises(ContractError):
                e.verify(req, frozen, output, recorded)
            with self.assertRaises(ContractError):
                e.verify(req, frozen, output, identity(output.read_bytes()))
            req.write_bytes(b'{}\n')
            with self.assertRaises(ContractError):
                e.analyze(req, frozen)

    def test_capture_identity_and_source_closure_cannot_be_patched(self):
        with tempfile.TemporaryDirectory() as tmp:
            request = Path(tmp)/'request.json'
            frozen = e.freeze(request)['request_identity']
            snapshot = parse_json(request.read_bytes(), canonical=True)
            for target in ('native_capture_identity', 'source', 'scaffold_source'):
                changed = copy.deepcopy(snapshot)
                if target == 'source':
                    key = 'src/trinite/lattice_observation/experiment.py'
                    changed['source'][key] = 'sha256:' + '0'*64
                elif target == 'scaffold_source':
                    key = 'src/trinite/lattice/scaffold.py'
                    changed['scaffold_source'][key] = 'sha256:' + '0'*64
                else:
                    changed[target] = 'sha256:' + '0'*64
                mutated = Path(tmp)/(target+'.json')
                mutated.write_bytes(json_bytes(changed))
                with self.subTest(target=target), self.assertRaises(ContractError):
                    e.request_at(mutated, identity(mutated.read_bytes()))
            self.assertEqual(e.request_at(request, frozen)['native_capture_identity'], snapshot['native_capture_identity'])

    def test_cli_prohibits_overwrites_and_verifies_without_mutation(self):
        script = ROOT/'scripts/run_lattice_observation.py'
        with tempfile.TemporaryDirectory() as tmp:
            request, output = Path(tmp)/'request.json', Path(tmp)/'analysis.json'
            cmd = [sys.executable, str(script)]
            freeze = subprocess.run([*cmd, '--freeze-request', str(request)], capture_output=True, text=True, timeout=30)
            self.assertEqual(freeze.returncode, 0, freeze.stderr)
            request_id = parse_json(freeze.stdout.encode())['request_identity']
            run = subprocess.run([*cmd, '--run', '--request', str(request),
                '--request-identity', request_id, '--output', str(output)],
                capture_output=True, text=True, timeout=30)
            self.assertEqual(run.returncode, 0, run.stderr)
            digest = parse_json(run.stdout.encode())['artifact_identity']
            before = (request.read_bytes(), output.read_bytes())
            verified = subprocess.run([*cmd, '--verify-only', '--request', str(request),
                '--request-identity', request_id, '--output', str(output),
                '--artifact-identity', digest], capture_output=True, text=True, timeout=30)
            self.assertEqual(verified.returncode, 0, verified.stderr)
            self.assertEqual(before, (request.read_bytes(), output.read_bytes()))
            denied = subprocess.run([*cmd, '--run', '--request', str(request),
                '--request-identity', request_id, '--output', str(output)],
                capture_output=True, text=True, timeout=30)
            self.assertNotEqual(denied.returncode, 0)
            link = Path(tmp)/'linked.json'
            link.symlink_to(request)
            rejected = subprocess.run([*cmd, '--verify-only', '--request', str(link),
                '--request-identity', request_id, '--output', str(output),
                '--artifact-identity', digest], capture_output=True, text=True, timeout=30)
            self.assertNotEqual(rejected.returncode, 0)

    def test_separate_package_and_no_new_kernels_or_network(self):
        allowed_relative = {(2, 'contracts'), (2, 'lattice.scaffold'), (2, 'reference_evidence'),
                            (2, 'reference_geometry')}
        root = ROOT/'src/trinite/lattice_observation'
        for file in root.glob('*.py'):
            for node in ast.walk(ast.parse(file.read_text())):
                if isinstance(node, ast.ImportFrom) and node.level:
                    self.assertIn((node.level, node.module), allowed_relative)
                elif isinstance(node, ast.ImportFrom):
                    self.assertIn(node.module.split('.')[0], sys.stdlib_module_names)
                elif isinstance(node, ast.Import):
                    for imp in node.names:
                        self.assertIn(imp.name.split('.')[0], sys.stdlib_module_names)
                        self.assertNotIn(imp.name.split('.')[0], {'socket','urllib','http','ssl','subprocess'})
        receipt = e.source_receipt()
        self.assertIn('scripts/run_lattice_observation.py', receipt)
        self.assertIn('src/trinite/lattice_observation/protocol.json', receipt)
        from trinite.lattice.scaffold import source_receipt as original_receipt
        self.assertEqual(e.source_receipt_scaffold(), original_receipt())


if __name__ == '__main__':
    unittest.main()
