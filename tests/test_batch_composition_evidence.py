"""Read-only public evidence audit; not a substitute for CPU inference replay."""
import lzma
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from trinite.contracts import identity, json_bytes, parse_json
from trinite.data import audit_bytes
from batch_composition_v1.validation import validate_report

MAX = 64 * 1024 * 1024
V1_COMMITMENT = 'sha256:6c20c17bd0b486b0cb75ffe5729838380fd4c0f3043af5c605d707dd0f581ece'
V2_COMMITMENT = 'sha256:b0276cd42dd85def986bdb25aa3f2525f6c0cea6db248266e41c12e2e85bee42'


def bounded(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular fixture file required')
    with path.open('rb') as stream:
        raw = stream.read(MAX + 1)
    if len(raw) > MAX:
        raise ValueError('fixture exceeds bound')
    return raw


def expanded(raw):
    reader = lzma.LZMADecompressor(memlimit=MAX)
    data = reader.decompress(raw, max_length=MAX + 1)
    if len(data) > MAX or not reader.eof or reader.unused_data:
        raise ValueError('expanded fixture exceeds bound or has trailing data')
    return data


class BatchCompositionEvidenceTests(unittest.TestCase):
    def packet(self, version, expected):
        folder = ROOT/'fixtures'/('batch-composition-' + version)
        raw = bounded(folder/'commitment.json')
        self.assertEqual(identity(raw), expected)
        commitment = parse_json(raw, canonical=True)
        actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
        self.assertEqual(actual, set(commitment['files']) | {'commitment.json'})
        for name, digest in commitment['files'].items():
            self.assertFalse(Path(name).is_absolute())
            self.assertNotIn('..', Path(name).parts)
            self.assertEqual(identity(bounded(folder/name)), digest, name)
        for name, info in commitment['compressed'].items():
            raw = expanded(bounded(folder/name))
            self.assertEqual(len(raw), info['uncompressed_bytes'])
            self.assertEqual(identity(raw), info['uncompressed_identity'])
        raw = expanded(bounded(folder/'report.json.xz'))
        self.assertEqual(identity(raw), commitment['report_identity'])
        report = parse_json(raw, canonical=True)
        request_raw = bounded(folder/'request.json')
        self.assertEqual(identity(request_raw), commitment['request_identity'])
        request = parse_json(request_raw, canonical=True)
        self.assertEqual(report['request'], request)
        return folder, commitment, report, request

    def test_original_failure_and_explicit_public_source_recovery_remain_exact(self):
        folder, commitment, report, request = self.packet('v1', V1_COMMITMENT)
        self.assertEqual(commitment['producer_commit'], '7be8c44288ea645ed8ac04e00abebbe452e2e567')
        self.assertIs(commitment['independent_inference_replay_completed'], False)
        self.assertEqual(report['schema'], 'trinite.batch-composition-report.v1')
        recovery = commitment['source_recovery']
        self.assertIs(recovery['producer_zip_published'], False)
        self.assertFalse((folder/'producer-source.zip').exists())
        self.assertEqual(recovery['commit'], commitment['producer_commit'])
        self.assertEqual(recovery['tree'], '07c93f7ad42f5d1dc13cd2c4f433cc748bf50fff')
        self.assertEqual(recovery['source_sha256'], request['source'])
        self.assertEqual(len(request['source']), 149)
        self.assertEqual(set(recovery['git_blobs']), set(request['source']))
        reviewed = parse_json(bounded(ROOT/'fixtures/batch-composition-v2/prepublication-review.json'), canonical=True)
        changed = set(reviewed['changed_source_paths'])
        self.assertEqual(changed, {p for p in request['source'] if request['source'][p] != reviewed['source'][p]})
        # All unchanged runtime/model sources remain available offline. The nine
        # old lane-revision blobs require the explicitly pinned public commit.
        for name, expected in request['source'].items():
            self.assertEqual(len(recovery['git_blobs'][name]), 40)
            if name not in changed:
                self.assertEqual(identity(bounded(ROOT/name)), expected, name)
        status = parse_json(bounded(folder/'failed-replay/status.json'), canonical=True)
        self.assertEqual(status['outcome'], 'failed')
        self.assertEqual(status['exit_code'], 1)
        self.assertIn(b'recomputed comparisons mismatch', bounded(folder/'failed-replay/worker.log'))
        self.assertFalse((folder/'failed-replay/replay-receipt.json').exists())
        audit = parse_json(bounded(folder/'failure-audit.json'), canonical=True)
        self.assertIs(audit['independent_inference_replay_completed'], False)
        self.assertEqual(audit['original_report_identity'], commitment['report_identity'])
        self.assertEqual(audit['violating_coordinate_count'], 0)
        self.assertEqual(identity(expanded(bounded(folder/'run/completed-cells.jsonl.xz'))), identity(b''.join(json_bytes(c) for c in report['cells'])))

    def test_new_source_bound_matrix_reports_replay_and_actual_data_admission(self):
        folder, commitment, report, request = self.packet('v2', V2_COMMITMENT)
        self.assertEqual(commitment['primary_python'], '3.12')
        self.assertEqual(set(commitment['matrix']), {'3.11', '3.12', '3.13'})
        self.assertEqual(commitment['producer_commit'], 'f05f8396a98fb77134e26486fa9482fcca0913b0')
        data_root = ROOT/'fixtures/formal-v1'
        admitted = audit_bytes(bounded(data_root/'dataset.json'), bounded(data_root/'manifest.json'))
        self.assertIs(admitted['verified'], True)
        stored_metadata = {}
        for name in commitment['files']:
            raw = bounded(folder/name)
            if name.endswith('.xz'):
                raw = expanded(raw)
            stored_metadata[name] = (identity(raw), len(raw))
        for version, matrix in commitment['matrix'].items():
            with self.subTest(python=version):
                request_raw = bounded(folder/matrix['request_path'])
                self.assertEqual(identity(request_raw), matrix['request_identity'])
                request = parse_json(request_raw, canonical=True)
                report_raw = expanded(bounded(folder/matrix['report_path']))
                self.assertEqual(identity(report_raw), matrix['report_identity'])
                report = parse_json(report_raw, canonical=True)
                self.assertEqual(report['schema'], 'trinite.batch-composition-report.v2')
                self.assertTrue(request['environment']['python'].startswith(version+'.'))
                for name, digest in request['source'].items():
                    self.assertEqual(identity(bounded(ROOT/name)), digest, name)
                self.assertEqual(request['protocol']['inputs']['dataset.json'], admitted['dataset_identity'])
                self.assertEqual(request['protocol']['inputs']['manifest.json'], admitted['manifest_identity'])
                self.assertEqual([r['split'] for r in request['workload']], ['train']*36 + ['validation']*6)
                self.assertTrue(all(set(r)=={'item_id','split','prompt','prompt_identity'} for r in request['workload']))
                checked = validate_report(report, request)
                self.assertIs(checked['conforms'], True)
                self.assertEqual(checked['budget'], {'physical_forward_calls':2500,'row_evaluations':6720,'active_decisions':6720})
                audit_path = 'independent-offline-audit.json' if version == '3.12' else 'matrix/'+version+'/independent-offline-audit.json'
                audit = parse_json(bounded(folder/audit_path), canonical=True)
                self.assertEqual(audit['report_identity'], matrix['report_identity'])
                self.assertEqual(audit['request_identity'], matrix['request_identity'])
                self.assertIs(audit['local_inference_executed'], False)
                self.assertIs(audit['read_only_inputs_unchanged'], True)
                self.assertEqual(audit['all_artifact_file_hashes'], {name:info['identity'] for name,info in matrix['members'].items()})
                self.assertEqual(len(matrix['members']), 19)
                actions = {}
                for member, info in matrix['members'].items():
                    self.assertFalse(Path(member).is_absolute())
                    self.assertNotIn('..', Path(member).parts)
                    self.assertEqual(stored_metadata[info['stored_path']], (info['identity'], info['bytes']))
                    self.assertEqual(info['encoding'], 'xz' if info['stored_path'].endswith('.xz') else 'identity')
                    if member.endswith('/invocation.json'):
                        value = parse_json(bounded(folder/info['stored_path']), canonical=True)
                        self.assertEqual(value['request_identity'], matrix['request_identity'])
                        self.assertEqual(value['deadline_seconds'], 300)
                        actions[member.rsplit('/',1)[0]] = value['action']
                self.assertEqual(set(actions.values()), {'run', 'verify'})
                for attempt, action in actions.items():
                    for name in ('computed-report.json','completed-cells.jsonl','completed-layouts.jsonl','worker.log','measurements.json','status.json'):
                        self.assertIn(attempt+'/'+name, matrix['members'])
                    self.assertEqual(matrix['members'][attempt+'/computed-report.json']['identity'], matrix['report_identity'])
                    status = parse_json(bounded(folder/matrix['members'][attempt+'/status.json']['stored_path']), canonical=True)
                    self.assertEqual(status['outcome'], 'completed')
                    self.assertEqual(status['exit_code'], 0)
                    if action == 'verify':
                        receipt = parse_json(bounded(folder/matrix['members'][attempt+'/replay-receipt.json']['stored_path']), canonical=True)
                        self.assertIs(receipt['exact_match'], True)
                        self.assertEqual(receipt['expected_identity'], matrix['report_identity'])
                        self.assertEqual(receipt['actual_identity'], matrix['report_identity'])
                        self.assertEqual(receipt['request_identity'], matrix['request_identity'])
                del report, report_raw
        self.assertIs(commitment['hosted_inference_replay_completed'], True)
        self.assertIs(commitment['independent_local_inference_replay_completed'], False)
