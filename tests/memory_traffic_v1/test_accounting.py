import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from trinite.contracts import ContractError, ModelConfig, json_bytes
from memory_traffic_v1 import accounting as a


def plan(**changes):
    value = {'config': ModelConfig(context_length=16, blocks=2, width=6, heads=2,
                                  feed_forward_width=9).to_dict(),
             'lane': 'ternary', 'batch': 2, 'prompt_tokens': 3, 'forward_calls': 4}
    value.update(changes)
    return value


class AccountingTests(unittest.TestCase):
    def test_counts_by_independent_tensor_inventory(self):
        value = a.report(plan())
        shapes = [(259, 6), (16, 6), (6,)]
        for _ in range(2):
            shapes += [(6, 6)]*4 + [(9, 6), (6, 9)] + [(6,)]*2
        counts = [__import__('math').prod(s) for s in shapes]
        self.assertEqual(value['parameters']['native_float32_parameter_bytes'], sum(counts)*4)
        self.assertEqual(value['parameters']['unique_parameters'], sum(counts))
        self.assertEqual(value['parameters']['tied_embedding_count'], 1)

    def test_per_matrix_packing_rounds_individually(self):
        p = plan(config=ModelConfig(context_length=8, blocks=1, width=3, heads=1,
                                   feed_forward_width=5).to_dict())
        r = a.report(p)['parameters']['hypothetical_fixed_2bit_ternary']
        self.assertEqual(r['matrix_code_bytes'], 4*3 + 2*4)
        self.assertEqual(r['float32_scale_bytes'], 24)
        self.assertEqual(r['payload_bytes'], r['matrix_code_bytes']+24+r['other_float32_bytes'])

    def test_lanes_have_identical_resident_float32_sizes(self):
        results = [a.report(plan(lane=x)) for x in ('dense', 'ternary', 'four-state')]
        self.assertEqual(len({r['parameters']['native_float32_parameter_bytes'] for r in results}), 1)
        self.assertIsNone(results[0]['parameters']['hypothetical_fixed_2bit_ternary'])
        self.assertIsNone(results[2]['parameters']['hypothetical_fixed_2bit_ternary'])
        self.assertEqual(results[0]['native_full_prefix_forwards'][0]['quantizer_materializations']['effective_float32_weights_all_calls_bytes'], 0)
        self.assertGreater(results[1]['native_full_prefix_forwards'][0]['quantizer_materializations']['effective_float32_weights_all_calls_bytes'], 0)

    def test_prefix_schedule_and_null_native_cache(self):
        r = a.report(plan())
        self.assertEqual([x['prefix_tokens'] for x in r['native_full_prefix_forwards']], [3,4,5,6])
        self.assertEqual(r['schedule']['processed_positions'], 2*(3+4+5+6))
        for row in r['native_full_prefix_forwards']:
            self.assertEqual(row['native_cache']['mode'], 'unsupported')
            self.assertIsNone(row['native_cache']['read_bytes'])
        self.assertIsNone(r['physical_dram_traffic_bytes'])
        self.assertIsNone(r['peak_live_activation_bytes'])

    def test_cache_append_and_reads_include_current_token(self):
        rows = a.report(plan())['hypothetical_dense_float32_cache']['rows']
        self.assertEqual([r['appended_tokens'] for r in rows], [3,1,1,1])
        for row in rows:
            self.assertEqual(row['occupied_kv_payload_bytes'], 2*2*2*row['cached_tokens']*6*4)
            self.assertEqual(row['dense_kv_operand_read_once_bytes'], row['occupied_kv_payload_bytes'])
        self.assertEqual(sum(r['kv_append_payload_bytes'] for r in rows), rows[-1]['occupied_kv_payload_bytes'])

    def test_heads_change_scores_not_kv_payload(self):
        p = plan()
        before = a.report(p)
        p['config']['heads'] = 3
        after = a.report(p)
        self.assertEqual(before['hypothetical_dense_float32_cache'], after['hypothetical_dense_float32_cache'])
        b = before['native_full_prefix_forwards'][0]['selected_activation_payload_bytes']
        c = after['native_full_prefix_forwards'][0]['selected_activation_payload_bytes']
        self.assertEqual(b['attention_scores_all_blocks']*3, c['attention_scores_all_blocks']*2)
        self.assertEqual(b['k_outputs_all_blocks'], c['k_outputs_all_blocks'])

    def test_activations_square_and_output_shapes(self):
        row = a.report(plan())['native_full_prefix_forwards'][0]
        sizes = row['selected_activation_payload_bytes']
        self.assertEqual(sizes['q_outputs_all_blocks'], 2*2*3*6*4)
        self.assertEqual(sizes['attention_scores_all_blocks'], 2*2*2*3*3*4)
        self.assertEqual(sizes['logits'], 2*3*259*4)
        self.assertEqual(sizes['ffn_up_outputs_all_blocks'], 2*2*3*9*4)
        self.assertEqual(row['logical_operand_payload_bytes']['tied_output_weight_once'],259*6*4)

    def test_one_forward_and_exact_context(self):
        r = a.report(plan(prompt_tokens=16, forward_calls=1))
        self.assertEqual(len(r['native_full_prefix_forwards']), 1)
        self.assertEqual(r['hypothetical_dense_float32_cache']['rows'][0]['appended_tokens'],16)
        with self.assertRaises(ContractError): a.report(plan(prompt_tokens=16,forward_calls=2))

    def test_strict_plan_validation(self):
        for key, bad in [('batch',True),('batch',0),('batch',9),('forward_calls',17),
                         ('forward_calls',1.0),('prompt_tokens',False),('lane','cached')]:
            with self.subTest(key=key,bad=bad), self.assertRaises(ContractError):
                a.report(plan(**{key:bad}))
        p=plan(); p['extra']=1
        with self.assertRaises(ContractError): a.report(p)
        p=plan(); del p['config']['dtype']
        with self.assertRaises(ContractError): a.report(p)

    def test_native_budgets_reject_without_allocation(self):
        with self.assertRaises(ContractError):
            a.report(plan(config=ModelConfig(width=1024,feed_forward_width=4096).to_dict()))
        with self.assertRaises(ContractError):
            a.report(plan(config=ModelConfig(width=32,heads=32,blocks=1).to_dict(),
                          batch=8,prompt_tokens=256,forward_calls=1))

    def test_source_pin_rejects_changed_model(self):
        with patch.dict(a.BASELINE, {'src/trinite/model.py':'0'*64}):
            with self.assertRaises(ContractError): a.report(plan())

    def test_foreign_contract_import_rejected(self):
        import trinite.contracts
        with patch.object(trinite.contracts, '__file__', '/foreign/contracts.py'):
            with self.assertRaises(ContractError): a.report(plan())

    def test_deterministic_and_input_unchanged(self):
        p=plan(); before=copy.deepcopy(p)
        self.assertEqual(json_bytes(a.report(p)),json_bytes(a.report(p)))
        self.assertEqual(p,before)

    def test_cli_isolated_no_backend_import_and_bounds(self):
        root=a.ROOT
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'plan.json'; path.write_bytes(json_bytes(plan()))
            command=[sys.executable,'-I','-B',str(root/'scripts/report_memory_traffic.py'),'--plan',str(path)]
            r=subprocess.run(command,capture_output=True,check=True)
            self.assertEqual(json.loads(r.stdout)['schema'],a.SCHEMA)
            self.assertEqual(r.stderr,b'')
            path.write_bytes(b' '*65537)
            self.assertEqual(subprocess.run(command,capture_output=True).returncode,2)
            path.write_text('{"batch":1,"batch":2}')
            self.assertEqual(subprocess.run(command,capture_output=True).returncode,2)

    def test_cli_rejects_fifo_directory_and_link_without_blocking(self):
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder)
            fifo=folder/'fifo'; os.mkfifo(fifo)
            target=folder/'plan.json'; target.write_bytes(json_bytes(plan()))
            link=folder/'link'; link.symlink_to(target)
            for path in (fifo,folder,link):
                r=subprocess.run([sys.executable,'-I','-B',str(a.ROOT/'scripts/report_memory_traffic.py'),
                                  '--plan',str(path)],capture_output=True,timeout=5)
                self.assertEqual(r.returncode,2)

    def test_accounting_import_does_not_load_torch(self):
        code="import sys; sys.path[:0]=['src','scripts']; import memory_traffic_v1.accounting; assert 'torch' not in sys.modules"
        subprocess.run([sys.executable,'-I','-B','-c',code],cwd=a.ROOT,check=True)


if __name__ == '__main__': unittest.main()
