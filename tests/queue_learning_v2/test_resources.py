"""Real harmless subprocesses test supervision, never a numerical experiment."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from queue_learning_v2 import resources


class SupervisorTests(unittest.TestCase):
    def run_code(self, code, seconds=.3, **kw):
        return resources.supervise([sys.executable, '-c', code], seconds, **kw)

    def test_success_and_reaped_cpu(self):
        r = self.run_code('print("hello")')
        self.assertEqual((r['exit_code'], r['failure'], r['stdout']), (0, None, 'hello\n'))
        self.assertEqual(len(r['reaped']), 1)
        self.assertGreater(r['peak_aggregate_rss_bytes'], 0)
        self.assertGreaterEqual(float.fromhex(r['child_user_seconds_hex']), 0)

    def test_closed_pipes_still_timeout(self):
        r = self.run_code('import os,time;os.close(1);os.close(2);time.sleep(9)')
        self.assertEqual(r['failure'], 'timeout')
        self.assertEqual(r['exit_code'], -signal.SIGKILL)

    def test_new_session_orphan_is_killed_and_reaped(self):
        r = self.run_code('import subprocess,sys;subprocess.Popen([sys.executable,"-c",'
                          '"import time;time.sleep(9)"], start_new_session=True)')
        self.assertEqual(r['failure'], 'descendants_after_worker_exit')
        self.assertEqual(len(r['reaped']), 2)
        for entry in r['reaped']:
            self.assertFalse(Path('/proc', str(entry['pid'])).exists())

    def test_output_bound(self):
        r = self.run_code('import os;os.write(1,b"x"*2000000)', 3)
        self.assertEqual(r['failure'], 'output_limit')
        self.assertEqual(len(r['stdout']), resources.OUTPUT_CAP)

    def test_process_rss_limit_and_highwater(self):
        with patch.object(resources, 'PROCESS_RSS', 32 << 20):
            r = self.run_code('import time;x=bytearray(64*1024*1024);time.sleep(9)', 3)
        self.assertEqual(r['failure'], 'rss_limit')
        self.assertEqual(r['exit_code'], -signal.SIGKILL)

    def test_update_deadline(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'progress.json'
            code = ('import json,time;from pathlib import Path;'
                    f'p=Path({str(p)!r});'
                    'p.write_text(json.dumps({"phase":"updates","step":1,"target_tokens":22,'
                    '"update_started_seconds_hex":time.monotonic().hex()},sort_keys=True,separators=(",",":"))+"\\n");'
                    'time.sleep(9)')
            r = self.run_code(code, 3, progress=p, update_seconds=.1)
            self.assertEqual(r['failure'], 'update_timeout')
            self.assertEqual(r['observed_updates'], 1)

    def test_proc_failure_still_reaps_leader(self):
        original = resources.process_table
        spawned = []
        popen = subprocess.Popen
        calls = 0
        def start(*a, **kw):
            proc = popen(*a, **kw)
            spawned.append(proc)
            return proc
        def table():
            nonlocal calls
            calls += 1
            if calls > 1:
                raise RuntimeError('synthetic proc unreadable')
            return original()
        with patch.object(resources, 'process_table', table), patch.object(resources.subprocess, 'Popen', start):
            with self.assertRaisesRegex(RuntimeError, 'proc unreadable'):
                self.run_code('import time;time.sleep(9)')
        self.assertEqual(spawned[0].returncode, -signal.SIGKILL)

    def test_private_cli_cannot_skip_supervisor(self):
        launcher = Path(__file__).resolve().parents[2] / 'scripts/run_queue_learning_v2.py'
        r = subprocess.run([sys.executable, '-I', str(launcher), '_run'], capture_output=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(b'internal worker requires', r.stderr)


if __name__ == '__main__':
    unittest.main()
