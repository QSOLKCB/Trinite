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

    def test_disappearing_proc_stat_accepts_enoent_and_esrch(self):
        # Linux can report either errno when a short-lived process disappears
        # between enumerating /proc and reading its stat file. No worker runs.
        for error in (FileNotFoundError(2, 'gone'), ProcessLookupError(3, 'gone')):
            with self.subTest(error=type(error).__name__):
                with (patch.object(Path, 'iterdir', return_value=iter([Path('/proc/12345')])),
                      patch.object(Path, 'read_text', side_effect=error)):
                    self.assertEqual(resources.process_table(), {})

    def test_proc_permission_and_unexpected_io_failure_still_stop_inspection(self):
        for error, expected in ((PermissionError(13, 'denied'), RuntimeError),
                                (OSError(5, 'IO failure'), OSError)):
            with self.subTest(error=type(error).__name__):
                with (patch.object(Path, 'iterdir', return_value=iter([Path('/proc/12345')])),
                      patch.object(Path, 'read_text', side_effect=error)):
                    with self.assertRaises(expected):
                        resources.process_table()

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
        # The former fire-and-forget parent could exit before the supervisor's
        # first /proc sample. A fast child startup/exit then made this test
        # nondeterministic: it had not established a *surviving* descendant.
        # Hold the parent until the supervisor actually sees both live
        # processes. The child still runs in an independent session.
        with tempfile.TemporaryDirectory() as d:
            spawned = Path(d) / 'spawned'
            release = Path(d) / 'release'
            code = (
                'import subprocess,sys,time;from pathlib import Path;'
                'child=subprocess.Popen([sys.executable,"-c",'
                '"import time;time.sleep(9)"],start_new_session=True);'
                f'Path({str(spawned)!r}).write_text(str(child.pid));'
                f'gate=Path({str(release)!r});'
                'while not gate.exists():time.sleep(.001)'
            )
            original = resources.process_table
            observed = []

            def sampled():
                table = original()
                if spawned.exists() and not release.exists():
                    try:
                        child_pid = int(spawned.read_text())
                    except (OSError, ValueError):
                        return table
                    child = table.get(child_pid)
                    parent = table.get(child['ppid']) if child else None
                    if (child and child['state'] not in ('Z', 'X') and
                            parent and parent['state'] not in ('Z', 'X') and
                            parent['ppid'] == os.getpid()):
                        observed.append((child_pid, child['ppid']))
                        release.write_text('observed')
                return table

            with patch.object(resources, 'process_table', sampled):
                r = self.run_code(code, seconds=3)
            self.assertEqual(len(observed), 1, 'orphan spawn was not observed alive')
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

    def update_deadline(self, *, slow_sample=False):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'progress.json'
            release = Path(d) / 'release'
            code = ('import json,time;from pathlib import Path;'
                    f'p=Path({str(p)!r});q=p.with_suffix(".next");'
                    f'release=Path({str(release)!r});')
            if slow_sample:
                code += '\nwhile not release.exists(): time.sleep(.001)\n'
            code += ('q.write_text(json.dumps({"phase":"updates","step":1,"target_tokens":22,'
                     '"update_started_seconds_hex":time.monotonic().hex()},sort_keys=True,separators=(",",":"))+"\\n");'
                     'q.replace(p);time.sleep(9)')
            original = resources.process_table
            calls = 0
            def sampled():
                nonlocal calls
                calls += 1
                if slow_sample and calls == 2:
                    release.write_text('publish now')
                    deadline = time.monotonic() + 2
                    while not p.exists() and time.monotonic() < deadline:
                        time.sleep(.001)
                    self.assertTrue(p.exists(), 'worker did not publish progress')
                return original()
            with patch.object(resources, 'process_table', sampled):
                r = self.run_code(code, 3, progress=p, update_seconds=.1)
            self.assertEqual(r['failure'], 'update_timeout')
            self.assertEqual(r['observed_updates'], 1)

    def test_update_deadline(self):
        self.update_deadline()

    def test_update_clock_published_during_slow_process_sample(self):
        self.update_deadline(slow_sample=True)

    def test_future_update_clock_is_still_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'progress.json'
            code = ('import json,time;from pathlib import Path;'
                    f'p=Path({str(p)!r});q=p.with_suffix(".next");'
                    'q.write_text(json.dumps({"phase":"updates","step":0,"target_tokens":0,'
                    '"update_started_seconds_hex":(time.monotonic()+10).hex()},sort_keys=True,separators=(",",":"))+"\\n");'
                    'q.replace(p);time.sleep(9)')
            with self.assertRaisesRegex(ValueError, 'worker update clock changed'):
                self.run_code(code, 3, progress=p)

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
