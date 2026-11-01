"""Linux bounded process-tree supervisor; RSS is sampled, AS is a hard rlimit.

Uses a child subreaper and PID start-time identities, including descendants that
start new sessions. Sampling cannot prove absence of transient RSS excursions.
This is a local cooperative worker boundary, not a hostile-code sandbox.
"""
import ctypes
import os
from pathlib import Path
import resource
import selectors
import signal
import subprocess
import time

import queue_evaluation_v1 as gate
from queue_learning_v1.artifacts import read_json

PROCESS_RSS = 1 << 30
TREE_RSS = 2 << 30
ADDRESS_SPACE = 4 << 30
OUTPUT_CAP = 1 << 18


def process_table():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():
            continue
        try:
            raw = (path / 'stat').read_text()
            # comm may contain spaces and parentheses.
            tail = raw[raw.rindex(')') + 2:].split()
            result[int(path.name)] = {'ppid': int(tail[1]), 'state': tail[0],
                'start': int(tail[19]), 'rss': int(tail[21]) * os.sysconf('SC_PAGE_SIZE'),
                'cpu_ticks': int(tail[11]) + int(tail[12])}
        except FileNotFoundError:
            continue  # Process disappeared during the snapshot.
        except (PermissionError, ValueError, IndexError) as error:
            raise RuntimeError('cannot inspect Linux process table') from error
    return result


def descendants(table, root, known):
    selected = {pid for pid, start in known.items()
                if pid in table and table[pid]['start'] == start}
    selected.add(root)
    while True:
        more = {pid for pid, row in table.items() if row['ppid'] in selected}
        if more <= selected:
            break
        selected |= more
    return {pid: table[pid]['start'] for pid in selected if pid in table}


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (ADDRESS_SPACE, ADDRESS_SPACE))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def supervise(argv, seconds, *, progress=None, update_seconds=120, poll_seconds=.01, max_updates=640, max_targets=20480):
    """Kill/reap on first limit; bounded stdout/stderr and cumulative wait4 usage.

    Called serially by the dedicated controller process. Subreaper adoption also
    tracks a worker's orphaned grandchildren, including new-session Rust gates.
    """
    gate.require(seconds > 0 and poll_seconds > 0, 'positive process deadline required')
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), 'Linux child subreaper required')
    started = time.monotonic()
    before = process_table()
    unrelated = {p: r['start'] for p, r in before.items() if r['ppid'] == os.getpid()}
    proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, start_new_session=True, preexec_fn=_limits,
        env={**os.environ, 'TRINITE_QUEUE_V2_SUPERVISOR': str(os.getpid())})
    selector = selectors.DefaultSelector()
    outputs = {'stdout': bytearray(), 'stderr': bytearray()}
    for stream, name in ((proc.stdout, 'stdout'), (proc.stderr, 'stderr')):
        os.set_blocking(stream.fileno(), False)
        selector.register(stream, selectors.EVENT_READ, name)
    known, cpu = {}, {}
    peak_process = peak_tree = 0
    failure, status = None, None
    reaped = []
    child_user = child_system = 0.0
    last_step = last_targets = 0
    update_start = None
    try:
        while True:
            now = time.monotonic()
            table = process_table()
            # Adopted descendants now have this dedicated supervisor as parent.
            adopted = {p: r['start'] for p, r in table.items()
                       if r['ppid'] == os.getpid() and unrelated.get(p) != r['start']}
            known.update(adopted)
            live = descendants(table, proc.pid, known)
            known.update(live)
            for pid in live:
                row = table[pid]
                cpu[(pid, row['start'])] = max(cpu.get((pid, row['start']), 0), row['cpu_ticks'])
            rss = [table[p]['rss'] for p in live]
            # Include this controller's own resident state in aggregate accounting.
            rss.append(table[os.getpid()]['rss'])
            peak_process = max(peak_process, *rss)
            peak_tree = max(peak_tree, sum(rss))
            if now - started >= seconds:
                failure = failure or 'timeout'
            if max(rss) >= PROCESS_RSS or sum(rss) >= TREE_RSS:
                failure = failure or 'rss_limit'
            if progress is not None and Path(progress).exists():
                value = read_json(Path(progress), 16384)
                step, targets = value['step'], value['target_tokens']
                gate.require(type(step) is int and type(targets) is int and
                             step >= last_step and targets >= last_targets,
                             'worker progress must be monotone integers')
                if step > max_updates or targets > max_targets:
                    failure = failure or 'work_budget'
                last_step, last_targets = step, targets
                marker = value['update_started_seconds_hex']
                if marker is not None:
                    candidate = float.fromhex(marker)
                    gate.require(started <= candidate <= now and
                                 (update_start is None or candidate == update_start),
                                 'worker update clock changed')
                    update_start = candidate
                if value['phase'] == 'updates' and (update_start is None or now - update_start >= update_seconds):
                    failure = failure or 'update_timeout'
            if failure:
                for pid, birth in live.items():
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            # wait4 gives cumulative accounting for every adopted child we reap.
            for pid in set(live) | {proc.pid}:
                try:
                    got, state, usage = os.wait4(pid, os.WNOHANG)
                except ChildProcessError:
                    continue
                if got:
                    if usage.ru_maxrss * 1024 >= PROCESS_RSS:
                        failure = failure or 'rss_limit'
                    peak_process = max(peak_process, usage.ru_maxrss * 1024)
                    child_user += usage.ru_utime
                    child_system += usage.ru_stime
                    reaped.append({'pid': got, 'start': known.get(pid), 'status': state,
                                   'peak_rss_kib': usage.ru_maxrss})
                    if got == proc.pid:
                        status = os.waitstatus_to_exitcode(state)
                        proc.returncode = status
            if status is not None:
                remaining = {p for p in live if p != proc.pid and table[p]['state'] != 'Z'}
                if remaining:
                    failure = failure or 'descendants_after_worker_exit'
                    for pid in remaining:
                        try:
                            os.kill(pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                if not remaining:
                    # Recheck adoption on the next pass until every known PID is gone.
                    current = process_table()
                    if not any(p != proc.pid and p in current and current[p]['start'] == birth
                               for p, birth in known.items()):
                        break
            for key, _ in selector.select(poll_seconds):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                remaining = OUTPUT_CAP - len(outputs[key.data])
                outputs[key.data].extend(chunk[:remaining])
                if len(chunk) > remaining:
                    failure = failure or 'output_limit'
            if not selector.get_map():
                time.sleep(poll_seconds)  # Closed pipes do not end supervision.
        # Drain available final bytes after the process has been reaped.
        for stream, name in ((proc.stdout, 'stdout'), (proc.stderr, 'stderr')):
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                room = OUTPUT_CAP - len(outputs[name])
                outputs[name].extend(chunk[:room])
                if len(chunk) > room:
                    failure = failure or 'output_limit'
    except BaseException:
        # Cleanup must work even if /proc itself became unreadable. The initial
        # process group plus retained start identities are a conservative fallback.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        cleanup = dict(known)
        try:
            table = process_table()
            cleanup.update(descendants(table, proc.pid, known))
            cleanup.update({p: r['start'] for p, r in table.items()
                            if r['ppid'] == os.getpid() and unrelated.get(p) != r['start']})
        except Exception:
            pass
        for pid in cleanup:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait()
        for pid in cleanup:
            if pid != proc.pid:
                try:
                    os.waitpid(pid, 0)
                except ChildProcessError:
                    pass
        raise
    finally:
        selector.close()
        proc.stdout.close()
        proc.stderr.close()
    elapsed = time.monotonic() - started
    if elapsed >= seconds:
        failure = failure or 'timeout'
    return {'exit_code': status, 'failure': failure,
        'limits': {'rss_bytes_per_process': PROCESS_RSS, 'aggregate_rss_bytes': TREE_RSS,
                   'address_space_bytes_per_process': ADDRESS_SPACE},
        'wall_seconds_hex': elapsed.hex(),
        'deadline_seconds_hex': float(seconds).hex(),
        'stdout': outputs['stdout'].decode('utf-8', 'replace'),
        'stderr': outputs['stderr'].decode('utf-8', 'replace'),
        'peak_process_rss_bytes': peak_process, 'peak_aggregate_rss_bytes': peak_tree,
        'child_user_seconds_hex': child_user.hex(), 'child_system_seconds_hex': child_system.hex(),
        'observed_process_cpu_seconds_hex': (sum(cpu.values()) / os.sysconf('SC_CLK_TCK')).hex(),
        'reaped': reaped, 'observed_updates': last_step, 'observed_target_tokens': last_targets,
        'scope': 'Linux sampled process-tree RSS including controller; hard inherited per-process AS; '
                 'wait4 cumulative reaped child CPU; sampling is not transient peak attestation'}
