"""Engineering-only constant-policy smoke. No model loading or scientific runs."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import bounded_controller_v1 as controller
import queue_evaluation_v1 as gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--constant', required=True)
    args = parser.parse_args()
    raw = sys.stdin.buffer.read(16385)
    gate.require(len(raw) <= 16384, 'request too large')
    request = gate.decode(raw)
    result = controller.run(controller.freeze(request), controller.ConstantPolicy(args.constant))
    result['policy_binding'] = dict(kind='ConstantPolicy', source_bound=True, configuration_bound=True,
                                    constant=args.constant, scope='local controller source closure only')
    sys.stdout.buffer.write(gate.encode(result))


if __name__ == '__main__':
    try:
        main()
    except (gate.GateError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2)
