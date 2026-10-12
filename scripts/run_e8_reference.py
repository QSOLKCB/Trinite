"""Print the exact E8 reference and an existing order-two geometry probe.

No training, checkpoint loading, model download or input selection takes place.
"""
import json

from trinite.e8_reference import reference_report
from trinite.geometry import metrics


def main():
    report = reference_report()
    # Equally spaced synthetic sample indices 0,1,2; x_i=i^2.
    trajectory = [[0, 0, 0], [1, 0, 0], [4, 0, 0]]
    report['second_difference_example'] = {
        'samples': trajectory,
        'stencil': [1, -2, 1],
        'sample_unit': 'synthetic integer index; spacing one',
        'order_2_hex': metrics(trajectory)['order_2_hex'],
        'interpretation': 'raw index second difference; not Menger curvature',
    }
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
