"""Print the exact E8 reference and an existing order-two geometry probe.

No training, checkpoint loading, model download or input selection takes place.
"""
import argparse
import json

from trinite.e8_reference import reference_report
from trinite.e8_reference import etq
from trinite.geometry import metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--etq-map', action='store_true',
                        help='print the complete compatible ETQ-303 event document as canonical JSON')
    args = parser.parse_args()
    if args.etq_map:
        print(etq.event_document_bytes().decode('utf-8'), end='')
        return
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
    graph = etq.lifted_graph()
    report['etq_reference'] = {
        'donor_commit': etq.SONIFICATION_COMMIT,
        'base_model': 'ETQ-101@2.0.0', 'event_protocol': 'ETQ-303@3.0.0',
        'base_sites': len(etq.site_registry()), 'tensor_states': graph['vertexCount'],
        'base_graph_edges': graph['baseEdgesPerLayer'],
        'lifted_graph_edges': graph['edgeCount'],
        'internal_triality_and_external_fibre_are_distinct': True,
        'phase_symbols': [etq.GAUSSIAN_SYMBOLS[g] for g in etq.PHASE_EXPONENTS],
        'commitments': etq.commitments(),
        'first_three_events': etq.canonical_events()[:3],
        'neural_mapping': 'not defined',
    }
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
