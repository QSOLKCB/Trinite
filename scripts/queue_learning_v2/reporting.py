"""V2 presentation of unchanged authoritative v1 rational score aggregation.

The original v1 summary remains retained byte-for-byte. This adapter changes
only the new summary schema and undefined (zero-denominator) rational metrics;
it does not change scoring, coverage, denominators or any historical producer.
"""
import queue_evaluation_v1 as gate


def normalize(summary):
    gate.require(type(summary) is dict and summary.get('schema') == 'trinite.queue-learning-summary.v1',
                 'v2 reporting requires the authoritative v1 score summary')

    def visit(value):
        if type(value) is dict:
            if set(value) == {'numerator', 'denominator'}:
                gate.require(type(value['numerator']) is int and type(value['denominator']) is int
                             and value['denominator'] >= 0, 'invalid rational score metric')
                return None if value['denominator'] == 0 else dict(value)
            return {key: visit(item) for key, item in value.items()}
        if type(value) is list:
            return [visit(item) for item in value]
        return value

    result = visit(summary)
    result['schema'] = 'trinite.queue-learning-summary.v2'
    return result
