"""Prospective fixed development recipe, not a fresh confirmation study."""

ARMS = ('control', 'candidate')
STRATA = ('familiar-answer', 'novel-answer', 'range-length')
SEEDS = (0, 1, 2)
STEPS = 4000
BATCH_SIZE = 8
MILESTONES = (0, 1000, 2000, 3000, 4000)
PROTOCOL = {
    'schema': 'trinite.addition-development-protocol.v1',
    'task': 'add', 'lane': 'dense', 'arms': list(ARMS), 'seeds': list(SEEDS),
    'steps': STEPS, 'batch_size': BATCH_SIZE, 'milestones': list(MILESTONES),
    'schedule': 'paired-slot-cycle.v1; fixed across initialization seeds',
    'cycle_visits': 40, 'cycles': 800, 'generation_cap': 5,
    'worker_seconds': 900, 'update_seconds_total': 480,
    'max_artifact_bytes': 16 * 1024 * 1024,
    'max_retained_payload_bytes': 32 * 1024 * 1024,
    'max_worker_log_bytes': 1024 * 1024,
    'update_time_enforcement': 'sampled at milestones; worker deadline is outer bound',
    'intervention': 'expand five unordered training families to ten at matched scheduled exposure',
    'matched': ['dense architecture', 'paired seeded initialization', 'optimizer',
                'update count', 'per-slot answer/carrier/input and target lengths',
                'cycle operand marginals and model-input token histogram',
                'raw greedy decoding and output cap'],
    'changed': ['unique training families', 'unique training rows',
                'per-example repetition and input co-occurrences'],
    'training_diagnostics': 'unique admitted rows for each arm; never repeated scheduled visits',
    'development': 'shared family-disjoint strata, excluded from both training arms; final checkpoint only',
    'selection': 'fixed final step; no election, retries, early stopping or answer repair',
    'exposure': 'public mathematical development facts; historical overlaps possible; not fresh confirmation',
    'cache_mode': 'unsupported', 'geometry_loss': '0',
    'fresh_confirmation': False, 'learning_adequate': False,
    'phase5_ready': False, 'release_ready': False,
}


def cells():
    """Fixed six-cell order; no outcome-dependent scheduling."""
    return tuple((seed, arm) for seed in SEEDS for arm in ARMS)
