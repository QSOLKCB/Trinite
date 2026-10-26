"""Freeze before outcomes. No search, adaptive repair, or target run in CI."""
ARMS = ('control', 'candidate')
MODES = {'control': 'sequence', 'candidate': 'coupled'}
SEEDS = (0, 1, 2)
STRATA = ('unseen', 'length3', 'length4')
STEPS = 4000
BATCH_SIZE = 8
MILESTONES = (0, 1000, 2000, 3000, 4000)
POSITION_CONFIG = {'schema': 'trinite.position-study-addressing.v1',
    'slots': 'trinite.arithmetic-position-slots.v1',
    'offset_policy': 'fixed-zero.v1', 'offset': 0,
    'cap_policy': 'prompt-operand-width-plus-two.v1',
    'adapter': 'trinite.position-study-live-prefix.v1'}
PROTOCOL = {
    'schema': 'trinite.position-study-protocol.v1',
    'hypothesis': 'equal-significance position coupling improves transfer conditional on common padded operands and reversed fixed-width answers',
    'arms': MODES, 'seeds': list(SEEDS), 'steps': STEPS, 'batch_size': BATCH_SIZE,
    'milestones': list(MILESTONES), 'position_configuration': POSITION_CONFIG,
    'schedule': 'position-study-family-order-width-cycle.v1',
    'cycle_visits': 160, 'cycles': 200, 'scheduled_visits': 32000,
    'scheduled_model_input_tokens': 307200, 'scheduled_target_tokens': 121600,
    'inventory': {'train': [64, 154], 'unseen': [24, 56],
                  'length3': [12, 23], 'length4': [12, 23]},
    'matched': ['all data and schedule', 'effective seeded parameters', 'optimizer',
                'architecture', 'updates and token budgets', 'generation cap policy'],
    'changed': ['position addressing only'],
    'selection': 'final step4000 only; no early stopping, election, retries or adaptive reruns',
    'training_diagnostics': 'unique training rows only at fixed milestones',
    'development': 'all probes excluded from updates, loss diagnostics and checkpoint selection; final only',
    'exposure': 'prospective development; historical0..12 facts are exposed, not fresh confirmation',
    'criteria': {'minimum_candidate_seen_percent': 95,
        'maximum_seen_regression_percentage_points': 2,
        'minimum_each_seed_unseen_gain_percentage_points': 10,
        'minimum_each_seed_length_gain_percentage_points': 10,
        'minimum_candidate_each_seed_length_percent': 25,
        'unit': 'all-renderings-correct semantic families; each seed and each stratum separately',
        'interpretation': 'descriptive support only if all criteria pass and all cells verify; otherwise mixed or unsupported; no significance or mechanism proof'},
    'numerical_tolerances': {'same_shape': 'exact', 'resume': 'exact model/AdamW/history',
        'cross_shape_absolute': '0.000002', 'cross_shape_relative': '0.00001'},
    'worker_seconds': 1800, 'update_seconds_total': 960,
    'update_time_enforcement': 'sampled at milestones; per-invocation native bound480; outer worker deadline1800',
    'max_artifact_bytes': 16 * 1024 * 1024,
    'max_retained_payload_bytes': 32 * 1024 * 1024,
    'max_worker_log_bytes': 1024 * 1024,
    'cache_mode': 'unsupported', 'geometry_loss': '0',
    'fresh_confirmation': False, 'learning_adequate': False,
    'phase5_ready': False, 'release_ready': False,
}
