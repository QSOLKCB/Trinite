"""V2 runtime-policy identity; all original data, scoring and budgets unchanged."""
from copy import deepcopy
from music_device_trial_v1.spec import (
    BACKENDS, SEED, TIMING_PHASES, PARITY_TOLERANCES, backend_name,
    score_cell, matrix_summary, train_rows, diagnostic_rows, development_rows,
    plan_for, dataset, schedule_receipt, PROTOCOL as ORIGINAL_PROTOCOL)

PROTOCOL = deepcopy(ORIGINAL_PROTOCOL)
PROTOCOL.update(schema='trinite.music-device-trial-protocol.v2',
    allocator_policy='exact pinned-PyTorch integer-byte quantization; requested and effective limits retained; no cap relaxation',
    historical_request_migration=False)
