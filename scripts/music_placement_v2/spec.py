"""Same frozen workload/admission, new prospective custody orchestration identity."""
from copy import deepcopy
from music_placement_v1.spec import (SEEDS, ARMS, CELLS, cell_name, plan_for,
    diagnostic_rows, train_rows, development_rows, schedule_receipt, dataset,
    require_admission, PROTOCOL as ORIGINAL_PROTOCOL)

PROTOCOL = deepcopy(ORIGINAL_PROTOCOL)
PROTOCOL['schema'] = 'trinite.music-placement-protocol.v2'
PROTOCOL['numerical_semantics'] = 'unchanged music_placement_v1 numerical/data/scoring; new request only'
PROTOCOL['partial_custody'] = 'no matrix seal or identity without all fixed records, no pending development and exact request/receipt inventory; raw partial bytes retained'
