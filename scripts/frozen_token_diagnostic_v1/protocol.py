"""Fixed observational recipe; cannot authorize training or select checkpoints."""
from addition_development_v1.protocol import STRATA

CLAIMS = {'optimizer_replayed': False, 'predictions_replayed': False,
          'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
          'fresh_confirmation': False}
PROTOCOL = {
    'schema': 'trinite.frozen-token-diagnostic-protocol.v1',
    'checkpoint_step': 4000, 'lane': 'dense', 'dtype': 'float32', 'device': 'cpu',
    'selection': 'all six addition development cells, all rows of all three strata; final4000 only',
    'strata': list(STRATA), 'rows_per_cell': 298, 'cells': 6,
    'conditioning': 'gold-prefix teacher forcing at every answer token including EOS; not raw generation',
    'rank': 'one-based descending logit; ties use lowest token ID',
    'raw_scores': 'original source-bound prompt-only greedy scores copied byte-for-byte; not rerun or repaired',
    'updates': 0, 'checkpoint_election': False, 'repair': False, 'cache_mode': 'unsupported',
    'forward_calls_per_row': 2, 'max_forward_calls_per_cell': 596,
    'worker_seconds': 900, 'max_worker_log_bytes': 1024*1024,
    'max_artifact_bytes': 16*1024*1024, 'max_retained_payload_bytes': 32*1024*1024,
    'progress_chunk_rows': 32,
    'exposure': 'already research-exposed development families; correlated renderings, not fresh confirmation',
    'acceptance': 'descriptive diagnostics only; no adequacy threshold or automatic release gate',
}
