"""Fixed exhaustive observational recipe; no training or checkpoint election."""
CLAIMS = {'optimizer_replayed': False, 'audit_replays_inference': False,
          'learning_adequate': False, 'phase5_ready': False, 'release_ready': False,
          'fresh_confirmation': False}
PROTOCOL = {
    'schema': 'trinite.failure-map-protocol.v1',
    'checkpoint_step': 4000, 'lane': 'dense', 'dtype': 'float32', 'device': 'cpu',
    'selection': 'all six final addition checkpoints; all 338 admitted rows',
    'rows_per_cell': 338, 'cells': 6, 'generation_cap': 5,
    'conditioning': 'prompt-only greedy generation; no gold prefix or answer repair',
    'raw_scores': 'original final training/development predictions retained unchanged and compared by exact row',
    'updates': 0, 'checkpoint_election': False, 'repair': False, 'cache_mode': 'unsupported',
    'max_forward_calls_per_cell': 1690,
    'worker_seconds': 900, 'max_worker_log_bytes': 1024*1024,
    'max_artifact_bytes': 16*1024*1024, 'max_retained_payload_bytes': 32*1024*1024,
    'progress_chunk_rows': 32,
    'exposure': 'historically research-exposed finite development corpus; seen anchors are not generalisation',
    'acceptance': 'descriptive per-case map only; consistency is not correctness; no release threshold',
}
