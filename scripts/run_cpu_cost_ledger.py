#!/usr/bin/env python3
"""Checkout-only, inference-only Stage A CPU cost ledger."""
import time
_START = (time.perf_counter_ns(), time.process_time_ns())
from cpu_cost_ledger_v1.controller import main
if __name__ == '__main__':
    raise SystemExit(main(startup_start=_START))
