"""Independent scalar four-state math, reusing the unchanged scalar decoder."""
import importlib.util
import math
from pathlib import Path
import struct

spec = importlib.util.spec_from_file_location('base_scalar_oracle', Path(__file__).resolve().parents[1]/'model/oracle.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


def effective(matrix, lane):
    if lane != 'four-state':
        raise ValueError('four-state oracle only')
    scale = f32(max(math.fsum(abs(v) for row in matrix for v in row)
                    / (2*sum(len(row) for row in matrix)), 1e-8))
    def level(v):
        unit = f32(v/scale)
        code = (3 if abs(unit) > 2 else 1)*(-1 if unit < 0 else 1)
        return f32(scale*code)
    return [[level(v) for v in row] for row in matrix]


base.effective = effective
forward = base.forward
hand_parameters = base.hand_parameters
