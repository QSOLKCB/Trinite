"""Independent integer-tick checker for raw symbolic score JSON values.

Does not import the generator, notation types, Fraction, audio or labels kernel.
A common integer grid is computed from the submitted denominators, then all
bar totals, tie edges and annotations are checked on that grid.
"""
from math import gcd, lcm


def _record(value, fields):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError("invalid oracle record")


def _int(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("invalid oracle integer")


def annotate(value):
    """Validate a raw score independently and return exact notation annotations."""
    _record(value, ("schema", "meter", "tempo_qpm", "bars"))
    if value["schema"] != "trinite.music-score.v1":
        raise ValueError("unknown score schema")
    _int(value["tempo_qpm"], 30, 300)
    meter = value["meter"]
    _record(meter, ("numerator", "denominator", "groups"))
    n, d = meter["numerator"], meter["denominator"]
    _int(n, 1, 16)
    _int(d, 1, 16)
    if (n, d) not in ((3, 4), (4, 4), (5, 8), (7, 8), (8, 16)):
        raise ValueError("unsupported meter")
    groups = meter["groups"]
    if type(groups) is not list or not 1 <= len(groups) <= 16:
        raise ValueError("invalid accent groups")
    for group in groups:
        _int(group, 1, 16)
    if sum(groups) != n:
        raise ValueError("incorrect group sum")
    bars = value["bars"]
    if type(bars) is not list or not 1 <= len(bars) <= 8:
        raise ValueError("invalid bars")
    grid = d
    for bar in bars:
        if type(bar) is not list or not 1 <= len(bar) <= 64:
            raise ValueError("invalid bar events")
        for e in bar:
            _record(e, ("duration", "pitch", "tie_in", "tie_out"))
            duration = e["duration"]
            if type(duration) is not list or len(duration) != 2:
                raise ValueError("invalid exact duration")
            a, b = duration
            _int(a, 1, 64)
            _int(b, 1, 64)
            if a > b or gcd(a, b) != 1:
                raise ValueError("duration must be positive, reduced and <= 1")
            grid = lcm(grid, b)
            if e["pitch"] is not None:
                _int(e["pitch"], 0, 127)
            if type(e["tie_in"]) is not bool or type(e["tie_out"]) is not bool:
                raise ValueError("invalid tie flags")
            if e["pitch"] is None and (e["tie_in"] or e["tie_out"]):
                raise ValueError("rest tie")

    def reduced(ticks):
        common = gcd(ticks, grid)
        return [ticks // common, grid // common]

    bar_ticks = n * (grid // d)
    attacks = rests = ties = 0
    prior_pitch, pending = None, False
    boundaries = [reduced(0)]
    accents = []
    for bar_index, bar in enumerate(bars):
        accent = bar_index * bar_ticks
        for group in groups:
            accents.append(reduced(accent))
            accent += group * (grid // d)
        used = 0
        for e in bar:
            a, b = e["duration"]
            used += a * (grid // b)
            if e["tie_in"] != pending or (pending and prior_pitch != e["pitch"]):
                raise ValueError("invalid tie edge")
            attacks += int(e["pitch"] is not None and not e["tie_in"])
            rests += int(e["pitch"] is None)
            ties += int(e["tie_out"])
            prior_pitch, pending = e["pitch"], e["tie_out"]
        if used != bar_ticks:
            raise ValueError("incorrect bar duration")
        boundaries.append(reduced((bar_index + 1) * bar_ticks))
    if pending:
        raise ValueError("dangling tie")
    return {"attacks": attacks, "rests": rests, "ties": ties,
            "bar_boundaries": boundaries, "group_onsets": accents,
            "duration": reduced(len(bars) * bar_ticks)}
