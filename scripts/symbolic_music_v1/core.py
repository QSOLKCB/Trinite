"""Bounded, exact monophonic notation; no model, geometry or audio dependency.

Durations and onsets are whole-note fractions. Meter grouping is notation,
whereas tempo is integer quarter notes per minute, independent of denominator.
"""
from dataclasses import dataclass
from fractions import Fraction
from math import gcd

SCHEMA = "trinite.music-score.v1"
SUPPORTED_METERS = ((3, 4), (4, 4), (5, 8), (7, 8), (8, 16))


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in [{low}, {high}]")


def _keys(value, expected, name):
    if type(value) is not dict or set(value) != set(expected):
        raise ValueError(f"invalid {name} fields")


@dataclass(frozen=True)
class Meter:
    numerator: int
    denominator: int
    groups: tuple[int, ...]

    def __post_init__(self):
        _integer(self.numerator, 1, 16, "meter numerator")
        _integer(self.denominator, 1, 16, "meter denominator")
        if (self.numerator, self.denominator) not in SUPPORTED_METERS:
            raise ValueError("unsupported meter")
        if type(self.groups) is not tuple or not 1 <= len(self.groups) <= 16:
            raise ValueError("groups must be a nonempty bounded tuple")
        for group in self.groups:
            _integer(group, 1, 16, "accent group")
        if sum(self.groups) != self.numerator:
            raise ValueError("accent groups must sum to the meter numerator")

    @property
    def duration(self):
        return Fraction(self.numerator, self.denominator)


@dataclass(frozen=True)
class Event:
    duration: Fraction
    pitch: int | None
    tie_in: bool = False
    tie_out: bool = False

    def __post_init__(self):
        if type(self.duration) is not Fraction or not 0 < self.duration <= 1:
            raise ValueError("duration must be an exact positive Fraction <= 1")
        if self.duration.denominator > 64:
            raise ValueError("duration denominator exceeds 64")
        if self.pitch is not None:
            _integer(self.pitch, 0, 127, "MIDI pitch")
        if type(self.tie_in) is not bool or type(self.tie_out) is not bool:
            raise ValueError("tie flags must be booleans")
        if self.pitch is None and (self.tie_in or self.tie_out):
            raise ValueError("rests cannot be tied")


@dataclass(frozen=True)
class Score:
    meter: Meter
    bars: tuple[tuple[Event, ...], ...]
    tempo: int = 120

    def __post_init__(self):
        validate(self)


def validate(score):
    """Reject malformed notation, incomplete bars and unmatched ties."""
    if type(score) is not Score or type(score.meter) is not Meter:
        raise ValueError("expected Score with Meter")
    score.meter.__post_init__()
    _integer(score.tempo, 30, 300, "quarter notes per minute")
    if type(score.bars) is not tuple or not 1 <= len(score.bars) <= 8:
        raise ValueError("bars must be a tuple of 1..8 bars")
    previous = None
    for bar in score.bars:
        if type(bar) is not tuple or not 1 <= len(bar) <= 64:
            raise ValueError("each bar must be a tuple of 1..64 events")
        total = Fraction(0)
        for event in bar:
            if type(event) is not Event:
                raise ValueError("expected Event")
            event.__post_init__()
            total += event.duration
            if event.tie_in != bool(previous and previous.tie_out):
                raise ValueError("unmatched tie")
            if event.tie_in and event.pitch != previous.pitch:
                raise ValueError("ties must connect identical pitches")
            previous = event
        if total != score.meter.duration:
            raise ValueError("each bar must exactly fill its notated meter")
    if previous.tie_out:
        raise ValueError("dangling final tie")


def timeline(score):
    """Return every notated event at its exact absolute whole-note onset."""
    validate(score)
    onset = Fraction(0)
    result = []
    for bar in score.bars:
        for event in bar:
            result.append((onset, event))
            onset += event.duration
    return tuple(result)


def pair(value):
    return [value.numerator, value.denominator]


def labels(score):
    """Exact notation diagnostics, not inferred meter or audio measurements.

    A group boundary is an accent *position*, not necessarily a note attack.
    A tied continuation contributes notation but never a new attack.
    """
    events = timeline(score)
    group_positions = []
    for bar_index in range(len(score.bars)):
        position = bar_index * score.meter.duration
        for group in score.meter.groups:
            group_positions.append(pair(position))
            position += Fraction(group, score.meter.denominator)
    return {
        "attacks": sum(e.pitch is not None and not e.tie_in for _, e in events),
        "rests": sum(e.pitch is None for _, e in events),
        "ties": sum(e.tie_out for _, e in events),
        "bar_boundaries": [pair(i * score.meter.duration)
                           for i in range(len(score.bars) + 1)],
        "group_onsets": group_positions,
        "duration": pair(len(score.bars) * score.meter.duration),
    }


def to_dict(score):
    validate(score)
    return {"schema": SCHEMA,
            "meter": {"numerator": score.meter.numerator,
                      "denominator": score.meter.denominator,
                      "groups": list(score.meter.groups)},
            "tempo_qpm": score.tempo,
            "bars": [[{"duration": pair(e.duration), "pitch": e.pitch,
                       "tie_in": e.tie_in, "tie_out": e.tie_out}
                      for e in bar] for bar in score.bars]}


def from_dict(value):
    """Strict JSON-value reader: no coercion, extra fields or unreduced pairs."""
    _keys(value, ("schema", "meter", "tempo_qpm", "bars"), "score")
    if value["schema"] != SCHEMA:
        raise ValueError("unknown score schema")
    m = value["meter"]
    _keys(m, ("numerator", "denominator", "groups"), "meter")
    if type(m["groups"]) is not list or not 1 <= len(m["groups"]) <= 16:
        raise ValueError("invalid group list")
    meter = Meter(m["numerator"], m["denominator"], tuple(m["groups"]))
    if type(value["bars"]) is not list or not 1 <= len(value["bars"]) <= 8:
        raise ValueError("invalid bar list")
    bars = []
    for bar in value["bars"]:
        if type(bar) is not list or not 1 <= len(bar) <= 64:
            raise ValueError("invalid event list")
        events = []
        for e in bar:
            _keys(e, ("duration", "pitch", "tie_in", "tie_out"), "event")
            d = e["duration"]
            if type(d) is not list or len(d) != 2:
                raise ValueError("duration must be a reduced integer pair")
            _integer(d[0], 1, 64, "duration numerator")
            _integer(d[1], 1, 64, "duration denominator")
            if gcd(*d) != 1:
                raise ValueError("duration must be reduced")
            events.append(Event(Fraction(*d), e["pitch"], e["tie_in"], e["tie_out"]))
        bars.append(tuple(events))
    return Score(meter, tuple(bars), value["tempo_qpm"])
