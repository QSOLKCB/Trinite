"""Exact cyclic-angle conformance module (Z/n additive group on an n-tick circle).

Standalone exact reference adapted from a user-supplied AI-assisted sketch.
It does not import or assume anything about Trinite's current state model.
See docs/CYCLIC-CONFORMANCE.md for contracts, provenance and limitations.

Design rules (from the design discussion):
  * Exact integer/rational arithmetic only. Non-integers are rejected, never truncated.
  * Changing n changes the group, not just the notation. Cross-grid mapping is
    allowed only where it is exact; otherwise it raises.
  * The inverse of the additive group is written -k (mod n), not k^-1.
  * A 60-tick circle is a chosen discretization, not a privileged geometry.
"""
from __future__ import annotations

import math
from fractions import Fraction
from typing import Union

Rational = Union[int, Fraction]


class CyclicGroupConformanceError(ValueError):
    """Raised on a non-exact value, a non-exact projection, or a cross-modulus operation."""


def _strict_int(x, what: str) -> int:
    # bool is a subclass of int in Python; reject it explicitly.
    if isinstance(x, bool) or not isinstance(x, int):
        raise CyclicGroupConformanceError(f"{what} must be an int, got {type(x).__name__} {x!r}")
    return x


def _strict_modulus(n, what: str = "modulus n") -> int:
    n = _strict_int(n, what)
    if n <= 0:
        raise CyclicGroupConformanceError(f"{what} must be a positive integer, got {n}")
    return n


def _exact_fraction(x, what: str) -> Fraction:
    """Accept int or Fraction only. Floats and strings are rejected: a float is a
    binary approximation of the author's intent, and we never round silently."""
    if isinstance(x, bool) or not isinstance(x, (int, Fraction)):
        raise CyclicGroupConformanceError(f"{what} must be an int or Fraction, got {type(x).__name__} {x!r}")
    return Fraction(x)


class CyclicElement:
    """Tick k on an n-tick circle, i.e. the element k + nZ of Z/nZ."""

    __slots__ = ("k", "n")

    def __init__(self, k: int, n: int):
        # Validate both arguments before writing anything, so a failure cannot
        # leave a partly-built object, and refuse to run on an already-built one.
        n = _strict_modulus(n)
        k = _strict_int(k, "tick k") % n  # canonical representative in [0, n)
        if hasattr(self, "n"):
            raise AttributeError(f"{type(self).__name__} is immutable; cannot reinitialise")
        object.__setattr__(self, "n", n)
        object.__setattr__(self, "k", k)

    def __setattr__(self, name, value):
        raise AttributeError(f"{type(self).__name__} is immutable")

    def __delattr__(self, name):
        raise AttributeError(f"{type(self).__name__} is immutable")

    # ---- notations for the same element -----------------------------------
    @classmethod
    def from_turn(cls, turn: Rational, n: int) -> "CyclicElement":
        """Rational fraction of a full turn (e.g. Fraction(1, 5)). Must land on a tick."""
        n = _strict_modulus(n)
        t = _exact_fraction(turn, "turn")
        ticks = t * n
        if ticks.denominator != 1:
            raise CyclicGroupConformanceError(f"{t} of a turn is not a tick on Z/{n}")
        return cls(int(ticks), n)

    @classmethod
    def from_degrees(cls, deg: Rational, n: int) -> "CyclicElement":
        return cls.from_turn(_exact_fraction(deg, "degrees") / 360, n)

    @classmethod
    def from_dms(cls, d: int, m: int, s: Rational, n: int) -> "CyclicElement":
        """Degrees, minutes, seconds (sexagesimal sub-units). Sign is carried by d only;
        m and s must be non-negative, as in 'minus 12 deg 30 min'."""
        _strict_int(d, "degrees"); _strict_int(m, "minutes")
        s = _exact_fraction(s, "seconds")
        if m < 0 or s < 0:
            raise CyclicGroupConformanceError("minutes and seconds must be non-negative; sign goes on degrees")
        mag = Fraction(abs(d)) + Fraction(m, 60) + s / 3600
        return cls.from_degrees(-mag if d < 0 else mag, n)

    @property
    def turn(self) -> Fraction:
        return Fraction(self.k, self.n)

    @property
    def degrees(self) -> Fraction:
        return self.turn * 360

    def dms(self) -> tuple[int, int, Fraction]:
        deg = self.degrees
        d = int(deg)  # deg >= 0 since k in [0, n)
        rem = (deg - d) * 60
        m = int(rem)
        s = (rem - m) * 60
        return d, m, s

    # ---- group structure -------------------------------------------------
    def __repr__(self) -> str:
        return f"CyclicElement({self.k}, n={self.n})"

    def __eq__(self, other) -> bool:
        return isinstance(other, CyclicElement) and self.n == other.n and self.k == other.k

    def __hash__(self) -> int:
        return hash((self.k, self.n))

    def _same_group(self, other: "CyclicElement") -> None:
        if not isinstance(other, CyclicElement) or self.n != other.n:
            raise CyclicGroupConformanceError(f"Cross-modulus operation: {self!r} vs {other!r}")

    def add(self, other: "CyclicElement") -> "CyclicElement":
        """Rotation composition: R_a then R_b is R_(a+b)."""
        self._same_group(other)
        return CyclicElement(self.k + other.k, self.n)

    def inverse(self) -> "CyclicElement":
        """Additive inverse -k (mod n), i.e. the inverse rotation. (Not a multiplicative k^-1.)"""
        return CyclicElement(-self.k, self.n)

    neg = inverse

    def sub(self, other: "CyclicElement") -> "CyclicElement":
        self._same_group(other)
        return CyclicElement(self.k - other.k, self.n)

    def order(self) -> int:
        """Smallest r >= 1 with r*k == 0 (mod n); equals n / gcd(n, k)."""
        return self.n // math.gcd(self.n, self.k)

    def order_bruteforce(self) -> int:
        """Diagnostic repeated addition; runtime is linear in the element order.
        The bounded conformance checker uses a separate integer-counter oracle.
        """
        acc, r = self, 1
        while acc.k != 0:
            acc, r = acc.add(self), r + 1
        return r

    def signed_displacement_to(self, other: "CyclicElement", half_turn_tie: str = "reject") -> int:
        """Shortest signed tick displacement d with self + d == other, abs(d) <= n/2 (either endpoint is allowed by its tie policy).
        When n is even and the displacement is exactly a half turn, +n/2 and -n/2 are
        both shortest. The tie rule must be chosen explicitly:
          'positive' -> +n/2, 'negative' -> -n/2, 'reject' -> raise."""
        if half_turn_tie not in ("positive", "negative", "reject"):
            raise ValueError(f"unknown half_turn_tie {half_turn_tie!r}; use 'positive', 'negative' or 'reject'")
        self._same_group(other)
        d = (other.k - self.k) % self.n
        if 2 * d < self.n:
            return d
        if 2 * d > self.n:
            return d - self.n
        if half_turn_tie == "positive":
            return d
        if half_turn_tie == "negative":
            return -d
        raise CyclicGroupConformanceError(f"half-turn tie on Z/{self.n}: choose a tie rule")

    # ---- cross-grid mapping -------------------------------------------------
    def project_exact(self, m: int) -> "CyclicElement":
        """Map tick k on Z/n to Z/m preserving the angle k/n == k'/m.
        Exact iff k*m is divisible by n. Anything else is rejected outright."""
        m = _strict_modulus(m, "target modulus m")
        num = self.k * m
        if num % self.n != 0:
            raise CyclicGroupConformanceError(
                f"Non-exact projection: tick {self.k} on Z/{self.n} is {self.turn} of a turn, "
                f"which is not a tick on Z/{m}.")
        return CyclicElement(num // self.n, m)


def to_sexagesimal(x: int) -> tuple[int, ...]:
    """Base-60 digits of a non-negative int, most significant first. 0 -> (0,)."""
    x = _strict_int(x, "value")
    if x < 0:
        raise CyclicGroupConformanceError("sexagesimal encoding is defined here for non-negative ints")
    digits = []
    while True:
        x, d = divmod(x, 60)
        digits.append(d)
        if x == 0:
            return tuple(reversed(digits))


def from_sexagesimal(digits) -> int:
    """Decode a finite list/tuple of base-60 digits; leading zeros are allowed.
    Re-encoding produces the unique canonical spelling. This is not a bijection
    on noncanonical input sequences. Every digit must be an int in [0, 60).
    """
    if not isinstance(digits, (tuple, list)):
        raise CyclicGroupConformanceError("digits must be a finite list or tuple")
    digits = tuple(digits)
    if not digits:
        raise CyclicGroupConformanceError("empty digit sequence")
    x = 0
    for d in digits:
        _strict_int(d, "digit")
        if not 0 <= d < 60:
            raise CyclicGroupConformanceError(f"digit {d} out of range [0, 60)")
        x = x * 60 + d
    return x


def embeds_exactly(n: int, m: int) -> bool:
    """Every tick of Z/n maps exactly into Z/m iff n divides m. Both must be valid moduli."""
    n = _strict_modulus(n, "modulus n"); m = _strict_modulus(m, "modulus m")
    return m % n == 0
