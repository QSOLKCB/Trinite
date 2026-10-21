from cyclic_conformance_v1.reference import (CyclicElement, CyclicGroupConformanceError,
    to_sexagesimal, from_sexagesimal, embeds_exactly)
from fractions import Fraction
import math
# ============================================================================
# Regression and conformance suite
# ============================================================================
import itertools
import random
import unittest

SMALL = list(range(1, 25))            # exhaustive axiom sweep
LARGE = [60, 61, 100, 360]            # sampled associativity, exhaustive everything else


class TestGroupAxioms(unittest.TestCase):
    def _elements(self, n):
        return [CyclicElement(i, n) for i in range(n)]

    def test_identity_inverse_closure(self):
        for n in SMALL + LARGE:
            e = CyclicElement(0, n)
            for a in self._elements(n):
                self.assertEqual(a.add(e), a); self.assertEqual(e.add(a), a)
                self.assertEqual(a.add(a.neg()), e); self.assertEqual(a.neg().add(a), e)
                self.assertEqual(a.neg().neg(), a)
                for b in self._elements(n):
                    ab = a.add(b)
                    self.assertTrue(0 <= ab.k < n)
                    self.assertEqual(ab, b.add(a))                 # Z/n is abelian
                    self.assertEqual(ab.sub(b), a)

    def test_associativity_exhaustive_small(self):
        for n in SMALL:
            els = self._elements(n)
            for a, b, c in itertools.product(els, repeat=3):
                self.assertEqual(a.add(b).add(c), a.add(b.add(c)))

    def test_associativity_sampled_large(self):
        rng = random.Random(0)
        for n in LARGE:
            for _ in range(5000):
                a, b, c = (CyclicElement(rng.randrange(n), n) for _ in range(3))
                self.assertEqual(a.add(b).add(c), a.add(b.add(c)))

    def test_hashable_and_set_semantics(self):
        s = {CyclicElement(1, 60), CyclicElement(61, 60), CyclicElement(-59, 60)}
        self.assertEqual(len(s), 1)
        self.assertNotEqual(CyclicElement(0, 60), CyclicElement(0, 360))  # different groups


class TestRotation(unittest.TestCase):
    def test_wraparound_example(self):
        # 350 deg + 25 deg -> 15 deg, on a 360-tick circle
        a = CyclicElement.from_degrees(350, 360); b = CyclicElement.from_degrees(25, 360)
        self.assertEqual(a.add(b).degrees, 15)

    def test_inverse_rotation(self):
        a = CyclicElement(12, 60)
        self.assertEqual(a.inverse(), CyclicElement(48, 60))           # -12 mod 60
        self.assertEqual(a.neg(), a.inverse())
        self.assertEqual(a.add(a.inverse()), CyclicElement(0, 60))

    def test_invalid_tie_policy_rejected_even_without_a_tie(self):
        z = CyclicElement(0, 60)
        with self.assertRaises(ValueError):
            z.signed_displacement_to(CyclicElement(10, 60), half_turn_tie="bogus")
        with self.assertRaises(ValueError):
            CyclicElement(0, 61).signed_displacement_to(CyclicElement(10, 61), half_turn_tie="bogus")

    def test_shortest_signed_displacement(self):
        z = CyclicElement(0, 60)
        self.assertEqual(z.signed_displacement_to(CyclicElement(10, 60)), 10)
        self.assertEqual(z.signed_displacement_to(CyclicElement(50, 60)), -10)
        self.assertEqual(z.signed_displacement_to(CyclicElement(29, 60)), 29)
        self.assertEqual(z.signed_displacement_to(CyclicElement(31, 60)), -29)
        # displacement composes back to the target
        for n in (7, 60, 61):
            for i in range(n):
                for j in range(n):
                    a, b = CyclicElement(i, n), CyclicElement(j, n)
                    d = a.signed_displacement_to(b, half_turn_tie="positive")
                    self.assertEqual(a.add(CyclicElement(d, n)), b)
                    self.assertLessEqual(abs(d), n // 2)

    def test_half_turn_tie_rule_is_explicit(self):
        z, h = CyclicElement(0, 60), CyclicElement(30, 60)
        with self.assertRaises(CyclicGroupConformanceError):
            z.signed_displacement_to(h)
        self.assertEqual(z.signed_displacement_to(h, "positive"), 30)
        self.assertEqual(z.signed_displacement_to(h, "negative"), -30)
        # odd n has no tie
        for j in range(61):
            CyclicElement(0, 61).signed_displacement_to(CyclicElement(j, 61))


class TestElementOrder(unittest.TestCase):
    def test_formula_against_bruteforce(self):
        for n in SMALL + LARGE:
            for k in range(n):
                e = CyclicElement(k, n)
                self.assertEqual(e.order(), e.order_bruteforce(), (k, n))
                self.assertEqual(e.order(), n // math.gcd(n, k))

    def test_z60_has_small_orders_z61_does_not(self):
        orders60 = {CyclicElement(k, 60).order() for k in range(1, 60)}
        self.assertTrue({2, 3, 4, 5, 6}.issubset(orders60))
        self.assertEqual(CyclicElement(12, 60).order(), 5)   # one-fifth turn
        self.assertEqual(CyclicElement(15, 60).order(), 4)   # one-quarter turn
        self.assertEqual({CyclicElement(k, 61).order() for k in range(1, 61)}, {61})


class TestNotations(unittest.TestCase):
    def test_same_element_many_spellings(self):
        n = 60
        target = CyclicElement(45, n)
        self.assertEqual(CyclicElement(-15, n), target)
        self.assertEqual(CyclicElement(105, n), target)
        self.assertEqual(CyclicElement.from_turn(Fraction(3, 4), n), target)
        self.assertEqual(CyclicElement.from_degrees(270, n), target)
        self.assertEqual(CyclicElement.from_degrees(-90, n), target)
        self.assertEqual(CyclicElement.from_dms(270, 0, 0, n), target)
        self.assertEqual(target.dms(), (270, 0, Fraction(0)))

    def test_fractions_of_a_circle_in_degrees(self):
        # The correction from the discussion: halves..sixths of a FULL circle.
        expected = {2: 180, 3: 120, 4: 90, 5: 72, 6: 60}
        for q, deg in expected.items():
            self.assertEqual(CyclicElement.from_turn(Fraction(1, q), 360).degrees, deg)
        # 30, 20, 15, 12, 10 are divisions of a 60-unit container (minutes of one degree).
        for q, mins in {2: 30, 3: 20, 4: 15, 5: 12, 6: 10}.items():
            self.assertEqual(Fraction(60, q), mins)
            self.assertEqual(CyclicElement.from_dms(0, mins, 0, 21600).turn, Fraction(1, q) / 360)

    def test_dms_round_trip_on_arcminute_grid(self):
        n = 360 * 60  # one tick per arcminute
        for k in range(0, n, 97):
            e = CyclicElement(k, n)
            d, m, s = e.dms()
            self.assertEqual(CyclicElement.from_dms(d, m, s, n), e)

    def test_non_tick_values_are_rejected_not_rounded(self):
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement.from_turn(Fraction(1, 7), 60)
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement.from_degrees(1, 60)          # 1 degree is 1/6 tick on Z/60
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement.from_dms(0, 30, 0, 360)      # half a degree on a degree grid

    def test_input_contract_is_strict(self):
        bad_ticks = [12.9, 12.0, "12", True, Fraction(12, 1), None]
        for bad in bad_ticks:
            with self.assertRaises(CyclicGroupConformanceError, msg=repr(bad)):
                CyclicElement(bad, 60)
        for bad_n in [60.0, 7.5, True, "60", 0, -1]:
            with self.assertRaises(CyclicGroupConformanceError, msg=repr(bad_n)):
                CyclicElement(5, bad_n)
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement.from_turn(0.2, 60)           # float turn rejected; use Fraction(1, 5)
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement(12, 60).project_exact(60.0)


class TestImmutability(unittest.TestCase):
    def test_attributes_cannot_change_after_validation(self):
        e = CyclicElement(12, 60)
        for name, val in [("k", 13), ("n", 61), ("extra", 1)]:
            with self.assertRaises(AttributeError):
                setattr(e, name, val)
        with self.assertRaises(AttributeError):
            del e.k
        self.assertEqual(e, CyclicElement(12, 60))

    def test_cannot_reinitialise(self):
        e = CyclicElement(12, 60)
        h, s = hash(e), {CyclicElement(12, 60)}
        with self.assertRaises(AttributeError):
            e.__init__(13, 60)
        with self.assertRaises((AttributeError, CyclicGroupConformanceError)):
            e.__init__(1.5, 61)
        self.assertEqual((e.k, e.n, hash(e)), (12, 60, h))
        self.assertIn(e, s)

    def test_failed_construction_has_no_partial_state(self):
        # Allocate without running __init__, then run a failing __init__: nothing may be written.
        e = CyclicElement.__new__(CyclicElement)
        with self.assertRaises(CyclicGroupConformanceError):
            e.__init__(1.5, 60)            # good n, bad k
        self.assertFalse(hasattr(e, "n")); self.assertFalse(hasattr(e, "k"))


class TestSexagesimal(unittest.TestCase):
    def test_round_trip_and_known_values(self):
        self.assertEqual(to_sexagesimal(0), (0,))
        self.assertEqual(to_sexagesimal(59), (59,))
        self.assertEqual(to_sexagesimal(60), (1, 0))
        self.assertEqual(to_sexagesimal(360), (6, 0))
        self.assertEqual(to_sexagesimal(3661), (1, 1, 1))     # 1h 1m 1s in seconds
        for x in list(range(0, 5000)) + [60**4 - 1, 60**4, 123456789]:
            self.assertEqual(from_sexagesimal(to_sexagesimal(x)), x)

    def test_representation_does_not_change_the_group(self):
        # The same tick, written in base 60 and base 10, is the same element and has
        # the same order. Notation is not structure.
        for k in range(360):
            self.assertEqual(CyclicElement(from_sexagesimal(to_sexagesimal(k)), 360), CyclicElement(k, 360))
        # a tick count on Z/60 is a single base-60 digit; on Z/3600 it is two
        self.assertTrue(all(len(to_sexagesimal(k)) == 1 for k in range(60)))
        self.assertTrue(all(len(to_sexagesimal(k)) <= 2 for k in range(3600)))

    def test_rejects_bad_digits(self):
        for bad in [(60,), (-1,), (1.0,), (True,), ()]:
            with self.assertRaises(CyclicGroupConformanceError, msg=repr(bad)):
                from_sexagesimal(bad)
        with self.assertRaises(CyclicGroupConformanceError):
            to_sexagesimal(-5)
        with self.assertRaises(CyclicGroupConformanceError):
            to_sexagesimal(5.0)


class TestCrossGrid(unittest.TestCase):
    def test_embeds_exactly_validates_both_moduli(self):
        for n, m in [(60, 0), (0, 60), (60, -60), (60, 60.0), (True, 60), ("60", 60)]:
            with self.assertRaises(CyclicGroupConformanceError, msg=(n, m)):
                embeds_exactly(n, m)

    def test_from_turn_validates_modulus_first(self):
        for bad_n in [60.0, 0, -1, True, "60"]:
            with self.assertRaises(CyclicGroupConformanceError, msg=repr(bad_n)):
                CyclicElement.from_turn(Fraction(1, 5), bad_n)
            with self.assertRaises(CyclicGroupConformanceError, msg=repr(bad_n)):
                CyclicElement.from_degrees(72, bad_n)

    def test_embedding_when_n_divides_m(self):
        for n, m in [(60, 360), (4, 60), (12, 60), (60, 21600), (1, 5)]:
            self.assertTrue(embeds_exactly(n, m))
            for k in range(n):
                img = CyclicElement(k, n).project_exact(m)
                self.assertEqual(img, CyclicElement(k * (m // n), m))
                self.assertEqual(img.turn, Fraction(k, n))          # angle preserved
                self.assertEqual(img.order(), CyclicElement(k, n).order())
            # embedding is a homomorphism: f(a+b) == f(a)+f(b)
            for a in range(n):
                for b in range(n):
                    A, B = CyclicElement(a, n), CyclicElement(b, n)
                    self.assertEqual(A.add(B).project_exact(m), A.project_exact(m).add(B.project_exact(m)))

    def test_nexus_z4_maps_to_ticks_0_15_30_45(self):
        self.assertEqual([CyclicElement(k, 4).project_exact(60).k for k in range(4)], [0, 15, 30, 45])

    def test_reverse_direction_is_exact_only_for_selected_ticks(self):
        self.assertFalse(embeds_exactly(360, 60))
        self.assertEqual(CyclicElement(6, 360).project_exact(60), CyclicElement(1, 60))
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement(7, 360).project_exact(60)
        ok = [k for k in range(360) if (k * 60) % 360 == 0]
        self.assertEqual(ok, list(range(0, 360, 6)))

    def test_shared_fifth_turn_between_60_and_100(self):
        fifth = CyclicElement(12, 60)
        img = fifth.project_exact(100)
        self.assertEqual(img, CyclicElement(20, 100))
        self.assertEqual(img.order(), 5)
        self.assertEqual(img.project_exact(60), fifth)                 # round trip
        self.assertFalse(embeds_exactly(60, 100) or embeds_exactly(100, 60))
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement(1, 100).project_exact(60)                    # 1/100 turn is not a 60-tick

    def test_fifth_turn_rejected_on_prime_61(self):
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement(12, 60).project_exact(61)
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement.from_turn(Fraction(1, 5), 61)
        # the only tick shared by Z/60 and Z/61 is the identity
        self.assertEqual([k for k in range(60) if (k * 61) % 60 == 0], [0])

    def test_exactness_criterion_matches_fraction_arithmetic(self):
        for n in (60, 61, 100, 360):
            for m in (60, 61, 100, 360):
                for k in range(n):
                    exact_by_fraction = (Fraction(k, n) * m).denominator == 1
                    try:
                        img = CyclicElement(k, n).project_exact(m)
                        self.assertTrue(exact_by_fraction); self.assertEqual(img.turn, Fraction(k, n))
                    except CyclicGroupConformanceError:
                        self.assertFalse(exact_by_fraction)

    def test_cross_modulus_arithmetic_is_refused(self):
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement(1, 60).add(CyclicElement(1, 360))
        with self.assertRaises(CyclicGroupConformanceError):
            CyclicElement(1, 60).signed_displacement_to(CyclicElement(1, 61))


if __name__ == "__main__":
    unittest.main(verbosity=1)
