"""Unit tests and randomised equivalence tests.

Run with:  python -m unittest discover -s tests -v   (from the project folder)
"""
import itertools
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cost import cost
from engine import apply_ordering, apply_rule, apply_fixpoint, clear_caches, explore
from generator import generate, generate_fuzz
from interpreter import equivalent, random_env, run, run_safe
from rules import N_RULES, RULE_NAMES
from tac import is_temp, parse, to_text, variables

CF, AS, SR, CP, RA, CSE, RS, DCE = range(8)


def rule_out(src, r):
    return to_text(apply_rule(parse(src), r)).strip().replace("    ", "")


class ParserAndCost(unittest.TestCase):
    def test_roundtrip(self):
        src = "x = 5\nt1 = a * -4\nL1:\nif a <= b goto L1\ngoto L1\nprint t1\n"
        code = parse(src)
        self.assertEqual(parse(to_text(code)), code)
        self.assertEqual(len(code), 6)

    def test_bad_line(self):
        with self.assertRaises(ValueError):
            parse("x = = 3")

    def test_cost_values(self):
        self.assertEqual(cost(parse("x = 1")), 1)
        self.assertEqual(cost(parse("x = a * b")), 4)
        self.assertEqual(cost(parse("x = a << 2")), 1)
        self.assertEqual(cost(parse("x = a / b")), 8)
        self.assertEqual(cost(parse("L1:\nprint x\ngoto L1")), 2)

    def test_interpreter(self):
        code = parse("x = a + 3\nif x > 5 goto L\nx = 0\nL:\nprint x\n")
        self.assertEqual(run(code, {"a": 10}), [13])
        self.assertEqual(run(code, {"a": 0}), [0])


class RuleBehaviour(unittest.TestCase):
    def test_constant_folding(self):
        self.assertEqual(rule_out("t1 = 2 + 3\nt2 = 7 / 2", CF), "t1 = 5\nt2 = 3")
        self.assertEqual(rule_out("t1 = 4 / 0", CF), "t1 = 4 / 0")           # never fold a fault
        self.assertEqual(rule_out("if 1 < 2 goto L\nL:", CF), "goto L\nL:")
        self.assertEqual(rule_out("if 3 < 2 goto L\nL:", CF), "L:")

    def test_algebraic(self):
        self.assertEqual(rule_out("x = a * 1\ny = 0 + a\nz = a - a\nw = a * 0", AS),
                         "x = a\ny = a\nz = 0\nw = 0")
        self.assertEqual(rule_out("x = a / 1\ny = a << 0\nz = a % 1", AS), "x = a\ny = a\nz = 0")

    def test_strength_reduction(self):
        self.assertEqual(rule_out("x = a * 8\ny = 4 * a\nz = a / 16\nw = a % 8", SR),
                         "x = a << 3\ny = a << 2\nz = a >> 4\nw = a & 7")
        self.assertEqual(rule_out("x = a * 6", SR), "x = a * 6")

    def test_copy_propagation(self):
        self.assertEqual(rule_out("y = a\nz = y + 1\nprint y", CP), "y = a\nz = a + 1\nprint a")
        self.assertEqual(rule_out("k = 4\nz = a * k", CP), "k = 4\nz = a * 4")
        # the copy source changes, so propagation must stop
        self.assertEqual(rule_out("y = a\na = 5\nz = y + 1", CP), "y = a\na = 5\nz = y + 1")
        # never past a label
        self.assertEqual(rule_out("y = a\nL:\nz = y", CP), "y = a\nL:\nz = y")

    def test_reassociation(self):
        self.assertEqual(rule_out("t = x + 3\nu = t + 5", RA), "t = x + 3\nu = x + 8")
        self.assertEqual(rule_out("t = x * 2\nu = t * 4", RA), "t = x * 2\nu = x * 8")
        self.assertEqual(rule_out("t = x + 3\nu = t - 3", RA), "t = x + 3\nu = x")
        self.assertEqual(rule_out("t = x + 3\nx = 9\nu = t + 5", RA), "t = x + 3\nx = 9\nu = t + 5")

    def test_cse(self):
        self.assertEqual(rule_out("t1 = a + b\nt2 = b + a", CSE), "t1 = a + b\nt2 = t1")
        self.assertEqual(rule_out("t1 = a + b\na = 1\nt2 = a + b", CSE), "t1 = a + b\na = 1\nt2 = a + b")

    def test_redundant_statements(self):
        self.assertEqual(rule_out("x = x\ny = 1", RS), "y = 1")
        self.assertEqual(rule_out("a = b\nb = a", RS), "a = b")
        self.assertEqual(rule_out("goto L\nL:", RS), "L:")
        self.assertEqual(rule_out("t1 = a + b\nt1 = a + b", RS), "t1 = a + b")

    def test_dce(self):
        self.assertEqual(rule_out("t1 = a * b\nx = a + 1\nprint x", DCE), "x = a + 1\nprint x")
        self.assertEqual(rule_out("x = a + 1\nx = b + 2\nprint x", DCE), "x = b + 2\nprint x")
        # an ordinary variable that is still live at the end of the segment is kept
        self.assertEqual(rule_out("x = a + 1\ngoto L\nL:\nprint x", DCE), "x = a + 1\ngoto L\nL:\nprint x")
        self.assertEqual(rule_out("goto L\nx = 1\ny = 2\nL:\nprint x", DCE), "goto L\nL:\nprint x")


class PhaseOrdering(unittest.TestCase):
    """Order of rules changes the result: the reason this project exists."""

    def test_reassoc_before_strength_reduction(self):
        code = parse("t1 = a * 2\nt2 = t1 * 4\nx = t2 + 0\nprint x\n")
        sr_first = apply_ordering(code, (SR, RA, AS, DCE))
        ra_first = apply_ordering(code, (RA, SR, AS, DCE))
        self.assertGreater(cost(sr_first), cost(ra_first))
        self.assertTrue(equivalent(code, sr_first))
        self.assertTrue(equivalent(code, ra_first))

    def test_propagate_before_fold(self):
        code = parse("t1 = 4\nx = t1 * 8\nprint x\n")
        good = apply_ordering(code, (CP, CF, SR, DCE))
        bad = apply_ordering(code, (CF, SR, DCE, CP))
        self.assertLess(cost(good), cost(bad))

    def test_explore_matches_brute_force(self):
        code = parse("t1 = a * 2\nt2 = t1 * 4\nt3 = 5\nx = t2 + t3\ny = a\nz = y + 1\nprint x\nprint z\n")
        clear_caches()
        best, order, worst = explore(code)
        costs = [cost(apply_ordering(code, p)) for p in itertools.permutations(range(N_RULES))]
        self.assertEqual(best, min(costs))
        self.assertEqual(worst, max(costs))
        self.assertEqual(cost(apply_ordering(code, order)), best)


class RandomEquivalence(unittest.TestCase):
    """Every rule, and random orderings, must preserve behaviour."""

    N = 250

    def test_generated_temporaries_are_segment_local(self):
        # Changing only the initial value of temporaries must not change the output.
        rng = random.Random(1)
        for seed in range(self.N):
            code = generate(seed)
            names = variables(code)
            env = random_env(names, rng)
            env2 = dict(env)
            for n in names:
                if is_temp(n):
                    env2[n] = env[n] + 7919
            self.assertEqual(run_safe(code, env), run_safe(code, env2), f"seed {seed}")

    def test_generated_programs_terminate_and_parse(self):
        for seed in range(self.N):
            code = generate(seed)
            self.assertEqual(parse(to_text(code)), code)
            status, _ = run_safe(code, random_env(variables(code), random.Random(seed)))
            self.assertEqual(status, "ok", f"seed {seed}")

    def test_each_rule_preserves_behaviour(self):
        for r in range(N_RULES):
            for seed in range(self.N):
                code = generate(seed)
                out = apply_rule(code, r)
                self.assertTrue(equivalent(code, out, seed=seed),
                                f"rule {RULE_NAMES[r]} broke seed {seed}")
                self.assertLessEqual(cost(out), cost(code), f"rule {RULE_NAMES[r]} raised cost, seed {seed}")

    def test_random_orderings_preserve_behaviour(self):
        rng = random.Random(7)
        for seed in range(self.N):
            code = generate(10_000 + seed)
            order = list(range(N_RULES))
            rng.shuffle(order)
            out = apply_ordering(code, order)
            self.assertTrue(equivalent(code, out, seed=seed), f"ordering {order} broke seed {seed}")
            fix = apply_fixpoint(code, order)
            self.assertTrue(equivalent(code, fix, seed=seed), f"fixpoint {order} broke seed {seed}")

    def test_fuzz_each_rule_preserves_behaviour(self):
        # dense programs over a tiny variable pool: definitions and uses collide constantly
        for r in range(N_RULES):
            for seed in range(1200):
                code = generate_fuzz(seed)
                out = apply_rule(code, r)
                self.assertTrue(equivalent(code, out, trials=8, seed=seed),
                                f"rule {RULE_NAMES[r]} broke fuzz seed {seed}")
                self.assertLessEqual(cost(out), cost(code))

    def test_fuzz_random_orderings_preserve_behaviour(self):
        rng = random.Random(11)
        for seed in range(1500):
            code = generate_fuzz(50_000 + seed)
            order = list(range(N_RULES))
            rng.shuffle(order)
            out = apply_ordering(code, order)
            self.assertTrue(equivalent(code, out, trials=8, seed=seed), f"{order} broke fuzz seed {seed}")

    def test_oracle_result_is_correct(self):
        for seed in range(40):
            code = generate(20_000 + seed)
            clear_caches()
            best, order, worst = explore(code)
            out = apply_ordering(code, order)
            self.assertEqual(cost(out), best)
            self.assertTrue(equivalent(code, out, seed=seed))
            self.assertLessEqual(best, worst)


if __name__ == "__main__":
    unittest.main()
