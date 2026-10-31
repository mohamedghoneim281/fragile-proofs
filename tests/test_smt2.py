import random
import unittest

from fragile.smt2 import Script, canon, declared_name, render, rename_symbols, split_script
from fragile.variants import planned_variants, render_variant

SAMPLE = """; header comment with (parens) and "quotes"
(set-info :source |a quoted ; source|)
(set-logic UFLIA)
(declare-fun f (Int) Int)
(declare-fun |weird name| (Int) Int)
(declare-const c Int)
(define-fun g ((x Int)) Int (+ x 1)) ; trailing comment
(assert (forall ((x Int)) (>= (f x) x)))
(assert (> (|weird name| c) 0))
; between asserts
(assert (< (f c) c))
(push 1)
(assert (= (g c) "f c ""c"" f"))
(assert (> c 0))
(pop 1)
(assert (> c 1))
(check-sat)
(get-model)
"""


class TestSplit(unittest.TestCase):
    def test_lossless(self):
        self.assertEqual(render(split_script(SAMPLE)), SAMPLE)

    def test_names(self):
        names = [c.name for c in split_script(SAMPLE) if c.is_cmd]
        self.assertEqual(names[:3], ["set-info", "set-logic", "declare-fun"])
        self.assertEqual(names.count("assert"), 6)
        self.assertEqual(names.count("check-sat"), 1)

    def test_unbalanced(self):
        with self.assertRaises(ValueError):
            split_script("(assert (> 1 0)")

    def test_declared(self):
        s = Script(SAMPLE)
        self.assertEqual(s.declared, ["f", "weird name", "c", "g"])
        self.assertEqual(declared_name("(declare-fun   f (Int) Int)"), "f")
        self.assertEqual(canon("|abc|"), "abc")


class TestRename(unittest.TestCase):
    def test_roundtrip(self):
        s = Script(SAMPLE)
        mapping = s.rename_mapping(random.Random(1))
        renamed = rename_symbols(s.text, mapping)
        self.assertNotEqual(renamed, s.text)
        inverse = {v: k for k, v in mapping.items()}
        # rename back; |weird name| comes back as the bare name 'weird name' in quotes-free form, so
        # compare on a version of the original with that symbol normalised
        back = rename_symbols(renamed, inverse)
        norm = SAMPLE.replace("|weird name|", "weird name")
        self.assertEqual(back, norm)

    def test_strings_and_comments_untouched(self):
        s = Script(SAMPLE)
        renamed = rename_symbols(s.text, s.rename_mapping(random.Random(2)))
        self.assertIn('"f c ""c"" f"', renamed)
        self.assertIn("; trailing comment", renamed)
        self.assertIn("|a quoted ; source|", renamed)
        self.assertIn("; header comment with (parens)", renamed)

    def test_injective_and_fresh(self):
        s = Script(SAMPLE)
        m = s.rename_mapping(random.Random(3))
        self.assertEqual(len(set(m.values())), len(m))
        self.assertFalse(set(m.values()) & s.atoms)

    def test_command_structure_unchanged(self):
        s = Script(SAMPLE)
        renamed = Script(rename_symbols(s.text, s.rename_mapping(random.Random(4))))
        self.assertEqual([c.name for c in s.chunks if c.is_cmd], [c.name for c in renamed.chunks if c.is_cmd])


class TestShuffle(unittest.TestCase):
    def test_runs(self):
        s = Script(SAMPLE)
        runs = s.assert_runs()
        # asserts: 3 consecutive (separated only by a comment), 2 inside push/pop, 1 alone after pop
        self.assertEqual([len(r) for r in runs], [3, 2])

    def test_never_crosses_boundaries(self):
        s = Script(SAMPLE)
        for seed in range(30):
            sh = Script(s.shuffled_text(random.Random(seed)))
            a = [c for c in s.chunks if c.is_cmd]
            b = [c for c in sh.chunks if c.is_cmd]
            self.assertEqual([c.name for c in a], [c.name for c in b])
            # non-assert commands stay in place
            for x, y in zip(a, b):
                if x.name != "assert":
                    self.assertEqual(x.text, y.text)
            # the multiset of assert texts is preserved within each run
            for run in s.assert_runs():
                self.assertEqual(sorted(s.chunks[i].text for i in run), sorted(sh.chunks[i].text for i in run))
            # the lone assert after pop never moves
            self.assertEqual(a[-3].text, b[-3].text) if a[-3].name == "assert" else None

    def test_actually_changes_order(self):
        s = Script(SAMPLE)
        outs = {s.shuffled_text(random.Random(i)) for i in range(20)}
        self.assertGreater(len(outs), 1)


class TestVariants(unittest.TestCase):
    def test_deterministic(self):
        s = Script(SAMPLE)
        for vid in planned_variants(s, ("rename", "shuffle", "reseed", "mix"), 3):
            self.assertEqual(render_variant(s, "q", vid, 7), render_variant(Script(SAMPLE), "q", vid, 7))

    def test_reseed_changes_only_seed(self):
        s = Script(SAMPLE)
        text, seed = render_variant(s, "q", "reseed:0", 0)
        self.assertEqual(text, SAMPLE)
        self.assertIsNotNone(seed)

    def test_distinct_variants_differ(self):
        s = Script(SAMPLE)
        texts = {render_variant(s, "q", f"rename:{i}", 0)[0] for i in range(10)}
        self.assertEqual(len(texts), 10)

    def test_infeasible_family_not_planned(self):
        s = Script("(set-logic QF_LIA)\n(assert (> 1 0))\n(check-sat)\n")
        v = planned_variants(s, ("rename", "shuffle", "reseed"), 2)
        self.assertEqual(v, ["orig:0", "reseed:0", "reseed:1"])


if __name__ == "__main__":
    unittest.main()
