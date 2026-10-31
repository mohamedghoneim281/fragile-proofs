import argparse
import contextlib
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from fragile.cli import cmd_evaluate
from fragile.collect import collect_command
from fragile.demo import make_demo
from fragile.grid import load_grid
from fragile.live import confirm_command, find_witnesses, hunt_command
from fragile.report import status_text
from fragile.smt2 import Script
from fragile.solver import Z3Solver, find_z3, parse_z3_output, success
from fragile.store import Store
from fragile.variants import planned_variants, render_variant


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def collect_args(qdir, out, **kw):
    d = dict(queries=str(qdir), project_depth=1, manifest=None, per_project=0, max_queries=0, max_mb=20.0,
             families="rename,shuffle,reseed", k=6, timeout=10.0, seed=0, success="unsat", force=False,
             solver="mock", solver_bin="z3", mem_mb=0, rlimit=0, out=str(out), workers=1, pin=False,
             control_repeats=3, max_hours=0, dry_run=False, retry_errors=False)
    d.update(kw)
    return argparse.Namespace(**d)


class TestParse(unittest.TestCase):
    def test_parse_z3_output(self):
        self.assertEqual(parse_z3_output("unsat\n(error \"model is not available\")\n", 0, 1, 10), "unsat")
        self.assertEqual(parse_z3_output("sat\n", 0, 1, 10), "sat")
        self.assertEqual(parse_z3_output("unknown\n(:reason-unknown \"incomplete\")\n", 0, 1, 10), "unknown")
        self.assertEqual(parse_z3_output("timeout\n", 0, 10, 10), "timeout")
        self.assertEqual(parse_z3_output("", 1, 0.1, 10), "error")
        self.assertEqual(parse_z3_output('(error "line 1 column 2: unknown constant")\n', 1, 0.1, 10), "error")
        self.assertEqual(parse_z3_output("", 0, 9.9, 10), "timeout")
        # memory limit hit: Z3 reports it on stderr with exit code 101 (observed with Z3 4.12.2)
        self.assertEqual(parse_z3_output("", 101, 5.8, 10, stderr='(error "out of memory")\n'), "memout")
        self.assertEqual(parse_z3_output('(error "out of memory")\n', 101, 5.8, 10), "memout")
        self.assertEqual(parse_z3_output("unsat\n", 0, 1, 10, stderr='(error "out of memory")'), "unsat")
        self.assertEqual(parse_z3_output("", 1, 0.1, 10, stderr="some other failure"), "error")

    def test_success(self):
        self.assertTrue(success("unsat"))
        self.assertFalse(success("sat"))
        self.assertTrue(success("sat", "solved"))
        self.assertFalse(success("unknown", "solved"))


class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.q = cls.tmp / "q"
        make_demo(cls.q, n_projects=6, per_project=10, seed=1)
        cls.out = cls.tmp / "out"
        with quiet():
            collect_command(collect_args(cls.q, cls.out))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_collect_is_resumable_and_idempotent(self):
        n = len(Store(self.out).read())
        with quiet():
            collect_command(collect_args(self.q, self.out))
        self.assertEqual(len(Store(self.out).read()), n)

    def test_settings_lock(self):
        with self.assertRaises(SystemExit):
            with quiet():
                collect_command(collect_args(self.q, self.out, timeout=20.0))

    def test_grid_loads(self):
        g = load_grid(self.out)
        self.assertEqual(g.n, 60)
        self.assertEqual(len(g.proj_names), 6)
        self.assertTrue(all(s.shape == c.shape for s, c in zip(g.pool_succ, g.pool_cost)))
        self.assertEqual(g.report["variant_errors"], 0)

    def test_status_mentions_synthetic(self):
        self.assertIn("SYNTHETIC", status_text(self.out))

    def test_evaluate_writes_results(self):
        ns = argparse.Namespace(out=str(self.out), seeds=3, folds=3, tau=0.1, label_mode="independent", no_tune=False,
                                tune_grid="small", tune_seeds=1, ablations=True, boot=50, quick=False)
        with quiet():
            cmd_evaluate(ns)
        res = json.loads((self.out / "results" / "eval.json").read_text())
        for name in ("uniform", "random", "runtime", "adaptive", "adaptive-tuned", "adaptive-nopool"):
            y = res["strategies"][name]["curve_mean"]
            self.assertTrue(all(b >= a - 1e-12 for a, b in zip(y, y[1:])))
            self.assertLessEqual(max(y), 1.0 + 1e-9)
        self.assertTrue((self.out / "results" / "summary.md").exists())
        self.assertIn("SYNTHETIC", (self.out / "results" / "summary.md").read_text())

    def test_confirm_writes_evidence(self):
        wit = find_witnesses(Store(self.out), "unsat")
        self.assertGreater(len(wit), 0)
        ns = argparse.Namespace(out=str(self.out), repeats=3, max=3, only_hunt=False, solver=None, solver_bin=None)
        with quiet():
            confirm_command(ns)
        conf = json.loads((self.out / "confirmations.json").read_text())
        self.assertEqual(len(conf), min(3, len(wit)))

    def test_retry_errors_replaces_error_record(self):
        out = self.tmp / "retry"
        with quiet():
            collect_command(collect_args(self.q, out))
        store = Store(out)
        victim = next(r for r in store.read() if r["variant"].startswith("rename") and r["repeat"] == 0)
        good = dict(victim)
        err = dict(victim, outcome="error")
        store.append(err)  # a later 'error' line now shadows the good measurement
        store.close()
        self.assertEqual(Store(out).latest()[(victim["query"], victim["variant"], 0)]["outcome"], "error")
        self.assertIn("1 runs ended in 'error'", status_text(out))
        with quiet():  # without the flag nothing is re-run
            collect_command(collect_args(self.q, out))
        self.assertEqual(Store(out).latest()[(victim["query"], victim["variant"], 0)]["outcome"], "error")
        with quiet():  # with the flag exactly that run is measured again
            collect_command(collect_args(self.q, out, retry_errors=True))
        again = Store(out).latest()[(victim["query"], victim["variant"], 0)]
        self.assertEqual(again["outcome"], good["outcome"])
        self.assertEqual(load_grid(out).report["variant_errors"], 0)

    def test_hunt_live_with_mock(self):
        out = self.tmp / "hunt"
        ns = collect_args(self.q, out, strategy="adaptive", budget_hours=0.2)
        with quiet():
            hunt_command(ns)
        h = json.loads((out / "hunt.json").read_text())
        self.assertGreater(h["tests"], 0)
        for f in h["flagged"]:
            self.assertNotEqual(f["success_variant"], f["failure_variant"])


@unittest.skipUnless(find_z3(), "z3 not installed")
class TestRealZ3(unittest.TestCase):
    TINY = """(set-logic UFLIA)
(declare-fun f (Int) Int)
(declare-const c Int)
(assert (forall ((x Int)) (>= (f x) x)))
(assert (< (f c) c))
(assert (> c 0))
(check-sat)
(get-model)
"""

    def test_all_variants_agree_on_easy_query(self):
        z3 = Z3Solver(mem_mb=2048)
        s = Script(self.TINY)
        for vid in planned_variants(s, ("rename", "shuffle", "reseed", "mix"), 3):
            text, seed = render_variant(s, "tiny", vid, 0)
            r = z3.run(text, seed, 10)
            self.assertEqual(r.outcome, "unsat", (vid, r.detail))
            self.assertIsNotNone(r.cpu_s)


if __name__ == "__main__":
    unittest.main()
