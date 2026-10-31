import csv
import gzip
import shutil
import tempfile
import unittest
from pathlib import Path

from fragile.mariposa import classify, prepare_mariposa
from fragile.solver import find_z3

Q = "(declare-const c Int)\n(assert (> c 0))\n(assert (< c 5))\n(check-sat)\n"


def write(root, rel, text=Q):
    p = Path(root) / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)
    return p


class TestPrepare(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rows(self, out):
        with open(Path(out) / "manifest.csv", newline="") as fh:
            return list(csv.DictReader(fh))

    def test_group_then_project_layout(self):
        for g in ("unstable_core", "stable_core"):
            for pr in ("D_KOMODO", "FS_VWASM", "D_LVBKV", "FS_DICE"):
                for i in range(2):
                    write(self.tmp / "b", f"{g}/{pr}/q{i}.smt2", Q + f"; {g}{pr}{i}\n")
        prepare_mariposa(self.tmp / "b", self.tmp / "out", log=lambda *a: None)
        rows = self.rows(self.tmp / "out")
        self.assertEqual(len(rows), 16)
        self.assertEqual({r["project"] for r in rows}, {"d_komodo", "fs_vwasm", "d_lvbkv", "fs_dice"})
        self.assertEqual({r["group"] for r in rows}, {"unstable_core", "stable_core"})

    def test_project_then_group_layout(self):
        for pr in ("komodo_d", "veribetrkv_l"):
            for g in ("unstable_ext", "stable_ext"):
                write(self.tmp / "b", f"{pr}/{g}/a.smt2", Q + f"; {pr}{g}\n")
        prepare_mariposa(self.tmp / "b", self.tmp / "out", log=lambda *a: None)
        rows = self.rows(self.tmp / "out")
        self.assertEqual({(r["project"], r["group"]) for r in rows},
                         {("komodo_d", "unstable_ext"), ("komodo_d", "stable_ext"),
                          ("veribetrkv_l", "unstable_ext"), ("veribetrkv_l", "stable_ext")})

    def test_flat_layout_uses_file_name(self):
        write(self.tmp / "b", "unstable_core/komodo_d__x.smt2", Q + "; 1\n")
        self.assertEqual(classify(Path("unstable_core/komodo_d__x.smt2")), ("komodo_d__x", "unstable_core"))

    def test_duplicates_dropped_core_preferred(self):
        write(self.tmp / "b", "unstable_ext/komodo_d/a.smt2")
        write(self.tmp / "b", "unstable_core/komodo_d/a.smt2")
        prepare_mariposa(self.tmp / "b", self.tmp / "out", log=lambda *a: None)
        rows = self.rows(self.tmp / "out")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["group"], "unstable_core")

    def test_gz_files(self):
        p = self.tmp / "b" / "stable_core" / "dice_f"
        p.mkdir(parents=True)
        with gzip.open(p / "q.smt2.gz", "wb") as fh:
            fh.write(Q.encode())
        prepare_mariposa(self.tmp / "b", self.tmp / "out", log=lambda *a: None)
        self.assertEqual(len(self.rows(self.tmp / "out")), 1)

    def test_empty_folder_exits_with_hint(self):
        (self.tmp / "e").mkdir()
        with self.assertRaises(SystemExit):
            prepare_mariposa(self.tmp / "e", self.tmp / "out", log=lambda *a: None)


class TestFindZ3(unittest.TestCase):
    def test_explicit_path(self):
        f = Path(tempfile.mkdtemp()) / "z3"
        f.write_text("#!/bin/sh\n")
        self.assertEqual(find_z3(str(f)), str(f))

    def test_missing(self):
        self.assertIsNone(find_z3("definitely-not-a-solver-xyz"))


if __name__ == "__main__":
    unittest.main()
