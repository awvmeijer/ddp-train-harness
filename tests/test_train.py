"""Tests that run without a GPU and without a launcher.

The distributed path itself is exercised by `make smoke`, which runs torchrun
with the gloo backend. These tests cover what is checkable in-process:
determinism, checkpoint round-trips, and that training actually reduces loss.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train import SyntheticRegression, build_model, parse_args, train  # noqa: E402


def args(**kw):
    argv = ["--backend", "none", "--epochs", "5", "--samples", "512", "--seed", "7"]
    for k, v in kw.items():
        argv.append(f"--{k.replace('_', '-')}")
        if v is not None:  # None means a bare flag like --resume
            argv.append(str(v))
    old = sys.argv
    sys.argv = ["train.py"] + argv
    try:
        return parse_args()
    finally:
        sys.argv = old


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_loss(self):
        a = train(args())
        b = train(args())
        self.assertEqual(a["final_loss"], b["final_loss"])

    def test_different_seed_different_loss(self):
        a = train(args())
        b = train(args(seed=8))
        self.assertNotEqual(a["final_loss"], b["final_loss"])

    def test_dataset_is_seed_deterministic(self):
        d1 = SyntheticRegression(64, 8, seed=3)
        d2 = SyntheticRegression(64, 8, seed=3)
        self.assertTrue((d1.x == d2.x).all())
        self.assertTrue((d1.y == d2.y).all())


class TestTraining(unittest.TestCase):
    def test_loss_decreases(self):
        with tempfile.TemporaryDirectory() as td:
            r = train(args(epochs=8, checkpoint_dir=td))
        # Relative, not absolute: convergence speed depends on lr and epochs,
        # but 8 epochs on a linear task must at least halve the loss.
        self.assertLess(r["final_loss"], 0.5 * r["first_loss"])

    def test_model_shapes(self):
        m = build_model(32)
        out = m(SyntheticRegression(4, 32, seed=1).x)
        self.assertEqual(tuple(out.shape), (4, 1))


class TestCheckpointing(unittest.TestCase):
    def test_resume_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            train(args(epochs=4, checkpoint_dir=td, checkpoint_every=2))
            # a second run with --resume starts from epoch 4's checkpoint
            resumed = train(args(epochs=6, checkpoint_dir=td, checkpoint_every=2, resume=None))
            self.assertLessEqual(resumed["epochs"], 6)
            self.assertTrue((Path(td) / "latest.pt").exists())

    def test_no_tmp_file_left_behind(self):
        with tempfile.TemporaryDirectory() as td:
            train(args(epochs=2, checkpoint_dir=td, checkpoint_every=1))
            self.assertEqual(list(Path(td).glob("*.tmp")), [])


class TestDistributedSmoke(unittest.TestCase):
    """Two-process gloo run through torchrun, the same thing CI does."""

    def test_torchrun_gloo_two_procs(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as td:
            result_file = Path(td) / "result.json"
            proc = subprocess.run(
                [
                    sys.executable, "-m", "torch.distributed.run",
                    "--nproc_per_node", "2", "--master_port", "29511",
                    str(root / "src" / "train.py"),
                    "--backend", "gloo", "--epochs", "3", "--samples", "256",
                    "--checkpoint-dir", td, "--result-file", str(result_file),
                ],
                capture_output=True, text=True, timeout=180,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
            self.assertTrue(result_file.exists())
            import json
            r = json.loads(result_file.read_text())
            self.assertEqual(r["world_size"], 2)
            self.assertEqual(r["backend"], "gloo")


if __name__ == "__main__":
    unittest.main()
