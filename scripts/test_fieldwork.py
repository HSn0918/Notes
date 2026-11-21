"""Learner-facing invariants for the offline AI Infra experiments."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLACEMENT = ROOT / "cloud-native/kubernetes/scheduling/placement-lab"
TRAINING = ROOT / "ai/training"


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FieldworkTest(unittest.TestCase):
    def test_shape_and_domain_have_independent_effects(self):
        lab = load_module(PLACEMENT / "placement.py")
        self.assertEqual((64, 1, 1, 1), lab.capacity(lab.SNAPSHOTS["packed"], 8))
        self.assertEqual((64, 1, 0, 0), lab.capacity(lab.SNAPSHOTS["fragmented"], 8))
        self.assertEqual((64, 1, 1, 0), lab.capacity(lab.SNAPSHOTS["split"], 8))
        self.assertEqual((1500, 23, 15, 15), lab.capacity(lab.SNAPSHOTS["fleet"], 8))
        self.assertEqual((64, 2, 2, 2), lab.capacity(lab.SNAPSHOTS["fragmented"], 4))

    def test_backfill_protects_head_and_conserves_capacity(self):
        queue = load_module(PLACEMENT / "queue_replay.py")
        fifo = queue.replay(queue.JOBS, "fifo")
        safe = queue.replay(queue.JOBS, "backfill")
        greedy = queue.replay(queue.JOBS, "greedy")
        self.assertEqual(6, fifo["large"])
        self.assertEqual(fifo["large"], safe["large"])
        self.assertEqual(2, safe["short"])
        self.assertGreater(greedy["large"], safe["large"])
        for starts in (fifo, safe, greedy):
            for j in queue.JOBS:
                self.assertGreaterEqual(starts[j.name], j.arrival)
            for t in starts.values():
                used = sum(j.nodes for j in queue.JOBS if starts[j.name] <= t < starts[j.name] + j.duration)
                self.assertLessEqual(used, queue.NODES)

    def test_restart_preserves_progress_not_just_weights(self):
        lab = load_module(TRAINING / "checkpoint-replay/replay.py")
        expected, sequence = lab.train(lab.State(), lab.TOTAL_STEPS)
        for cut in range(1, lab.TOTAL_STEPS):
            state, prefix = lab.train(lab.State(), cut)
            result, suffix = lab.train(state, lab.TOTAL_STEPS - cut)
            self.assertEqual(expected, result)
            self.assertEqual(sequence, prefix + suffix)
            _, wrong = lab.train(lab.replace(state, cursor=0), lab.TOTAL_STEPS - cut)
            self.assertEqual(cut % len(lab.SAMPLES) == 0, suffix == wrong)
        demo = subprocess.check_output([sys.executable, str(TRAINING / "checkpoint-replay/replay.py")], text=True)
        self.assertIn("full: next_sample=1 same_order=True weight_error=0\n", demo)
        momentum = next(line for line in demo.splitlines() if line.startswith("omit_momentum:"))
        self.assertIn("same_order=True", momentum)
        self.assertGreater(float(momentum.split("weight_error=")[1]), 0.2)

    def test_even_and_uneven_token_splits(self):
        for cut in (1, 2, 3):
            cmd = [sys.executable, str(TRAINING / "sharded-gradient/sharded_gradient.py"), "--left-tokens", str(cut)]
            self.assertIn("SELF-CHECK: PASS", subprocess.check_output(cmd, text=True))
