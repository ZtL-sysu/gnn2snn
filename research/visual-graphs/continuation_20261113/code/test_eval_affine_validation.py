"""Synthetic-only checks; these do not load CIFAR10 or a teacher checkpoint."""
from pathlib import Path
import importlib.util

import numpy as np

SCRIPT = Path(__file__).with_name("eval_affine_validation.py")
SPEC = importlib.util.spec_from_file_location("eval_affine_validation", SCRIPT)
EVAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EVAL)


def test_metric_row_perfect_prediction():
    labels = np.array([0, 1, 2])
    logits = np.array([[3., 0., -1.], [0., 2., -2.], [-1., 0., 4.]])
    row = EVAL.metric_row(logits, logits.copy(), labels)
    assert row["accuracy"] == 1.0
    assert row["teacher_accuracy"] == 1.0
    assert row["teacher_agreement"] == 1.0
    assert row["logit_mse"] == 0.0
    assert abs(row["teacher_to_output_kl"]) < 1e-12


def test_calibration_path_remapping(tmp_path):
    source = tmp_path / "source"
    data = tmp_path / "data"
    teacher = tmp_path / "teacher"
    for p in (source / "public_baselines", data / "CIFAR10" / "processed", teacher):
        p.mkdir(parents=True)
    assert EVAL.remap_bound_input(
        r"C:\old\takeover_20261112\public_baselines\gated_net.py", source, data, teacher
    ) == (source / "public_baselines" / "gated_net.py").resolve()
    assert EVAL.remap_bound_input(
        r"C:\old\data\pyg_bench\CIFAR10\processed\train_data.pt", source, data, teacher
    ) == (data / "CIFAR10" / "processed" / "train_data.pt").resolve()
    assert EVAL.remap_bound_input(
        r"C:\old\takeover\full_teacher_runs\run41\best.pt", source, data, teacher
    ) == (teacher / "best.pt").resolve()
