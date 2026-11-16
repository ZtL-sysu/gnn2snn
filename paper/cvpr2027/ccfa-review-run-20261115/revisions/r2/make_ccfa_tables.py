"""Create review-round-1 derived tables from frozen experimental rows."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(parents=True, exist_ok=True)
RUNS = ROOT / "research/20261115_decision/runs"


def hard_rows() -> pd.DataFrame:
    frames = [
        pd.read_csv(RUNS / "intervention_v2/MNIST41/graphs.csv"),
        pd.read_csv(RUNS / "intervention_v2/CIFAR1041/graphs.csv"),
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame["message_improved"] = frame.changed_message_mse < frame.baseline_message_mse
    frame["margin_worse"] = frame.changed_margin < frame.baseline_margin
    frame["adverse"] = frame.message_improved & frame.margin_worse
    frame["prediction_changed"] = frame.baseline_pred != frame.changed_pred
    frame["agreement_lost"] = (
        (frame.baseline_pred == frame.teacher_class)
        & (frame.changed_pred != frame.teacher_class)
    )
    frame["agreement_gained"] = (
        (frame.baseline_pred != frame.teacher_class)
        & (frame.changed_pred == frame.teacher_class)
    )
    return frame


def hard_stratified(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (dataset, readout, selector, direction), group in frame.groupby(
        ["dataset", "T", "selector", "direction"], observed=True
    ):
        improved = group[group.message_improved]
        rows.append(
            {
                "dataset": dataset,
                "readout": readout,
                "selector": selector,
                "direction": direction,
                "attempts": len(group),
                "message_improved": len(improved),
                "adverse": int(improved.margin_worse.sum()),
                "adverse_rate_percent": 100 * improved.margin_worse.mean(),
                "prediction_flips": int((improved.prediction_changed).sum()),
                "agreement_lost": int((improved.agreement_lost).sum()),
                "agreement_gained": int((improved.agreement_gained).sum()),
                "median_margin_delta": float(
                    (improved.changed_margin - improved.baseline_margin).median()
                ),
            }
        )
    return pd.DataFrame(rows)


def hard_per_graph(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame[frame["T"] == 1].copy()
    frame["any_adverse"] = frame.adverse
    frame["any_flip"] = frame.message_improved & frame.prediction_changed
    per_graph = frame.groupby(["dataset", "index"], observed=True).agg(
        any_improved=("message_improved", "max"),
        any_adverse=("any_adverse", "max"),
        any_prediction_change=("any_flip", "max"),
    )
    return per_graph.groupby(level=0, observed=True).mean().reset_index()


def temporal_rows() -> pd.DataFrame:
    rows = []
    for dataset in ("MNIST", "CIFAR10"):
        for direction, run in (
            ("halve", "prefix_diagnostic"),
            ("double", "prefix_double"),
        ):
            root = RUNS / run / f"{dataset}41"
            base = np.load(root / "baseline_T2.npz")
            alt = np.load(root / "all_eligible_T2.npz")
            base_pred = base["logits"].argmax(1)
            alt_pred = alt["logits"].argmax(1)
            teacher_pred = base["teacher"].argmax(1)
            rows.append(
                {
                    "dataset": dataset,
                    "direction": direction,
                    "policy": "all_eligible",
                    "train_graphs": len(base_pred),
                    "eligible_first_site_coordinates": int(
                        base["eligible_by_channel"].sum()
                    ),
                    "local_T2_decode_max_error": float(
                        alt["first_decode_error_max"].max()
                    ),
                    "base_accuracy_percent": 100
                    * np.mean(base_pred == base["labels"]),
                    "changed_accuracy_percent": 100
                    * np.mean(alt_pred == base["labels"]),
                    "base_agreement_percent": 100
                    * np.mean(base_pred == teacher_pred),
                    "changed_agreement_percent": 100
                    * np.mean(alt_pred == teacher_pred),
                    "prediction_changes": int((base_pred != alt_pred).sum()),
                    "agreement_lost": int(
                        ((base_pred == teacher_pred) & (alt_pred != teacher_pred)).sum()
                    ),
                    "agreement_gained": int(
                        ((base_pred != teacher_pred) & (alt_pred == teacher_pred)).sum()
                    ),
                    "whole_network_events_base": int(base["events"].sum()),
                    "whole_network_events_changed": int(alt["events"].sum()),
                }
            )
    return pd.DataFrame(rows)


def write(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / name, index=False, float_format="%.8g")


def main() -> None:
    hard = hard_rows()
    write(hard_stratified(hard), "hard_intervention_stratified.csv")
    write(hard_per_graph(hard), "hard_intervention_per_graph.csv")
    write(temporal_rows(), "temporal_decision_effects.csv")


if __name__ == "__main__":
    main()
