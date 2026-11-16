"""Verify manuscript integrity and recompute all R2 headline numbers."""
from pathlib import Path
import hashlib
import json
import re

import fitz
import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[5]
SRC = RUN / "source"
DATA = RUN / "data"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


checks: dict[str, object] = {}

for name in ("main", "supplement"):
    pdf = fitz.open(SRC / f"{name}.pdf")
    text = "\n".join(page.get_text() for page in pdf)
    checks[f"{name}_pages"] = len(pdf)
    checks[f"{name}_anonymous"] = not any(
        token in text for token in ("ZtL", "xhf", "github.com", "C:\\", "Institution1")
    )
    checks[f"{name}_resolved"] = not any(
        token in text for token in ("??", "[?]", "TBD", "undefined citation")
    )
    checks[f"{name}_fonts_embedded"] = all(
        " yes " in line
        for line in (
            __import__("subprocess")
            .run(["pdffonts", str(SRC / f"{name}.pdf")], capture_output=True, text=True)
            .stdout.splitlines()[2:]
        )
        if line.strip()
    )
    log = (SRC / f"{name}.log").read_text(errors="replace")
    checks[f"{name}_clean_log"] = not re.search(
        r"Overfull|LaTeX Error|undefined", log, re.I
    )

main_text = "\n".join(page.get_text() for page in fitz.open(SRC / "main.pdf"))
checks["llm_before_references"] = main_text.index("LLM Usage Statement") < main_text.index(
    "References\n"
)

bib_keys = set(re.findall(r"@\w+\{([^,]+),", (SRC / "references.bib").read_text()))
cited = {
    key
    for group in re.findall(r"\\cite\{([^}]+)\}", (SRC / "main.tex").read_text())
    for key in group.split(",")
}
checks["citation_keys"] = cited <= bib_keys

intervention = pd.concat(
    [
        pd.read_csv(
            RUN
            / f"artifact/hard_intervention_rows/{dataset}41_graphs.csv"
        )
        for dataset in ("MNIST", "CIFAR10")
    ],
    ignore_index=True,
)
intervention["improved"] = (
    intervention.changed_message_mse < intervention.baseline_message_mse
)
intervention["worse"] = intervention.changed_margin < intervention.baseline_margin
h1 = intervention[intervention["T"] == 1]
for dataset, expected in (("MNIST", (811, 332)), ("CIFAR10", (928, 296))):
    group = h1[h1.dataset == dataset]
    observed = (
        int(group.improved.sum()),
        int((group.improved & group.worse).sum()),
    )
    checks[f"headline_{dataset.lower()}"] = observed == expected

strata = pd.read_csv(DATA / "hard_intervention_stratified.csv")
checks["strata_counts"] = int(strata[strata.readout == 1].adverse.sum()) == 332 + 296

temporal = pd.read_csv(DATA / "temporal_decision_effects.csv")
checks["temporal_counts"] = {
    "mnist_halve": int(
        temporal[(temporal.dataset == "MNIST") & (temporal.direction == "halve")]
        .prediction_changes.iloc[0]
    )
    == 6,
    "cifar_halve": int(
        temporal[(temporal.dataset == "CIFAR10") & (temporal.direction == "halve")]
        .prediction_changes.iloc[0]
    )
    == 16,
}

manifest = {
    "generated": "ccfa R2 verification",
    "source_hashes": {
        name: sha(SRC / name)
        for name in (
            "main.tex",
            "supplement.tex",
            "references.bib",
            "main.pdf",
            "supplement.pdf",
        )
    },
    "table_hashes": {path.name: sha(path) for path in sorted(DATA.glob("*.csv"))},
    "checks": checks,
}
artifact = RUN / "artifact"
artifact_manifest = json.loads((RUN / "artifact-manifest.json").read_text(encoding="utf-8"))
for rel, entry in artifact_manifest["files"].items():
    path = artifact / rel
    checks[f"artifact_{rel}"] = (
        path.exists()
        and path.stat().st_size == entry["bytes"]
        and sha(path) == entry["sha256"]
    )

for dataset, run_name in (
    ("MNIST", "prefix_diagnostic"),
    ("CIFAR10", "prefix_diagnostic"),
    ("MNIST", "prefix_double"),
    ("CIFAR10", "prefix_double"),
):
    direction = "halve" if run_name == "prefix_diagnostic" else "double"
    base = np.load(artifact / run_name / f"{dataset}41" / "baseline_T2.npz")
    alt = np.load(artifact / run_name / f"{dataset}41" / "all_eligible_T2.npz")
    row = temporal[
        (temporal.dataset == dataset) & (temporal.direction == direction)
    ].iloc[0]
    base_pred, alt_pred = base["logits"].argmax(1), alt["logits"].argmax(1)
    teacher_pred = base["teacher"].argmax(1)
    checks[f"endpoint_{dataset.lower()}_{direction}"] = {
        "graphs": len(base_pred) == int(row.train_graphs),
        "eligible": int(base["eligible_by_channel"].sum())
        == int(row.eligible_first_site_coordinates),
        "local_equality": float(alt["first_decode_error_max"].max())
        == float(row.local_T2_decode_max_error),
        "event_count": np.array_equal(base["first_events"], alt["first_events"]),
        "prediction_changes": int((base_pred != alt_pred).sum())
        == int(row.prediction_changes),
        "agreement_lost": int(
            ((base_pred == teacher_pred) & (alt_pred != teacher_pred)).sum()
        )
        == int(row.agreement_lost),
        "agreement_gained": int(
            ((base_pred != teacher_pred) & (alt_pred == teacher_pred)).sum()
        )
        == int(row.agreement_gained),
        "events": int(base["events"].sum())
        == int(row.whole_network_events_base)
        and int(alt["events"].sum()) == int(row.whole_network_events_changed),
    }

source_text = (SRC / "main.tex").read_text(encoding="utf-8")
supp_text = (SRC / "supplement.tex").read_text(encoding="utf-8")
checks["positioning"] = (
    "PD-Quant" in source_text
    and "Pred-Quant" not in source_text
    and "single-spike phase-coding converter" in source_text
    and "does not match first-prefix profile or absolute amplitude" in source_text
    and "whole-network activity" in source_text
)
checks["llm_disclosure"] = (
    "Kimi K3.1 Pro" in source_text
    and "DeepSeek V4.1 Flash" in source_text
    and "GPT-6-Luna" not in source_text
)
checks["artifact_claim"] = "anonymous review package" in supp_text

checks["all_pass"] = all(
    value
    for value in checks.values()
    if isinstance(value, bool)
    or (isinstance(value, dict) and all(value.values()))
)
(RUN / "audit/verification.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(checks, indent=2))
if not checks["all_pass"]:
    raise SystemExit(1)
