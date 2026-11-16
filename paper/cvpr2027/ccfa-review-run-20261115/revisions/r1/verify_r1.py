"""Verify manuscript integrity and recompute all R1 headline numbers."""
from pathlib import Path
import hashlib
import json
import re

import fitz
import numpy as np
import pandas as pd

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[4]
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
            ROOT
            / f"research/20261115_decision/runs/intervention_v2/{dataset}41/graphs.csv"
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
    "generated": "ccfa R1 verification",
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
checks["all_pass"] = all(
    value
    for value in checks.values()
    if isinstance(value, bool)
    or (isinstance(value, dict) and all(value.values()))
)
(RUN / "audit/verification-r1.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(checks, indent=2))
if not checks["all_pass"]:
    raise SystemExit(1)
