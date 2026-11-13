"""Emit the research-github-sync fixed-column METRICS.csv for a control run.

The evidence column points at the de-identified public report, not the local
original. LF line endings are explicit so `git diff --check` stays clean.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

HEADER = ("task_id,agent_id,run_id,dataset,split,coverage,method,teacher_seed,"
          "student_seed,T,metric,value,unit,evidence,protocol_sha256,status").split(",")
PROTOCOL_SHA256 = "8573171a65a55fb10c044d315eeba70101eab643e208d01941014de254904543"
EVIDENCE = ("review-suggestions/grok-bot/f-research-v1-run-20261113-residual-prefix/"
            "results/e1-seed41-controls-v1/report.public.json")
STATUS = "measured_single_seed; E1 acceptance pending seeds42-43 and complete cost"


def main() -> None:
    run_dir = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    report = json.loads((run_dir / "report.json").read_text(encoding="utf-8"))
    with out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(HEADER)
        for row in report["rows"]:
            for metric, unit, value in (
                ("accuracy", "fraction", row["accuracy"]),
                ("teacher_agreement", "fraction", row["teacher_agreement"]),
                ("logit_mse", "squared_logit", row["logit_mse"]),
            ):
                writer.writerow(["visual-conversion", "20261113-wb-resume", "e1-seed41-controls-v1",
                                 "CIFAR10", "official_val", "all_5000_graphs", row["mode"], 41, "na",
                                 row["T"], metric, format(value, ".10g"), unit, EVIDENCE,
                                 PROTOCOL_SHA256, STATUS])
    print(json.dumps({"rows": len(report["rows"]), "metric_rows": len(out.read_text().splitlines()) - 1,
                      "output": str(out)}))


if __name__ == "__main__":
    main()
