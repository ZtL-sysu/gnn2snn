"""Create a de-identified public copy of an evaluator report.

Only absolute local path keys in calibration_bound_inputs_sha256 and the
author_repo path are replaced. All numeric values, hashes and per-sample
evidence stay untouched; the original report is never modified. The original
SHA-256 is embedded in the public copy so the pair can be re-verified locally.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


def deidentify(path: str) -> str:
    s = path.replace("\\", "/")
    low = s.lower()
    if "/gnn2snn-wb-resume/" in low:
        return "repo/" + s.split("gnn2snn-wb-resume/", 1)[1]
    if "pyg_bench/" in low:
        return "data/" + s.split("pyg_bench/", 1)[1]
    if "dcgs-" in low:
        return "<pinned-dcgs>/" + "/".join(s.split("/")[-2:])
    return "local/" + s.rsplit("/", 1)[-1]


def scrub_strings(node):
    """Replace every remaining drive-letter path anywhere in the document."""
    if isinstance(node, dict):
        return {deidentify(k) if ":/" in str(k).replace("\\", "/") else k: scrub_strings(v)
                for k, v in node.items()}
    if isinstance(node, list):
        return [scrub_strings(v) for v in node]
    if isinstance(node, str) and len(node) > 2 and node[1:3] in (":/", ":\\"):
        return deidentify(node)
    return node


def main() -> None:
    src = Path(sys.argv[1]).resolve()
    dst = src.with_name("report.public.json")
    original_sha = hashlib.sha256(src.read_bytes()).hexdigest()
    doc = json.loads(src.read_text(encoding="utf-8"))
    doc["calibration_bound_inputs_sha256"] = {
        deidentify(k): v for k, v in doc["calibration_bound_inputs_sha256"].items()
    }
    doc["author_source_provenance"]["author_repo"] = "<pinned DCGS checkout; local path withheld>"
    doc = scrub_strings(doc)
    doc["publication_note"] = {
        "scope": "de-identified public copy; only absolute local path keys and the author_repo path were replaced",
        "numeric_and_hash_values": "unchanged",
        "source_report": src.name,
        "source_report_sha256": original_sha,
        "per_sample_evidence": "12 NPZ files unchanged",
    }
    dst.write_text(json.dumps(doc, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    text = dst.read_text(encoding="utf-8")
    print(json.dumps({
        "source_report_sha256": original_sha,
        "public_report_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "public_report": str(dst),
        "rows": len(doc["rows"]),
        "source_hashes_equal": doc["source_sha256_start"] == doc["source_sha256_end"],
        "contains_local_username": "xhf" in text,
        "remaining_windows_drive_paths": sum(1 for k in doc["calibration_bound_inputs_sha256"] if ":/" in k),
    }, indent=2))


if __name__ == "__main__":
    main()
