"""Create a de-identified public copy of any JSON evidence artifact (v2).

Generalises deidentify_report_v1.py to the multiseed E1 reports and the
per-seed calibration JSONs, whose absolute-path keys live in different
sections (source_sha256_start/end, input_hashes_start/end, data.root,
teacher_run, checkpoint.path, author_repo).

Only absolute local path *strings/keys* are replaced. Every numeric value,
hash, array and scientific field stays byte-for-byte identical. The original
file is never modified; its SHA-256 is embedded in the public copy.
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
    if "/takeover_20261112/" in low:
        return "legacy/" + s.split("/takeover_20261112/", 1)[1]
    return "local/" + s.rsplit("/", 1)[-1]


def is_abs_local(s: str) -> bool:
    norm = s.replace("\\", "/")
    return len(norm) > 2 and norm[1:3] == ":/"


def scrub(node):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            nk = deidentify(k) if is_abs_local(str(k)) else k
            if nk in out:
                raise RuntimeError(f"de-identification key collision: {nk}")
            out[nk] = scrub(v)
        return out
    if isinstance(node, list):
        return [scrub(v) for v in node]
    if isinstance(node, str) and is_abs_local(node):
        return deidentify(node)
    return node


def count_drive_paths(node) -> int:
    if isinstance(node, dict):
        return sum(count_drive_paths(k) + count_drive_paths(v) for k, v in node.items())
    if isinstance(node, list):
        return sum(count_drive_paths(v) for v in node)
    return 1 if isinstance(node, str) and is_abs_local(node) else 0


def main() -> None:
    src = Path(sys.argv[1]).resolve()
    dst = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else src.with_name(src.stem + ".public.json")
    if dst.exists():
        raise FileExistsError(f"refusing existing public copy: {dst}")
    raw = src.read_bytes()
    original_sha = hashlib.sha256(raw).hexdigest()
    doc = json.loads(raw.decode("utf-8"))
    public = scrub(doc)
    public["publication_note"] = {
        "scope": "de-identified public copy; only absolute local path strings/keys replaced",
        "numeric_and_hash_values": "unchanged",
        "source_artifact": src.name,
        "source_artifact_sha256": original_sha,
        "deidentify_tool": "deidentify_v2.py",
    }
    text = json.dumps(public, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(json.dumps({
        "source_artifact": str(src),
        "source_artifact_sha256": original_sha,
        "public_artifact": str(dst),
        "public_artifact_sha256": hashlib.sha256(dst.read_bytes()).hexdigest(),
        "remaining_windows_drive_paths": count_drive_paths(public),
        "contains_local_username": str(Path.home().name).lower() in text.lower(),
        "rows": len(public.get("rows", [])),
    }, indent=2))


if __name__ == "__main__":
    main()
