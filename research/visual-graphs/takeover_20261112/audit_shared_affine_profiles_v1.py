"""Independent audit of the four frozen shared-threshold train-only profiles.

Reads JSON/Markdown and hashes only; it does not load official data or models.
Both outputs are written exclusively after all four records pass.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
CAL_PATH = HERE / "affine_cal1.json"
AUDIT_JSON = HERE / "affine-six-profile-audit-v1.json"
AUDIT_MD = HERE / "affine-six-profile-audit-v1.md"
AUDIT_SCRIPT = HERE / "audit_affine_profiles_v1.py"
CLARIFICATION = HERE / "affine-screen-decision-clarification-v1.md"
PROTOCOL = HERE / "affine-base-fulltrain-protocol-v1.md"
OUT_JSON = HERE / "affine-shared-profile-audit-v1.json"
OUT_MD = HERE / "affine-shared-profile-audit-v1.md"
MODES = ("first_analog", "centered")
SCALES = (1, 8)
PREFIXES = (1, 2, 4)
N, BATCHES, FINAL, NODES, EDGES, WIDTH = 45000, 352, 72, 5294119, 42352952, 70
SHARED_SCOPE = "all_45000_official_train_graphs_label_free_fixed_order_shared_threshold_control"


class AuditError(RuntimeError):
    pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profile_path(mode, scale):
    return HERE / f"affine_{mode}_shared_s{scale}_tr1.json"


def load_base_auditor():
    spec = importlib.util.spec_from_file_location("_affine_profile_audit_v1", AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise AuditError("cannot load existing six-profile audit helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def expected_shared_thresholds(cal, scale):
    out = []
    for layer in range(1, 4):
        for kind in ("node", "edge"):
            raw = cal["calibration"]["channel_thresholds"][f"raw_{kind}_{layer}"]
            csum = cal["calibration"]["channel_thresholds"][f"branch_cumsum_{kind}_{layer}"]
            if len(raw) != WIDTH or len(csum) != WIDTH:
                raise AuditError(f"calibration threshold width mismatch at layer{layer}.{kind}")
            out.append([max(float(a), float(b)) * scale for a, b in zip(raw, csum)])
    return out


def close_vector(actual, wanted, label):
    if len(actual) != 6:
        raise AuditError(f"{label}: expected six shared site vectors")
    for i, (got, want) in enumerate(zip(actual, wanted)):
        if len(got) != WIDTH:
            raise AuditError(f"{label}: wrong threshold width at site {i}")
        for j, (a, b) in enumerate(zip(got, want)):
            if (not math.isfinite(float(a)) or float(a) <= 0 or
                    not math.isclose(float(a), float(b), rel_tol=2e-6, abs_tol=1e-8)):
                raise AuditError(f"{label}: shared max(raw,cumsum)*scale mismatch at site {i}, channel {j}")


def required_bound_paths():
    paths = [CAL_PATH, AUDIT_JSON, AUDIT_MD, AUDIT_SCRIPT, CLARIFICATION, PROTOCOL]
    paths.extend(HERE / n for n in (
        "gated_affine_shared_profile_v1.py", "check_gated_affine_shared_profile_v1.py",
        "gated_affine_profile_v1.py", "check_gated_affine_profile_v1.py",
        "gated_affine_base_v1.py", "check_gated_affine_base_v1.py",
        "gated_affine_cal_v1.py", "check_gated_affine_cal_v1.py",
        "activation_histogram.py", "gated_frozen_gate_calibrate.py", "gated_teacher_model.py"))
    paths.extend(profile_path(m, s) for m in ("generic", "first_analog", "centered") for s in SCALES)
    return {str(Path(p).resolve()) for p in paths}


def verify_shared_doc(base, doc, mode, scale, cal, cache):
    label = f"{mode}/shared-s{scale}"
    if (doc.get("scope") != SHARED_SCOPE or doc.get("threshold_rule") !=
            "per-channel max(raw_abs, branch_cumsum) × fixed scale"):
        raise AuditError(f"{label}: scope or threshold rule mismatch")
    # Reuse the frozen six-profile audit's detailed checks without changing it.
    normalized = dict(doc)
    normalized["scope"] = "all_45000_official_train_graphs_label_free_fixed_order"
    record = base.validate_profile(normalized, mode, scale, cal, cache, verify_files=True)
    close_vector(doc.get("threshold_vectors", []), expected_shared_thresholds(cal, scale), label)
    bound = doc.get("input_hashes_start", {})
    if not isinstance(bound, dict) or bound != doc.get("input_hashes_end"):
        raise AuditError(f"{label}: missing or changed start/end input hashes")
    missing = required_bound_paths() - set(bound)
    if missing:
        raise AuditError(f"{label}: missing required provenance hashes: " + ", ".join(sorted(missing)))
    if doc.get("prior_profile_audit") != str(AUDIT_JSON.resolve()) or doc.get("prior_profile_audit_sha256") != sha(AUDIT_JSON):
        raise AuditError(f"{label}: six-profile audit JSON binding mismatch")
    if doc.get("prior_profile_audit_md_sha256") != sha(AUDIT_MD):
        raise AuditError(f"{label}: six-profile audit Markdown binding mismatch")
    for m in ("generic", "first_analog", "centered"):
        for s in SCALES:
            entry = doc.get("prior_profile_hashes", {}).get(f"{m}_s{s}")
            if entry != sha(profile_path(m, s)):
                raise AuditError(f"{label}: prior six-profile hash mismatch for {m}/s{s}")
    if doc.get("calibration_sha256") != sha(CAL_PATH):
        raise AuditError(f"{label}: calibration hash mismatch")
    return record


def compare_records(records):
    by = {(x["mode"], x["scale"]): x for x in records}
    out = {}
    for scale in SCALES:
        base = by[("first_analog", scale)]["profiles"]
        cand = by[("centered", scale)]["profiles"]
        out[str(scale)] = {}
        for t in PREFIXES:
            a, b = base[str(t)], cand[str(t)]
            out[str(scale)][str(t)] = {
                "first_analog": a,
                "centered": b,
                "centered_minus_first_analog": {
                    "agreement": b["teacher_class_agreement"] - a["teacher_class_agreement"],
                    "logit_mse": b["logit_mse"] - a["logit_mse"],
                    "teacher_to_candidate_kl": b["teacher_to_candidate_kl"] - a["teacher_to_candidate_kl"],
                    "prospective_projection_adds": b["prospective_projection_adds"] - a["prospective_projection_adds"],
                }}
    return out


def load_and_audit():
    missing = [str(profile_path(m, s)) for m in MODES for s in SCALES if not profile_path(m, s).is_file()]
    if missing:
        raise AuditError("four-control audit refused; missing JSON(s): " + ", ".join(missing))
    if OUT_JSON.exists() or OUT_MD.exists():
        raise AuditError("refusing existing audit outputs; no overwrite")
    for p in (CAL_PATH, AUDIT_JSON, AUDIT_MD, AUDIT_SCRIPT, CLARIFICATION, PROTOCOL):
        if not p.is_file():
            raise AuditError("missing audit dependency: " + str(p))
    base = load_base_auditor()
    cache = {}
    cal = base.verify_calibration(json.loads(CAL_PATH.read_text(encoding="utf-8")), cache)
    old_audit = json.loads(AUDIT_JSON.read_text(encoding="utf-8"))
    if old_audit.get("status") != "complete" or old_audit.get("overall_shared_threshold_controls_trigger") is not True:
        raise AuditError("six-profile audit did not authorize shared-threshold controls")
    old_profiles = {(x["mode"], x["scale"]): x for x in old_audit.get("profiles", [])}
    if set(old_profiles) != {(m, s) for m in ("generic", "first_analog", "centered") for s in SCALES}:
        raise AuditError("six-profile audit does not bind exactly six source profiles")
    for (mode, scale), profile in old_profiles.items():
        if profile.get("profile_sha256") != sha(profile_path(mode, scale)):
            raise AuditError(f"six-profile source profile changed: {mode}/s{scale}")
    docs, records = {}, []
    for mode in MODES:
        for scale in SCALES:
            doc = json.loads(profile_path(mode, scale).read_text(encoding="utf-8"))
            docs[(mode, scale)] = doc
            records.append(verify_shared_doc(base, doc, mode, scale, cal, cache))
    ref = docs[("first_analog", 1)]
    for (mode, scale), doc in docs.items():
        if doc.get("teacher") != ref.get("teacher") or doc.get("calibration_sha256") != ref.get("calibration_sha256") or doc.get("author_source") != ref.get("author_source"):
            raise AuditError("shared profiles do not share the same teacher/calibration/author source")
    for scale in SCALES:
        a = docs[("first_analog", scale)]["threshold_vectors"]
        b = docs[("centered", scale)]["threshold_vectors"]
        if a != b:
            raise AuditError(f"same-scale controls do not use exactly the same threshold vectors: scale {scale}")
    return cal, records, compare_records(records)


def make_report(cal, records, comparisons):
    return {"status": "complete", "scope": "four shared-threshold CIFAR10 seed41 label-free training profiles",
            "ground_truth_accuracy_or_labels_used": False,
            "calibration_sha256": sha(CAL_PATH),
            "six_profile_audit_sha256": sha(AUDIT_JSON),
            "shared_profile_sha256": {f"{m}_s{s}": sha(profile_path(m, s)) for m in MODES for s in SCALES},
            "expected_coverage": {"graphs": N, "batches": BATCHES, "final_batch_graphs": FINAL,
                                  "nodes": NODES, "directed_edges": EDGES, "prefixes": list(PREFIXES)},
            "profiles": records, "same_scale_prefix_comparisons": comparisons,
            "qualification": "Training-only teacher-fidelity, event-count, and prospective projection ADD diagnostics; shared-threshold controls do not establish accuracy, efficient execution, latency, energy, or novelty.",
            "audit_script_sha256": sha(Path(__file__))}


def markdown(report):
    lines = ["# Affine-base shared-profile audit v1", "",
             "All four predeclared shared-threshold controls passed integrity and statistics checks. Values are teacher agreement, logit MSE, teacher-to-candidate KL and prospective projection ADDs; no ground-truth labels, validation or test were used.", "",
             "| Scale | T | Mode | Teacher agreement | Logit MSE | Teacher→candidate KL | Prospective projection ADDs |",
             "|---:|---:|---|---:|---:|---:|---:|"]
    for s in SCALES:
        for t in PREFIXES:
            for m in MODES:
                x = report["same_scale_prefix_comparisons"][str(s)][str(t)][m]
                lines.append(f"| {s} | {t} | {m} | {x['teacher_class_agreement']:.9f} | {x['logit_mse']:.12g} | {x['teacher_to_candidate_kl']:.12g} | {x['prospective_projection_adds']} |")
    lines += ["", "## Same-scale differences (centered − first analog)", "",
              "| Scale | T | Agreement | Logit MSE | KL | Prospective projection ADDs |", "|---:|---:|---:|---:|---:|---:|"]
    for s in SCALES:
        for t in PREFIXES:
            d = report["same_scale_prefix_comparisons"][str(s)][str(t)]["centered_minus_first_analog"]
            lines.append(f"| {s} | {t} | {d['agreement']:.9g} | {d['logit_mse']:.12g} | {d['teacher_to_candidate_kl']:.12g} | {d['prospective_projection_adds']} |")
    lines += ["", "Each mode/scale passed 45,000 graph, 352 batch, final-batch, node/edge-row, three-prefix, scalar/sign/level event partition, metric-sum, ADD formula, source/head-call, deterministic runtime, calibration, teacher, author-source and input-hash checks. The six threshold vectors are exactly shared within each scale and equal per-channel `max(raw_abs, branch_cumsum) × scale` from the frozen calibration. This audit does not trigger further threshold search.", "",
              f"Calibration SHA-256: `{report['calibration_sha256']}`. Six-profile audit SHA-256: `{report['six_profile_audit_sha256']}`.", "",
              "Prospective ADDs count one coded stream and are not total compute, measured runtime, latency or energy. These are one-teacher, training-only diagnostics; no accuracy or novelty claim follows."]
    return "\n".join(lines) + "\n"


def write_exclusive(report):
    if OUT_JSON.exists() or OUT_MD.exists():
        raise AuditError("refusing existing audit outputs; no overwrite")
    j = json.dumps(report, indent=2, allow_nan=False) + "\n"
    m = markdown(report)
    made = []
    try:
        with OUT_JSON.open("x", encoding="utf-8") as f:
            f.write(j)
        made.append(OUT_JSON)
        with OUT_MD.open("x", encoding="utf-8") as f:
            f.write(m)
        made.append(OUT_MD)
    except Exception:
        for p in made:
            try:
                p.unlink()
            except OSError:
                pass
        raise


def self_test():
    base = load_base_auditor()
    fake_cal = {"calibration": {"channel_thresholds": {}}}
    for layer in range(4):
        for kind in ("node", "edge"):
            fake_cal["calibration"]["channel_thresholds"][f"raw_{kind}_{layer}"] = [1.0 + layer] * WIDTH
    for layer in range(1, 4):
        for kind in ("node", "edge"):
            fake_cal["calibration"]["channel_thresholds"][f"branch_cumsum_{kind}_{layer}"] = [2.0 + layer] * WIDTH
    records = []
    for scale in SCALES:
        shared = expected_shared_thresholds(fake_cal, scale)
        for mode in MODES:
            doc = base.synthetic_profile(mode, scale, agreement=2 if mode == "first_analog" else 3,
                                         mse=.001 if mode == "first_analog" else .0005)
            doc["scope"] = SHARED_SCOPE
            doc["threshold_rule"] = "per-channel max(raw_abs, branch_cumsum) × fixed scale"
            doc["threshold_vectors"] = shared
            copy_doc = dict(doc)
            copy_doc["scope"] = "all_45000_official_train_graphs_label_free_fixed_order"
            rec = base.validate_profile(copy_doc, mode, scale, None, {}, verify_files=False,
                                        n=3, batches=2, final=1, nodes=5, edges=9)
            close_vector(doc["threshold_vectors"], shared, "synthetic")
            records.append(rec)
    comparisons = compare_records(records)
    assert comparisons["1"]["1"]["centered_minus_first_analog"]["agreement"] > 0
    bad = list(expected_shared_thresholds(fake_cal, 1))
    bad[0] = list(bad[0]); bad[0][0] *= 2
    try:
        close_vector(bad, expected_shared_thresholds(fake_cal, 1), "fault")
    except AuditError:
        pass
    else:
        raise AssertionError("mismatched shared threshold accepted")
    print("PASS: shared-threshold expectation, full profile statistics reuse, pairwise prefix comparison, and mismatch rejection; synthetic only")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true", help="in-memory synthetic checks only")
    args = p.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        cal, records, comparisons = load_and_audit()
        report = make_report(cal, records, comparisons)
        write_exclusive(report)
        print(json.dumps({"status": "complete", "json": str(OUT_JSON), "markdown": str(OUT_MD)}, indent=2))
        return 0
    except Exception as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
