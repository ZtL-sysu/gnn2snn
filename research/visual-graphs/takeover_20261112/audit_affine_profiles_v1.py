"""Independent audit of six frozen affine-base training-only profile JSONs.

This script reads result/calibration/provenance JSON and hashes bound files. It
does not load a dataset, labels, weights, logits, or execute a model. It writes
the JSON/Markdown audit pair only after all six profiles pass validation.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
CAL_PATH = HERE / "affine_cal1.json"
MODES = ("generic", "first_analog", "centered")
SCALES = (1, 8)
PREFIXES = (1, 2, 4)
N, BATCHES, FINAL, NODES, EDGES, WIDTH = 45000, 352, 72, 5294119, 42352952, 70
PINNED_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"
OUT_JSON = HERE / "affine-six-profile-audit-v1.json"
OUT_MD = HERE / "affine-six-profile-audit-v1.md"


class AuditError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def profile_path(mode, scale):
    return HERE / f"affine_{mode}_s{scale}_tr1.json"


def site_names(mode):
    start = 0 if mode == "generic" else 1
    return [f"layer{i}.{kind}" for i in range(start, 4) for kind in ("node", "edge")]


def number(x, name, nonnegative=False):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x):
        raise AuditError(f"{name}: expected finite number")
    if nonnegative and x < 0:
        raise AuditError(f"{name}: expected nonnegative number")
    return float(x)


def integer(x, name, minimum=0):
    if type(x) is not int or x < minimum:
        raise AuditError(f"{name}: expected integer >= {minimum}")
    return x


def verify_hashmap(doc, cache, label):
    start, end = doc.get("input_hashes_start"), doc.get("input_hashes_end")
    if not isinstance(start, dict) or not start or start != end:
        raise AuditError(f"{label}: input hash start/end missing or unequal")
    for raw, expected in start.items():
        path = Path(raw)
        if not path.is_absolute() or not path.is_file():
            raise AuditError(f"{label}: missing/nonabsolute bound input: {raw}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise AuditError(f"{label}: malformed SHA256 for {raw}")
        resolved = str(path.resolve())
        actual = cache.get(resolved)
        if actual is None:
            actual = sha256(path)
            cache[resolved] = actual
        if actual != expected:
            raise AuditError(f"{label}: bound hash mismatch: {raw}")
    return start


def verify_calibration(cal, cache):
    if (cal.get("status") != "complete" or cal.get("dataset") != "CIFAR10" or
            cal.get("seed") != 41 or
            cal.get("scope") != "all_45000_official_train_graphs_label_free_two_pass"):
        raise AuditError("calibration status/dataset/teacher/scope mismatch")
    c = cal.get("calibration", {})
    if (c.get("passes") != 2 or c.get("pass_graph_counts") != [N, N] or
            c.get("pass_batch_counts") != [BATCHES, BATCHES] or c.get("bins") != 512 or
            c.get("quantile") != .999 or c.get("inactive_channel_floor") != 1e-6):
        raise AuditError("calibration passes/coverage/histogram protocol mismatch")
    totals = c.get("pass_totals", [])
    if (len(totals) != 2 or totals[0] != totals[1] or totals[0].get("nodes") != NODES or
            totals[0].get("directed_edges") != EDGES or totals[0].get("final_batch_graphs") != FINAL):
        raise AuditError("calibration pass totals mismatch")
    expected_sites = {f"raw_{k}_{i}" for i in range(4) for k in ("node", "edge")} | {
        f"branch_cumsum_{k}_{i}" for i in range(1, 4) for k in ("node", "edge")}
    sites = c.get("sites", {})
    if set(sites) != expected_sites or set(c.get("channel_thresholds", {})) != expected_sites:
        raise AuditError("calibration 14-site set mismatch")
    if c.get("pass_rows_per_site") != [c.get("activation_rows_per_site")]*2:
        raise AuditError("calibration pass row maps differ")
    for name, v in sites.items():
        rows = NODES if "_node_" in name else EDGES
        counts, glob = v.get("channel_counts"), v.get("global_counts")
        maxima, theta = v.get("maxima"), v.get("per_channel_thresholds")
        if (v.get("rows") != rows or v.get("channels") != WIDTH or len(counts) != WIDTH or
                any(len(ch) != 512 for ch in counts) or len(glob) != 512 or
                len(maxima) != WIDTH or len(theta) != WIDTH):
            raise AuditError(f"calibration shape/rows mismatch at {name}")
        for ch, (channel, maximum, threshold) in enumerate(zip(counts, maxima, theta)):
            if any(type(x) is not int or x < 0 for x in channel) or sum(channel) != rows:
                raise AuditError(f"calibration channel histogram partition mismatch at {name}[{ch}]")
            if (not math.isfinite(float(maximum)) or maximum < 0 or
                    not math.isfinite(float(threshold)) or threshold <= 0):
                raise AuditError(f"calibration invalid maximum/threshold at {name}[{ch}]")
            target, running, hit = math.ceil(rows*.999), 0, None
            for b, count in enumerate(channel):
                running += count
                if running >= target:
                    hit = b
                    break
            if hit is None:
                raise AuditError(f"calibration channel quantile missing at {name}[{ch}]")
            wanted = max((hit+1)*float(maximum)/512, 1e-6)
            if not math.isclose(wanted, float(threshold), rel_tol=2e-7, abs_tol=1e-10):
                raise AuditError(f"calibration .999 channel threshold mismatch at {name}[{ch}]")
        if any(type(x) is not int or x < 0 for x in glob) or sum(glob) != rows*WIDTH:
            raise AuditError(f"calibration global histogram partition mismatch at {name}")
        target, running, hit = math.ceil(rows*WIDTH*.999), 0, None
        for b, count in enumerate(glob):
            running += count
            if running >= target:
                hit = b
                break
        if hit is None:
            raise AuditError(f"calibration global quantile missing at {name}")
        want = max((hit+1)*max(float(x) for x in maxima)/512, 1e-6)
        if not math.isclose(want, float(v.get("global_threshold")), rel_tol=2e-7, abs_tol=1e-10):
            raise AuditError(f"calibration .999 global threshold mismatch at {name}")
    verify_hashmap(cal, cache, "calibration")
    if sha256(CAL_PATH) != cal.get("calibration_sha256", sha256(CAL_PATH)):
        raise AuditError("calibration self-hash mismatch")
    return cal


def verify_binding(doc, cal, cache, label):
    if doc.get("calibration") != str(CAL_PATH.resolve()) or doc.get("calibration_sha256") != sha256(CAL_PATH):
        raise AuditError(f"{label}: calibration path/hash mismatch")
    teacher = doc.get("teacher", {})
    if teacher != cal.get("teacher"):
        raise AuditError(f"{label}: profile/calibration teacher differs")
    run = Path(teacher.get("run", ""))
    meta_path, done_path, ckpt_path = run/"metadata.json", run/"done.json", run/"best.pt"
    if not run.is_absolute() or not all(x.is_file() for x in (meta_path, done_path, ckpt_path)):
        raise AuditError(f"{label}: missing/nonabsolute teacher binding")
    meta, done = json.loads(meta_path.read_text()), json.loads(done_path.read_text())
    if teacher.get("metadata") != meta or teacher.get("done") != done:
        raise AuditError(f"{label}: teacher metadata/done mismatch")
    if (meta.get("seed") != 41 or done.get("status") != "finished" or
            sha256(ckpt_path) != teacher.get("checkpoint_sha256") or
            done.get("best_checkpoint_sha256") != teacher.get("checkpoint_sha256")):
        raise AuditError(f"{label}: teacher checkpoint mismatch")
    author = doc.get("author_source", {})
    if author.get("commit") != PINNED_COMMIT or author.get("worktree_clean_except_ignored_bytecode") is not True:
        raise AuditError(f"{label}: pinned author source mismatch")
    for key, cls in (("relu_neuron", "MTH_diff"), ("line_neuron", "MTH_diff_line")):
        item = author.get("source_files", {}).get(key, {})
        path = Path(author.get("author_repo", ""))/item.get("path", "")
        if (not path.is_absolute() or not path.is_file() or
                item.get("head_blob_oid") != item.get("worktree_blob_oid") or
                sha256(path) != item.get("sha256") or
                Path(author.get("loaded_classes", {}).get(cls, "")).resolve() != path.resolve()):
            raise AuditError(f"{label}: author source file/hash/path mismatch: {key}")
    hashes = verify_hashmap(doc, cache, label)
    cal_hashes = cal.get("input_hashes_start", {})
    if any(hashes.get(path) != digest for path, digest in cal_hashes.items()):
        raise AuditError(f"{label}: calibration input bindings not retained")
    for name, digest in meta.get("provenance", {}).get("source_sha256", {}).items():
        if not any(Path(path).name == name and h == digest for path, h in hashes.items()):
            raise AuditError(f"{label}: missing teacher source hash {name}")
    for name in ("pre_filter.pt", "pre_transform.pt", "train_data.pt"):
        digest = meta["provenance"]["processed_data_sha256"][name]
        if not any(Path(path).name == name and h == digest for path, h in hashes.items()):
            raise AuditError(f"{label}: missing processed-data hash {name}")
    return author, hashes


def expected_thresholds(cal, mode, scale):
    family = "branch_cumsum" if mode == "centered" else "raw"
    start = 0 if mode == "generic" else 1
    out = []
    for layer in range(start, 4):
        for kind in ("node", "edge"):
            key = f"{family}_{kind}_{layer}"
            out.append([float(x)*scale for x in cal["calibration"]["channel_thresholds"][key]])
    return out


def validate_site_stats(st, rows, t, name):
    for k in ("positive", "negative", "zero", "nonzero_scalar_events", "nonzero_rows"):
        integer(st.get(k), f"{name}/T{t}/{k}")
    pos, neg, zero = st["positive"], st["negative"], st["zero"]
    nz, nr = st["nonzero_scalar_events"], st["nonzero_rows"]
    if (pos+neg != nz or zero+nz != rows*WIDTH*t or nr > rows*t or
            nr > nz or nz > nr*WIDTH):
        raise AuditError(f"{name}/T{t}: scalar/nonzero-row partition mismatch")
    number(st.get("sum_abs_graded_amplitude"), f"{name}/T{t}/amplitude", True)
    levels = st.get("signed_threshold_level_events")
    if not isinstance(levels, dict):
        raise AuditError(f"{name}/T{t}: missing signed level partition")
    pos_levels = neg_levels = 0
    valid = {f"{s}2^-{i}" for s in ("+", "-") for i in range(8)}
    for key, count in levels.items():
        if key not in valid:
            raise AuditError(f"{name}/T{t}: invalid level key {key}")
        integer(count, f"{name}/T{t}/{key}")
        if key.startswith("+"):
            pos_levels += count
        else:
            neg_levels += count
    if (pos_levels != pos or neg_levels != neg or pos_levels+neg_levels != nz):
        raise AuditError(f"{name}/T{t}: level/sign partition mismatch")


def validate_profile(doc, mode, scale, cal, cache, verify_files=True,
                     n=N, batches=BATCHES, final=FINAL, nodes=NODES, edges=EDGES):
    label = f"{mode}/s{scale}"
    if (doc.get("status") != "complete" or doc.get("dataset") != "CIFAR10" or
            doc.get("seed") != 41 or doc.get("mode") != mode or doc.get("threshold_scale") != scale or
            doc.get("scope") != "all_45000_official_train_graphs_label_free_fixed_order" or
            doc.get("executed_T") != 4 or doc.get("prefix_time_grid") != list(PREFIXES)):
        raise AuditError(f"{label}: status/scope/setting mismatch")
    if verify_files:
        author, _ = verify_binding(doc, cal, cache, label)
        runtime = doc.get("runtime", {})
        if runtime.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or runtime.get("deterministic") is not True:
            raise AuditError(f"{label}: deterministic runtime metadata mismatch")
    else:
        author = doc.get("author_source", {})
    names = site_names(mode)
    if set(doc.get("profiles", {})) != {str(t) for t in PREFIXES}:
        raise AuditError(f"{label}: prefix set mismatch")
    results = {}
    old_stats = {}
    for t in PREFIXES:
        row = doc["profiles"][str(t)]
        for field, want in (("prefix_T", t), ("graphs", n), ("batches", batches),
                            ("nodes", nodes), ("directed_edges", edges), ("final_batch_graphs", final)):
            if row.get(field) != want:
                raise AuditError(f"{label}/T{t}: {field} mismatch")
        stats = row.get("sites", {})
        if list(stats) != names:
            raise AuditError(f"{label}/T{t}: site set/order mismatch")
        for site in names:
            validate_site_stats(stats[site], nodes if site.endswith(".node") else edges, t, site)
        old_stats[str(t)] = stats
        sums = row.get("metric_sums", {})
        if set(sums) != {"agree", "squared_error", "kl_sum", "graphs"} or sums.get("graphs") != n:
            raise AuditError(f"{label}/T{t}: metric sum coverage mismatch")
        agree = integer(sums.get("agree"), f"{label}/T{t}/agree")
        if agree > n:
            raise AuditError(f"{label}/T{t}: agreement count exceeds graphs")
        sq = number(sums.get("squared_error"), f"{label}/T{t}/squared_error", True)
        kl_sum = number(sums.get("kl_sum"), f"{label}/T{t}/kl_sum")
        if kl_sum < -1e-10:
            raise AuditError(f"{label}/T{t}: negative KL sum")
        agreement, mse, kl = agree/n, sq/(n*10), kl_sum/n
        for field, value in (("teacher_class_agreement", agreement), ("logit_mse", mse),
                             ("teacher_to_candidate_kl", kl)):
            if not math.isclose(number(row.get(field), f"{label}/T{t}/{field}"), value,
                                rel_tol=2e-12, abs_tol=1e-14):
                raise AuditError(f"{label}/T{t}: {field} does not reproduce from metric sums")
        adds = WIDTH * sum((4 if site.endswith(".node") else 1) * stats[site]["nonzero_scalar_events"] for site in names)
        if row.get("prospective_projection_adds") != adds:
            raise AuditError(f"{label}/T{t}: projection ADD formula mismatch")
        if row.get("prefix_logical_active_calls") != len(names)*t*batches:
            raise AuditError(f"{label}/T{t}: prefix logical active calls mismatch")
        results[str(t)] = {"teacher_class_agreement": agreement, "logit_mse": mse,
                           "teacher_to_candidate_kl": kl, "prospective_projection_adds": adds,
                           "metric_sums": {"agree": agree, "squared_error": sq,
                                           "kl_sum": kl_sum, "graphs": n}}
    for t1, t2 in (("1", "2"), ("2", "4")):
        for site in names:
            a, b = old_stats[t1][site], old_stats[t2][site]
            for k in ("positive", "negative", "zero", "nonzero_scalar_events", "nonzero_rows"):
                if b[k] < a[k]:
                    raise AuditError(f"{label}: cumulative {k} decreases at {site}")
            if b["sum_abs_graded_amplitude"] < a["sum_abs_graded_amplitude"]:
                raise AuditError(f"{label}: cumulative amplitude decreases at {site}")
            for key, count in a["signed_threshold_level_events"].items():
                if b["signed_threshold_level_events"].get(key, 0) < count:
                    raise AuditError(f"{label}: cumulative level count decreases at {site}/{key}")
    units = len(names)
    if doc.get("source_units_per_batch") != units:
        raise AuditError(f"{label}: source units/batch mismatch")
    for field, want in (("actual_source_warmup_calls", units*batches),
                        ("actual_active_calls", units*4*batches),
                        ("diagnostic_head_calls", 3*batches),
                        ("normal_endpoint_head_calls", batches)):
        if doc.get(field) != want:
            raise AuditError(f"{label}: {field} mismatch")
    if doc.get("source_head_scope") != "three diagnostic prefix heads; normal endpoint head once per batch":
        raise AuditError(f"{label}: head-call accounting scope mismatch")
    thresholds = doc.get("threshold_vectors", [])
    expected = expected_thresholds(cal, mode, scale) if cal else []
    if len(thresholds) != units or (expected and len(expected) != units):
        raise AuditError(f"{label}: threshold vector count mismatch")
    for i, vector in enumerate(thresholds):
        if len(vector) != WIDTH or any(not math.isfinite(float(x)) or float(x) <= 0 for x in vector):
            raise AuditError(f"{label}: invalid threshold vector {i}")
        if expected:
            for j, (actual, wanted) in enumerate(zip(vector, expected[i])):
                if not math.isclose(float(actual), wanted, rel_tol=2e-6, abs_tol=1e-8):
                    raise AuditError(f"{label}: calibration threshold mismatch site {i}, channel {j}")
    return {"mode": mode, "scale": scale, "site_count": units,
            "actual_source_warmup_calls": doc["actual_source_warmup_calls"],
            "actual_active_calls": doc["actual_active_calls"],
            "diagnostic_head_calls": doc["diagnostic_head_calls"],
            "normal_endpoint_head_calls": doc["normal_endpoint_head_calls"],
            "profiles": results,
            "profile_sha256": sha256(profile_path(mode, scale)) if verify_files else None,
            "author_source_commit": author.get("commit")}


def conditional_gate(audits):
    by = {(a["mode"], a["scale"]): a for a in audits}
    decision = {}
    for scale in SCALES:
        base = by[("first_analog", scale)]["profiles"]
        cand = by[("centered", scale)]["profiles"]
        checks, fidelity_all, add_all, strict_any = {}, True, True, False
        for t in PREFIXES:
            b, c = base[str(t)], cand[str(t)]
            ap = c["teacher_class_agreement"] >= b["teacher_class_agreement"]
            mp = c["logit_mse"] <= b["logit_mse"]
            dp = c["prospective_projection_adds"] <= b["prospective_projection_adds"]
            strict = c["teacher_class_agreement"] > b["teacher_class_agreement"] or c["logit_mse"] < b["logit_mse"]
            checks[str(t)] = {"agreement_non_degrade": ap, "mse_non_degrade": mp,
                              "projection_adds_non_increase": dp,
                              "strict_fidelity_improvement": strict,
                              "first_analog": b, "centered": c}
            fidelity_all &= ap and mp
            add_all &= dp
            strict_any |= strict
        trigger = fidelity_all and strict_any and add_all
        decision[str(scale)] = {"prefixes": checks,
            "all_prefixes_fidelity_non_degrade": bool(fidelity_all),
            "at_least_one_strict_fidelity_improvement": bool(strict_any),
            "all_prefixes_projection_adds_non_increase": bool(add_all),
            "shared_threshold_controls_trigger": bool(trigger)}
    decision["overall_shared_threshold_controls_trigger"] = any(
        decision[str(s)]["shared_threshold_controls_trigger"] for s in SCALES)
    decision["decision_rule"] = (
        "At one same scale, centered must have agreement >= and MSE <= first_analog at every T1/T2/T4, "
        "at least one strict fidelity improvement over the three prefixes, and projection ADDs <= at every prefix.")
    return decision


def make_markdown(report):
    lines = ["# Affine-base six-profile audit v1", "",
        "Status: independent audit of all six saved training-only profiles. No labels or ground-truth accuracy were used. Reported values are teacher-class agreement, logit MSE, teacher-to-candidate KL and prospective single-stream projection ADDs.", "",
        "## Prefix results", "",
        "| Mode | Scale | T | Teacher-class agreement | Logit MSE | Teacher→candidate KL | Prospective projection ADDs |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for mode in MODES:
        for scale in SCALES:
            p = next(x for x in report["profiles"] if x["mode"] == mode and x["scale"] == scale)
            for t in PREFIXES:
                x = p["profiles"][str(t)]
                lines.append(f"| {mode} | {scale} | {t} | {x['teacher_class_agreement']:.9f} | {x['logit_mse']:.12g} | {x['teacher_to_candidate_kl']:.12g} | {x['prospective_projection_adds']} |")
    lines += ["", "Agreement is a fraction, not top-1 accuracy. ADDs are `70 × Σ(4 × node events + edge events)` across coded input sites; this is one prospective arithmetic component, not total compute, latency or energy.", "",
        "## Frozen conditional-control decision", "",
        "| Scale | Agreement ≥ and MSE ≤ at all prefixes | At least one strict fidelity improvement | Projection ADDs ≤ at all prefixes | Shared-θ controls |",
        "|---:|---|---|---|---|"]
    d = report["conditional_controls"]
    for scale in SCALES:
        x = d[str(scale)]
        lines.append(f"| {scale} | {x['all_prefixes_fidelity_non_degrade']} | {x['at_least_one_strict_fidelity_improvement']} | {x['all_prefixes_projection_adds_non_increase']} | {'TRIGGER' if x['shared_threshold_controls_trigger'] else 'does not trigger'} |")
    lines += ["", f"Overall: **{'TRIGGER' if d['overall_shared_threshold_controls_trigger'] else 'does not trigger'}**. Decisions use unrounded JSON values and the conservative all-prefix rule. A trigger only authorizes the four shared-threshold train-only control trajectories in the frozen protocol.", "",
        "## Integrity and scope", "",
        f"All six profiles passed status, coverage (45,000 graphs / 352 batches / final batch 72), node/edge rows ({NODES:,} / {EDGES:,}), prefix, runtime, input-hash, teacher/checkpoint, calibration, author-source, threshold, event-partition, call-count, metric-sum and projection-ADD checks. Calibration SHA-256: `{report['calibration_sha256']}`.", "",
        "The profiles are fixed-order, label-free training-split diagnostics for one frozen seed-41 CIFAR10 GatedGCN teacher. No ground-truth accuracy, validation or test is used. This dense reference does not establish efficient execution, latency, energy or novelty.", "",
        "## Profile JSON hashes", ""]
    for x in report["profiles"]:
        lines.append(f"- `{x['mode']}` scale {x['scale']}: `{x['profile_sha256']}`")
    return "\n".join(lines) + "\n"


def load_all_profiles():
    missing = [str(profile_path(m, s)) for m in MODES for s in SCALES if not profile_path(m, s).is_file()]
    if missing:
        raise AuditError("six-profile audit refused; missing JSON(s): " + ", ".join(missing))
    if OUT_JSON.exists() or OUT_MD.exists():
        raise AuditError("refusing existing audit output; no overwrite")
    if not CAL_PATH.is_file():
        raise AuditError("missing affine_cal1.json")
    cache = {}
    cal = verify_calibration(json.loads(CAL_PATH.read_text(encoding="utf-8")), cache)
    audits, docs = [], {}
    for mode in MODES:
        for scale in SCALES:
            doc = json.loads(profile_path(mode, scale).read_text(encoding="utf-8"))
            docs[(mode, scale)] = doc
            audits.append(validate_profile(doc, mode, scale, cal, cache))
    reference = docs[("generic", 1)]
    for doc in docs.values():
        if (doc["teacher"] != reference["teacher"] or
                doc["calibration_sha256"] != reference["calibration_sha256"] or
                doc["author_source"] != reference["author_source"]):
            raise AuditError("cross-profile teacher/calibration/author-source binding mismatch")
    for scale in SCALES:
        generic_later = docs[("generic", scale)]["threshold_vectors"][2:]
        first_analog = docs[("first_analog", scale)]["threshold_vectors"]
        if generic_later != first_analog:
            raise AuditError(f"first_analog scale {scale} does not reuse generic raw thresholds for layers 1-3")
    return cal, audits


def write_exclusive(report):
    if OUT_JSON.exists() or OUT_MD.exists():
        raise AuditError("refusing existing audit output; no overwrite")
    jtext = json.dumps(report, indent=2, allow_nan=False) + "\n"
    mtext = make_markdown(report)
    made = []
    try:
        with OUT_JSON.open("x", encoding="utf-8") as f:
            f.write(jtext)
        made.append(OUT_JSON)
        with OUT_MD.open("x", encoding="utf-8") as f:
            f.write(mtext)
        made.append(OUT_MD)
    except Exception:
        for path in made:
            try:
                path.unlink()
            except OSError:
                pass
        raise


def synthetic_profile(mode, scale, agreement=2, mse=.001, add_adjust=0):
    names = site_names(mode)
    profiles = {}
    for t in PREFIXES:
        stats = {}
        for name in names:
            rows = 5 if name.endswith(".node") else 9
            nz = rows*WIDTH*t
            stats[name] = {"positive":nz,"negative":0,"zero":0,
                "nonzero_scalar_events":nz,"nonzero_rows":rows*t,
                "sum_abs_graded_amplitude":float(nz),
                "signed_threshold_level_events":{"+2^-0":nz}}
        adds = WIDTH*sum((4 if n.endswith(".node") else 1)*v["nonzero_scalar_events"] for n,v in stats.items())+add_adjust
        agree = max(0,min(3,agreement))
        kl_sum=.3
        profiles[str(t)] = {"prefix_T":t,"graphs":3,"batches":2,"nodes":5,"directed_edges":9,
            "final_batch_graphs":1,"sites":stats,
            "metric_sums":{"agree":agree,"squared_error":mse*30,"kl_sum":kl_sum,"graphs":3},
            "teacher_class_agreement":agree/3,"logit_mse":mse,"teacher_to_candidate_kl":kl_sum/3,
            "prospective_projection_adds":adds,"prefix_logical_active_calls":len(names)*t*2}
    return {"status":"complete","dataset":"CIFAR10","seed":41,"mode":mode,"threshold_scale":scale,
        "scope":"all_45000_official_train_graphs_label_free_fixed_order","executed_T":4,
        "prefix_time_grid":list(PREFIXES),"source_units_per_batch":len(names),
        "actual_source_warmup_calls":len(names)*2,"actual_active_calls":len(names)*4*2,
        "diagnostic_head_calls":6,"normal_endpoint_head_calls":2,
        "source_head_scope":"three diagnostic prefix heads; normal endpoint head once per batch",
        "profiles":profiles,"threshold_vectors":[[float(scale)]*WIDTH for _ in names],
        "author_source":{"commit":PINNED_COMMIT}}


def self_test():
    fake_cal={"calibration":{"channel_thresholds":{}}}
    for i in range(4):
        for k in ("node","edge"):
            fake_cal["calibration"]["channel_thresholds"][f"raw_{k}_{i}"]=[1.]*WIDTH
    for i in range(1,4):
        for k in ("node","edge"):
            fake_cal["calibration"]["channel_thresholds"][f"branch_cumsum_{k}_{i}"]=[1.]*WIDTH
    docs={}
    for scale in SCALES:
        docs[("generic",scale)] = synthetic_profile("generic",scale)
        docs[("first_analog",scale)] = synthetic_profile("first_analog",scale)
        docs[("centered",scale)] = synthetic_profile("centered",scale,agreement=3,mse=.0005)
    audits=[]
    for (mode,scale),doc in docs.items():
        audits.append(validate_profile(doc,mode,scale,fake_cal,{},verify_files=False,
            n=3,batches=2,final=1,nodes=5,edges=9))
    d=conditional_gate(audits)
    if not d["overall_shared_threshold_controls_trigger"]:
        raise AssertionError("synthetic Pareto example should trigger")
    # A single adverse ADD prefix also blocks the trigger.
    by={(a["mode"],a["scale"]):a for a in audits}
    by[("centered",8)]["profiles"]["4"]["prospective_projection_adds"] += 1
    if conditional_gate(audits)["8"]["shared_threshold_controls_trigger"]:
        raise AssertionError("one ADD regression must suppress same-scale trigger")
    bad=docs[("centered",1)]
    bad["profiles"]["2"]["metric_sums"]["agree"]=0
    bad["profiles"]["2"]["teacher_class_agreement"]=0.
    audits=[validate_profile(doc,mode,scale,fake_cal,{},verify_files=False,
        n=3,batches=2,final=1,nodes=5,edges=9) for (mode,scale),doc in docs.items()]
    if d2:=conditional_gate(audits):
        if d2["1"]["shared_threshold_controls_trigger"]:
            raise AssertionError("one bad middle prefix must suppress same-scale trigger")
    corrupt=synthetic_profile("generic",1)["profiles"]["1"]["sites"]["layer0.node"]
    corrupt["zero"] += 1
    try:
        validate_site_stats(corrupt,5,1,"synthetic")
    except AuditError:
        pass
    else:
        raise AssertionError("corrupt scalar partition accepted")
    print("PASS: synthetic metric sums, event partitions, ADD formula, and conservative all-prefix gate")


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test",action="store_true",help="in-memory synthetic checks only")
    args=p.parse_args(argv)
    try:
        if args.self_test:
            self_test()
            return 0
        cal, audits = load_all_profiles()
        report={"status":"complete","scope":"six frozen label-free CIFAR10 seed41 training-only profiles",
            "ground_truth_accuracy_or_labels_used":False,"calibration_sha256":sha256(CAL_PATH),
            "expected_coverage":{"graphs":N,"batches":BATCHES,"final_batch_graphs":FINAL,
                "nodes":NODES,"directed_edges":EDGES,"prefixes":list(PREFIXES)},
            "profiles":audits,"conditional_controls":conditional_gate(audits),
            "qualification":"Teacher-fidelity diagnostics and projected linear ADD counts only; no validation/test, accuracy, novelty, energy or latency claim."}
        write_exclusive(report)
        print(json.dumps({"status":"complete","json":str(OUT_JSON),"markdown":str(OUT_MD),
            "shared_threshold_controls_trigger":report["conditional_controls"]["overall_shared_threshold_controls_trigger"]},indent=2))
        return 0
    except Exception as exc:
        print(f"REFUSED: {exc}",file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
