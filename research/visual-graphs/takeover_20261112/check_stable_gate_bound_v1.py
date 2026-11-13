"""Reproducible synthetic checks for stable-normalized-gate-bound-v1.md.

CPU-only, NumPy-only. Does not access project data, models, checkpoints, or GPU.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform

import numpy as np

SEED = 20261113
RANDOM_CASES = 20_000
TOL = 2e-12
HERE = Path(__file__).resolve().parent
REPORT = HERE / "stable_gate_bound_check_v1.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _log_sigmoid(z: np.ndarray) -> np.ndarray:
    # Stable for both very negative and very positive float64 logits.
    return -np.logaddexp(0.0, -np.asarray(z, dtype=np.float64))


def _probabilities(logits: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Normalized sigmoid weights and dummy mass, computed in log domain."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.size == 0 and eps <= 0:
        raise ValueError("empty-neighbor aggregate is undefined at eps=0")
    logw = _log_sigmoid(logits)
    if eps > 0:
        logw = np.concatenate((logw, np.array([math.log(eps)], dtype=np.float64)))
    if logw.size == 0:
        raise ValueError("no positive total weight")
    shifted = np.exp(logw - np.max(logw))
    p = shifted / shifted.sum(dtype=np.float64)
    return p, logw


def _messages(logits: np.ndarray, eps: float, b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    p, _ = _probabilities(logits, eps)
    b = np.asarray(b, dtype=np.float64)
    if eps > 0:
        b = np.concatenate((b, np.array([0.0], dtype=np.float64)))
    return float(np.dot(p, b)), p, b


def _check_case(name: str, z: np.ndarray, zp: np.ndarray, b: np.ndarray,
                bp: np.ndarray, eps: float, eta: float,
                failures: list[dict], margins: dict[str, tuple[float, str]]) -> None:
    z = np.asarray(z, dtype=np.float64)
    zp = np.asarray(zp, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    bp = np.asarray(bp, dtype=np.float64)
    if z.shape != zp.shape or z.shape != b.shape or b.shape != bp.shape:
        raise ValueError(f"{name}: shape mismatch")
    if eta < 0 or not np.isfinite(eta) or eps < 0 or not np.isfinite(eps):
        raise ValueError(f"{name}: invalid eta/epsilon")
    if z.size and np.max(np.abs(zp - z)) > eta + TOL:
        raise ValueError(f"{name}: declared eta does not cover the logit perturbation")
    if z.size == 0 and eps <= 0:
        # Deliberate undefined-domain case; this is reported separately.
        return

    m, p, b_aug = _messages(z, eps, b)
    mp, pp, bp_aug = _messages(zp, eps, bp)
    tv = 0.5 * float(np.abs(pp - p).sum(dtype=np.float64))
    tv_bound = math.tanh(eta / 2.0)
    if bp_aug.size:
        db = float(np.max(np.abs(bp_aug - b_aug)))
        value_range = float(max(0.0, np.max(bp_aug)) - min(0.0, np.min(bp_aug)))
    else:
        db = value_range = 0.0
    msg_error = abs(mp - m)
    msg_bound = db + value_range * tv_bound
    tv_margin = tv_bound - tv
    msg_margin = msg_bound - msg_error
    margins["tv"] = min((tv_margin, name), margins.get("tv", (math.inf, "")), key=lambda x: x[0])
    margins["message"] = min((msg_margin, name), margins.get("message", (math.inf, "")), key=lambda x: x[0])
    tv_tol = TOL * max(1.0, tv, tv_bound)
    msg_tol = TOL * max(1.0, msg_error, msg_bound)
    if tv_margin < -tv_tol:
        failures.append({"case": name, "bound": "tv", "observed": tv, "upper": tv_bound,
                         "margin": tv_margin, "tolerance": tv_tol})
    if msg_margin < -msg_tol:
        failures.append({"case": name, "bound": "message", "observed": msg_error, "upper": msg_bound,
                         "margin": msg_margin, "tolerance": msg_tol})


def _likelihood_ratio_extremal(eta: float) -> tuple[float, float]:
    """Exact two-point likelihood-ratio extremizer for the TV lemma."""
    if eta == 0:
        return 0.0, 0.0
    # Baseline mass is chosen so E_p[r]=1 for r in {exp(-eta), exp(eta)}.
    p_hi = 1.0 / (math.exp(eta) + 1.0)
    p = np.array([1.0 - p_hi, p_hi], dtype=np.float64)
    r = np.array([math.exp(-eta), math.exp(eta)], dtype=np.float64)
    q = p * r
    q /= q.sum()
    tv = 0.5 * float(np.abs(q - p).sum())
    bound = math.tanh(eta / 2.0)
    return tv, bound


def main() -> int:
    rng = np.random.default_rng(SEED)
    failures: list[dict] = []
    margins: dict[str, tuple[float, str]] = {}
    eta_grid = [0.0, 1e-12, 1e-8, 1e-4, 0.1, 1.0, 10.0, 40.0]
    boundary_checks = 0

    # Explicit eta boundaries, signed messages, positive dummy mass, eps=0,
    # and very small sigmoid gates. These stress the bound endpoints.
    for j, eta in enumerate(eta_grid):
        z = np.array([-650.0, -649.5, 0.0, 649.0], dtype=np.float64)
        signs = np.array([-1.0, 1.0, -1.0, 1.0])
        direction = np.array([-1.0, 1.0, 0.5, -0.5])
        zp = z + eta * direction
        b = np.array([-3.0, 0.25, 5.0, -1.5]) * signs
        bp = b + np.array([0.5, -0.75, 1.0, -2.0])
        eps = 0.0 if j % 2 else 1e-300
        _check_case(f"eta_boundary_{eta:g}_eps_{eps:g}", z, zp, b, bp, eps, eta,
                    failures, margins)
        boundary_checks += 1
        tv, tv_bound = _likelihood_ratio_extremal(eta)
        if abs(tv - tv_bound) > TOL * max(1.0, tv_bound):
            failures.append({"case": f"likelihood_ratio_extremal_{eta:g}",
                             "bound": "tv_extremal_equality", "observed": tv,
                             "upper": tv_bound, "margin": tv_bound - tv})
        boundary_checks += 1

    # Explicit empty-neighbor semantics: eps>0 gives exactly zero aggregate;
    # eps=0 has no defined quotient and is recorded as out of theorem scope.
    for eta in eta_grid:
        _check_case(f"empty_positive_eps_eta_{eta:g}", np.array([]), np.array([]),
                    np.array([]), np.array([]), 1e-300, eta, failures, margins)
        boundary_checks += 1

    # Random logits include tiny/near-saturated sigmoid gates. Message values
    # are signed and the dummy message is always exactly zero when eps>0.
    for i in range(RANDOM_CASES):
        n = int(rng.integers(1, 26))
        eta = float(10.0 ** rng.uniform(-12.0, 1.6))
        z = rng.uniform(-650.0, 650.0, size=n).astype(np.float64)
        zp = z + rng.uniform(-eta, eta, size=n)
        b = rng.normal(0.0, 5.0, size=n)
        bp = b + rng.normal(0.0, 10.0 ** rng.uniform(-12.0, 2.0), size=n)
        eps = 0.0 if i % 5 == 0 else float(10.0 ** rng.uniform(-300.0, 5.0))
        _check_case(f"random_{i}", z, zp, b, bp, eps, eta, failures, margins)

    # Random positive-eps empty sets and eps=0 empty undefined case coverage.
    empty_eps_zero_undefined = True
    try:
        _probabilities(np.array([], dtype=np.float64), 0.0)
        empty_eps_zero_undefined = False
    except ValueError:
        pass
    if not empty_eps_zero_undefined:
        failures.append({"case": "empty_eps_zero", "bound": "undefined_case_guard"})

    result = {
        "status": "pass" if not failures else "fail",
        "check": "normalized_sigmoid_gate_tv_and_message_bound",
        "seed": SEED,
        "random_cases": RANDOM_CASES,
        "boundary_checks": boundary_checks,
        "total_defined_cases": RANDOM_CASES + boundary_checks,
        "float_dtype": "numpy.float64",
        "epsilon_cases": ["zero with nonempty incoming set", "1e-300 dummy mass", "random 1e-300..1e5"],
        "logit_cases": ["eta grid 0..40", "random logits in [-650,650]", "tiny and near-saturated sigmoid gates"],
        "message_cases": ["signed", "perturbed signed", "dummy message zero", "empty neighbor with positive epsilon"],
        "empty_eps_zero": "undefined unless caller separately defines aggregate as zero; checked as guarded out-of-scope case",
        "recurrence_scope": {
            "applies_to": ["same incoming edge set", "positive sigmoid gates", "fixed positive epsilon or nonempty set at epsilon=0", "eval-mode fixed weights/BN", "identity raw residual skips", "real-arithmetic error envelopes"],
            "local_decoder_error": "kappa bounds decoder minus its own converted raw target; consumer-input envelope is delta_raw+abs(kappa)",
            "message_range": "converted B(h) messages plus zero dummy; no teacher hidden state required",
            "not_certified": ["floating-point sigmoid/underflow/accumulation", "source neuron threshold decisions", "readout-head class margin", "runtime or energy"],
        },
        "worst_tv_margin": {"value": margins.get("tv", (None, None))[0], "case": margins.get("tv", (None, None))[1]},
        "worst_message_margin": {"value": margins.get("message", (None, None))[0], "case": margins.get("message", (None, None))[1]},
        "failure_count": len(failures),
        "failures": failures,
        "runtime": {"python": platform.python_version(), "numpy": np.__version__, "device": "cpu", "official_data_loaded": False, "gpu_used": False},
        "script_sha256": _sha256(Path(__file__).resolve()),
    }
    REPORT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "random_cases", "boundary_checks", "worst_tv_margin", "worst_message_margin", "failure_count", "script_sha256")}, sort_keys=True))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())

