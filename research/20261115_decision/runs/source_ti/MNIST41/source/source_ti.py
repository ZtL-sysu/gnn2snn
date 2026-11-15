"""Pinned-author threshold initialization (TI) adapter for six graph sites.

Statistics and the 100-step threshold update are executed by the unmodified
author ``ThreHook`` implementation.  The adapter only validates inputs,
records explicit source fallback masks, and applies the author's converter
mapping of negative raw thresholds to zero.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import nn

PINNED_COMMIT = "5e0e4c73f1d0e8e17d81befa24ff947deec1f4be"
EXPECTED_CONFIG = "configs/resnet/resnet34_config_test_channel_mth_diff.yaml"


class SourceTIError(RuntimeError):
    """Input/source failure with a concise machine-readable diagnostic."""


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_author_api(author_repo: Path):
    """Dynamically load the exact pinned threshold_getter module."""
    repo = Path(author_repo).resolve()
    source = repo / "converter" / "threshold_getter.py"
    if not source.is_file():
        raise SourceTIError(f"missing author source file: {source}")
    try:
        result = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SourceTIError(f"cannot verify author git revision at {repo}: {exc}") from exc
    commit = result.stdout.strip()
    if commit != PINNED_COMMIT:
        raise SourceTIError(f"author repository revision mismatch: expected {PINNED_COMMIT}, found {commit}")
    tracked_source_files = [
        "converter/threshold_getter.py", "main.py", "converter/converter.py", EXPECTED_CONFIG,
    ]
    dirty = subprocess.run(["git", "-C", str(repo), "diff", "--quiet", "HEAD", "--", *tracked_source_files],
                           capture_output=True, text=True)
    if dirty.returncode != 0:
        raise SourceTIError("pinned TI/config/converter source files have local tracked modifications")

    repo_text = str(repo)
    inserted = repo_text not in sys.path
    if inserted:
        sys.path.insert(0, repo_text)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    module_name = "_pinned_gnn2snn_threshold_getter_5e0e4c73"
    old_utils = sys.modules.get("utils")
    try:
        module = sys.modules.get(module_name)
        if module is None:
            # Pinned threshold_getter.py imports the repository's broad utils
            # package at module scope, but never references it in this file.
            # Give that unused import an explicit empty namespace to prevent
            # unrelated utility side effects/name collisions; TI functions and
            # ThreHook remain the original dynamically imported source objects.
            sys.modules["utils"] = types.ModuleType("utils")
            spec = importlib.util.spec_from_file_location(module_name, source)
            if spec is None or spec.loader is None:
                raise SourceTIError(f"cannot create import spec for {source}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
    except Exception as exc:
        sys.modules.pop(module_name, None)
        raise SourceTIError(f"failed importing pinned threshold_getter.py: {exc}") from exc
    finally:
        if old_utils is None:
            sys.modules.pop("utils", None)
        else:
            sys.modules["utils"] = old_utils
        sys.dont_write_bytecode = old_dont_write_bytecode
        if inserted:
            try:
                sys.path.remove(repo_text)
            except ValueError:
                pass
    return module, commit, source


def _config_evidence(repo: Path) -> dict[str, Any]:
    """Record source defaults relevant to TI and conversion scale semantics."""
    config_path = repo / EXPECTED_CONFIG
    main_path = repo / "main.py"
    converter_path = repo / "converter" / "converter.py"
    config = config_path.read_text(encoding="utf-8") if config_path.is_file() else ""
    main = main_path.read_text(encoding="utf-8") if main_path.is_file() else ""
    converter = converter_path.read_text(encoding="utf-8") if converter_path.is_file() else ""

    def active_value(key: str, default: str | None = None) -> str | None:
        for line in config.splitlines():
            stripped = line.strip()
            if stripped.startswith("#") or ":" not in stripped:
                continue
            lhs, rhs = stripped.split(":", 1)
            if lhs.strip() == key:
                return rhs.split("#", 1)[0].strip().strip("'\"")
        return default

    level_default = "layer"
    if "default='layer'" not in main and 'default="layer"' not in main:
        level_default = "unknown; inspect pinned main.py"
    return {
        "config_path": EXPECTED_CONFIG if config else None,
        "config_sha256": _sha256(config_path) if config_path.is_file() else None,
        "threshold_mode": active_value("threshold_mode"),
        "threshold_level_config_value": active_value("threshold_level"),
        "threshold_level_cli_default": level_default,
        "neuron_name": active_value("neuron_name"),
        "num_thresholds": active_value("num_thresholds"),
        "threshold_scale_config_value": active_value("threshold_scale"),
        "conversion_scale_semantics": "for ResNet MTH converter, raw ThreHook.scale is clamped by scale>=0 and passed directly as neuron thresh; threshold_scale is not read by converter.py",
        "converter_source_sha256": _sha256(converter_path) if converter_path.is_file() else None,
        "adaptation_note": "this graph protocol explicitly uses channel level; the pinned ImageNet example config defaults to 4 thresholds and threshold_scale 1, so 8-level graph use is an adaptation, not a literal ImageNet endpoint reproduction",
    }


def fit_source_ti(
    preactivation_batches: list[list[torch.Tensor]],
    effective_steps: int,
    author_repo: Path,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Fit six per-channel TI thresholds from cached teacher preactivations.

    Args:
        preactivation_batches: outer list is calibration batch order; each
            inner list has six CPU tensors shaped ``[N,C]``.  For node sites N
            is the number of nodes in that batch; for readout sites N is the
            number of graphs.  This exactly matches source ``level='channel'``
            moments and its ``num_batches_tracked += x.shape[0]`` weighting.
        effective_steps: source ``calculate_better(..., n=...)`` argument,
            typically horizon * 2**levels, decided by the caller.
        author_repo: pinned author checkout.

    Returns:
        Six channel vectors after the source converter's finite negative to
        zero mapping, plus JSON-serializable raw moments, raw source thresholds,
        source fallback masks, and provenance.  No epsilon or clipping repair
        is applied beyond the pinned converter's ``scale >= 0`` rule.
    """
    if not isinstance(effective_steps, int) or isinstance(effective_steps, bool) or effective_steps <= 0:
        raise SourceTIError(f"effective_steps must be a positive integer, got {effective_steps!r}")
    if not preactivation_batches:
        raise SourceTIError("preactivation_batches is empty")

    api, commit, source_path = load_author_api(author_repo)
    hooks = [api.ThreHook(mode="var", level="channel", out_layer=nn.ReLU()) for _ in range(6)]
    channel_counts: list[int | None] = [None] * 6
    batch_counts = [0] * 6
    row_counts = [0] * 6
    for batch_i, batch in enumerate(preactivation_batches):
        if not isinstance(batch, (list, tuple)) or len(batch) != 6:
            raise SourceTIError(f"batch {batch_i} must contain exactly six preactivation tensors")
        for site, x in enumerate(batch):
            if not isinstance(x, torch.Tensor) or x.ndim != 2:
                shape = getattr(x, "shape", None)
                raise SourceTIError(f"batch {batch_i} site {site} must be Tensor[N,C], got {shape}")
            if x.device.type != "cpu":
                raise SourceTIError(f"batch {batch_i} site {site} must be CPU cached tensor, got {x.device}")
            if not x.is_floating_point() or x.shape[0] <= 0 or x.shape[1] <= 0:
                raise SourceTIError(f"batch {batch_i} site {site} must be nonempty floating Tensor[N,C]")
            if not bool(torch.isfinite(x).all()):
                raise SourceTIError(f"batch {batch_i} site {site} contains nonfinite preactivations")
            c = int(x.shape[1])
            if channel_counts[site] is None:
                channel_counts[site] = c
            elif channel_counts[site] != c:
                raise SourceTIError(f"site {site} channel count changed {channel_counts[site]} -> {c}")
            with torch.no_grad():
                hooks[site](x)
            batch_counts[site] += 1
            row_counts[site] += int(x.shape[0])

    thresholds: list[torch.Tensor] = []
    site_meta: list[dict[str, Any]] = []
    for site, hook in enumerate(hooks):
        if hook.mean is None or hook.var is None:
            raise SourceTIError(f"site {site} has no source moments")
        mean = hook.mean.detach().clone()
        var = hook.var.detach().clone()
        if not bool(torch.isfinite(mean).all()) or not bool(torch.isfinite(var).all()):
            raise SourceTIError(f"site {site} source moments are nonfinite")

        # Capture exactly the source method's terminal NaN fallback without
        # changing calculate_better or the 100-iteration scale arithmetic.
        original_calculate = api.calculate_better
        nan_fallback = torch.zeros_like(mean, dtype=torch.bool)

        def tracked_calculate(m, v, scale, n=64):
            factor = original_calculate(m, v, scale, n=n)
            candidate = scale * factor
            nan_fallback.logical_or_(torch.isnan(candidate))
            return factor

        api.calculate_better = tracked_calculate
        try:
            with torch.no_grad():
                hook.get_scale_from_var(T=effective_steps)
        finally:
            api.calculate_better = original_calculate

        raw = hook.scale.detach().clone()
        if not bool(torch.isfinite(raw).all()):
            raise SourceTIError(
                f"site {site} source TI ended nonfinite after its NaN->initial-1 fallback; "
                f"nan_fallback_mask={nan_fallback.reshape(-1).tolist()}, final={raw.reshape(-1).tolist()}"
            )
        if not bool(torch.equal(hook.num_batches_tracked.cpu(), torch.tensor(row_counts[site]))):
            raise SourceTIError(f"site {site} source observation count mismatch")

        raw_vec = raw.reshape(-1)
        fallback_vec = nan_fallback.reshape(-1)
        negative_mask = raw_vec < 0
        zero_mask = raw_vec == 0
        # Converter source: thre = scale * (scale >= 0).float(). Preserve raw
        # values in metadata and return its exact resulting values, including 0.
        converted = raw_vec * (raw_vec >= 0).to(raw_vec.dtype)
        thresholds.append(converted.expand(int(channel_counts[site])).clone())
        site_meta.append({
            "site": site,
            "channels": int(channel_counts[site]),
            "source_level": "channel",
            "source_moment_axis": "dim=0, keepdim=True for [N,C]",
            "batch_count": batch_counts[site],
            "source_num_batches_tracked_rows": int(hook.num_batches_tracked.item()),
            "total_scalar_values": int(row_counts[site] * int(channel_counts[site])),
            "mean": mean.reshape(-1).tolist(),
            "variance_population_unbiased_false": var.reshape(-1).tolist(),
            "raw_source_threshold_after_100_iterations": raw_vec.tolist(),
            "source_nan_fallback_mask": fallback_vec.tolist(),
            "source_nan_fallback_count": int(fallback_vec.sum().item()),
            "raw_negative_mask": negative_mask.tolist(),
            "raw_negative_count": int(negative_mask.sum().item()),
            "raw_zero_mask": zero_mask.tolist(),
            "raw_zero_count": int(zero_mask.sum().item()),
            "converter_threshold_after_negative_to_zero_mapping": converted.tolist(),
        })

    source_sha = _sha256(source_path)
    metadata: dict[str, Any] = {
        "status": "ok",
        "author_commit": commit,
        "threshold_getter_path": str(source_path),
        "threshold_getter_sha256": source_sha,
        "source_method": "unmodified ThreHook(mode='var', level='channel').forward moments and ThreHook.get_scale_from_var",
        "source_calculate_better_sha256": source_sha,
        "effective_steps_n": effective_steps,
        "scale_iterations": 100,
        "initial_scale": 1.0,
        "source_nan_policy": "after 100 source iterations, ThreHook replaces NaN scale entries with the initial scale (1.0); mask/count are instrumented without changing source arithmetic",
        "converter_nonpositive_policy": "pinned converter maps finite negative scale to zero; zero remains zero; no epsilon repair",
        "config_evidence": _config_evidence(Path(author_repo).resolve()),
        "sites": site_meta,
    }
    return thresholds, metadata
