"""R7/v2 regression test: reproduce the v1 subset-mask crash and verify the fix.

The failed v1 GPU run (logs/e5-falsifier-failure-v1/) raised:
  RuntimeError: The size of tensor a (9278) must match the size of tensor b
  (15019) at non-singleton dimension 0
at Acc.add() because a per-element tensor `tol` was not sliced with the same
boolean mask as `actual`/`bound`.

This test is CPU-only, writes no report, and touches no published artifact.
It loads v1 and v2 by file path so the published v1 bytes stay untouched.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parent


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, str(CODE / filename))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    v1 = load("e5_falsifier_v1_under_test", "e5_falsifier_v1.py")
    v2 = load("e5_falsifier_v2_under_test", "e5_falsifier_v2.py")

    torch.manual_seed(0)
    n_nodes, channels = 15019, 3
    actual = torch.rand(n_nodes, channels)
    bound = torch.rand(n_nodes, channels) + 1.0
    tol_abs, tol_rel = v2.TOL_ABS, v2.TOL_REL
    tol = tol_abs + tol_rel * bound            # full-length per-element tensor
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[:9278] = True                         # exactly the failing split

    # 1. v1 reproduces the observed crash (kept as evidence, not "fixed").
    try:
        v1.Acc().add(actual, bound, tol, mask=mask)
    except RuntimeError as exc:
        assert "must match the size of tensor" in str(exc), str(exc)
        print("v1_REPRODUCED", str(exc).split("\n")[0])
    else:
        raise AssertionError("v1 unexpectedly passed; defect no longer reproducible")

    # 2. v2 no longer crashes and computes the identical well-defined quantity.
    a2 = v2.Acc()
    a2.add(actual, bound, tol, mask=mask)
    a_sel, b_sel, t_sel = actual[mask], bound[mask], tol[mask]
    expected_excess = float((a_sel - (b_sel + t_sel)).clamp(min=0).max().item())
    expected_violations = int((a_sel > b_sel + t_sel).sum().item())
    assert a2.count == int(mask.sum().item()) * channels, a2.count
    assert a2.violations == expected_violations, (a2.violations, expected_violations)
    assert abs(a2.max_excess - expected_excess) < 1e-9, (a2.max_excess, expected_excess)
    print("v2_MASKED_TENSOR_TOL_PASS", {"count": a2.count,
                                       "violations": a2.violations,
                                       "max_excess": a2.max_excess})

    # 3. The other call shapes must be unchanged.
    s = v2.Acc()
    s.add(actual, bound, tol)                  # no mask, tensor tol
    assert s.count == n_nodes * channels
    g = v2.Acc()
    g.add(actual[:10, :1], bound[:10, :1], tol_abs)   # scalar tol
    assert g.count == 10
    print("v2_OTHER_CALL_SHAPES_PASS")

    # 4. Scientific parameters are byte-for-byte identical between versions.
    for attr in ("TIMESTEPS", "TOL_ABS", "TOL_REL", "PROTOCOL_SHA",
                 "ADDENDUM_SHA", "DCGS_COMMIT", "CSV_COLUMNS"):
        assert getattr(v1, attr) == getattr(v2, attr), attr
    print("v2_PARAMETERS_IDENTICAL_PASS")


if __name__ == "__main__":
    main()
