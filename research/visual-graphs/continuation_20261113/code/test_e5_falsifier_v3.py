"""R7b/v3 regression test: reproduce BOTH historical E5 falsifier defects.

v1 (bytes bound by the R6c manifest) crashed at the first subset
accumulation:
  RuntimeError: The size of tensor a (9278) must match the size of tensor b
  (15019) at non-singleton dimension 0
because a per-element tensor `tol` was not sliced with the same boolean mask
as `actual`/`bound`.  v2 fixed that but crashed AFTER all 40 GPU batches:
  AttributeError: 'Acc' object has no attribute 'items'
because its nested dump_acc() unpacked two nested layers although the caller
already passes ONE {subset: Acc} layer.

This test is CPU-only, writes no report, and touches no published artifact.
It loads v1/v2/v3 by file path so the manifest-bound v1/v2 bytes stay
untouched.
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
    v1 = load("e5_falsifier_v1_under_test_v3", "e5_falsifier_v1.py")
    v2 = load("e5_falsifier_v2_under_test_v3", "e5_falsifier_v2.py")
    v3 = load("e5_falsifier_v3_under_test_v3", "e5_falsifier_v3.py")

    # --- defect 1: v1 mask / tensor-tol mismatch -------------------------
    torch.manual_seed(0)
    n_nodes, channels = 15019, 3
    actual = torch.rand(n_nodes, channels)
    bound = torch.rand(n_nodes, channels) + 1.0
    tol_abs, tol_rel = v3.TOL_ABS, v3.TOL_REL
    tol = tol_abs + tol_rel * bound
    mask = torch.zeros(n_nodes, dtype=torch.bool)
    mask[:9278] = True                         # exactly the failing split
    try:
        v1.Acc().add(actual, bound, tol, mask=mask)
    except RuntimeError as exc:
        assert "must match the size of tensor" in str(exc), str(exc)
        print("v1_DEFECT1_REPRODUCED", str(exc).split("\n")[0])
    else:
        raise AssertionError("v1 unexpectedly passed; defect no longer reproducible")

    a3 = v3.Acc()
    a3.add(actual, bound, tol, mask=mask)
    a_sel, b_sel, t_sel = actual[mask], bound[mask], tol[mask]
    assert a3.count == int(mask.sum().item()) * channels
    assert a3.violations == int((a_sel > b_sel + t_sel).sum().item())
    assert abs(a3.max_excess -
               float((a_sel - (b_sel + t_sel)).clamp(min=0).max().item())) < 1e-9
    s = v3.Acc()
    s.add(actual, bound, tol)                  # no mask, tensor tol
    assert s.count == n_nodes * channels
    g = v3.Acc()
    g.add(actual[:10, :1], bound[:10, :1], tol_abs)   # scalar tol
    assert g.count == 10
    print("v3_DEFECT1_FIXED_AND_OTHER_CALL_SHAPES_PASS")

    # --- defect 2: v2 report serialisation -------------------------------
    accum_v2 = v2.new_accum()
    one_layer = accum_v2["raw_vs_decoded"]     # exactly what run() passes
    assert set(one_layer) == {"all", "degree_low", "degree_high",
                              "sharpness_low", "sharpness_high"}
    try:
        # verbatim v2 nested dump_acc expression (kept local inside v2.run)
        {k: {s: x.as_dict() for s, x in sub.items()} for k, sub in one_layer.items()}
    except AttributeError as exc:
        assert "has no attribute 'items'" in str(exc), str(exc)
        print("v2_DEFECT2_REPRODUCED", str(exc))
    else:
        raise AssertionError("v2 serialisation unexpectedly passed; defect not reproducible")

    dumped = v3.dump_acc(v3.new_accum()["raw_vs_decoded"])
    assert set(dumped) == set(one_layer)
    for subset, payload in dumped.items():
        assert isinstance(payload, dict), subset
        assert set(payload) == {"elements", "bound_violations", "max_actual",
                                "max_bound", "max_excess_over_bound",
                                "max_pre_bn_layer_error"}, subset
    # the exact report-building comprehensions from run() must now succeed
    accum = {str(T): v3.new_accum() for T in v3.TIMESTEPS}
    gate_bound = {str(T): v3.dump_acc(accum[str(T)]["raw_vs_decoded"])
                  for T in v3.TIMESTEPS}
    stale_bound = {str(T): v3.dump_acc(accum[str(T)]["fresh_vs_stale"])
                   for T in v3.TIMESTEPS}
    assert len(gate_bound) == len(stale_bound) == len(v3.TIMESTEPS)
    # two-layer shape tolerance (defensive; schema unchanged)
    two_layer = v3.dump_acc(v3.new_accum())
    assert set(two_layer) == {"raw_vs_decoded", "fresh_vs_stale"}
    assert set(two_layer["raw_vs_decoded"]) == set(one_layer)
    print("v3_DEFECT2_FIXED_AND_REPORT_SCHEMA_PASS")

    # --- scientific parameters unchanged across all three versions -------
    for attr in ("TIMESTEPS", "TOL_ABS", "TOL_REL", "PROTOCOL_SHA",
                 "ADDENDUM_SHA", "DCGS_COMMIT", "CSV_COLUMNS", "OUT"):
        assert getattr(v1, attr) == getattr(v2, attr) == getattr(v3, attr), attr
    # the v2 fix must still be present in v3 (no regression on defect 1)
    assert "torch.is_tensor(tol)" in (CODE / "e5_falsifier_v3.py").read_text(encoding="utf-8")
    print("v1_V2_V3_PARAMETERS_IDENTICAL_PASS")


if __name__ == "__main__":
    main()
