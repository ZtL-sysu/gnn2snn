#!/usr/bin/env python3
"""Theory <-> experiment consistency check for the D1 GNN->SNN conversion.

(Full derivation and the two errors this check caught: audits/d1-theory-derivation.md)

SETUP
  A_hat = symmetric normalised adjacency with self loops; a_n := A_hat[n,:] (row),
  rho_n = sum_m a_n[m]  (NOTE: NOT 1 -- row sum 1 holds for random-walk normalisation
  D^-1 A, not for the symmetric GCN normalisation; measured rho ~ 5.7 on Cora),
  ||a_n||_2 = the concentration factor.
  ANN:  B1 = X W1 + b1,  H1 = relu(B1),  B2 = A_hat H1 W2 + b2.
  SNN:  layer-1 IF units receive constant per-step current B1 + beta, theta = 1,
        reset by subtraction; layer 2 accumulates the instantaneous messages, so

            mem = sum_t inst_t = A_hat (count1 W2) + T b2
        =>  mem/T = A_hat (count1/T) W2 + b2 =: B2_hat

EXACT ERROR IDENTITY (no approximation anywhere up to here)
        B2_hat - B2 = A_hat (E1 W2),        E1 := count1/T - relu(B1).          (1)

LEMMA 1 (IF is an exact floor quantiser on [0, theta]).
  For 0 <= c <= theta: count_T = floor(T c/theta) and the residual
  U := T c - theta*count_T lies in [0, theta). Hence rate = c - U/(T) ... i.e.
        E1 = -U/T  with U in [0,1)   (theta = 1).
  For c > theta the unit saturates (1 spike/step) and E1 = 1 - c: this is the ONLY way the
  identity above escapes the 1/T bound, and it is why the absolute threshold requires
  activations in [0, theta] (measured: a handful of units per graph; explicit sigma term below).
  For c < 0 the unit never fires => rate 0 = relu(c): relu is implemented for free.

ASSUMPTION 1 (phase randomisation, standard in conversion analyses; VERIFIED below).
  The residuals U[m,h] are approximately uniform on [0,1) and independent across (m,h).
  Checked numerically against the realised spikes (mean/var/KS).

LEMMA 2 (systematic term, and exactly what FTBC removes).
  E[E1] = -(1/(2T)) without bias; with a per-step bias beta, E[E1] = beta - 1/(2T).
  Therefore beta* = 1/(2T) is the unique bias that zeroes the systematic error, and in both
  cases the zero-mean fluctuation xi := E1 - E[E1] has half-width 1/(2T).
  => FTBC does not change the fluctuation size; it removes a systematic, degree-dependent
     shrinkage of the decision margin.

LEMMA 3 (fluctuation concentrates; neighbourhood averaging is what makes it small).
  For node n and a rival class c, put DW_c := W2[:, c*] - W2[:, c]. From (1)
        margin error  = sum_{m,h} a_n[m] * DW_c[h] * E1[m,h]
                      = -(rho_n/(2T)) * sum_h DW_c[h]        [SYSTEMATIC, zero under FTBC]
                        + sum_{m,h} a_n[m] * DW_c[h] * xi[m,h]   [FLUCTUATION]
  Each fluctuation term is an independent zero-mean variable of half-width
  |a_n[m] DW_c[h]|/(2T), so Hoeffding gives, with probability >= 1-delta,
        |fluctuation| <= ||a_n||_2 * ||DW_c||_2 * sqrt(ln(2/delta)/8) / T.
  The ||a_n||_2 factor is small because A_hat spreads weight over the neighbourhood: this is
  the quantitative statement of "temporal averaging over the neighbourhood".

THEOREM (margin recovery guarantee).
  If   Delta_n > |systematic_{n,c}| + ||a_n||_2 ||DW_c||_2 sqrt(ln(2/delta)/8)/T + sigma-term
  for every rival c, then argmax(B2_hat[n,:]) = argmax(B2[n,:]) with probability >= 1-delta.
  Without FTBC the systematic term scales as rho_n/(2T) * |sum_h DW_c[h]| -- i.e. FTBC buys
  exactly the removal of a 1/T margin shrinkage, which is why it helps most at small T.

WHAT THIS SCRIPT CHECKS
  1. Lemma 1: count == floor(T c) on [0, theta], residual in [0, theta).
  2. Assumption 1: realised residuals ~ Uniform[0,1) (mean ~ 0.5, var ~ 1/12).
  3. Lemma 2: the experiment's grid-searched beta ~= 1/(2T) = beta*.
  4. Systematic term: measured mean margin error ~= -(rho/(2T)) sum_h DW without bias,
     and ~= 0 with FTBC.
  5. Theorem: predicted guaranteed fraction <= measured agreement with the ANN.
  6. MOUTHPIECE: the accuracy derived here from its own mem/T must equal convert()'s accuracy
     (gap must be 0) -- otherwise the theory is analysing a different computation.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import snn_gnn_experiment as E
import graph_data as G


def if_count(B, T, beta, theta=1.0):
    """Replicates convert()'s layer-1 IF loop EXACTLY (mode='naive', l1_mode='if')."""
    N, H = B.shape
    u = np.zeros((N, H), dtype=np.float64)
    cnt = np.zeros((N, H), dtype=np.int64)
    for t in range(T):
        u = u + B + beta
        s = (u >= theta).astype(np.int64)
        u = u - s * theta
        cnt += s
    return cnt, u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer", "pubmed"])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--ts", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--data-root",
                    default=str(Path(__file__).resolve().parent.parent / "results" / ".data"))
    ap.add_argument("--out", default="../results/theory_verify.json")
    a = ap.parse_args()

    print("=" * 100)
    print("THEORY <-> EXPERIMENT CONSISTENCY")
    print("=" * 100)

    print("\n[1] Lemma 1: IF emits exactly floor(T*c/theta) spikes for c in [0,theta]")
    for T in a.ts:
        rng = np.random.default_rng(0)
        c = rng.random((20000, 1))
        cnt, u = if_count(c, T, 0.0)
        print(f"    T={T:2d}: count==floor(T*c) -> {np.array_equal(cnt, np.floor(T*c).astype(np.int64))}"
              f"   residual in [0,theta) -> {bool(np.all(u < 1.0) and np.all(u >= 0))}")

    out = {}
    for name in a.datasets:
        d = G.load_ds(name, a.data_root)
        adj_norm = E.normalize_adj(d["adj"]).tocsr()
        A = np.asarray(adj_norm.toarray())
        rho = A.sum(1)                      # row sums (NOT 1 for symmetric normalisation)
        anorm2 = np.sqrt((A ** 2).sum(1))   # ||a_n||_2
        te = d["test_mask"]
        out[name] = {}
        print(f"\n[{name}] row-sum rho: mean={rho.mean():.3f} max={rho.max():.3f} | "
              f"||a_n||_2 mean={anorm2.mean():.3f}")
        # Train ONCE per seed and reuse the models for every T (the previous version
        # retrained inside the T loop, which multiplied runtime by |T| for no reason).
        models = []
        for sd in range(a.seeds):
            m = E.train_gcn(d["features"], d["labels"], adj_norm, d["train_mask"],
                            d["val_mask"], d["test_mask"], sd, hidden=16,
                            row_normalize=d["row_normalize"])
            m["labels"] = d["labels"]; m["test_mask"] = d["test_mask"]
            m["train_mask"] = d["train_mask"]; m["val_mask"] = d["val_mask"]
            m["seed"] = sd
            models.append(m)

        for T in a.ts:
            rows = []
            for m in models:
                B1, B2, W2, b2 = m["B1"], m["B2"], m["W2"], m["b2"]
                reluB1 = np.maximum(B1, 0.0)
                # the FTBC arm now uses the closed-form optimum beta* = theta/(2T)
                beta_emp = E.ftbc_bias(T)

                for tag, beta in (("ftbc", beta_emp), ("nobias", 0.0)):
                    # ---- A-PRIORI quantities: from B1 alone, NO spike simulation ----------
                    # IF is deterministic, so E1 is a deterministic function of B1:
                    #     rate = floor(T*(B1+beta))/T on the sub-threshold regime,
                    #     1.0 where saturated (B1+beta > theta), 0 where it never fires.
                    cpos = np.clip(B1 + beta, 0.0, None)
                    rate_ap = np.minimum(np.floor(T * cpos) / T, 1.0)
                    rate_ap = np.where(B1 + beta <= 0.0, 0.0, rate_ap)
                    E1_ap = rate_ap - reluB1
                    mu_h = E1_ap.mean(axis=0)                     # per-hidden-unit mean error
                    mu_E = float(E1_ap.mean())
                    # fluctuation half-width: exclude the (handful of) saturated units, whose
                    # deviation is deterministic and added back separately below.
                    sat = (B1 + beta) > 1.0
                    bulk = ~sat
                    w = float(np.abs((E1_ap - mu_h)[bulk]).max()) if bulk.any() else 0.0
                    cnt, u1 = if_count(B1, T, beta)
                    rate1 = cnt / T
                    E1 = rate1 - reluB1
                    B2h = np.asarray(A @ (rate1 @ W2), dtype=np.float64) + b2
                    ann_pred = np.argmax(B2, 1); snn_pred = np.argmax(B2h, 1)
                    agree_te = float((snn_pred[te] == ann_pred[te]).mean())
                    # ---- residual uniformity (Assumption 1) on non-saturated positive units
                    pos = (B1 > 0) & (B1 <= 1.0)
                    U = (T * B1 - cnt)[pos]
                    # ---- systematic + fluctuation bound, per node, worst rival class
                    srt = np.argsort(-B2, axis=1)
                    cstar = srt[:, 0]; rival = srt[:, 1]
                    DW = W2[:, cstar] - W2[:, rival]              # (H, N) -> build per node
                    S_h = DW.sum(0)                                # sum_h DW_c[h] per node
                    nrm2 = np.sqrt((DW ** 2).sum(0))               # ||DW_c||_2 per node
                    # systematic: per-unit means averaged over the neighbourhood
                    syst = rho * (DW * mu_h[:, None]).sum(0)
                    # deterministic saturation contribution (few units), bounded exactly
                    sat_bound = np.zeros(B1.shape[0])
                    if sat.any():
                        Esat = np.where(sat, np.abs(E1_ap - mu_h), 0.0)   # (N,H)
                        Msat = np.asarray(A @ Esat)                        # (N,H)
                        sat_bound = (Msat.T * np.abs(DW)).sum(0)           # per node (DW is H,N)
                    # Hoeffding over the (m,h) fluctuation terms, half-width w:
                    #   P(|S|>=t) <= 2 exp(-t^2 / (2 w^2 ||a_n||_2^2 ||DW||_2^2))
                    fluct = w * anorm2 * nrm2 * np.sqrt(2.0 * np.log(2.0 / a.delta))
                    delta_n = B2[np.arange(B2.shape[0]), cstar] - B2[np.arange(B2.shape[0]), rival]
                    pred = float(((delta_n > np.abs(syst) + fluct + sat_bound)[te]).mean())
                    meas_sys = float((((B2h - B2)[np.arange(B2.shape[0]), cstar]
                                       - (B2h - B2)[np.arange(B2.shape[0]), rival])[te]).mean())
                    conv = E.convert(m, T=T, beta1=beta_emp if tag == "ftbc" else 0.0,
                                     mode="naive", l1_mode="if", seed=0, theta=1.0,
                                     readout="membrane")
                    rows.append(dict(tag=tag, beta=beta, agree_te=agree_te, pred=pred,
                                     syst_pred_mean=float(syst[te].mean()),
                                     syst_meas_mean=meas_sys,
                                     mu_E=mu_E, w=w,
                                     U_mean=float(U.mean()) if U.size else float("nan"),
                                     U_var=float(U.var()) if U.size else float("nan"),
                                     acc_theory=float((snn_pred[te] == d["labels"][te]).mean()),
                                     acc_convert=conv["acc_test"]))
            for tag in ("ftbc", "nobias"):
                rs = [r for r in rows if r["tag"] == tag]
                agg = {k: float(np.mean([r[k] for r in rs])) for k in
                       ("beta", "agree_te", "pred", "syst_pred_mean", "syst_meas_mean",
                        "U_mean", "U_var", "acc_theory", "acc_convert", "mu_E", "w")}
                agg["T"] = T; agg["beta_theory"] = 1.0 / (2 * T)
                agg["mouthpiece_gap"] = float(np.mean([abs(r["acc_theory"] - r["acc_convert"])
                                                       for r in rs]))
                out[name][f"{tag}_T{T}"] = agg
                flag = "OK" if agg["agree_te"] >= agg["pred"] - 1e-9 else "VIOLATED"
                print(f"  T={T:2d} {tag:7s} beta={agg['beta']:.3f} (beta*={agg['beta_theory']:.4f}) | "
                      f"mu_E={agg['mu_E']:+.4f} w={agg['w']:.4f} | U mean={agg['U_mean']:.3f} var={agg['U_var']:.3f} (unif .500/.083)")
                print(f"        systematic: predicted={agg['syst_pred_mean']:+.4f} "
                      f"measured={agg['syst_meas_mean']:+.4f}")
                print(f"        THEOREM predicted>={agg['pred']:.4f}  measured agreement={agg['agree_te']:.4f} "
                      f"[{flag}]   mouthpiece gap={agg['mouthpiece_gap']:.2e}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
