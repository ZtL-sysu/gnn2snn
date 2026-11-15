"""CPU-only conformance tests for the pinned-author TI adapter."""
from __future__ import annotations

import unittest
from pathlib import Path

import torch
from torch import nn

from source_ti import fit_source_ti, load_author_api

AUTHOR = Path("C:/code/gnn2snn-dcgs-reference")


def make_batches(seed: int = 19):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    batches = []
    shapes = [(5, 4), (9, 4), (3, 4)]
    for n, _ in shapes:
        site_batch = []
        for site in range(6):
            x = torch.randn((n, 2 + site), generator=gen, dtype=torch.float32)
            x = x + (site - 2.0) * 0.31
            site_batch.append(x)
        batches.append(site_batch)
    return batches


def regroup(batches, cuts):
    out = []
    for site in range(6):
        joined = torch.cat([b[site] for b in batches], dim=0)
        out.append([joined[a:b] for a, b in zip(cuts[:-1], cuts[1:])])
    # transpose from per-site groups to outer batch -> six site tensors
    per_site = out
    return [[per_site[s][i] for s in range(6)] for i in range(len(cuts) - 1)]


class SourceTIConformance(unittest.TestCase):
    def test_moments_and_threshold_match_source_hook(self):
        batches = make_batches()
        thresholds, meta = fit_source_ti(batches, effective_steps=256, author_repo=AUTHOR)
        api, _, _ = load_author_api(AUTHOR)
        self.assertEqual(meta["scale_iterations"], 100)
        self.assertEqual(meta["effective_steps_n"], 256)
        self.assertEqual(len(thresholds), 6)
        for site in range(6):
            hook = api.ThreHook(mode="var", level="channel", out_layer=nn.ReLU())
            with torch.no_grad():
                for batch in batches:
                    hook(batch[site])
                hook.get_scale_from_var(T=256)
            entry = meta["sites"][site]
            self.assertTrue(torch.equal(hook.mean.reshape(-1), torch.tensor(entry["mean"])))
            self.assertTrue(torch.equal(hook.var.reshape(-1), torch.tensor(entry["variance_population_unbiased_false"])))
            self.assertTrue(torch.equal(hook.scale.reshape(-1), torch.tensor(entry["raw_source_threshold_after_100_iterations"])))
            expected = hook.scale.reshape(-1) * (hook.scale.reshape(-1) >= 0).to(hook.scale.dtype)
            self.assertTrue(torch.equal(thresholds[site], expected.expand(thresholds[site].numel())))

    def test_moments_are_invariant_to_batch_regrouping(self):
        original = make_batches(seed=27)
        merged = regroup(original, [0, 4, 11, 17])
        _, a = fit_source_ti(original, effective_steps=512, author_repo=AUTHOR)
        _, b = fit_source_ti(merged, effective_steps=512, author_repo=AUTHOR)
        for site in range(6):
            for field in ("mean", "variance_population_unbiased_false"):
                self.assertTrue(torch.allclose(torch.tensor(a["sites"][site][field]),
                                               torch.tensor(b["sites"][site][field]),
                                               rtol=2e-6, atol=2e-7), (site, field))
            # The source TI recurrence is nonlinear and runs in source float32;
            # tiny regrouping-roundoff can amplify in scale, so only moments are
            # required to agree at this boundary.

    def test_constant_variance_records_source_nan_fallback(self):
        batches = []
        for n in (4, 7):
            batches.append([torch.zeros((n, 3), dtype=torch.float32) for _ in range(6)])
        thresholds, meta = fit_source_ti(batches, effective_steps=256, author_repo=AUTHOR)
        for site in range(6):
            self.assertEqual(meta["sites"][site]["source_nan_fallback_count"], 3)
            self.assertEqual(meta["sites"][site]["raw_source_threshold_after_100_iterations"], [1.0] * 3)
            self.assertTrue(torch.equal(thresholds[site], torch.ones(3)))


if __name__ == "__main__":
    unittest.main()
