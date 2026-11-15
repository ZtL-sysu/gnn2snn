"""CPU tests for fixed-feature graph-conditioned swap prediction."""
from __future__ import annotations

import unittest

import numpy as np
import torch

from swap_predictor import graph_cache, pair_features, ridge_fit, ridge_predict, choose_pairs


def graph_fixture():
    # Graph 0: bidirectional 3-node star; graph 1: one bidirectional edge.
    src = np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int64)
    dst = np.asarray([1, 2, 0, 1, 4, 3], dtype=np.int64)
    edge = torch.tensor(np.stack((src, dst)), dtype=torch.long)
    ptr = torch.tensor([0, 3, 5], dtype=torch.long)
    return edge, ptr, [73, 91]


class SwapPredictorTests(unittest.TestCase):
    def test_graph_cache_matches_explicit_normalized_adjacency(self):
        edge, ptr, gids = graph_fixture()
        cache = graph_cache(edge, ptr, gids, "graph")
        src, dst = edge.numpy(); p = ptr.numpy()
        n = int(p[-1]);
        out = np.bincount(src, minlength=n); indeg = np.bincount(dst, minlength=n)
        expected_mass = np.zeros(n); expected_mass2 = np.zeros(n)
        for lo, hi in zip(p[:-1], p[1:]):
            A = np.zeros((hi-lo, hi-lo), dtype=np.float64)
            for u, v in zip(src, dst):
                if lo <= u < hi:
                    A[v-lo, u-lo] = 1 / np.sqrt(max(out[u], 1) * max(indeg[v], 1))
            mass = np.ones(hi-lo) @ A
            mass2 = np.ones(hi-lo) @ A @ A
            expected_mass[lo:hi] = mass
            expected_mass2[lo:hi] = mass2
        self.assertTrue(np.allclose(cache["mass"], expected_mass, atol=1e-14))
        self.assertTrue(np.allclose(cache["mass2"], expected_mass2, atol=1e-14))
        self.assertEqual(cache["cache_nbytes"], sum(v.nbytes for v in cache.values() if isinstance(v, np.ndarray)))

    def test_shuffled_cache_is_deterministic_and_does_not_mutate_graph(self):
        edge, ptr, gids = graph_fixture(); before = edge.clone()
        a = graph_cache(edge, ptr, gids, "shuffled")
        b = graph_cache(edge, ptr, gids, "shuffled")
        self.assertTrue(torch.equal(edge, before))
        self.assertTrue(np.array_equal(a["shuffled_mass"], b["shuffled_mass"]))
        self.assertTrue(np.array_equal(a["shuffled_mass2"], b["shuffled_mass2"]))
        for lo, hi in zip(ptr[:-1].tolist(), ptr[1:].tolist()):
            self.assertEqual(sorted(a["mass"][lo:hi]), sorted(a["shuffled_mass"][lo:hi]))
            self.assertEqual(sorted(a["mass2"][lo:hi]), sorted(a["shuffled_mass2"][lo:hi]))

    def test_pair_feature_order_sources_and_channel_onehot(self):
        edge, ptr_t, gids = graph_fixture(); ptr = ptr_t.numpy()
        cache = graph_cache(edge, ptr_t, gids, "graph")
        events = np.zeros((5, 146), dtype=np.float64)
        membrane = np.zeros_like(events)
        theta = np.full(146, 2.0)
        # Nodes 1 and 2 share outdegree 1 and channel-7 values differ.
        events[0, 7] = 1.; events[1, 7] = 3.; events[2, 7] = -1.
        membrane[:, 7] = [4., 2., 0., 0., 0.]
        pairs = np.full((2, 2, 3), -1, dtype=np.int64)
        pairs[0, 0] = (1, 2, 7)
        X, valid = pair_features(events, membrane, theta, pairs, ptr, cache, "graph")
        self.assertEqual(X.shape, (2, 2, 162)); self.assertEqual(valid.tolist(), [[True, False], [False, False]])
        x = X[0, 0]
        d = (-1. - 3.) / 2.
        md = (2. - 0.) / 2.
        expected_base = [d, md, d*d, d*md, 3/2, -1/2, 2/2, 1,
                         events[:3,7].mean()/2, 1., membrane[:3,7].mean()/2,
                         np.log1p(3), 2/2]
        self.assertTrue(np.allclose(x[:13], expected_base))
        self.assertTrue(np.allclose(x[13:16], [d*(cache['mass'][1]-cache['mass'][2]),
            d*(cache['mass'][1]*2-cache['mass'][2]*0)/2,
            d*(cache['mass2'][1]-cache['mass2'][2])]))
        self.assertEqual(float(x[16+7]), 1.)
        self.assertEqual(float(x[16+6]), 0.)
        self.assertEqual(int(np.count_nonzero(x[16:])), 1)
        self.assertTrue(np.all(X[1,1] == 0.))
        # Distinguish |mu|-|mv| from |mu-mv| in the registered non-graph slot.
        membrane[1,7],membrane[2,7]=-2.,1.
        nongraph=graph_cache(edge,ptr_t,gids,'nongraph')
        ng,_=pair_features(events,membrane,theta,pairs,ptr,nongraph,'nongraph')
        self.assertEqual(ng[0,0,15],d*(2.-1.)/2.)

    def test_ridge_known_linear_system_and_constant_feature(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(500, 162)); X[:, 9] = 4.0
        B = rng.normal(scale=.15, size=(162, 8)); bias = rng.normal(size=8)
        Y = X @ B + bias
        model = ridge_fit(X, Y, alpha=.001)
        self.assertEqual(model['mean'].shape, (162,))
        self.assertEqual(model['std'].shape, (162,))
        self.assertEqual(model['coef'].shape, (163, 8))
        self.assertEqual(float(model['std'][9]), 1.)
        self.assertTrue(np.isfinite(model['coef']).all())
        self.assertTrue(np.allclose(ridge_predict(model, X), Y, atol=2e-3, rtol=2e-3))
        self.assertTrue(np.allclose(ridge_predict(model, X[:2,None,:]), Y[:2,None,:], atol=2e-3, rtol=2e-3))
        with self.assertRaises(ValueError):
            ridge_fit(X, Y, alpha=.002)

    def test_selector_requires_all_predicted_constraints(self):
        pred = np.zeros((4, 2, 8), dtype=np.float64)
        valid = np.ones((4, 2), dtype=bool)
        # Graph 0: candidates 0 and 1 feasible; choose candidate 1 (return 2).
        pred[0, 0] = [.5, 0., 0., 0., 0., 0., 0., 0.]
        pred[1, 0] = [.8, .1, -1., -1., -1., -1., -1., -1.]
        pred[2, 0] = [2., -.01, -1., -1., -1., -1., -1., -1.]  # T2 fails
        # Graph 1 has positive T1 but a positive cost head, so no feasible swap.
        pred[0, 1] = [1., 0., .001, -1., -1., -1., -1., -1.]
        chosen = choose_pairs(pred, valid)
        self.assertEqual(chosen.tolist(), [2, 0])
        valid[1, 0] = False
        self.assertEqual(choose_pairs(pred, valid).tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
