"""CPU tests for support- and alphabet-preserving event swaps."""
from __future__ import annotations

import unittest
import importlib.util
from pathlib import Path

import torch

from event_swap import apply_swaps, propose_swaps


def fixture():
    events = torch.tensor([
        [1., 4.], [2., 5.], [1., 6.], [2., 4.], [0., 5.], [-1., 0.],
        [3., 2.], [4., 3.], [3., 2.], [4., 3.], [0., 0.], [0., 1.],
    ], dtype=torch.float32)
    ptr = torch.tensor([0, 6, 12], dtype=torch.long)
    degree = torch.tensor([3, 3, 3, 3, 2, 2, 4, 4, 4, 4, 2, 2], dtype=torch.long)
    return events, ptr, degree


class EventSwapTests(unittest.TestCase):
    def test_fast_proposals_match_frozen_capacity_bank_implementation(self):
        path=Path(__file__).parent/'runs/swap_oracle/MNIST41/source/event_swap.py'
        spec=importlib.util.spec_from_file_location('frozen_swap_sampler',path)
        frozen=importlib.util.module_from_spec(spec);spec.loader.exec_module(frozen)
        events,ptr,degree=fixture()
        for seed in (7,20261117):
            self.assertEqual(propose_swaps(events,ptr,degree,[101,202],count=8,seed=seed),
                frozen.propose_swaps(events,ptr,degree,[101,202],count=8,seed=seed))

    def test_swaps_preserve_support_histogram_deliveries_and_input(self):
        events, ptr, degree = fixture()
        original = events.clone()
        candidates = propose_swaps(events, ptr, degree, [101, 202], count=8)
        self.assertEqual(len(candidates), 8)
        self.assertTrue(all(len(candidate) == 2 for candidate in candidates))
        for candidate in candidates:
            changed = apply_swaps(events, candidate, ptr, degree)
            self.assertTrue(torch.equal(events, original))
            self.assertTrue(torch.equal(changed != 0, events != 0))
            for lo, hi in ((0, 6), (6, 12)):
                for ch in range(events.shape[1]):
                    self.assertEqual(sorted(changed[lo:hi, ch].tolist()), sorted(events[lo:hi, ch].tolist()))
                    self.assertEqual(int((changed[lo:hi, ch] != 0).sum()), int((events[lo:hi, ch] != 0).sum()))
                    self.assertEqual(float((changed[lo:hi, ch].abs() * degree[lo:hi]).sum()),
                                     float((events[lo:hi, ch].abs() * degree[lo:hi]).sum()))
                    self.assertEqual(int(((changed[lo:hi, ch] != 0).long() * degree[lo:hi]).sum()),
                                     int(((events[lo:hi, ch] != 0).long() * degree[lo:hi]).sum()))
            for gi, pair in enumerate(candidate):
                if pair is not None:
                    u, v, ch = pair
                    self.assertEqual(int(degree[u]), int(degree[v]))
                    self.assertNotEqual(float(events[u, ch]), float(events[v, ch]))
                    self.assertNotEqual(float(events[u, ch]), 0.)
                    self.assertNotEqual(float(events[v, ch]), 0.)
                    self.assertTrue((int(ptr[gi]) <= u < int(ptr[gi + 1])) and
                                    (int(ptr[gi]) <= v < int(ptr[gi + 1])))
        for gi in range(2):
            pairs = [c[gi] for c in candidates if c[gi] is not None]
            unordered = [(min(u, v), max(u, v), ch) for u, v, ch in pairs]
            self.assertEqual(len(unordered), len(set(unordered)))

    def test_graph_id_rng_is_independent_of_batch_regrouping(self):
        events, ptr, degree = fixture()
        full = propose_swaps(events, ptr, degree, [101, 202], count=8, seed=7)
        graph_events = events[6:12].clone()
        graph_degree = degree[6:12].clone()
        alone = propose_swaps(graph_events, torch.tensor([0, 6]), graph_degree, [202], count=8, seed=7)
        for k in range(8):
            p_full, p_alone = full[k][1], alone[k][0]
            if p_full is None or p_alone is None:
                self.assertIsNone(p_full); self.assertIsNone(p_alone)
            else:
                u, v, ch = p_full
                a, b, ch2 = p_alone
                self.assertEqual((u - 6, v - 6, ch), (a, b, ch2))

    def test_empty_eligibility_returns_none_and_count_zero_is_empty(self):
        x = torch.tensor([[1.], [1.], [0.]], dtype=torch.float32)
        p = torch.tensor([0, 3], dtype=torch.long)
        d = torch.tensor([2, 2, 2], dtype=torch.long)
        self.assertEqual(propose_swaps(x, p, d, [9], count=0), [])
        self.assertEqual(propose_swaps(x, p, d, [9], count=3), [[None], [None], [None]])

    def test_apply_rejects_invalid_pairs(self):
        events, ptr, degree = fixture()
        invalid = [
            [(0, 6, 0), None],     # crosses graphs
            [(0, 4, 0), None],     # degree mismatch
            [(0, 0, 0), None],     # same node
            [(0, 4, 8), None],     # channel out of range
            [(0, 5, 0), None],     # node 5 nonzero but degree mismatch
            [(4, 4, 1), None],     # same node
        ]
        for plan in invalid:
            with self.subTest(plan=plan), self.assertRaises(ValueError):
                apply_swaps(events, plan, ptr, degree)
        with self.assertRaises(ValueError):
            apply_swaps(events, [(0, 2, 0), None], ptr, degree)  # equal symbol
        with self.assertRaises(ValueError):
            apply_swaps(events, [(0, 1, 0), (0, 1, 0)], ptr, degree)  # graph membership mismatch
        with self.assertRaises(ValueError):
            apply_swaps(events, [(4, 5, 0), None], ptr, degree)  # one zero event


if __name__ == "__main__":
    unittest.main()
