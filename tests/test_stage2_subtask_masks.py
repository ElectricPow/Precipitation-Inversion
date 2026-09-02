"""Contract tests for leakage-safe Stage-2 R0 subtask masks."""

from __future__ import annotations

from dataclasses import replace
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from precipitation_inversion.data.stage2_subtask_masks import (  # noqa: E402
    Stage2GRRoutingMasks,
    build_stage2_gr_routing_masks,
    build_stage2_subtask_masks,
)


def _numpy_example() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    # Flat toy voxels are sufficient to verify the same boolean algebra used
    # for real (nscan,nray,z) arrays. Positions 6--7 lie outside the audit
    # geometry and therefore must be false in every routed/supervised subset.
    gr_value = np.array([1, 0, 0, 1, 0, 0, 1, 0], dtype=bool)
    gr_interp = np.array([1, 1, 0, 1, 1, 0, 1, 1], dtype=bool)
    routing_domain = np.array([1, 1, 1, 1, 1, 1, 0, 0], dtype=bool)
    label_domain = routing_domain.copy()
    dpr_value = np.array([1, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
    dpr_dbz = np.array(
        [10.0, 36.0, 50.0, np.nan, np.nan, 35.0, np.nan, np.nan],
        dtype=np.float32,
    )
    return gr_value, gr_interp, routing_domain, label_domain, dpr_value, dpr_dbz


class Stage2GRRoutingMaskTests(unittest.TestCase):
    def test_numpy_observed_near_far_are_exhaustive_in_domain(self) -> None:
        gr, interp, domain, _, _, _ = _numpy_example()

        routing = build_stage2_gr_routing_masks(gr, interp, domain_mask=domain)

        np.testing.assert_array_equal(
            routing.observed, [1, 0, 0, 1, 0, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            routing.near, [0, 1, 0, 0, 1, 0, 0, 0]
        )
        np.testing.assert_array_equal(
            routing.far, [0, 0, 1, 0, 0, 1, 0, 0]
        )
        self.assertEqual(
            routing.counts(),
            {"domain": 6, "observed": 2, "near": 2, "far": 2},
        )
        self.assertTrue(all("dpr" not in name for name in routing.as_dict()))

    def test_default_domain_is_all_true_and_torch_backend_is_preserved(self) -> None:
        gr = torch.tensor([[True, False, False]])
        interp = torch.tensor([[True, True, False]])

        routing = build_stage2_gr_routing_masks(gr, interp)

        self.assertIsInstance(routing.domain, torch.Tensor)
        self.assertEqual(routing.domain.dtype, torch.bool)
        self.assertEqual(routing.domain.device, gr.device)
        torch.testing.assert_close(routing.domain, torch.ones_like(gr))
        torch.testing.assert_close(
            routing.observed, torch.tensor([[True, False, False]])
        )
        torch.testing.assert_close(
            routing.near, torch.tensor([[False, True, False]])
        )
        torch.testing.assert_close(
            routing.far, torch.tensor([[False, False, True]])
        )

    def test_routing_rejects_bad_dtype_shape_backend_and_partition(self) -> None:
        boolean = np.ones((2, 3), dtype=bool)
        with self.assertRaisesRegex(TypeError, "boolean dtype"):
            build_stage2_gr_routing_masks(
                boolean, np.ones((2, 3), dtype=np.float32)
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            build_stage2_gr_routing_masks(boolean, np.ones((2, 4), dtype=bool))
        with self.assertRaisesRegex(TypeError, "same NumPy or PyTorch"):
            build_stage2_gr_routing_masks(boolean, torch.ones(2, 3, dtype=torch.bool))

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            Stage2GRRoutingMasks(
                domain=np.array([True, True]),
                observed=np.array([True, False]),
                near=np.array([True, True]),  # voxel 0 belongs to two routes
                far=np.array([False, False]),
            )


class Stage2SubtaskMaskTests(unittest.TestCase):
    def test_q_regions_gap_outside_and_strong_tail_have_exact_semantics(self) -> None:
        gr, interp, route_domain, label_domain, dpr, dpr_dbz = _numpy_example()
        routing = build_stage2_gr_routing_masks(
            gr, interp, domain_mask=route_domain
        )

        masks = build_stage2_subtask_masks(
            routing,
            label_domain,
            dpr,
            dpr_dbz,
            strong_dbz_threshold=35.0,
        )

        np.testing.assert_array_equal(masks.echo_support, [1, 1, 1, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(masks.no_echo, [0, 0, 0, 1, 1, 0, 0, 0])
        np.testing.assert_array_equal(masks.q11_overlap, [1, 0, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(masks.q10_gr_only, [0, 0, 0, 1, 0, 0, 0, 0])
        np.testing.assert_array_equal(masks.q01_dpr_only, [0, 1, 1, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(masks.q00_neither, [0, 0, 0, 0, 1, 0, 0, 0])
        np.testing.assert_array_equal(masks.dpr_only_gap, [0, 1, 0, 0, 0, 0, 0, 0])
        np.testing.assert_array_equal(
            masks.dpr_only_outside, [0, 0, 1, 0, 0, 1, 0, 0]
        )
        np.testing.assert_array_equal(masks.strong_echo, [0, 1, 1, 0, 0, 1, 0, 0])
        np.testing.assert_array_equal(
            masks.non_strong_echo, [1, 0, 0, 0, 0, 0, 0, 0]
        )
        counts = masks.counts()
        self.assertEqual(counts["label_domain"], 6)
        self.assertEqual(counts["q01_dpr_only"], 3)
        self.assertEqual(counts["dpr_only_gap"], 1)
        self.assertEqual(counts["dpr_only_outside"], 2)
        self.assertEqual(counts["strong_echo"], 3)

    def test_torch_masks_keep_batch_shape_and_match_numpy_contract(self) -> None:
        gr, interp, route_domain, label_domain, dpr, dpr_dbz = _numpy_example()
        # Mimic a small batched 3-D tensor: (B=1,C=1,D=2,H=2,Z=2).
        tensor_shape = (1, 1, 2, 2, 2)
        routing = build_stage2_gr_routing_masks(
            torch.from_numpy(gr.reshape(tensor_shape)),
            torch.from_numpy(interp.reshape(tensor_shape)),
            domain_mask=torch.from_numpy(route_domain.reshape(tensor_shape)),
        )
        masks = build_stage2_subtask_masks(
            routing,
            torch.from_numpy(label_domain.reshape(tensor_shape)),
            torch.from_numpy(dpr.reshape(tensor_shape)),
            torch.from_numpy(dpr_dbz.reshape(tensor_shape)),
        )

        self.assertEqual(masks.shape, tensor_shape)
        self.assertEqual(masks.echo_support.dtype, torch.bool)
        self.assertEqual(masks.strong_echo.sum().item(), 3)
        self.assertEqual(masks.dpr_only_gap.sum().item(), 1)
        self.assertEqual(masks.dpr_only_outside.sum().item(), 2)

    def test_gr_routing_is_invariant_to_all_dpr_label_changes(self) -> None:
        gr, interp, route_domain, label_domain, dpr, dpr_dbz = _numpy_example()
        routing = build_stage2_gr_routing_masks(
            gr, interp, domain_mask=route_domain
        )
        before = {name: values.copy() for name, values in routing.as_dict().items()}

        build_stage2_subtask_masks(routing, label_domain, dpr, dpr_dbz)
        changed_dpr = ~dpr
        changed_dbz = np.full(dpr_dbz.shape, 60.0, dtype=np.float32)
        build_stage2_subtask_masks(
            routing, label_domain, changed_dpr, changed_dbz
        )

        for name, expected in before.items():
            np.testing.assert_array_equal(routing.as_dict()[name], expected)
        # The deployable mapping deliberately has no target-derived key.
        forbidden_tokens = ("dpr", "label", "support", "echo", "strong")
        self.assertFalse(
            any(
                token in name
                for name in routing.as_dict()
                for token in forbidden_tokens
            )
        )

    def test_q01_gap_outside_cannot_include_q00_or_q11(self) -> None:
        gr, interp, route_domain, label_domain, dpr, dpr_dbz = _numpy_example()
        routing = build_stage2_gr_routing_masks(
            gr, interp, domain_mask=route_domain
        )
        masks = build_stage2_subtask_masks(
            routing, label_domain, dpr, dpr_dbz
        )

        invalid_gap = masks.dpr_only_gap | masks.q00_neither
        with self.assertRaisesRegex(ValueError, "gap/outside"):
            replace(masks, dpr_only_gap=invalid_gap)
        invalid_outside = masks.dpr_only_outside | masks.q11_overlap
        with self.assertRaisesRegex(ValueError, "gap/outside"):
            replace(masks, dpr_only_outside=invalid_outside)

    def test_invalid_label_domain_dbz_dtype_shape_and_values_are_rejected(self) -> None:
        gr, interp, route_domain, label_domain, dpr, dpr_dbz = _numpy_example()
        routing = build_stage2_gr_routing_masks(
            gr, interp, domain_mask=route_domain
        )
        outside_domain = label_domain.copy()
        outside_domain[-1] = True
        with self.assertRaisesRegex(ValueError, "subset"):
            build_stage2_subtask_masks(
                routing, outside_domain, dpr, dpr_dbz
            )
        with self.assertRaisesRegex(TypeError, "floating-point"):
            build_stage2_subtask_masks(
                routing, label_domain, dpr, np.zeros(8, dtype=np.int16)
            )
        with self.assertRaisesRegex(ValueError, "shape"):
            build_stage2_subtask_masks(
                routing, label_domain, dpr, np.zeros(9, dtype=np.float32)
            )
        invalid_dbz = dpr_dbz.copy()
        invalid_dbz[1] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            build_stage2_subtask_masks(
                routing, label_domain, dpr, invalid_dbz
            )
        with self.assertRaisesRegex(ValueError, "threshold"):
            build_stage2_subtask_masks(
                routing,
                label_domain,
                dpr,
                dpr_dbz,
                strong_dbz_threshold=np.inf,
            )


if __name__ == "__main__":
    unittest.main()
