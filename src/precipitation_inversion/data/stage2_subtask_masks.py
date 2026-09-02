"""Leakage-safe mask contracts for the decomposed Stage-2 (R0) audit.

The original Stage-2 pipeline computes useful masks, but it exposes all of
them in one dictionary.  That makes it too easy for a future model input to
accidentally include a mask derived from DPR labels.  R0 therefore separates
the contract into two explicit layers:

``Stage2GRRoutingMasks``
    Contains only information available from GR at deployment time.  It routes
    every voxel to a directly observed, nearby/reachable, or far/unreachable
    region.

``Stage2SubtaskMasks``
    Adds DPR label-domain and reflectivity information for loss calculation
    and offline audit.  These arrays are supervision-only and must never be
    stacked into model inputs.

Both NumPy arrays and PyTorch tensors are supported.  A single object may not
mix the two backends, and torch tensors must also reside on one device.  All
masks keep their source shape, normally ``(nscan, nray, z)`` before batching or
``(B, 1, D, H, Z)`` after collation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch


MaskArray = np.ndarray | torch.Tensor


def _backend(values: Any, *, name: str) -> str:
    if isinstance(values, np.ndarray):
        return "numpy"
    if isinstance(values, torch.Tensor):
        return "torch"
    raise TypeError(f"{name} must be a numpy.ndarray or torch.Tensor")


def _shape(values: MaskArray) -> tuple[int, ...]:
    return tuple(int(size) for size in values.shape)


def _device(values: MaskArray) -> torch.device | None:
    return values.device if isinstance(values, torch.Tensor) else None


def _is_bool(values: MaskArray) -> bool:
    if isinstance(values, torch.Tensor):
        return values.dtype == torch.bool
    return values.dtype == np.bool_


def _any(values: MaskArray) -> bool:
    if isinstance(values, torch.Tensor):
        return bool(torch.any(values).item())
    return bool(np.any(values))


def _equal(left: MaskArray, right: MaskArray) -> bool:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        return bool(torch.equal(left, right))
    assert isinstance(right, np.ndarray)
    return bool(np.array_equal(left, right))


def _count(values: MaskArray) -> int:
    if isinstance(values, torch.Tensor):
        return int(values.sum(dtype=torch.int64).item())
    return int(values.sum(dtype=np.int64))


def _validate_arrays(
    arrays: Mapping[str, MaskArray],
    *,
    require_bool: bool = True,
) -> tuple[str, tuple[int, ...], torch.device | None]:
    """Validate a common backend, shape, device, and optionally bool dtype."""

    if not arrays:
        raise ValueError("at least one array is required")
    first_name, first = next(iter(arrays.items()))
    backend = _backend(first, name=first_name)
    shape = _shape(first)
    device = _device(first)
    for name, values in arrays.items():
        if _backend(values, name=name) != backend:
            raise TypeError("all masks must use the same NumPy or PyTorch backend")
        if _shape(values) != shape:
            raise ValueError(
                f"all masks must have shape {shape}; {name} has {_shape(values)}"
            )
        if _device(values) != device:
            raise ValueError("all torch tensors must be on the same device")
        if require_bool and not _is_bool(values):
            raise TypeError(f"{name} must have boolean dtype")
    return backend, shape, device


def _assert_partition(
    domain: MaskArray,
    parts: Mapping[str, MaskArray],
    *,
    description: str,
) -> None:
    """Require ``parts`` to be mutually exclusive and exhaustive in domain."""

    arrays = {"domain": domain, **parts}
    backend, _, _ = _validate_arrays(arrays)
    if backend == "torch":
        assert isinstance(domain, torch.Tensor)
        tensors = [part.to(torch.int16) for part in parts.values()]
        membership = torch.stack(tensors, dim=0).sum(dim=0)
        expected = domain.to(torch.int16)
    else:
        assert isinstance(domain, np.ndarray)
        membership = sum(
            (part.astype(np.int16, copy=False) for part in parts.values()),
            start=np.zeros_like(domain, dtype=np.int16),
        )
        expected = domain.astype(np.int16, copy=False)
    if not _equal(membership, expected):
        raise ValueError(
            f"{description} masks must be mutually exclusive and exhaustive "
            "inside their domain, and false outside it"
        )


@dataclass(frozen=True)
class Stage2GRRoutingMasks:
    """GR-only deployment routing masks.

    All fields have the same shape and are derived without consulting DPR.

    ``domain``
        Input geometry in which routing is defined.  It is all ``True`` when
        omitted by :func:`build_stage2_gr_routing_masks`.
    ``observed``
        ``domain & gr_value``: a physical sparse-GR dBZ exists at this voxel.
    ``near``
        ``domain & ~gr_value & gr_interp_value``: no direct GR value exists,
        but the GR-only interpolation proxy can reach this voxel.  This is a
        routing proxy, not a verified elevation-gap label.
    ``far``
        ``domain & ~gr_value & ~gr_interp_value``: neither direct GR nor its
        interpolation proxy reaches the voxel.  This does not mean that no
        precipitation exists there.
    """

    domain: MaskArray
    observed: MaskArray
    near: MaskArray
    far: MaskArray

    def __post_init__(self) -> None:
        _validate_arrays(
            {
                "domain": self.domain,
                "observed": self.observed,
                "near": self.near,
                "far": self.far,
            }
        )
        _assert_partition(
            self.domain,
            {"observed": self.observed, "near": self.near, "far": self.far},
            description="GR routing",
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape(self.domain)

    def counts(self) -> dict[str, int]:
        """Return JSON-safe counts for the input-only routing regions."""

        return {
            "domain": _count(self.domain),
            "observed": _count(self.observed),
            "near": _count(self.near),
            "far": _count(self.far),
        }

    def as_dict(self) -> dict[str, MaskArray]:
        """Return a stable mapping containing no DPR-derived information."""

        return {
            "gr_routing_domain": self.domain,
            "gr_route_observed": self.observed,
            "gr_route_near": self.near,
            "gr_route_far": self.far,
        }


@dataclass(frozen=True)
class Stage2SubtaskMasks:
    """Supervision-only R0 partitions layered on a GR-only routing object.

    ``label_domain`` says where DPR labels are trustworthy/evaluable.  Within
    it, ``echo_support`` and ``no_echo`` are exhaustive.  Q11/Q10/Q01/Q00 use
    the conventional first bit for GR physical-value presence and the second
    bit for DPR echo presence.  ``dpr_only_gap`` and ``dpr_only_outside`` split
    Q01 by the GR-only near/far routing proxy.  ``strong_echo`` is defined from
    raw physical DPR dBZ and is always a subset of ``echo_support``.

    These masks may be used to select labels, losses, and metrics.  None is a
    deployable input feature; only ``routing.as_dict()`` is label-free.
    """

    routing: Stage2GRRoutingMasks
    label_domain: MaskArray
    echo_support: MaskArray
    no_echo: MaskArray
    q11_overlap: MaskArray
    q10_gr_only: MaskArray
    q01_dpr_only: MaskArray
    q00_neither: MaskArray
    dpr_only_gap: MaskArray
    dpr_only_outside: MaskArray
    strong_echo: MaskArray
    non_strong_echo: MaskArray
    strong_dbz_threshold: float

    def __post_init__(self) -> None:
        masks = {
            "label_domain": self.label_domain,
            "echo_support": self.echo_support,
            "no_echo": self.no_echo,
            "q11_overlap": self.q11_overlap,
            "q10_gr_only": self.q10_gr_only,
            "q01_dpr_only": self.q01_dpr_only,
            "q00_neither": self.q00_neither,
            "dpr_only_gap": self.dpr_only_gap,
            "dpr_only_outside": self.dpr_only_outside,
            "strong_echo": self.strong_echo,
            "non_strong_echo": self.non_strong_echo,
        }
        backend, shape, device = _validate_arrays(masks)
        route_backend, route_shape, route_device = _validate_arrays(
            {
                "routing.domain": self.routing.domain,
                "routing.observed": self.routing.observed,
                "routing.near": self.routing.near,
                "routing.far": self.routing.far,
            }
        )
        if (backend, shape, device) != (route_backend, route_shape, route_device):
            raise ValueError(
                "supervision masks and GR routing must share backend, shape, "
                "and torch device"
            )
        if not np.isfinite(self.strong_dbz_threshold):
            raise ValueError("strong_dbz_threshold must be finite")
        if _any(self.label_domain & ~self.routing.domain):
            raise ValueError("label_domain must be a subset of routing.domain")

        _assert_partition(
            self.label_domain,
            {"echo_support": self.echo_support, "no_echo": self.no_echo},
            description="DPR echo/support",
        )
        _assert_partition(
            self.label_domain,
            {
                "q11_overlap": self.q11_overlap,
                "q10_gr_only": self.q10_gr_only,
                "q01_dpr_only": self.q01_dpr_only,
                "q00_neither": self.q00_neither,
            },
            description="Q11/Q10/Q01/Q00",
        )
        _assert_partition(
            self.q01_dpr_only,
            {
                "dpr_only_gap": self.dpr_only_gap,
                "dpr_only_outside": self.dpr_only_outside,
            },
            description="DPR-only gap/outside",
        )
        _assert_partition(
            self.echo_support,
            {
                "strong_echo": self.strong_echo,
                "non_strong_echo": self.non_strong_echo,
            },
            description="strong/non-strong DPR echo",
        )

        expected = {
            "q11_overlap": self.label_domain
            & self.routing.observed
            & self.echo_support,
            "q10_gr_only": self.label_domain
            & self.routing.observed
            & self.no_echo,
            "q01_dpr_only": self.label_domain
            & ~self.routing.observed
            & self.echo_support,
            "q00_neither": self.label_domain
            & ~self.routing.observed
            & self.no_echo,
            "dpr_only_gap": self.q01_dpr_only & self.routing.near,
            "dpr_only_outside": self.q01_dpr_only & self.routing.far,
        }
        for name, values in expected.items():
            if not _equal(getattr(self, name), values):
                raise ValueError(f"{name} is inconsistent with its mask semantics")

    @property
    def shape(self) -> tuple[int, ...]:
        return _shape(self.label_domain)

    def counts(self) -> dict[str, int]:
        """Return stable counts for R0 audit JSON/CSV output."""

        result = {
            "label_domain": _count(self.label_domain),
            "echo_support": _count(self.echo_support),
            "no_echo": _count(self.no_echo),
            "q11_overlap": _count(self.q11_overlap),
            "q10_gr_only": _count(self.q10_gr_only),
            "q01_dpr_only": _count(self.q01_dpr_only),
            "q00_neither": _count(self.q00_neither),
            "dpr_only_gap": _count(self.dpr_only_gap),
            "dpr_only_outside": _count(self.dpr_only_outside),
            "strong_echo": _count(self.strong_echo),
            "non_strong_echo": _count(self.non_strong_echo),
        }
        result.update(
            {f"route_{name}": count for name, count in self.routing.counts().items()}
        )
        return result

    def as_dict(self) -> dict[str, MaskArray]:
        """Return supervision masks; the name warns callers about leakage."""

        return {
            "label_domain": self.label_domain,
            "echo_support": self.echo_support,
            "no_echo": self.no_echo,
            "q11_overlap": self.q11_overlap,
            "q10_gr_only": self.q10_gr_only,
            "q01_dpr_only": self.q01_dpr_only,
            "q00_neither": self.q00_neither,
            "dpr_only_gap": self.dpr_only_gap,
            "dpr_only_outside": self.dpr_only_outside,
            "strong_echo": self.strong_echo,
            "non_strong_echo": self.non_strong_echo,
        }


def build_stage2_gr_routing_masks(
    gr_value_mask: MaskArray,
    gr_interp_value_mask: MaskArray,
    *,
    domain_mask: MaskArray | None = None,
) -> Stage2GRRoutingMasks:
    """Build the deployable observed/near/far partition from GR only.

    Parameters use masks, never dBZ values, so the function has no possible DPR
    label dependency.  If ``domain_mask`` is omitted, routing covers the full
    source array.  Padding/core restrictions should be supplied explicitly
    when they are relevant to an audit.
    """

    backend, shape, _ = _validate_arrays(
        {
            "gr_value_mask": gr_value_mask,
            "gr_interp_value_mask": gr_interp_value_mask,
        }
    )
    if domain_mask is None:
        if backend == "torch":
            assert isinstance(gr_value_mask, torch.Tensor)
            domain_mask = torch.ones_like(gr_value_mask, dtype=torch.bool)
        else:
            domain_mask = np.ones(shape, dtype=bool)
    _validate_arrays(
        {
            "gr_value_mask": gr_value_mask,
            "gr_interp_value_mask": gr_interp_value_mask,
            "domain_mask": domain_mask,
        }
    )

    observed = domain_mask & gr_value_mask
    missing = domain_mask & ~gr_value_mask
    near = missing & gr_interp_value_mask
    far = missing & ~gr_interp_value_mask
    return Stage2GRRoutingMasks(
        domain=domain_mask,
        observed=observed,
        near=near,
        far=far,
    )


def build_stage2_subtask_masks(
    routing: Stage2GRRoutingMasks,
    label_domain_mask: MaskArray,
    dpr_value_mask: MaskArray,
    dpr_dbz: np.ndarray | torch.Tensor,
    *,
    strong_dbz_threshold: float = 35.0,
) -> Stage2SubtaskMasks:
    """Add DPR supervision partitions to an already-built GR routing object.

    ``dpr_dbz`` must contain a finite physical value wherever
    ``label_domain_mask & dpr_value_mask`` is true.  Values outside that mask
    are intentionally ignored.  This function returns audit/loss masks only;
    it must not be called while assembling deployable model input channels.
    """

    if not np.isfinite(strong_dbz_threshold):
        raise ValueError("strong_dbz_threshold must be finite")
    backend, shape, device = _validate_arrays(
        {
            "label_domain_mask": label_domain_mask,
            "dpr_value_mask": dpr_value_mask,
        }
    )
    route_backend, route_shape, route_device = _validate_arrays(
        {
            "routing.domain": routing.domain,
            "routing.observed": routing.observed,
            "routing.near": routing.near,
            "routing.far": routing.far,
        }
    )
    if (backend, shape, device) != (route_backend, route_shape, route_device):
        raise ValueError(
            "DPR supervision arrays and GR routing must share backend, shape, "
            "and torch device"
        )
    if _backend(dpr_dbz, name="dpr_dbz") != backend:
        raise TypeError("dpr_dbz and masks must use the same backend")
    if _shape(dpr_dbz) != shape:
        raise ValueError(f"dpr_dbz must have shape {shape}; got {_shape(dpr_dbz)}")
    if _device(dpr_dbz) != device:
        raise ValueError("dpr_dbz and masks must be on the same torch device")
    if isinstance(dpr_dbz, torch.Tensor):
        if not (dpr_dbz.dtype.is_floating_point or dpr_dbz.dtype.is_complex):
            raise TypeError("dpr_dbz must have floating-point dtype")
        if dpr_dbz.dtype.is_complex:
            raise TypeError("dpr_dbz must have real floating-point dtype")
    elif not np.issubdtype(dpr_dbz.dtype, np.floating):
        raise TypeError("dpr_dbz must have floating-point dtype")

    if _any(label_domain_mask & ~routing.domain):
        raise ValueError("label_domain_mask must be a subset of routing.domain")
    echo_support = label_domain_mask & dpr_value_mask
    if isinstance(dpr_dbz, torch.Tensor):
        finite = torch.isfinite(dpr_dbz)
    else:
        finite = np.isfinite(dpr_dbz)
    if _any(echo_support & ~finite):
        raise ValueError("dpr_dbz must be finite wherever DPR echo/support is true")

    no_echo = label_domain_mask & ~dpr_value_mask
    q11 = echo_support & routing.observed
    q10 = no_echo & routing.observed
    q01 = echo_support & ~routing.observed
    q00 = no_echo & ~routing.observed
    gap = q01 & routing.near
    outside = q01 & routing.far
    strong = echo_support & (dpr_dbz >= float(strong_dbz_threshold))
    non_strong = echo_support & ~strong

    return Stage2SubtaskMasks(
        routing=routing,
        label_domain=label_domain_mask,
        echo_support=echo_support,
        no_echo=no_echo,
        q11_overlap=q11,
        q10_gr_only=q10,
        q01_dpr_only=q01,
        q00_neither=q00,
        dpr_only_gap=gap,
        dpr_only_outside=outside,
        strong_echo=strong,
        non_strong_echo=non_strong,
        strong_dbz_threshold=float(strong_dbz_threshold),
    )


__all__ = [
    "MaskArray",
    "Stage2GRRoutingMasks",
    "Stage2SubtaskMasks",
    "build_stage2_gr_routing_masks",
    "build_stage2_subtask_masks",
]
