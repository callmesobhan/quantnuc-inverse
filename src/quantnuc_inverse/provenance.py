"""SHA-256 cryptographic provenance ledger and hard physical abstention gates.

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel

from quantnuc_inverse import __version__
from quantnuc_inverse.ir import (
    RESEARCH_DISCLAIMER,
    AbstentionCode,
    AbstentionError,
    ProvenanceRecord,
)


def _normalize_for_json(obj: Any) -> Any:
    """Convert Pydantic models, NumPy arrays, and scalars to deterministic JSON structures."""
    if isinstance(obj, BaseModel):
        return _normalize_for_json(obj.model_dump(mode="json"))
    if isinstance(obj, dict):
        return {str(k): _normalize_for_json(v) for k, v in sorted(obj.items(), key=lambda x: str(x[0]))}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "shape": list(obj.shape),
            "dtype": str(obj.dtype),
            "sha256": hashlib.sha256(np.ascontiguousarray(obj).tobytes()).hexdigest(),
        }
    if isinstance(obj, (np.floating, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return str(val)
        return round(val, 12)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def compute_sha256(payload: Any) -> str:
    """Compute a deterministic SHA-256 hex digest of any structured payload."""
    canonical = json.dumps(
        _normalize_for_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_provenance_record(
    operation: str,
    inputs: Any,
    outputs: Any,
    *,
    data_source_tag: str = "ANALYTICAL_PHYSICS_MODEL",
    citations: list[str] | None = None,
) -> ProvenanceRecord:
    """Construct an immutable SHA-256 provenance record for a physics calculation."""
    return ProvenanceRecord(
        record_id=f"qn-{uuid.uuid4().hex[:12]}",
        engine_version=__version__,
        timestamp_utc=datetime.now(UTC).isoformat(),
        operation=operation,
        input_sha256=compute_sha256(inputs),
        output_sha256=compute_sha256(outputs),
        data_source_tag=data_source_tag,
        citations=citations or [],
        disclaimer=RESEARCH_DISCLAIMER,
    )


class ProvenanceLedger:
    """Append-only hash-chained provenance ledger for auditable quantitative workflows."""

    def __init__(self) -> None:
        self._records: list[ProvenanceRecord] = []
        self._chain_hashes: list[str] = []

    @property
    def records(self) -> list[ProvenanceRecord]:
        return list(self._records)

    @property
    def head_sha256(self) -> str:
        if not self._chain_hashes:
            return hashlib.sha256(b"quantnuc-inverse-genesis").hexdigest()
        return self._chain_hashes[-1]

    def append(self, record: ProvenanceRecord) -> str:
        """Append a record and advance the ledger SHA-256 hash chain."""
        prev = self.head_sha256
        link = compute_sha256({"prev": prev, "record": record})
        self._records.append(record)
        self._chain_hashes.append(link)
        return link

    def verify_integrity(self) -> bool:
        """Verify that every hash link in the ledger matches its deterministic chain."""
        prev = hashlib.sha256(b"quantnuc-inverse-genesis").hexdigest()
        for rec, expected_hash in zip(self._records, self._chain_hashes, strict=True):
            recomputed = compute_sha256({"prev": prev, "record": rec})
            if recomputed != expected_hash:
                return False
            prev = recomputed
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine_version": __version__,
            "disclaimer": RESEARCH_DISCLAIMER,
            "record_count": len(self._records),
            "head_sha256": self.head_sha256,
            "integrity_verified": self.verify_integrity(),
            "entries": [
                {
                    "chain_sha256": h,
                    "record": r.model_dump(mode="json"),
                }
                for r, h in zip(self._records, self._chain_hashes, strict=True)
            ],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return target


# --- Hard Quantitative Abstention Gates ---


def assert_decay_corrected(decay_correction: str | None, modality: str = "PT") -> None:
    """Reject PET/SPECT datasets lacking decay correction to scan or administration time."""
    if modality == "PT":
        normalized = (decay_correction or "").strip().upper()
        if normalized not in {"START", "ADMIN"}:
            raise AbstentionError(
                AbstentionCode.MISSING_DECAY_CORRECTION,
                f"PET DecayCorrection tag is '{decay_correction}'; must be 'START' or 'ADMIN' for quantitative SUV.",
                details={"decay_correction": decay_correction, "modality": modality},
            )


def assert_calibrated_suv_metadata(
    *,
    units: str | None,
    patient_weight_kg: float | None,
    injected_dose_bq: float | None,
    half_life_s: float | None,
    rescale_slope: float | None,
) -> None:
    """Reject DICOM datasets missing calibrated BQML scaling or radiopharmaceutical metadata."""
    missing: list[str] = []
    if (units or "").strip().upper() != "BQML":
        missing.append(f"Units={units!r} (expected 'BQML')")
    if patient_weight_kg is None or patient_weight_kg <= 0.0:
        missing.append("PatientWeight")
    if injected_dose_bq is None or injected_dose_bq <= 0.0:
        missing.append("RadionuclideTotalDose")
    if half_life_s is None or half_life_s <= 0.0:
        missing.append("RadionuclideHalfLife")
    if rescale_slope is None or rescale_slope <= 0.0:
        missing.append("RescaleSlope")

    if missing:
        raise AbstentionError(
            AbstentionCode.UNCALIBRATED_SUV_METADATA,
            "Cannot compute calibrated SUV; invalid or missing DICOM quantitative metadata: "
            + ", ".join(missing),
            details={"invalid_fields": missing},
        )


def assert_lesion_nyquist_recoverable(
    *,
    diameter_mm: float,
    voxel_size_mm: tuple[float, float, float],
    effective_fwhm_mm: float,
    min_voxel_ratio: float = 1.5,
    min_fwhm_ratio: float = 0.75,
) -> None:
    """Abstain when lesion diameter falls below spatial Nyquist sampling or PSF recoverability."""
    max_voxel = max(voxel_size_mm)
    if diameter_mm < min_voxel_ratio * max_voxel or diameter_mm < min_fwhm_ratio * effective_fwhm_mm:
        raise AbstentionError(
            AbstentionCode.SUB_NYQUIST_LESION_UNRECOVERABLE,
            (
                f"Lesion diameter {diameter_mm:.2f} mm is sub-Nyquist / unrecoverable for "
                f"max voxel {max_voxel:.2f} mm and effective PSF FWHM {effective_fwhm_mm:.2f} mm."
            ),
            details={
                "diameter_mm": diameter_mm,
                "max_voxel_mm": max_voxel,
                "effective_fwhm_mm": effective_fwhm_mm,
                "min_voxel_ratio": min_voxel_ratio,
                "min_fwhm_ratio": min_fwhm_ratio,
            },
        )


def assert_ct_ac_within_tolerance(
    *,
    apparent_suv_percent_error: float,
    breathing_shift_mm: float,
    max_percent_error: float = 20.0,
) -> None:
    """Abstain when CT-PET/SPECT respiratory AC mismatch induces severe attenuation bias."""
    if abs(apparent_suv_percent_error) > max_percent_error:
        raise AbstentionError(
            AbstentionCode.SEVERE_CT_AC_MISMATCH,
            (
                f"CT-AC respiratory phase shift ({breathing_shift_mm:+.1f} mm) induces "
                f"{apparent_suv_percent_error:+.1f}% attenuation-corrected SUV bias "
                f"(exceeds +/-{max_percent_error:.1f}% quantitative tolerance)."
            ),
            details={
                "breathing_shift_mm": breathing_shift_mm,
                "apparent_suv_percent_error": apparent_suv_percent_error,
                "max_percent_error": max_percent_error,
            },
        )


def assert_earl_corridor_feasible(
    *,
    feasible: bool,
    scanner_id: str,
    standard_name: str,
    max_violation: float,
) -> None:
    """Abstain when no non-negative post-reconstruction filter can bring scanner into corridor."""
    if not feasible:
        raise AbstentionError(
            AbstentionCode.EARL_CORRIDOR_INFEASIBLE,
            (
                f"Scanner '{scanner_id}' cannot be harmonized into {standard_name} corridor "
                f"via post-reconstruction filtering (residual violation={max_violation:.4f})."
            ),
            details={
                "scanner_id": scanner_id,
                "standard_name": standard_name,
                "max_violation": max_violation,
            },
        )


def assert_sufficient_timepoints_for_tiac(
    times_hours: list[float],
    min_points: int = 2,
) -> None:
    """Abstain when too few or non-monotonic post-injection timepoints are provided for TIAC fitting."""
    if len(times_hours) < min_points:
        raise AbstentionError(
            AbstentionCode.INSUFFICIENT_TIMEPOINTS_FOR_TIAC,
            f"Need at least {min_points} distinct post-injection timepoints for TIAC integration; got {len(times_hours)}.",
            details={"n_timepoints": len(times_hours), "min_points": min_points},
        )
    if any(times_hours[i] >= times_hours[i + 1] for i in range(len(times_hours) - 1)):
        raise AbstentionError(
            AbstentionCode.INSUFFICIENT_TIMEPOINTS_FOR_TIAC,
            "Time-activity curve timestamps must be strictly increasing for TIAC dosimetry fitting.",
            details={"times_hours": times_hours},
        )
