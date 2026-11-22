"""Strictly typed Pydantic v2 Physics Intermediate Representation (IR) for quantnuc-inverse.

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RESEARCH_DISCLAIMER = "Research Software — NOT A MEDICAL DEVICE"


class AbstentionCode(StrEnum):
    """Hard quantitative safety & validity abstention gates."""

    UNCALIBRATED_SUV_METADATA = "UNCALIBRATED_SUV_METADATA"
    SEVERE_CT_AC_MISMATCH = "SEVERE_CT_AC_MISMATCH"
    SUB_NYQUIST_LESION_UNRECOVERABLE = "SUB_NYQUIST_LESION_UNRECOVERABLE"
    EARL_CORRIDOR_INFEASIBLE = "EARL_CORRIDOR_INFEASIBLE"
    MISSING_DECAY_CORRECTION = "MISSING_DECAY_CORRECTION"
    INSUFFICIENT_TIMEPOINTS_FOR_TIAC = "INSUFFICIENT_TIMEPOINTS_FOR_TIAC"


class AbstentionError(ValueError):
    """Exception raised when a hard physical or metadata validity gate is violated."""

    def __init__(
        self,
        code: AbstentionCode,
        message: str,
        details: dict[str, Any] | None = None,
        provenance_sha256: str | None = None,
    ) -> None:
        super().__init__(f"[{code.value}] {message}")
        self.code = code
        self.message = message
        self.details = details or {}
        self.provenance_sha256 = provenance_sha256

    def to_dict(self) -> dict[str, Any]:
        """Serialize structured abstention payload for JSON-RPC / audit logs."""
        return {
            "abstained": True,
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
            "provenance_sha256": self.provenance_sha256,
            "disclaimer": RESEARCH_DISCLAIMER,
        }


class ProvenanceRecord(BaseModel):
    """Cryptographic SHA-256 provenance record attached to every physics output."""

    model_config = ConfigDict(frozen=True)

    record_id: str = Field(..., min_length=4)
    engine_version: str = Field(default="0.2.0")
    timestamp_utc: str
    operation: str
    input_sha256: str = Field(..., min_length=64, max_length=64)
    output_sha256: str = Field(..., min_length=64, max_length=64)
    data_source_tag: str = Field(default="ANALYTICAL_PHYSICS_MODEL")
    citations: list[str] = Field(default_factory=list)
    disclaimer: str = Field(default=RESEARCH_DISCLAIMER)


ScannerModality = Literal["PET_SIPM", "PET_PMT", "PET_TOF_PSF", "SPECT_CZT", "SPECT_NAI"]
IsotopeName = Literal["F-18", "Ga-68", "Zr-89", "I-124", "Lu-177", "Tc-99m", "I-131", "Y-90"]


class ScannerSpec(BaseModel):
    """Physical imaging system specification for PET or quantitative SPECT/CT."""

    model_config = ConfigDict(frozen=True)

    scanner_id: str = Field(..., min_length=1)
    vendor_model: str = Field(..., min_length=1)
    modality: ScannerModality
    psf_fwhm_xy_mm: float = Field(..., gt=0.5, le=25.0)
    psf_fwhm_z_mm: float = Field(..., gt=0.5, le=25.0)
    voxel_size_mm: tuple[float, float, float] = Field(
        default=(2.0, 2.0, 2.0),
        description="Voxel spacing (dz, dy, dx) in mm.",
    )
    radial_psf_degradation_mm_per_10cm: float = Field(
        default=0.45,
        ge=0.0,
        le=8.0,
        description="Off-axis parallax (PET) or depth-dependent CDR (SPECT) broadening in mm per 10 cm.",
    )
    tof_resolution_ps: float | None = Field(
        default=None,
        gt=0.0,
        le=2000.0,
        description="Time-of-flight coincidence timing resolution in picoseconds (PET only).",
    )
    has_psf_modeling: bool = Field(default=False)
    gibbs_ringing_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=0.60,
        description="Dimensionless PSF-induced oscillatory Gibbs edge overshoot amplitude.",
    )
    sensitivity_cps_per_kbq: float = Field(default=12.0, gt=0.01)
    isotope: IsotopeName = Field(default="F-18")
    include_positron_range: bool = Field(
        default=False,
        description="Include isotope-specific positron-range blur scaled by medium density (Conti & Eriksson 2016).",
    )
    medium_density_g_cm3: float = Field(
        default=1.0,
        gt=0.05,
        le=3.0,
        description="Local tissue density in g/cm^3 (1.0 water/soft tissue, ~0.27 inflated lung).",
    )

    @model_validator(mode="after")
    def _validate_voxel_spacing(self) -> ScannerSpec:
        if any(v <= 0.0 or v > 20.0 for v in self.voxel_size_mm):
            raise ValueError("All voxel_size_mm entries must be in (0, 20] mm.")
        return self


class ReconConfig(BaseModel):
    """Iterative reconstruction and post-filtering configuration."""

    model_config = ConfigDict(frozen=True)

    algorithm: Literal["OSEM", "PSF_OSEM", "TOF_PSF_OSEM", "BSREM"] = Field(default="OSEM")
    iterations: int = Field(default=3, ge=1, le=30)
    subsets: int = Field(default=16, ge=1, le=64)
    post_filter_fwhm_mm: float = Field(default=4.0, ge=0.0, le=20.0)
    filter_type: Literal["gaussian", "butterworth"] = Field(default="gaussian")
    butterworth_order: int = Field(default=2, ge=1, le=12)
    bsrem_beta: float | None = Field(default=None, ge=0.0, le=5000.0)
    convergence_rate_k: float = Field(default=0.055, gt=0.001, le=1.0)


class NEMASphereSpec(BaseModel):
    """Single NEMA NU 2-2018 IEC Body Phantom fillable sphere specification."""

    model_config = ConfigDict(frozen=True)

    diameter_mm: float = Field(..., gt=0.0, le=100.0)
    center_mm: tuple[float, float, float] = Field(
        ...,
        description="Sphere center (z, y, x) in mm relative to phantom center.",
    )
    true_activity_bq_ml: float = Field(..., gt=0.0)
    background_activity_bq_ml: float = Field(..., ge=0.0)
    wall_thickness_mm: float = Field(default=1.0, ge=0.0, le=3.0)


class LesionROI(BaseModel):
    """Observed clinical or phantom lesion region of interest for inverse recovery."""

    model_config = ConfigDict(frozen=True)

    lesion_id: str = Field(..., min_length=1)
    diameter_mm: float = Field(..., gt=0.0, le=200.0)
    observed_suv_mean: float = Field(..., ge=0.0)
    observed_suv_max: float | None = Field(default=None, ge=0.0)
    observed_suv_peak: float | None = Field(default=None, ge=0.0)
    background_suv: float = Field(..., ge=0.0)
    radial_offset_mm: float = Field(default=0.0, ge=0.0, le=350.0)
    delineation_error_mm: float = Field(default=0.0, ge=-10.0, le=10.0)
    is_near_lung_diaphragm: bool = Field(default=False)
    ct_pet_shift_mm: float = Field(default=0.0, ge=-30.0, le=30.0)


class EARLCorridorSpec(BaseModel):
    """EANM/EARL, QIBA, or MIRD-23 multi-sphere Recovery Coefficient acceptability corridor."""

    model_config = ConfigDict(frozen=True)

    standard_name: Literal["EARL1", "EARL2", "QIBA_FDG_PET", "MIRD23_SPECT"]
    sphere_diameters_mm: list[float]
    rc_mean_lower: list[float]
    rc_mean_upper: list[float]
    rc_max_lower: list[float]
    rc_max_upper: list[float]
    rc_peak_lower: list[float] | None = None
    rc_peak_upper: list[float] | None = None
    reference_citation: str

    @model_validator(mode="after")
    def _validate_corridor_lengths(self) -> EARLCorridorSpec:
        n = len(self.sphere_diameters_mm)
        for field_name in ("rc_mean_lower", "rc_mean_upper", "rc_max_lower", "rc_max_upper"):
            if len(getattr(self, field_name)) != n:
                raise ValueError(f"{field_name} length must match sphere_diameters_mm ({n}).")
        for lo, hi in zip(self.rc_mean_lower, self.rc_mean_upper, strict=True):
            if lo > hi:
                raise ValueError("rc_mean_lower must not exceed rc_mean_upper.")
        for lo, hi in zip(self.rc_max_lower, self.rc_max_upper, strict=True):
            if lo > hi:
                raise ValueError("rc_max_lower must not exceed rc_max_upper.")
        if (self.rc_peak_lower is not None or self.rc_peak_upper is not None) and (
            self.rc_peak_lower is None
            or self.rc_peak_upper is None
            or len(self.rc_peak_lower) != n
            or len(self.rc_peak_upper) != n
        ):
            raise ValueError("rc_peak_lower and rc_peak_upper must both match sphere count.")
        return self


class ActivityRecoveryResult(BaseModel):
    """Inverse partial-volume activity recovery output with calibrated bias-variance intervals."""

    model_config = ConfigDict(frozen=True)

    lesion_id: str
    method: Literal["GTM_EXACT", "MULLER_GARTNER", "BAYESIAN_REGULARIZED"]
    observed_activity: float
    recovered_activity_nominal: float
    recovered_activity_p05: float
    recovered_activity_p95: float
    recovery_coefficient_estimated: float
    gibbs_overshoot_factor: float
    forward_reprojection_residual: float
    total_activity_conservation_error: float
    condition_number: float
    provenance: ProvenanceRecord


class ACArtifactReport(BaseModel):
    """CT-PET/SPECT respiratory attenuation-correction phase mismatch forensics report."""

    model_config = ConfigDict(frozen=True)

    breathing_shift_mm: float
    photon_energy_kev: float
    mu_lung_per_cm: float
    mu_liver_per_cm: float
    delta_mu_integral: float
    attenuation_bias_factor: float
    apparent_suv_percent_error: float
    artifact_polarity: Literal["FALSE_COLD", "FALSE_HOT", "NEGLIGIBLE"]
    exceeds_abstention_threshold: bool
    provenance: ProvenanceRecord


class HarmonizationFilterResult(BaseModel):
    """Synthesized cross-scanner harmonization filter and pre/post recovery coefficients."""

    model_config = ConfigDict(frozen=True)

    scanner_id: str
    standard_name: str
    feasible: bool
    filter_type: Literal["gaussian", "butterworth"]
    optimal_post_filter_fwhm_mm: float
    butterworth_cutoff_cycles_per_mm: float | None = None
    recommended_iterations: int
    recommended_bsrem_beta: float | None = None
    pre_filter_rc_mean: dict[str, float]
    pre_filter_rc_max: dict[str, float]
    pre_filter_rc_peak: dict[str, float]
    post_filter_rc_mean: dict[str, float]
    post_filter_rc_max: dict[str, float]
    post_filter_rc_peak: dict[str, float]
    max_corridor_violation_before: float
    max_corridor_violation_after: float
    sharpness_retention_ratio: float
    background_cov_pct: float = Field(default=0.0, ge=0.0)
    max_allowed_cov_pct: float = Field(default=15.0, gt=0.0)
    noise_compliant: bool = Field(default=True)
    cnr_score: float = Field(default=0.0, ge=0.0)
    provenance: ProvenanceRecord


class ParetoHarmonizationPoint(BaseModel):
    """Single point on the EARL RC-compliance vs. background-noise COV & CNR Pareto frontier."""

    model_config = ConfigDict(frozen=True)

    post_filter_fwhm_mm: float
    iterations: int
    max_corridor_violation: float
    background_cov_pct: float
    cnr_score: float
    sharpness_retention_ratio: float
    feasible_earl_and_noise: bool


class VolumetricCTACReport(BaseModel):
    """3D bilinear HU-to-mu(E) ray-traced CT-AC respiratory mismatch report."""

    model_config = ConfigDict(frozen=True)

    shift_mm: tuple[float, float, float]
    photon_energy_kev: float
    roi_mean_bias_factor: float
    roi_mean_percent_error: float
    max_abs_percent_error: float
    artifact_polarity: Literal["FALSE_COLD", "FALSE_HOT", "NEGLIGIBLE"]
    exceeds_abstention_threshold: bool
    provenance: ProvenanceRecord


class TimeActivityPoint(BaseModel):
    """Single quantitative post-injection time-activity sample for theranostic TIAC fitting."""

    model_config = ConfigDict(frozen=True)

    time_hours: float = Field(..., ge=0.0, le=2000.0)
    observed_activity_mbq_ml: float = Field(..., ge=0.0)
    background_activity_mbq_ml: float = Field(default=0.0, ge=0.0)


class TheranosticDoseResult(BaseModel):
    """PVE-propagated MIRD-23/26 theranostic tumor absorbed dose (Gy) with 90% uncertainty interval."""

    model_config = ConfigDict(frozen=True)

    lesion_id: str
    isotope: Literal["Lu-177", "I-131", "Y-90"]
    lesion_diameter_mm: float
    lesion_mass_g: float
    fit_model: Literal["MONO_EXPONENTIAL", "BI_EXPONENTIAL"]
    effective_half_life_h: float
    s_factor_mgy_per_mbq_s: float
    uncorrected_tiac_mbq_h_per_ml: float
    pvc_tiac_mbq_h_per_ml: float
    uncorrected_dose_gy: float
    pvc_dose_nominal_gy: float
    pvc_dose_p05_gy: float
    pvc_dose_p95_gy: float
    pve_dose_underestimation_pct: float
    provenance: ProvenanceRecord
