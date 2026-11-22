"""quantnuc-inverse: Quantitative PET/SPECT Image-Formation, PVE, PSF-Gibbs & EARL Harmonization Engine.

Research Software — NOT A MEDICAL DEVICE.
"""

__version__ = "0.2.0"

from quantnuc_inverse.dicom_io import (
    QuantDicomVolume,
    build_synthetic_ct_dicom,
    build_synthetic_dicom_series,
    build_synthetic_nucmed_dicom,
    load_ct_dicom_series_3d,
    load_dicom_series_3d,
    load_quantitative_nucmed_dicom,
    resample_ct_to_pet_grid,
)
from quantnuc_inverse.earl_harmonizer import (
    build_reference_scanner_fleet,
    compute_harmonization_pareto_frontier,
    estimate_background_cov_pct,
    get_corridor_spec,
    synthesize_harmonization_filter,
)
from quantnuc_inverse.forward_image_formation import (
    apply_forward_system_operator,
    build_nema_iec_phantom_dro,
    compute_3d_ct_ac_bias_volume,
    compute_ct_ac_mismatch_report,
    compute_sphere_recovery_coefficients,
    ct_hu_to_linear_attenuation_cm,
    isotope_positron_range_fwhm_mm,
    simulate_nema_rc_curve,
)
from quantnuc_inverse.ir import (
    RESEARCH_DISCLAIMER,
    AbstentionCode,
    AbstentionError,
    ACArtifactReport,
    ActivityRecoveryResult,
    EARLCorridorSpec,
    HarmonizationFilterResult,
    LesionROI,
    NEMASphereSpec,
    ParetoHarmonizationPoint,
    ProvenanceRecord,
    ReconConfig,
    ScannerSpec,
    TheranosticDoseResult,
    TimeActivityPoint,
    VolumetricCTACReport,
)
from quantnuc_inverse.provenance import ProvenanceLedger, compute_sha256
from quantnuc_inverse.pve_inverse_solver import (
    apply_muller_gartner_pvc,
    apply_rbv_pvc,
    build_necrotic_tumor_phantom,
    invert_lesion_activity,
    solve_gtm_multi_region,
)
from quantnuc_inverse.sobol_sensitivity import SobolSensitivityResult, run_sobol_sensitivity
from quantnuc_inverse.theranostics_dosimetry import (
    compute_mird_sphere_s_factor,
    compute_pve_propagated_theranostic_dose,
    fit_tiac_mbq_h_per_ml,
)

__all__ = [
    "ACArtifactReport",
    "AbstentionCode",
    "AbstentionError",
    "ActivityRecoveryResult",
    "EARLCorridorSpec",
    "HarmonizationFilterResult",
    "LesionROI",
    "NEMASphereSpec",
    "ParetoHarmonizationPoint",
    "ProvenanceLedger",
    "ProvenanceRecord",
    "QuantDicomVolume",
    "RESEARCH_DISCLAIMER",
    "ReconConfig",
    "ScannerSpec",
    "SobolSensitivityResult",
    "TheranosticDoseResult",
    "TimeActivityPoint",
    "VolumetricCTACReport",
    "__version__",
    "apply_forward_system_operator",
    "apply_muller_gartner_pvc",
    "apply_rbv_pvc",
    "build_necrotic_tumor_phantom",
    "build_nema_iec_phantom_dro",
    "build_reference_scanner_fleet",
    "build_synthetic_ct_dicom",
    "build_synthetic_dicom_series",
    "build_synthetic_nucmed_dicom",
    "compute_3d_ct_ac_bias_volume",
    "compute_ct_ac_mismatch_report",
    "compute_harmonization_pareto_frontier",
    "compute_mird_sphere_s_factor",
    "compute_pve_propagated_theranostic_dose",
    "compute_sha256",
    "compute_sphere_recovery_coefficients",
    "ct_hu_to_linear_attenuation_cm",
    "estimate_background_cov_pct",
    "fit_tiac_mbq_h_per_ml",
    "get_corridor_spec",
    "invert_lesion_activity",
    "isotope_positron_range_fwhm_mm",
    "load_ct_dicom_series_3d",
    "load_dicom_series_3d",
    "load_quantitative_nucmed_dicom",
    "resample_ct_to_pet_grid",
    "run_sobol_sensitivity",
    "simulate_nema_rc_curve",
    "solve_gtm_multi_region",
    "synthesize_harmonization_filter",
]
