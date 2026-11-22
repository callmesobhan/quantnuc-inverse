"""quantnuc-inverse: Quantitative PET/SPECT Image-Formation, PVE, PSF-Gibbs & EARL Harmonization Engine.

Research Software — NOT A MEDICAL DEVICE.
"""

__version__ = "0.1.0"

from quantnuc_inverse.dicom_io import (
    QuantDicomVolume,
    build_synthetic_ct_dicom,
    build_synthetic_nucmed_dicom,
    load_quantitative_nucmed_dicom,
)
from quantnuc_inverse.earl_harmonizer import (
    build_reference_scanner_fleet,
    get_corridor_spec,
    synthesize_harmonization_filter,
)
from quantnuc_inverse.forward_image_formation import (
    apply_forward_system_operator,
    build_nema_iec_phantom_dro,
    compute_ct_ac_mismatch_report,
    compute_sphere_recovery_coefficients,
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
    ProvenanceRecord,
    ReconConfig,
    ScannerSpec,
)
from quantnuc_inverse.provenance import ProvenanceLedger, compute_sha256
from quantnuc_inverse.pve_inverse_solver import (
    apply_muller_gartner_pvc,
    invert_lesion_activity,
    solve_gtm_multi_region,
)
from quantnuc_inverse.sobol_sensitivity import SobolSensitivityResult, run_sobol_sensitivity

__all__ = [
    "ACArtifactReport",
    "AbstentionCode",
    "AbstentionError",
    "ActivityRecoveryResult",
    "EARLCorridorSpec",
    "HarmonizationFilterResult",
    "LesionROI",
    "NEMASphereSpec",
    "ProvenanceLedger",
    "ProvenanceRecord",
    "QuantDicomVolume",
    "RESEARCH_DISCLAIMER",
    "ReconConfig",
    "ScannerSpec",
    "SobolSensitivityResult",
    "__version__",
    "apply_forward_system_operator",
    "apply_muller_gartner_pvc",
    "build_nema_iec_phantom_dro",
    "build_reference_scanner_fleet",
    "build_synthetic_ct_dicom",
    "build_synthetic_nucmed_dicom",
    "compute_ct_ac_mismatch_report",
    "compute_sha256",
    "compute_sphere_recovery_coefficients",
    "get_corridor_spec",
    "invert_lesion_activity",
    "load_quantitative_nucmed_dicom",
    "run_sobol_sensitivity",
    "simulate_nema_rc_curve",
    "solve_gtm_multi_region",
    "synthesize_harmonization_filter",
]
