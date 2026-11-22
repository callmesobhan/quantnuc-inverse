"""PVE-Propagated MIRD-23 / MIRD-26 Theranostic Absorbed-Dose (Gy) & TIAC Engine.

Connects partial-volume-corrected multi-timepoint quantitative SPECT/PET activity recovery
directly into MIRD unit-density sphere S-factors and Time-Integrated Activity Coefficient (TIAC)
integration for radiopharmaceutical therapy (`177Lu`, `131I`, `90Y`).

References:
- Dewaraja et al. (2012) MIRD Pamphlet No. 23, *J Nucl Med* 53(8):1310-1325, DOI:10.2967/jnumed.111.100123
- Ljungberg et al. (2016) MIRD Pamphlet No. 26 (177Lu SPECT Dosimetry), *J Nucl Med* 57(1):151-162, DOI:10.2967/jnumed.115.159012

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.optimize import curve_fit

from quantnuc_inverse.ir import (
    LesionROI,
    ReconConfig,
    ScannerSpec,
    TheranosticDoseResult,
    TimeActivityPoint,
)
from quantnuc_inverse.provenance import (
    assert_sufficient_timepoints_for_tiac,
    build_provenance_record,
)
from quantnuc_inverse.pve_inverse_solver import invert_lesion_activity

TheranosticIsotope = Literal["Lu-177", "I-131", "Y-90"]

# Physical decay constants & self-irradiation beta/gamma constants (ICRP-107 / MIRD-26)
# delta_beta_mgy_g_per_mbq_s = 1.6021766e-13 [J/MeV] * E_mean_beta [MeV] * 1e6 [Bq/MBq] * 1e3 [g/kg] * 1e3 [mGy/Gy]
#                            = 0.16021766 * E_mean_beta [MeV]
THERANOSTIC_ISOTOPE_PHYSICS: dict[TheranosticIsotope, dict[str, float]] = {
    "Lu-177": {
        "half_life_h": 159.528,  # 6.647 days
        "mean_beta_mev": 0.1479,
        "beta_range_mm": 0.67,
        "gamma_self_const": 0.00055,  # small 208/113 keV photon self-absorption term
    },
    "I-131": {
        "half_life_h": 192.552,  # 8.023 days
        "mean_beta_mev": 0.1917,
        "beta_range_mm": 0.90,
        "gamma_self_const": 0.0052,  # substantial 364 keV gamma self-absorption
    },
    "Y-90": {
        "half_life_h": 64.053,
        "mean_beta_mev": 0.9331,
        "beta_range_mm": 4.90,  # significant beta escape in small <20 mm spheres
        "gamma_self_const": 0.0,
    },
}


def compute_mird_sphere_s_factor(
    diameter_mm: float,
    isotope: TheranosticIsotope = "Lu-177",
    tissue_density_g_cm3: float = 1.04,
) -> tuple[float, float]:
    """Compute MIRD self-irradiation sphere S-factor [mGy/(MBq*s)] and tumor mass [g].

    Includes finite-sphere beta absorbed fraction phi_beta(R / R_csda) and photon self-absorption
    calibrated against OLINDA/MIRD-26 unit-density sphere tables.
    """
    if diameter_mm <= 0.0:
        raise ValueError("diameter_mm must be strictly positive.")
    phys = THERANOSTIC_ISOTOPE_PHYSICS[isotope]
    radius_mm = diameter_mm / 2.0
    volume_ml = (math.pi / 6.0) * (diameter_mm / 10.0) ** 3
    mass_g = volume_ml * tissue_density_g_cm3

    # Beta absorbed fraction in a sphere of radius R for CSDA range R_beta:
    # Accurate analytical approximation to Monte Carlo sphere phi_beta(x) where x = R / R_beta
    x = radius_mm / max(phys["beta_range_mm"], 0.05)
    phi_beta = 1.0 - 0.75 / max(x, 0.25) + 0.12 / (max(x, 0.25) ** 2)
    phi_beta = float(np.clip(phi_beta, 0.15, 0.998))

    delta_beta = 0.16021766 * phys["mean_beta_mev"]  # mGy * g / (MBq * s)
    # Photon self-absorption grows roughly linearly with sphere radius (~mu_en * R)
    s_beta = (delta_beta * phi_beta) / mass_g
    s_gamma = (phys["gamma_self_const"] * (radius_mm / 10.0)) / mass_g
    return float(s_beta + s_gamma), float(mass_g)


def fit_tiac_mbq_h_per_ml(
    times_h: list[float],
    activities_mbq_ml: list[float],
    isotope: TheranosticIsotope = "Lu-177",
    fit_model: Literal["AUTO", "MONO_EXPONENTIAL", "BI_EXPONENTIAL"] = "AUTO",
) -> tuple[float, float, Literal["MONO_EXPONENTIAL", "BI_EXPONENTIAL"]]:
    """Fit post-injection time-activity curve and return (TIAC [MBq*h/mL], T_eff [h], chosen_model).

    Enforces physical decay ceiling (lambda_eff >= lambda_phys) and raises
    `AbstentionError(INSUFFICIENT_TIMEPOINTS_FOR_TIAC)` when < 2 valid timepoints are supplied.
    """
    assert_sufficient_timepoints_for_tiac(times_h, min_points=2)
    t = np.asarray(times_h, dtype=np.float64)
    a = np.maximum(np.asarray(activities_mbq_ml, dtype=np.float64), 1e-6)
    t_half_phys = THERANOSTIC_ISOTOPE_PHYSICS[isotope]["half_life_h"]
    lam_phys = math.log(2.0) / t_half_phys

    # Decide mono-exponential vs bi-exponential uptake+clearance
    use_biexp = (fit_model == "BI_EXPONENTIAL" and len(t) >= 3) or (
        fit_model == "AUTO" and len(t) >= 4 and int(np.argmax(a)) > 0
    )

    if use_biexp:
        def _biexp(x: np.ndarray, a0: float, lam_eff: float, lam_up: float) -> np.ndarray:
            return a0 * (np.exp(-lam_eff * x) - np.exp(-lam_up * x))

        p0 = [float(np.max(a) * 1.4), max(lam_phys * 1.3, 0.008), 0.15]
        bounds = ([0.0, lam_phys, 0.02], [float(np.max(a) * 20.0), 1.0, 5.0])
        try:
            popt, _ = curve_fit(_biexp, t, a, p0=p0, bounds=bounds, maxfev=4000)
            a0_fit, lam_eff_fit, lam_up_fit = float(popt[0]), float(popt[1]), float(popt[2])
            if lam_up_fit > lam_eff_fit * 1.05:
                tiac = a0_fit * (1.0 / lam_eff_fit - 1.0 / lam_up_fit)
                t_eff_h = math.log(2.0) / lam_eff_fit
                return float(max(0.0, tiac)), float(t_eff_h), "BI_EXPONENTIAL"
        except Exception:
            pass

    # Mono-exponential log-linear fit with physical decay floor (lambda_eff >= lambda_phys)
    log_a = np.log(a)
    slope, intercept = np.polyfit(t, log_a, 1)
    lam_eff = max(lam_phys, float(-slope))
    # Re-estimate optimal A0 for clamped lambda_eff via least squares: A0 = sum(a * e^-lt) / sum(e^-2lt)
    basis = np.exp(-lam_eff * t)
    a0 = float(np.dot(a, basis) / max(float(np.dot(basis, basis)), 1e-12))
    tiac = a0 / lam_eff
    t_eff_h = math.log(2.0) / lam_eff
    return float(max(0.0, tiac)), float(t_eff_h), "MONO_EXPONENTIAL"


def compute_pve_propagated_theranostic_dose(
    *,
    lesion_id: str,
    lesion_diameter_mm: float,
    timepoints: list[TimeActivityPoint],
    scanner: ScannerSpec,
    recon: ReconConfig,
    isotope: TheranosticIsotope = "Lu-177",
    fit_model: Literal["AUTO", "MONO_EXPONENTIAL", "BI_EXPONENTIAL"] = "AUTO",
    radial_offset_mm: float = 40.0,
    tissue_density_g_cm3: float = 1.04,
) -> TheranosticDoseResult:
    """Propagate Rousset GTM PVE recovery [p05, nominal, p95] into MIRD-23/26 tumor absorbed dose (Gy)."""
    times = [tp.time_hours for tp in timepoints]
    assert_sufficient_timepoints_for_tiac(times, min_points=2)

    obs_acts = [tp.observed_activity_mbq_ml for tp in timepoints]
    pvc_nom_acts: list[float] = []
    pvc_p05_acts: list[float] = []
    pvc_p95_acts: list[float] = []

    for idx, tp in enumerate(timepoints):
        roi = LesionROI(
            lesion_id=f"{lesion_id}_t{idx}",
            diameter_mm=lesion_diameter_mm,
            observed_suv_mean=tp.observed_activity_mbq_ml,
            background_suv=tp.background_activity_mbq_ml,
            radial_offset_mm=radial_offset_mm,
        )
        rec = invert_lesion_activity(
            roi,
            scanner,
            recon,
            method="GTM_EXACT",
            seed=42 + idx,
        )
        pvc_nom_acts.append(rec.recovered_activity_nominal)
        pvc_p05_acts.append(rec.recovered_activity_p05)
        pvc_p95_acts.append(rec.recovered_activity_p95)

    uncorr_tiac, _, _ = fit_tiac_mbq_h_per_ml(times, obs_acts, isotope=isotope, fit_model=fit_model)
    pvc_tiac, t_eff_h, chosen_model = fit_tiac_mbq_h_per_ml(
        times, pvc_nom_acts, isotope=isotope, fit_model=fit_model
    )
    pvc_tiac_p05, _, _ = fit_tiac_mbq_h_per_ml(times, pvc_p05_acts, isotope=isotope, fit_model=chosen_model)
    pvc_tiac_p95, _, _ = fit_tiac_mbq_h_per_ml(times, pvc_p95_acts, isotope=isotope, fit_model=chosen_model)

    s_factor, mass_g = compute_mird_sphere_s_factor(
        lesion_diameter_mm, isotope=isotope, tissue_density_g_cm3=tissue_density_g_cm3
    )
    volume_ml = mass_g / tissue_density_g_cm3

    # Dose [Gy] = TIAC [MBq*h/mL] * Volume [mL] * 3600 [s/h] * S [mGy/(MBq*s)] * 1e-3 [Gy/mGy]
    factor_gy = volume_ml * 3600.0 * s_factor * 1e-3
    uncorr_dose_gy = uncorr_tiac * factor_gy
    pvc_dose_nom = pvc_tiac * factor_gy
    pvc_dose_p05 = min(pvc_dose_nom, pvc_tiac_p05 * factor_gy)
    pvc_dose_p95 = max(pvc_dose_nom, pvc_tiac_p95 * factor_gy)

    underest_pct = ((uncorr_dose_gy - pvc_dose_nom) / max(pvc_dose_nom, 1e-9)) * 100.0

    prov = build_provenance_record(
        operation="mird26_pve_propagated_theranostic_dose",
        inputs={
            "lesion_id": lesion_id,
            "lesion_diameter_mm": lesion_diameter_mm,
            "isotope": isotope,
            "n_timepoints": len(timepoints),
            "scanner_id": scanner.scanner_id,
        },
        outputs={
            "uncorrected_dose_gy": uncorr_dose_gy,
            "pvc_dose_nominal_gy": pvc_dose_nom,
            "pvc_dose_p05_gy": pvc_dose_p05,
            "pvc_dose_p95_gy": pvc_dose_p95,
            "pve_dose_underestimation_pct": underest_pct,
        },
        data_source_tag="MIRD26_THERANOSTICS_DOSIMETRY_MODEL",
        citations=[
            "Dewaraja et al. (2012) MIRD Pamphlet No. 23, DOI:10.2967/jnumed.111.100123",
            "Ljungberg et al. (2016) MIRD Pamphlet No. 26, DOI:10.2967/jnumed.115.159012",
            "Rousset, Ma, & Evans (1998) J Nucl Med 39(5):904-911, PMID:9591599",
        ],
    )

    return TheranosticDoseResult(
        lesion_id=lesion_id,
        isotope=isotope,
        lesion_diameter_mm=float(lesion_diameter_mm),
        lesion_mass_g=float(round(mass_g, 3)),
        fit_model=chosen_model,
        effective_half_life_h=float(round(t_eff_h, 2)),
        s_factor_mgy_per_mbq_s=float(s_factor),
        uncorrected_tiac_mbq_h_per_ml=float(round(uncorr_tiac, 4)),
        pvc_tiac_mbq_h_per_ml=float(round(pvc_tiac, 4)),
        uncorrected_dose_gy=float(round(uncorr_dose_gy, 3)),
        pvc_dose_nominal_gy=float(round(pvc_dose_nom, 3)),
        pvc_dose_p05_gy=float(round(pvc_dose_p05, 3)),
        pvc_dose_p95_gy=float(round(pvc_dose_p95, 3)),
        pve_dose_underestimation_pct=float(round(underest_pct, 2)),
        provenance=prov,
    )
