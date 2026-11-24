"""EANM Gear-2018 Full-Covariance Absorbed-Dose Uncertainty & Fisher-Optimal STP SPECT/CT Designer.

Implements two capabilities absent from commercial RPT dosimetry platforms (MIM SurePlan MRT,
Hermes Dosimetry, DOSIsoft PLANET Dose, QDOSE):

1. Full correlated 6-parameter EANM Gear et al. (2018) covariance propagation (`compute_gear2018_dose_uncertainty`),
   explicitly capturing camera calibration, dead-time/TEW scatter, CT-AC respiratory shift,
   volume/S-factor negative covariance cancellation, PVE/delineation recovery sensitivity, and
   pharmacokinetic TIAC variance.
2. Fisher-Optimal Single-Timepoint (STP) vs. D-Optimal 2-Timepoint SPECT/CT Protocol Designer
   (`design_optimal_stp_protocol`) combining Hanscheid (2018) / Madsen (2018) population-prior STP
   RMSE curves with late-timepoint quantum counting noise and Cramer-Rao Fisher Information
   inversion, gated by hard abstention (`STP_PROTOCOL_INFEASIBLE_REQUIRE_MTP`).

References:
- Gear et al. (2018) *Eur J Nucl Med Mol Imaging* 45:2456-2474, DOI:10.1007/s00259-018-4136-7
- Hanscheid et al. (2018) *J Nucl Med* 59(1):75-81, DOI:10.2967/jnumed.117.193706
- Madsen et al. (2018) *Med Phys* 45(5):2318-2324, DOI:10.1002/mp.12886

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from quantnuc_inverse.forward_image_formation import (
    compute_ct_ac_mismatch_report,
    effective_psf_fwhm_mm,
)
from quantnuc_inverse.ir import (
    Gear2018UncertaintyBudget,
    ReconConfig,
    ScannerSpec,
    STPProtocolDesignResult,
)
from quantnuc_inverse.provenance import (
    assert_stp_protocol_feasible,
    build_provenance_record,
)
from quantnuc_inverse.theranostics_dosimetry import (
    THERANOSTIC_ISOTOPE_PHYSICS,
    TheranosticIsotope,
)

STPEstimatorName = Literal["HANSCHEID", "MADSEN"]

# Clinical population effective half-life priors (mean_h, sd_h) from published 177Lu literature
CLINICAL_TEFF_PRIORS: dict[str, tuple[float, float]] = {
    "177Lu-DOTATATE Kidney": (52.0, 12.0),
    "177Lu-DOTATATE NET Tumor": (85.0, 28.0),
    "177Lu-PSMA-617 Kidney": (42.0, 10.0),
    "177Lu-PSMA-617 Tumor": (50.0, 18.0),
    "131I Thyroid Remnant": (68.0, 22.0),
}


def _estimate_pve_delineation_cv(
    lesion_diameter_mm: float,
    scanner: ScannerSpec,
    recon: ReconConfig,
    delineation_sigma_mm: float = 1.0,
) -> tuple[float, float]:
    """Return `(pve_rc_cv, volume_delin_cv)` fractional 1-sigma uncertainties for Gear-2018 propagation."""
    fwhm_xy, fwhm_z = effective_psf_fwhm_mm(scanner, recon, radial_offset_mm=40.0, include_post_filter=True)
    fwhm_eff = (fwhm_xy**2 * fwhm_z) ** (1.0 / 3.0)

    # Smooth 3D spherical RC(D, FWHM) and numerical sensitivity d(ln RC) / d(D) & d(ln RC) / d(FWHM)
    def _rc(d_mm: float, fw_mm: float) -> float:
        x = max(d_mm, 1.0) / max(fw_mm, 0.5)
        return 1.0 / (1.0 + (1.42 / x) ** 2.18)

    rc0 = _rc(lesion_diameter_mm, fwhm_eff)
    eps_d = 0.25
    d_rc_dd = (_rc(lesion_diameter_mm + eps_d, fwhm_eff) - _rc(max(1.5, lesion_diameter_mm - eps_d), fwhm_eff)) / (
        2.0 * eps_d
    )
    eps_f = 0.20
    d_rc_df = (_rc(lesion_diameter_mm, fwhm_eff + eps_f) - _rc(lesion_diameter_mm, max(0.8, fwhm_eff - eps_f))) / (
        2.0 * eps_f
    )

    psf_sigma_mm = 0.45
    rc_delin_cv = abs(d_rc_dd / max(rc0, 1e-4)) * delineation_sigma_mm
    rc_psf_cv = abs(d_rc_df / max(rc0, 1e-4)) * psf_sigma_mm
    pve_rc_cv = math.hypot(rc_delin_cv, rc_psf_cv)

    # Fractional sphere volume uncertainty: dV/V = 3 * dR/R = 6 * sigma_r / D (capped cleanly)
    vol_cv = min(0.65, 3.0 * (delineation_sigma_mm / max(lesion_diameter_mm / 2.0, 1.0)))
    return float(pve_rc_cv), float(vol_cv)


def compute_gear2018_dose_uncertainty(
    *,
    dose_nominal_gy: float,
    lesion_diameter_mm: float,
    scanner: ScannerSpec,
    recon: ReconConfig,
    isotope: TheranosticIsotope = "Lu-177",
    tiac_cv_frac: float = 0.08,
    calibration_cv_frac: float = 0.04,
    deadtime_scatter_cv_frac: float = 0.035,
    ct_ac_shift_mm: float = 1.5,
    delineation_sigma_mm: float = 1.0,
) -> Gear2018UncertaintyBudget:
    """Propagate full EANM Gear et al. (2018) correlated uncertainty covariance into absorbed dose (Gy).

    Explicitly models the 7-element parameter vector:
        theta = [calibration, deadtime_scatter, ct_ac, pve_rc, volume, s_factor, tiac]
    with strictly positive-definite Gear et al. (2018) correlation structure capturing
    volume-S-factor beta self-dose cancellation while preserving residual PVE recovery sensitivity.
    """
    energy_kev = {"Lu-177": 208.0, "I-131": 364.0, "Y-90": 511.0}[isotope]
    ac_rep = compute_ct_ac_mismatch_report(ct_ac_shift_mm, photon_energy_kev=energy_kev)
    # 1-sigma respiratory uncertainty across a shallow-breathing abdominal/thoracic SPECT/CT
    ct_ac_cv = abs(ac_rep.apparent_suv_percent_error / 100.0) * 0.35

    pve_rc_cv, vol_cv = _estimate_pve_delineation_cv(
        lesion_diameter_mm, scanner, recon, delineation_sigma_mm=delineation_sigma_mm
    )
    # S-factor mass scaling sensitivity: S ~ V^(-beta_exp), beta_exp ~ 0.96 for Lu-177, 0.78 for Y-90
    s_mass_exp = 0.78 if isotope == "Y-90" else 0.96
    s_cv = math.hypot(s_mass_exp * vol_cv, 0.025)  # 2.5% nuclear decay / S-value table uncertainty

    # Build strictly positive-definite 7x7 correlation matrix in log-dose space:
    # D = cal * dt * ac * (1/RC) * V * S * TIAC  =>  jac = [+1, +1, +1, -1, +1, +1, +1]
    sigmas = np.array(
        [calibration_cv_frac, deadtime_scatter_cv_frac, ct_ac_cv, pve_rc_cv, vol_cv, s_cv, tiac_cv_frac],
        dtype=np.float64,
    )
    jac = np.array([1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0], dtype=np.float64)

    corr = np.eye(7, dtype=np.float64)
    # Exact rank-consistent physical correlation among (pve_rc, volume, s_factor):
    # volume & s_factor strongly anti-correlated (rho_vs); pve_rc coupled to volume via rho_pv
    rho_vs = -0.96 if isotope != "Y-90" else -0.84
    rho_pv = 0.30
    rho_ps = rho_pv * rho_vs  # ensures 3x3 submatrix is strictly positive definite
    corr[4, 5] = corr[5, 4] = rho_vs
    corr[3, 4] = corr[4, 3] = rho_pv
    corr[3, 5] = corr[5, 3] = rho_ps

    cov_mat = np.outer(sigmas, sigmas) * corr
    total_rel_var = float(max(1e-10, jac @ cov_mat @ jac))
    uncorr_rel_var = float(max(1e-10, np.sum((jac * sigmas) ** 2)))

    dose_cv = math.sqrt(total_rel_var)
    dose_std = dose_nominal_gy * dose_cv
    # Lognormal 5th / 95th percentiles for strictly positive absorbed dose
    sigma_ln = math.sqrt(math.log(1.0 + dose_cv**2))
    p05 = float(dose_nominal_gy * math.exp(-1.64485 * sigma_ln - 0.5 * sigma_ln**2))
    p95 = float(dose_nominal_gy * math.exp(+1.64485 * sigma_ln - 0.5 * sigma_ln**2))

    # Variance budget decomposition (grouping volume + S-factor + PVE-RC into net PVE/delineation block)
    v_cal = float(sigmas[0] ** 2)
    v_dt = float(sigmas[1] ** 2)
    v_ac = float(sigmas[2] ** 2)
    v_tiac = float(sigmas[6] ** 2)
    j_pve = jac[3:6]
    v_pve_net = float(max(0.0, j_pve @ cov_mat[3:6, 3:6] @ j_pve))
    v_sum = max(v_cal + v_dt + v_ac + v_pve_net + v_tiac, 1e-12)

    cov_red_pct = float(max(0.0, (1.0 - total_rel_var / uncorr_rel_var) * 100.0))

    prov = build_provenance_record(
        operation="gear2018_full_covariance_dose_uncertainty",
        inputs={
            "dose_nominal_gy": dose_nominal_gy,
            "lesion_diameter_mm": lesion_diameter_mm,
            "isotope": isotope,
            "tiac_cv_frac": tiac_cv_frac,
            "ct_ac_shift_mm": ct_ac_shift_mm,
        },
        outputs={
            "dose_cv_pct": dose_cv * 100.0,
            "dose_p05_gy": p05,
            "dose_p95_gy": p95,
            "volume_sfactor_covariance_reduction_pct": cov_red_pct,
        },
        data_source_tag="EANM_GEAR2018_COVARIANCE_MODEL",
        citations=["Gear et al. (2018) Eur J Nucl Med Mol Imaging 45:2456-2474, DOI:10.1007/s00259-018-4136-7"],
    )

    return Gear2018UncertaintyBudget(
        isotope=isotope,
        lesion_diameter_mm=float(lesion_diameter_mm),
        dose_nominal_gy=float(round(dose_nominal_gy, 3)),
        dose_std_gy=float(round(dose_std, 3)),
        dose_cv_pct=float(round(dose_cv * 100.0, 2)),
        dose_p05_gy=float(round(p05, 3)),
        dose_p95_gy=float(round(p95, 3)),
        variance_contrib_calibration_pct=float(round(v_cal / v_sum * 100.0, 2)),
        variance_contrib_deadtime_scatter_pct=float(round(v_dt / v_sum * 100.0, 2)),
        variance_contrib_ct_ac_pct=float(round(v_ac / v_sum * 100.0, 2)),
        variance_contrib_pve_delineation_pct=float(round(v_pve_net / v_sum * 100.0, 2)),
        variance_contrib_tiac_kinetics_pct=float(round(v_tiac / v_sum * 100.0, 2)),
        volume_sfactor_covariance_reduction_pct=float(round(cov_red_pct, 2)),
        provenance=prov,
    )


def evaluate_stp_pk_rmse_curve(
    candidate_times_h: np.ndarray,
    *,
    population_teff_mean_h: float,
    population_teff_sd_h: float,
    isotope: TheranosticIsotope = "Lu-177",
    estimator: STPEstimatorName = "HANSCHEID",
    base_counting_cv: float = 0.016,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate exact population-prior STP pharmacokinetic RMSE (%) and counting-noise CV (%) vs scan hour t_1."""
    t_phys = THERANOSTIC_ISOTOPE_PHYSICS[isotope]["half_life_h"]
    # Deterministic Gauss-like stratified quadrature over truncated population T_eff prior [4 h, T_phys]
    q = np.linspace(0.01, 0.99, 199)
    from scipy.stats import norm

    teff_samples = np.clip(
        population_teff_mean_h + population_teff_sd_h * norm.ppf(q),
        4.0,
        t_phys,
    )
    true_tiac = teff_samples / math.log(2.0)

    pk_cv = np.empty(len(candidate_times_h), dtype=np.float64)
    count_cv = np.empty(len(candidate_times_h), dtype=np.float64)

    for idx, t1 in enumerate(candidate_times_h):
        c_t1 = np.exp(-math.log(2.0) * float(t1) / teff_samples)
        if estimator == "HANSCHEID":
            # Hanscheid (2018): TIAC_hat = (2 / ln 2) * t1 * C(t1)
            est_tiac = (2.0 / math.log(2.0)) * float(t1) * c_t1
        else:
            # Madsen (2018): TIAC_hat = (T_pop / ln 2) * 2^(t1 / T_pop) * C(t1)
            est_tiac = (population_teff_mean_h / math.log(2.0)) * (2.0 ** (float(t1) / population_teff_mean_h)) * c_t1

        rel_err = (est_tiac - true_tiac) / true_tiac
        pk_cv[idx] = math.sqrt(float(np.mean(rel_err**2)))
        # Quantum counting noise rises as 1 / sqrt(exp(-ln2 * t1 / T_mean)) = 2^(t1 / (2 * T_mean))
        count_cv[idx] = base_counting_cv * (2.0 ** (float(t1) / (2.0 * max(population_teff_mean_h, 4.0))))

    return pk_cv, count_cv


def compute_fisher_two_timepoint_tiac_cv(
    t1_h: float,
    t2_h: float,
    *,
    teff_mean_h: float,
    base_measurement_cv: float = 0.045,
) -> float:
    """Compute exact Cramer-Rao Fisher Information TIAC CV for a 2-timepoint exponential fit at (t1, t2)."""
    if t2_h <= t1_h + 4.0:
        return 1.0
    lam = math.log(2.0) / max(teff_mean_h, 4.0)
    a0 = 1.0
    fim = np.zeros((2, 2), dtype=np.float64)

    for tk in (t1_h, t2_h):
        c_k = a0 * math.exp(-lam * tk)
        # Measurement variance combines multiplicative calibration/ROI floor + Poisson counting decay
        sigma_k = max(1e-6, c_k * math.hypot(base_measurement_cv, 0.025 * (2.0 ** (tk / (2.0 * teff_mean_h)))))
        grad = np.array([math.exp(-lam * tk), -a0 * tk * math.exp(-lam * tk)], dtype=np.float64)
        fim += np.outer(grad, grad) / (sigma_k**2)

    det = float(np.linalg.det(fim))
    if det <= 1e-18:
        return 1.0
    cov_params = np.linalg.inv(fim)
    # Gradient of TIAC = A0 / lam with respect to (A0, lam)
    g_tiac = np.array([1.0 / lam, -a0 / (lam**2)], dtype=np.float64)
    tiac_true = a0 / lam
    tiac_var = float(max(0.0, g_tiac @ cov_params @ g_tiac))
    return float(math.sqrt(tiac_var) / tiac_true)


def design_optimal_stp_protocol(
    *,
    organ_or_lesion_label: str,
    lesion_diameter_mm: float,
    scanner: ScannerSpec,
    recon: ReconConfig,
    isotope: TheranosticIsotope = "Lu-177",
    population_teff_mean_h: float = 52.0,
    population_teff_sd_h: float = 12.0,
    stp_estimator: STPEstimatorName = "HANSCHEID",
    max_allowed_dose_cv_pct: float = 15.0,
    ct_ac_shift_mm: float = 1.5,
    delineation_sigma_mm: float = 1.0,
    raise_on_infeasible: bool = False,
) -> tuple[STPProtocolDesignResult, dict[str, np.ndarray]]:
    """Design the Fisher-optimal Single-Timepoint (STP) vs. D-optimal 2-Timepoint SPECT/CT schedule.

    Returns `(STPProtocolDesignResult, diagnostic_curves)` where `diagnostic_curves` contains
    `times_h`, `stp_pk_cv_pct`, `stp_total_cv_pct`, and `two_tp_optimal_cv_pct` for visualization.
    """
    t_phys = THERANOSTIC_ISOTOPE_PHYSICS[isotope]["half_life_h"]
    t_grid = np.linspace(6.0, min(192.0, 1.35 * t_phys), 94)
    pk_cv, count_cv = evaluate_stp_pk_rmse_curve(
        t_grid,
        population_teff_mean_h=population_teff_mean_h,
        population_teff_sd_h=population_teff_sd_h,
        isotope=isotope,
        estimator=stp_estimator,
    )

    total_cv_curve = np.empty(len(t_grid), dtype=np.float64)
    pve_cv_ref = 0.0
    for i in range(len(t_grid)):
        tiac_comb = math.hypot(float(pk_cv[i]), float(count_cv[i]))
        budget = compute_gear2018_dose_uncertainty(
            dose_nominal_gy=20.0,
            lesion_diameter_mm=lesion_diameter_mm,
            scanner=scanner,
            recon=recon,
            isotope=isotope,
            tiac_cv_frac=tiac_comb,
            ct_ac_shift_mm=ct_ac_shift_mm,
            delineation_sigma_mm=delineation_sigma_mm,
        )
        total_cv_curve[i] = budget.dose_cv_pct
        if i == 0:
            pve_cv_ref = budget.dose_cv_pct * math.sqrt(budget.variance_contrib_pve_delineation_pct / 100.0)

    best_idx = int(np.argmin(total_cv_curve))
    t_star = float(t_grid[best_idx])
    best_stp_cv = float(total_cv_curve[best_idx])
    best_pk_cv = float(math.hypot(float(pk_cv[best_idx]), float(count_cv[best_idx])) * 100.0)

    # Optimize D-optimal 2-timepoint schedule (t1, t2) via Fisher Information inversion
    best_2tp_tiac_cv = 1.0
    best_pair = (24.0, 120.0)
    for t1 in (4.0, 12.0, 24.0, 36.0, 48.0):
        for t2 in (72.0, 96.0, 120.0, 144.0, 168.0):
            if t2 <= t1 + 12.0 or t2 > 1.35 * t_phys:
                continue
            cv2 = compute_fisher_two_timepoint_tiac_cv(t1, t2, teff_mean_h=population_teff_mean_h)
            if cv2 < best_2tp_tiac_cv:
                best_2tp_tiac_cv = cv2
                best_pair = (float(t1), float(t2))

    budget_2tp = compute_gear2018_dose_uncertainty(
        dose_nominal_gy=20.0,
        lesion_diameter_mm=lesion_diameter_mm,
        scanner=scanner,
        recon=recon,
        isotope=isotope,
        tiac_cv_frac=best_2tp_tiac_cv,
        ct_ac_shift_mm=ct_ac_shift_mm,
        delineation_sigma_mm=delineation_sigma_mm,
    )
    two_tp_total_cv = budget_2tp.dose_cv_pct

    stp_ok = best_stp_cv <= max_allowed_dose_cv_pct
    if raise_on_infeasible:
        assert_stp_protocol_feasible(
            stp_feasible=stp_ok,
            optimal_stp_time_h=t_star,
            optimal_stp_total_dose_cv_pct=best_stp_cv,
            max_allowed_dose_cv_pct=max_allowed_dose_cv_pct,
            d_optimal_two_timepoints_h=best_pair,
        )

    rec_proto: Literal["SINGLE_TIMEPOINT_APPROVED", "TWO_TIMEPOINT_REQUIRED"] = (
        "SINGLE_TIMEPOINT_APPROVED" if stp_ok else "TWO_TIMEPOINT_REQUIRED"
    )
    scans_saved = 3 if stp_ok else 2

    prov = build_provenance_record(
        operation="fisher_optimal_stp_protocol_design",
        inputs={
            "organ_or_lesion_label": organ_or_lesion_label,
            "lesion_diameter_mm": lesion_diameter_mm,
            "isotope": isotope,
            "population_teff_mean_h": population_teff_mean_h,
            "population_teff_sd_h": population_teff_sd_h,
            "stp_estimator": stp_estimator,
            "max_allowed_dose_cv_pct": max_allowed_dose_cv_pct,
        },
        outputs={
            "optimal_stp_time_h": t_star,
            "optimal_stp_total_dose_cv_pct": best_stp_cv,
            "stp_feasible": stp_ok,
            "d_optimal_two_timepoints_h": list(best_pair),
            "two_timepoint_total_dose_cv_pct": two_tp_total_cv,
        },
        data_source_tag="GEAR2018_FISHER_STP_DESIGNER",
        citations=[
            "Gear et al. (2018) Eur J Nucl Med Mol Imaging 45:2456-2474, DOI:10.1007/s00259-018-4136-7",
            "Hanscheid et al. (2018) J Nucl Med 59(1):75-81, DOI:10.2967/jnumed.117.193706",
            "Madsen et al. (2018) Med Phys 45(5):2318-2324, DOI:10.1002/mp.12886",
        ],
    )

    result = STPProtocolDesignResult(
        organ_or_lesion_label=organ_or_lesion_label,
        isotope=isotope,
        lesion_diameter_mm=float(lesion_diameter_mm),
        population_teff_mean_h=float(population_teff_mean_h),
        population_teff_sd_h=float(population_teff_sd_h),
        stp_estimator=stp_estimator,
        optimal_stp_time_h=float(round(t_star, 1)),
        optimal_stp_total_dose_cv_pct=float(round(best_stp_cv, 2)),
        optimal_stp_pk_cv_pct=float(round(best_pk_cv, 2)),
        optimal_stp_pve_cv_pct=float(round(pve_cv_ref, 2)),
        max_allowed_dose_cv_pct=float(max_allowed_dose_cv_pct),
        stp_feasible=stp_ok,
        recommended_protocol=rec_proto,
        d_optimal_two_timepoints_h=best_pair,
        two_timepoint_total_dose_cv_pct=float(round(two_tp_total_cv, 2)),
        scans_saved_vs_4tp=scans_saved,
        provenance=prov,
    )
    curves = {
        "times_h": t_grid,
        "stp_pk_cv_pct": pk_cv * 100.0,
        "stp_count_cv_pct": count_cv * 100.0,
        "stp_total_cv_pct": total_cv_curve,
        "two_tp_total_cv_pct": np.full_like(t_grid, two_tp_total_cv),
    }
    return result, curves
