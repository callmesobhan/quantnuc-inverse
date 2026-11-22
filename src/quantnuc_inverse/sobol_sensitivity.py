"""Exact Saltelli-Jansen QMC Global Sobol' Sensitivity Analysis for Quantitative PET/SPECT Bias.

Separates total variance in observed/apparent lesion SUV bias across four physical drivers:
1. `psf_fwhm_mm`: Scanner PSF / CDR spatial resolution uncertainty.
2. `delineation_error_mm`: Lesion ROI boundary delineation error (+-1 voxel).
3. `gibbs_ringing_strength`: PSF-OSEM edge-overshoot (Gibbs ringing) uncertainty.
4. `ct_ac_shift_mm`: CT-PET/SPECT respiratory attenuation-correction phase shift.

Implements Saltelli et al. (2010), *Computer Physics Communications* 181(2):259-270,
DOI:10.1016/j.cpc.2009.09.018 (Jansen 1999 total-order estimator + Saltelli 2010 first-order estimator).

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats.qmc import Sobol

from quantnuc_inverse.ir import ProvenanceRecord
from quantnuc_inverse.provenance import build_provenance_record

PARAMETER_NAMES: tuple[str, ...] = (
    "psf_fwhm_mm",
    "delineation_error_mm",
    "gibbs_ringing_strength",
    "ct_ac_shift_mm",
)


@dataclass(frozen=True)
class SobolSensitivityResult:
    """First-order (S1), total-order (ST), and interaction Sobol' indices with bootstrap 95% CIs."""

    lesion_diameter_mm: float
    n_base_samples: int
    total_evaluations: int
    parameter_names: tuple[str, ...]
    s1: dict[str, float]
    st: dict[str, float]
    interactions: dict[str, float]
    s1_ci95: dict[str, tuple[float, float]]
    st_ci95: dict[str, tuple[float, float]]
    output_mean_suv: float
    output_std_suv: float
    provenance: ProvenanceRecord


def _evaluate_suv_forward_batch(
    samples: np.ndarray,
    *,
    lesion_diameter_mm: float,
    true_suv: float,
    background_suv: float,
    photon_energy_kev: float = 511.0,
) -> np.ndarray:
    """Vectorized physical forward model mapping (PSF, delineation, Gibbs, CT-AC) -> apparent SUV.

    Captures exact physical coupling:
    - PVE recovery coefficient as a smooth 3D spherical function of (D_eff / FWHM).
    - PSF-Gibbs oscillatory edge overshoot peaking at D ~ 2.8 * FWHM.
    - Exact CT-AC respiratory line-integral attenuation bias exp(int delta_mu ds).
    """
    psf_fwhm = samples[:, 0]
    delin_err = samples[:, 1]
    gibbs = samples[:, 2]
    ct_shift = samples[:, 3]

    # Effective delineated diameter (boundary error affects both spill-out and ROI averaging)
    d_eff = np.maximum(2.5, lesion_diameter_mm + 2.0 * delin_err)
    x = d_eff / np.maximum(psf_fwhm, 0.5)

    # 3D spherical PVE recovery coefficient + ROI boundary inclusion dilution
    # When delineation expands beyond true lesion (delin_err > 0), extra background voxels dilute ROI mean
    rc_pve = 1.0 / (1.0 + (1.42 / np.maximum(x, 0.05)) ** 2.18)
    vol_ratio_dilute = np.where(
        delin_err > 0.0,
        (lesion_diameter_mm / d_eff) ** 3,
        1.0 + 0.12 * np.clip(-delin_err / max(lesion_diameter_mm, 1.0), 0.0, 0.5),
    )
    # Effective contrast recovery with Gibbs edge overshoot (peaks when D/FWHM ~ 2.8)
    gibbs_boost = 0.62 * gibbs * np.exp(-0.5 * ((x - 2.85) / 1.35) ** 2)

    # Apparent SUV before CT-AC mismatch
    contrast = true_suv - background_suv
    suv_pve_gibbs = background_suv + contrast * np.clip(
        rc_pve * vol_ratio_dilute + gibbs_boost, 0.02, 1.45
    )

    # Exact CT-AC respiratory dome attenuation bias factor exp(int delta_mu ds)
    mu_diff = 0.070 if photon_energy_kev >= 400.0 else 0.099  # cm^-1 (liver - lung)
    r_dome = 80.0
    abs_shift = np.minimum(np.abs(ct_shift), r_dome)
    chord_cm = (2.0 * np.sqrt(np.maximum(0.0, 2.0 * r_dome * abs_shift - abs_shift**2)) * 0.55) / 10.0
    # Positive shift = lung mu on liver -> negative delta_mu_integral (false-cold); negative -> false-hot
    delta_mu_int = np.where(ct_shift >= 0.0, -mu_diff * chord_cm, mu_diff * chord_cm)
    ac_bias = np.exp(delta_mu_int)

    return suv_pve_gibbs * ac_bias


def run_sobol_sensitivity(
    lesion_diameter_mm: float,
    *,
    true_suv: float = 8.0,
    background_suv: float = 2.0,
    bounds: dict[str, tuple[float, float]] | None = None,
    n_base_samples: int = 1024,
    photon_energy_kev: float = 511.0,
    seed: int = 2026,
    n_bootstrap: int = 200,
) -> SobolSensitivityResult:
    """Run Saltelli-Jansen QMC global Sobol' sensitivity analysis on quantitative SUV variance.

    Requires `n_base_samples` to be a power of 2 for balanced Sobol QMC nets.
    """
    if n_base_samples < 32 or (n_base_samples & (n_base_samples - 1)) != 0:
        raise ValueError("n_base_samples must be a power of 2 >= 32 (e.g., 256, 512, 1024).")

    default_bounds: dict[str, tuple[float, float]] = {
        "psf_fwhm_mm": (3.2, 7.2),
        "delineation_error_mm": (-2.0, 2.0),  # +/- 1 typical 2 mm PET voxel
        "gibbs_ringing_strength": (0.0, 0.35),
        "ct_ac_shift_mm": (-5.5, 5.5),
    }
    param_bounds = {**default_bounds, **(bounds or {})}
    k = len(PARAMETER_NAMES)
    lo = np.array([param_bounds[p][0] for p in PARAMETER_NAMES], dtype=np.float64)
    hi = np.array([param_bounds[p][1] for p in PARAMETER_NAMES], dtype=np.float64)
    if np.any(hi <= lo):
        raise ValueError("All Sobol parameter upper bounds must strictly exceed lower bounds.")

    # Draw 2*k dimensional Sobol sequence to generate independent A and B matrices
    m_pow = int(round(math.log2(n_base_samples)))
    sampler = Sobol(d=2 * k, scramble=True, seed=seed)
    u_all = sampler.random_base2(m=m_pow)
    u_a = u_all[:, :k]
    u_b = u_all[:, k:]

    mat_a = lo + u_a * (hi - lo)
    mat_b = lo + u_b * (hi - lo)

    y_a = _evaluate_suv_forward_batch(
        mat_a,
        lesion_diameter_mm=lesion_diameter_mm,
        true_suv=true_suv,
        background_suv=background_suv,
        photon_energy_kev=photon_energy_kev,
    )
    y_b = _evaluate_suv_forward_batch(
        mat_b,
        lesion_diameter_mm=lesion_diameter_mm,
        true_suv=true_suv,
        background_suv=background_suv,
        photon_energy_kev=photon_energy_kev,
    )

    y_ab_list: list[np.ndarray] = []
    for i in range(k):
        mat_ab = mat_a.copy()
        mat_ab[:, i] = mat_b[:, i]
        y_ab_list.append(
            _evaluate_suv_forward_batch(
                mat_ab,
                lesion_diameter_mm=lesion_diameter_mm,
                true_suv=true_suv,
                background_suv=background_suv,
                photon_energy_kev=photon_energy_kev,
            )
        )

    y_pooled = np.concatenate([y_a, y_b])
    var_y = float(np.var(y_pooled, ddof=1))
    if var_y <= 1e-12:
        var_y = 1e-12

    s1_dict: dict[str, float] = {}
    st_dict: dict[str, float] = {}
    inter_dict: dict[str, float] = {}
    s1_ci: dict[str, tuple[float, float]] = {}
    st_ci: dict[str, tuple[float, float]] = {}

    rng = np.random.default_rng(seed + 99)
    n = n_base_samples

    for i, name in enumerate(PARAMETER_NAMES):
        y_ab = y_ab_list[i]
        # Jansen (1999) / Saltelli (2010) low-variance variance-based estimators for S1 and ST
        s1_val = float(1.0 - 0.5 * np.mean((y_b - y_ab) ** 2) / var_y)
        st_val = float(0.5 * np.mean((y_a - y_ab) ** 2) / var_y)
        s1_clamped = float(np.clip(s1_val, 0.0, 1.0))
        st_clamped = float(np.clip(st_val, 0.0, 1.0))
        s1_dict[name] = s1_clamped
        st_dict[name] = max(st_clamped, s1_clamped)
        inter_dict[name] = float(max(0.0, st_dict[name] - s1_dict[name]))

        # Bootstrap 95% CI
        boot_s1 = np.empty(n_bootstrap, dtype=np.float64)
        boot_st = np.empty(n_bootstrap, dtype=np.float64)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            v_b = float(np.var(np.concatenate([y_a[idx], y_b[idx]]), ddof=1))
            v_b = max(v_b, 1e-12)
            boot_s1[b] = np.clip(1.0 - 0.5 * np.mean((y_b[idx] - y_ab[idx]) ** 2) / v_b, 0.0, 1.0)
            boot_st[b] = np.clip(0.5 * np.mean((y_a[idx] - y_ab[idx]) ** 2) / v_b, 0.0, 1.0)
        s1_ci[name] = (float(np.percentile(boot_s1, 2.5)), float(np.percentile(boot_s1, 97.5)))
        st_ci[name] = (float(np.percentile(boot_st, 2.5)), float(np.percentile(boot_st, 97.5)))

    prov = build_provenance_record(
        operation="saltelli_jansen_sobol_sensitivity",
        inputs={
            "lesion_diameter_mm": lesion_diameter_mm,
            "true_suv": true_suv,
            "background_suv": background_suv,
            "bounds": param_bounds,
            "n_base_samples": n_base_samples,
            "seed": seed,
        },
        outputs={"s1": s1_dict, "st": st_dict, "interactions": inter_dict},
        data_source_tag="ANALYTICAL_PHYSICS_MODEL",
        citations=["Saltelli et al. (2010) Comput Phys Commun 181(2):259-270, DOI:10.1016/j.cpc.2009.09.018"],
    )

    return SobolSensitivityResult(
        lesion_diameter_mm=float(lesion_diameter_mm),
        n_base_samples=n_base_samples,
        total_evaluations=n_base_samples * (k + 2),
        parameter_names=PARAMETER_NAMES,
        s1=s1_dict,
        st=st_dict,
        interactions=inter_dict,
        s1_ci95=s1_ci,
        st_ci95=st_ci,
        output_mean_suv=float(np.mean(y_pooled)),
        output_std_suv=float(np.std(y_pooled, ddof=1)),
        provenance=prov,
    )
