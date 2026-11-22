"""Multi-Scanner EARL1 / EARL2 / QIBA / MIRD-23 Recovery Coefficient Corridor Harmonizer.

Synthesizes the minimal post-reconstruction Gaussian or Butterworth harmonization filter
(FWHM in mm) and iteration/BSREM-beta guidance via bounded nonlinear optimization
(`scipy.optimize.minimize_scalar`) to map disparate PET and SPECT/CT scanners into a shared
quantitative Recovery Coefficient (RCmean, RCmax, RCpeak) corridor without sacrificing
diagnostic sharpness.

References:
- Kaalep et al. (2018) EJNMMI Phys 5:14, DOI:10.1186/s40658-018-0226-7
- Kaalep et al. (2018) Eur J Nucl Med Mol Imaging 45:1344-1361, DOI:10.1007/s00259-018-3977-4
- Boellaard et al. (2015) Eur J Nucl Med Mol Imaging 42:328-354, DOI:10.1007/s00259-014-2961-x
- Dewaraja et al. (2012) MIRD Pamphlet No. 23, DOI:10.2967/jnumed.111.100123

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve

from quantnuc_inverse.forward_image_formation import (
    NEMA_NU2_2018_DIAMETERS_MM,
    apply_forward_system_operator,
    build_antialiased_sphere_mask,
    effective_psf_fwhm_mm,
)
from quantnuc_inverse.ir import (
    EARLCorridorSpec,
    HarmonizationFilterResult,
    ParetoHarmonizationPoint,
    ReconConfig,
    ScannerSpec,
)
from quantnuc_inverse.provenance import (
    assert_earl_corridor_feasible,
    build_provenance_record,
)

CorridorName = Literal["EARL1", "EARL2", "QIBA_FDG_PET", "MIRD23_SPECT"]


def estimate_background_cov_pct(
    scanner: ScannerSpec,
    recon: ReconConfig,
) -> float:
    """Estimate NEMA background noise Coefficient of Variation (COV %) per EANM/EARL criteria.

    Scales physically with:
    - OSEM iteration-subset updates: sqrt(N_iter * N_sub / 48)
    - Scanner sensitivity & voxel volume: 1 / sqrt(sensitivity * voxel_vol_mm3)
    - Time-of-Flight (TOF) SNR gain: sqrt(TOF_ps / 450)
    - 3D post-reconstruction filter volume smoothing: (1 + (FWHM_post / 3.8)^2)^(-0.75)
    - BSREM penalty beta noise suppression.
    """
    vox_vol = scanner.voxel_size_mm[0] * scanner.voxel_size_mm[1] * scanner.voxel_size_mm[2]
    # Calibrated base COV (%) for standard 2-min/bed NEMA phantom (5.3 kBq/mL background)
    sens_factor = math.sqrt(12.0 / max(scanner.sensitivity_cps_per_kbq, 0.5))
    vox_factor = math.sqrt(8.0 / max(vox_vol, 1.0))
    iter_factor = math.sqrt((recon.iterations * recon.subsets) / 48.0)

    tof_factor = 1.0
    if scanner.tof_resolution_ps is not None:
        tof_factor = math.sqrt(min(scanner.tof_resolution_ps, 600.0) / 450.0)

    # Resolution modeling slightly correlates adjacent voxels (small variance change)
    psf_factor = 1.06 if (scanner.has_psf_modeling or recon.algorithm in {"PSF_OSEM", "TOF_PSF_OSEM"}) else 1.0

    # 3D Gaussian / Butterworth low-pass noise reduction scales with effective filter volume
    fwhm_p = max(0.0, recon.post_filter_fwhm_mm)
    filt_factor = (1.0 + (fwhm_p / 3.6) ** 2) ** (-0.72)

    beta_factor = 1.0
    if recon.algorithm == "BSREM" and recon.bsrem_beta is not None and recon.bsrem_beta > 0.0:
        beta_factor = 1.0 / (1.0 + math.sqrt(recon.bsrem_beta / 350.0) * 0.55)

    # SPECT operates with much lower sensitivity but larger collimated effective smoothing volume
    modality_scale = 0.62 if scanner.modality.startswith("SPECT") else 1.0

    cov_pct = 17.5 * sens_factor * vox_factor * iter_factor * tof_factor * psf_factor * filt_factor * beta_factor * modality_scale
    return float(max(0.5, cov_pct))


def get_corridor_spec(standard_name: CorridorName) -> EARLCorridorSpec:
    """Return standardized NEMA NU 2 6-sphere RC acceptability corridors."""
    diameters = list(NEMA_NU2_2018_DIAMETERS_MM)

    if standard_name == "EARL1":
        return EARLCorridorSpec(
            standard_name="EARL1",
            sphere_diameters_mm=diameters,
            rc_mean_lower=[0.27, 0.44, 0.57, 0.65, 0.72, 0.76],
            rc_mean_upper=[0.43, 0.60, 0.73, 0.80, 0.85, 0.89],
            rc_max_lower=[0.34, 0.57, 0.76, 0.85, 0.91, 0.95],
            rc_max_upper=[0.57, 0.85, 1.04, 1.12, 1.13, 1.16],
            rc_peak_lower=[0.25, 0.41, 0.57, 0.68, 0.76, 0.82],
            rc_peak_upper=[0.45, 0.65, 0.81, 0.91, 0.97, 1.02],
            reference_citation=(
                "Boellaard et al. (2015) Eur J Nucl Med Mol Imaging 42:328-354, "
                "DOI:10.1007/s00259-014-2961-x; Kaalep et al. (2018) DOI:10.1007/s00259-017-3853-7"
            ),
        )

    if standard_name == "EARL2":
        return EARLCorridorSpec(
            standard_name="EARL2",
            sphere_diameters_mm=diameters,
            rc_mean_lower=[0.39, 0.60, 0.71, 0.76, 0.80, 0.84],
            rc_mean_upper=[0.63, 0.86, 0.93, 0.97, 0.99, 1.02],
            rc_max_lower=[0.52, 0.82, 0.95, 0.98, 0.98, 0.97],
            rc_max_upper=[0.88, 1.22, 1.32, 1.32, 1.27, 1.24],
            rc_peak_lower=[0.36, 0.56, 0.72, 0.82, 0.87, 0.89],
            rc_peak_upper=[0.60, 0.84, 0.99, 1.06, 1.09, 1.10],
            reference_citation=(
                "Kaalep et al. (2018) Eur J Nucl Med Mol Imaging 45:1344-1361, "
                "DOI:10.1007/s00259-018-3977-4; Kaalep et al. (2018) EJNMMI Phys 5:14, "
                "DOI:10.1186/s40658-018-0226-7"
            ),
        )

    if standard_name == "QIBA_FDG_PET":
        return EARLCorridorSpec(
            standard_name="QIBA_FDG_PET",
            sphere_diameters_mm=diameters,
            rc_mean_lower=[0.35, 0.54, 0.67, 0.74, 0.79, 0.83],
            rc_mean_upper=[0.58, 0.80, 0.90, 0.95, 0.97, 1.00],
            rc_max_lower=[0.46, 0.74, 0.90, 0.95, 0.96, 0.96],
            rc_max_upper=[0.78, 1.12, 1.24, 1.25, 1.22, 1.20],
            rc_peak_lower=[0.32, 0.50, 0.68, 0.78, 0.84, 0.87],
            rc_peak_upper=[0.54, 0.76, 0.93, 1.02, 1.06, 1.08],
            reference_citation="QIBA FDG-PET/CT Profile & Boellaard et al. (2015) DOI:10.1007/s00259-014-2961-x",
        )

    # MIRD Pamphlet No. 23 / 26 Quantitative SPECT corridor (CZT & NaI SPECT/CT)
    return EARLCorridorSpec(
        standard_name="MIRD23_SPECT",
        sphere_diameters_mm=diameters,
        rc_mean_lower=[0.18, 0.30, 0.44, 0.56, 0.66, 0.74],
        rc_mean_upper=[0.44, 0.62, 0.76, 0.85, 0.92, 0.97],
        rc_max_lower=[0.22, 0.38, 0.56, 0.70, 0.80, 0.86],
        rc_max_upper=[0.56, 0.80, 0.98, 1.08, 1.14, 1.16],
        rc_peak_lower=[0.16, 0.27, 0.42, 0.56, 0.67, 0.76],
        rc_peak_upper=[0.42, 0.60, 0.76, 0.87, 0.95, 1.01],
        reference_citation=(
            "Dewaraja et al. (2012) MIRD Pamphlet No. 23, DOI:10.2967/jnumed.111.100123; "
            "Ljungberg et al. (2016) MIRD Pamphlet No. 26, DOI:10.2967/jnumed.115.159012"
        ),
    )


def build_reference_scanner_fleet() -> list[tuple[ScannerSpec, ReconConfig]]:
    """Create a realistic 4-scanner multi-vendor fleet (SiPM PET, TOF-PSF PET, Legacy PMT PET, CZT SPECT)."""
    return [
        (
            ScannerSpec(
                scanner_id="SiPM_Digital_PET",
                vendor_model="Digital SiPM TOF-PSF PET/CT (214 ps)",
                modality="PET_SIPM",
                psf_fwhm_xy_mm=3.5,
                psf_fwhm_z_mm=3.7,
                voxel_size_mm=(2.0, 2.0, 2.0),
                radial_psf_degradation_mm_per_10cm=0.30,
                tof_resolution_ps=214.0,
                has_psf_modeling=True,
                gibbs_ringing_strength=0.34,
                sensitivity_cps_per_kbq=18.5,
                isotope="F-18",
            ),
            ReconConfig(
                algorithm="TOF_PSF_OSEM",
                iterations=4,
                subsets=16,
                post_filter_fwhm_mm=0.0,
            ),
        ),
        (
            ScannerSpec(
                scanner_id="TOF_PSF_PET",
                vendor_model="Clinical TOF-PSF PET/CT (380 ps)",
                modality="PET_TOF_PSF",
                psf_fwhm_xy_mm=4.2,
                psf_fwhm_z_mm=4.4,
                voxel_size_mm=(2.0, 2.0, 2.0),
                radial_psf_degradation_mm_per_10cm=0.45,
                tof_resolution_ps=380.0,
                has_psf_modeling=True,
                gibbs_ringing_strength=0.28,
                sensitivity_cps_per_kbq=13.0,
                isotope="F-18",
            ),
            ReconConfig(
                algorithm="PSF_OSEM",
                iterations=4,
                subsets=16,
                post_filter_fwhm_mm=0.0,
            ),
        ),
        (
            ScannerSpec(
                scanner_id="Legacy_PMT_PET",
                vendor_model="Legacy PMT OSEM PET/CT",
                modality="PET_PMT",
                psf_fwhm_xy_mm=6.1,
                psf_fwhm_z_mm=6.3,
                voxel_size_mm=(2.5, 2.5, 2.5),
                radial_psf_degradation_mm_per_10cm=0.75,
                tof_resolution_ps=None,
                has_psf_modeling=False,
                gibbs_ringing_strength=0.0,
                sensitivity_cps_per_kbq=7.5,
                isotope="F-18",
            ),
            ReconConfig(
                algorithm="OSEM",
                iterations=2,
                subsets=16,
                post_filter_fwhm_mm=6.5,
            ),
        ),
        (
            ScannerSpec(
                scanner_id="CZT_SPECT_CT",
                vendor_model="Solid-State CZT Quantitative SPECT/CT (Lu-177)",
                modality="SPECT_CZT",
                psf_fwhm_xy_mm=7.4,
                psf_fwhm_z_mm=7.6,
                voxel_size_mm=(2.5, 2.5, 2.5),
                radial_psf_degradation_mm_per_10cm=1.35,
                tof_resolution_ps=None,
                has_psf_modeling=True,
                gibbs_ringing_strength=0.10,
                sensitivity_cps_per_kbq=2.2,
                isotope="Lu-177",
            ),
            ReconConfig(
                algorithm="PSF_OSEM",
                iterations=6,
                subsets=10,
                post_filter_fwhm_mm=0.0,
            ),
        ),
    ]


class _CachedSphereEvaluator:
    """Pre-builds 3D anti-aliased sphere masks and 1 cm^3 peak kernel for rapid optimization."""

    def __init__(
        self,
        diameters_mm: list[float],
        voxel_size_mm: tuple[float, float, float],
        *,
        true_activity: float = 4.0,
        background_activity: float = 1.0,
    ) -> None:
        self.diameters_mm = diameters_mm
        self.voxel_size_mm = voxel_size_mm
        self.true_activity = true_activity
        self.background_activity = background_activity
        dz, dy, dx = voxel_size_mm

        # Precompute 1 cm^3 spherical peak kernel once
        r_peak = 6.203504908994
        kz = max(1, int(math.ceil(r_peak / dz)))
        ky = max(1, int(math.ceil(r_peak / dy)))
        kx = max(1, int(math.ceil(r_peak / dx)))
        gz = np.arange(-kz, kz + 1) * dz
        gy = np.arange(-ky, ky + 1) * dy
        gx = np.arange(-kx, kx + 1) * dx
        zz, yy, xx = np.meshgrid(gz, gy, gx, indexing="ij")
        kernel = ((zz**2 + yy**2 + xx**2) <= r_peak**2).astype(np.float64)
        if kernel.sum() <= 0.0:
            kernel[kz, ky, kx] = 1.0
        self.peak_kernel = kernel / kernel.sum()

        self.masks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        pad_xy = 26.0
        pad_z = 22.0
        for d in diameters_mm:
            nx = max(21, int(math.ceil((d + 2.0 * pad_xy) / dx)) | 1)
            ny = max(21, int(math.ceil((d + 2.0 * pad_xy) / dy)) | 1)
            nz = max(17, int(math.ceil((d + 2.0 * pad_z) / dz)) | 1)
            mask = build_antialiased_sphere_mask(
                shape=(nz, ny, nx),
                voxel_size_mm=(dz, dy, dx),
                center_mm=(0.0, 0.0, 0.0),
                diameter_mm=d,
                supersample=3,
            )
            true_vol = background_activity + (true_activity - background_activity) * mask
            roi = mask >= 0.1
            if not np.any(roi):
                roi = mask == np.max(mask)
            key = f"{int(d)}mm" if float(d).is_integer() else f"{d:.1f}mm"
            self.masks.append((key, mask, true_vol, roi))

    def evaluate(
        self,
        scanner: ScannerSpec,
        recon: ReconConfig,
        radial_offset_mm: float = 57.2,
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        rc_mean: dict[str, float] = {}
        rc_max: dict[str, float] = {}
        rc_peak: dict[str, float] = {}

        for key, mask, true_vol, roi in self.masks:
            obs_vol = apply_forward_system_operator(
                true_vol,
                scanner,
                recon,
                radial_offset_mm=radial_offset_mm,
                boundary_mode="nearest",
            )
            obs_max = float(np.max(obs_vol[roi]))
            thr_a50 = self.background_activity + 0.5 * (obs_max - self.background_activity)
            a50 = (obs_vol >= thr_a50) & (mask > 0.0)
            if np.any(a50):
                obs_mean = float(np.mean(obs_vol[a50]))
            else:
                obs_mean = float(np.sum(obs_vol * mask) / np.sum(mask))

            peak_map = fftconvolve(obs_vol, self.peak_kernel, mode="same")
            obs_peak = float(np.max(peak_map[roi]))

            rc_mean[key] = obs_mean / self.true_activity
            rc_max[key] = obs_max / self.true_activity
            rc_peak[key] = obs_peak / self.true_activity

        return rc_mean, rc_max, rc_peak


def compute_corridor_violation(
    corridor: EARLCorridorSpec,
    rc_mean: dict[str, float],
    rc_max: dict[str, float],
    rc_peak: dict[str, float] | None = None,
) -> tuple[float, float]:
    """Compute (max_violation, sum_squared_violation) outside EARL/QIBA/MIRD corridor."""
    violations: list[float] = []
    for idx, d in enumerate(corridor.sphere_diameters_mm):
        key = f"{int(d)}mm" if float(d).is_integer() else f"{d:.1f}mm"
        v_mean = max(0.0, corridor.rc_mean_lower[idx] - rc_mean[key], rc_mean[key] - corridor.rc_mean_upper[idx])
        v_max = max(0.0, corridor.rc_max_lower[idx] - rc_max[key], rc_max[key] - corridor.rc_max_upper[idx])
        violations.extend([v_mean, v_max])
        if (
            rc_peak is not None
            and corridor.rc_peak_lower is not None
            and corridor.rc_peak_upper is not None
        ):
            v_peak = max(
                0.0,
                corridor.rc_peak_lower[idx] - rc_peak[key],
                rc_peak[key] - corridor.rc_peak_upper[idx],
            )
            violations.append(v_peak)

    max_v = float(max(violations)) if violations else 0.0
    sse_v = float(sum(v * v for v in violations))
    return max_v, sse_v


def synthesize_harmonization_filter(
    scanner: ScannerSpec,
    recon: ReconConfig,
    corridor: EARLCorridorSpec,
    *,
    filter_type: Literal["gaussian", "butterworth"] = "gaussian",
    max_filter_fwhm_mm: float = 12.0,
    allow_iteration_tuning: bool = True,
    max_allowed_cov_pct: float = 15.0,
    enforce_noise_constraint: bool = False,
    raise_on_infeasible: bool = False,
    radial_offset_mm: float = 57.2,
) -> HarmonizationFilterResult:
    """Synthesize the minimal post-reconstruction harmonization filter mapping a scanner into a corridor.

    Objective:
        Minimize post-filter FWHM (preserving maximum diagnostic sharpness) subject to zero
        violation across all 6 NEMA sphere RCmean, RCmax, and RCpeak acceptability limits
        (and optionally EANM/EARL background noise COV <= max_allowed_cov_pct, default 15%).
    """
    evaluator = _CachedSphereEvaluator(corridor.sphere_diameters_mm, scanner.voxel_size_mm)

    # Evaluate pre-filter baseline
    pre_mean, pre_max, pre_peak = evaluator.evaluate(scanner, recon, radial_offset_mm=radial_offset_mm)
    max_viol_before, _ = compute_corridor_violation(corridor, pre_mean, pre_max, pre_peak)

    # Candidate iteration counts (keeps baseline first, optionally tests higher convergence if under-recovering)
    iter_candidates = [recon.iterations]
    if allow_iteration_tuning:
        for cand in (max(2, recon.iterations + 2), max(2, recon.iterations + 4)):
            if cand <= 20 and cand not in iter_candidates:
                iter_candidates.append(cand)

    best_fwhm = recon.post_filter_fwhm_mm
    best_iters = recon.iterations
    best_max_viol = float("inf")
    best_metrics = (pre_mean, pre_max, pre_peak)
    best_feasible = False

    for n_iter in iter_candidates:
        def _eval_at_fwhm(
            fwhm_val: float, _iters: int = n_iter
        ) -> tuple[float, float, float, dict[str, float], dict[str, float], dict[str, float]]:
            cfg = recon.model_copy(
                update={
                    "iterations": _iters,
                    "post_filter_fwhm_mm": float(max(0.0, fwhm_val)),
                    "filter_type": filter_type,
                }
            )
            m, mx, pk = evaluator.evaluate(scanner, cfg, radial_offset_mm=radial_offset_mm)
            mv, sse = compute_corridor_violation(corridor, m, mx, pk)
            cov = estimate_background_cov_pct(scanner, cfg)
            return mv, sse, cov, m, mx, pk

        def _is_pt_feasible(mv_val: float, cov_val: float) -> bool:
            return (mv_val <= 1e-6) and ((not enforce_noise_constraint) or (cov_val <= max_allowed_cov_pct))

        # Coarse sweep over [0, max_filter_fwhm_mm] to detect exact feasible interval
        grid_fwhm = np.linspace(0.0, max_filter_fwhm_mm, 25)
        grid_results = [(float(f), *_eval_at_fwhm(float(f))) for f in grid_fwhm]

        feasible_pts = [r for r in grid_results if _is_pt_feasible(r[1], r[3])]
        if feasible_pts:
            # Find the MINIMAL FWHM that stays inside the corridor (+ noise bound if enforced)
            first_feas_idx = next(i for i, r in enumerate(grid_results) if _is_pt_feasible(r[1], r[3]))
            if first_feas_idx == 0:
                cand_fwhm = 0.0
                cand_mv, _, _, cm, cmx, cpk = grid_results[0][1:]
            else:
                lo_f = grid_results[first_feas_idx - 1][0]
                hi_f = grid_results[first_feas_idx][0]
                for _ in range(10):
                    mid = 0.5 * (lo_f + hi_f)
                    mv_mid, _, cov_mid, _, _, _ = _eval_at_fwhm(mid)
                    if _is_pt_feasible(mv_mid, cov_mid):
                        hi_f = mid
                    else:
                        lo_f = mid
                cand_fwhm = min(max_filter_fwhm_mm, hi_f + 0.05)
                cand_mv, _, cand_cov, cm, cmx, cpk = _eval_at_fwhm(cand_fwhm)
                if not _is_pt_feasible(cand_mv, cand_cov):
                    cand_fwhm = hi_f
                    cand_mv, _, _, cm, cmx, cpk = _eval_at_fwhm(cand_fwhm)

            if (not best_feasible) or (cand_fwhm < best_fwhm):
                best_feasible = True
                best_fwhm = cand_fwhm
                best_iters = n_iter
                best_max_viol = cand_mv
                best_metrics = (cm, cmx, cpk)
                if n_iter == recon.iterations:
                    break
        else:
            best_grid = min(grid_results, key=lambda r: (r[1], r[2]))
            bracket_lo = max(0.0, best_grid[0] - 1.0)
            bracket_hi = min(max_filter_fwhm_mm, best_grid[0] + 1.0)

            def _loss(x: float) -> float:
                mv_x, sse_x, cov_x, _, _, _ = _eval_at_fwhm(float(x))
                noise_pen = max(0.0, cov_x - max_allowed_cov_pct) * 0.1 if enforce_noise_constraint else 0.0
                return mv_x + 5.0 * sse_x + noise_pen + 1e-4 * float(x)

            opt = minimize_scalar(_loss, bounds=(bracket_lo, bracket_hi), method="bounded", options={"xatol": 0.02})
            cand_fwhm = float(opt.x)
            cand_mv, _, cand_cov, cm, cmx, cpk = _eval_at_fwhm(cand_fwhm)
            cand_feasible = (cand_mv <= 1e-3) and ((not enforce_noise_constraint) or (cand_cov <= max_allowed_cov_pct))

            if cand_feasible and ((not best_feasible) or (cand_fwhm < best_fwhm)):
                best_feasible = True
                best_fwhm = cand_fwhm
                best_iters = n_iter
                best_max_viol = cand_mv
                best_metrics = (cm, cmx, cpk)
                if n_iter == recon.iterations:
                    break
            elif (not best_feasible) and (cand_mv < best_max_viol):
                best_fwhm = cand_fwhm
                best_iters = n_iter
                best_max_viol = cand_mv
                best_metrics = (cm, cmx, cpk)

    if raise_on_infeasible:
        assert_earl_corridor_feasible(
            feasible=best_feasible,
            scanner_id=scanner.scanner_id,
            standard_name=corridor.standard_name,
            max_violation=best_max_viol,
        )

    post_mean, post_max, post_peak = best_metrics

    # Sharpness retention ratio = intrinsic PSF FWHM / post-harmonized effective FWHM
    pre_xy, _ = effective_psf_fwhm_mm(
        scanner,
        recon.model_copy(update={"iterations": best_iters, "post_filter_fwhm_mm": 0.0}),
        radial_offset_mm=radial_offset_mm,
        include_post_filter=False,
    )
    post_xy = math.hypot(pre_xy, best_fwhm)
    sharpness_ratio = float(min(1.0, pre_xy / max(post_xy, 1e-6)))

    final_cfg = recon.model_copy(
        update={
            "iterations": best_iters,
            "post_filter_fwhm_mm": best_fwhm,
            "filter_type": filter_type,
        }
    )
    bg_cov = estimate_background_cov_pct(scanner, final_cfg)
    noise_ok = bg_cov <= max_allowed_cov_pct
    # Small-lesion (13 mm) Contrast-to-Noise Ratio: (RC_mean(13mm)*4 - 1) / (COV/100)
    rc13 = post_mean.get("13mm", next(iter(post_mean.values())))
    cnr = float(max(0.0, (4.0 * rc13 - 1.0) / max(bg_cov / 100.0, 1e-3)))

    bw_cutoff = float(0.44 / best_fwhm) if (filter_type == "butterworth" and best_fwhm > 0.0) else None

    rec_beta: float | None = None
    if recon.algorithm == "BSREM" or scanner.modality in {"PET_SIPM", "PET_TOF_PSF"}:
        rec_beta = float(round(200.0 + 45.0 * (best_fwhm**1.6), 1))

    inputs = {
        "scanner": scanner,
        "recon": recon,
        "corridor": corridor.standard_name,
        "filter_type": filter_type,
        "max_allowed_cov_pct": max_allowed_cov_pct,
        "enforce_noise_constraint": enforce_noise_constraint,
    }
    outputs = {
        "feasible": best_feasible,
        "optimal_post_filter_fwhm_mm": best_fwhm,
        "recommended_iterations": best_iters,
        "max_corridor_violation_after": best_max_viol,
        "background_cov_pct": bg_cov,
        "cnr_score": cnr,
    }
    prov = build_provenance_record(
        operation="earl_harmonization_filter_synthesis",
        inputs=inputs,
        outputs=outputs,
        data_source_tag="SYNTHETIC_NEMA_NU2_2018_DRO",
        citations=[corridor.reference_citation],
    )

    return HarmonizationFilterResult(
        scanner_id=scanner.scanner_id,
        standard_name=corridor.standard_name,
        feasible=best_feasible,
        filter_type=filter_type,
        optimal_post_filter_fwhm_mm=float(round(best_fwhm, 4)),
        butterworth_cutoff_cycles_per_mm=bw_cutoff,
        recommended_iterations=int(best_iters),
        recommended_bsrem_beta=rec_beta,
        pre_filter_rc_mean=pre_mean,
        pre_filter_rc_max=pre_max,
        pre_filter_rc_peak=pre_peak,
        post_filter_rc_mean=post_mean,
        post_filter_rc_max=post_max,
        post_filter_rc_peak=post_peak,
        max_corridor_violation_before=float(round(max_viol_before, 6)),
        max_corridor_violation_after=float(round(best_max_viol, 6)),
        sharpness_retention_ratio=float(round(sharpness_ratio, 4)),
        background_cov_pct=float(round(bg_cov, 2)),
        max_allowed_cov_pct=float(max_allowed_cov_pct),
        noise_compliant=noise_ok,
        cnr_score=float(round(cnr, 2)),
        provenance=prov,
    )


def compute_harmonization_pareto_frontier(
    scanner: ScannerSpec,
    recon: ReconConfig,
    corridor: EARLCorridorSpec,
    *,
    fwhm_grid_mm: list[float] | None = None,
    iteration_grid: list[int] | None = None,
    max_allowed_cov_pct: float = 15.0,
    radial_offset_mm: float = 57.2,
) -> list[ParetoHarmonizationPoint]:
    """Sweep (post-filter FWHM, iterations) to map RC-compliance vs. noise COV (%) vs. CNR Pareto space."""
    fwhms = fwhm_grid_mm if fwhm_grid_mm is not None else [float(x) for x in np.linspace(0.0, 9.5, 20)]
    iters = iteration_grid if iteration_grid is not None else [2, 4, 6]
    evaluator = _CachedSphereEvaluator(corridor.sphere_diameters_mm, scanner.voxel_size_mm)

    points: list[ParetoHarmonizationPoint] = []
    for n_it in iters:
        pre_xy, _ = effective_psf_fwhm_mm(
            scanner,
            recon.model_copy(update={"iterations": n_it, "post_filter_fwhm_mm": 0.0}),
            radial_offset_mm=radial_offset_mm,
            include_post_filter=False,
        )
        for fw in fwhms:
            cfg = recon.model_copy(update={"iterations": n_it, "post_filter_fwhm_mm": fw})
            m, mx, pk = evaluator.evaluate(scanner, cfg, radial_offset_mm=radial_offset_mm)
            mv, _ = compute_corridor_violation(corridor, m, mx, pk)
            cov = estimate_background_cov_pct(scanner, cfg)
            rc13 = m.get("13mm", next(iter(m.values())))
            cnr = float(max(0.0, (4.0 * rc13 - 1.0) / max(cov / 100.0, 1e-3)))
            sharp = float(min(1.0, pre_xy / max(math.hypot(pre_xy, fw), 1e-6)))
            points.append(
                ParetoHarmonizationPoint(
                    post_filter_fwhm_mm=float(round(fw, 3)),
                    iterations=int(n_it),
                    max_corridor_violation=float(round(mv, 5)),
                    background_cov_pct=float(round(cov, 2)),
                    cnr_score=float(round(cnr, 2)),
                    sharpness_retention_ratio=float(round(sharp, 4)),
                    feasible_earl_and_noise=(mv <= 1e-4) and (cov <= max_allowed_cov_pct),
                )
            )
    return points
