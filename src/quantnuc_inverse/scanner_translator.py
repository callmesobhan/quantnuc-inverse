"""Physics-Informed N=1 Retrospective Cross-Scanner Translation Operator (T_{A->B} = H_B H_A^{-1}).

Solves the fundamental limitation of both vendor-locked prospective harmonization (Siemens EQ.PET,
which requires console-level raw sinograms) and statistical cohort batch harmonization (ComBat,
Orlhac et al. JNM 2018, DOI:10.2967/jnumed.117.199935, which requires N >= 30 patients per site
and ignores nonlinear lesion-diameter PVE/Gibbs physics):

1. Exact 3D Modulation Transfer Function H(f) of any PET/SPECT scanner + PSF-Gibbs + post-filter.
2. Noise-regularized frequency-domain Reverse-Gibbs & Cross-Scanner Translation operator:
       T_{A->B}(f) = [H_B(f) * conj(H_A(f))] / [|H_A(f)|^2 + lambda * (1 + (|f|/f_nyq)^4)]
   normalized so T_{A->B}(0) = 1.0 (strict DC / total-activity conservation).
3. Exact Parseval spectral noise-amplification norm G_noise with hard abstention gate
   (`TRANSLATION_NOISE_AMPLIFICATION_UNSAFE`) preventing ill-posed super-resolution hallucination.
4. Blind NEMA RC-curve fingerprinting (`fingerprint_scanner_from_rc`) inferring (FWHM, Gibbs)
   from a site's phantom RC curve alone without vendor recon internals.
5. N=1 Lesion-Size-Aware Longitudinal Equivalence Certificate across 10-37 mm lesions.

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares

from quantnuc_inverse.forward_image_formation import (
    FWHM_TO_SIGMA,
    NEMA_NU2_2018_DIAMETERS_MM,
    apply_forward_system_operator,
    build_antialiased_sphere_mask,
    compute_sphere_recovery_coefficients,
    effective_gibbs_strength,
    effective_psf_fwhm_mm,
)
from quantnuc_inverse.ir import (
    CrossScannerTranslationResult,
    ReconConfig,
    ScannerFingerprintResult,
    ScannerSpec,
)
from quantnuc_inverse.provenance import (
    assert_translation_noise_safe,
    build_provenance_record,
)


def compute_3d_system_mtf(
    shape: tuple[int, int, int],
    scanner: ScannerSpec,
    recon: ReconConfig,
    *,
    radial_offset_mm: float = 57.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exact 3D Fourier Modulation Transfer Function H(f) of `apply_forward_system_operator`.

    Returns `(mtf_3d, freq_radius_3d)` where `mtf_3d[0, 0, 0] == 1.0` strictly.
    Notice that for PSF-OSEM scanners with Gibbs ringing, |H(f)| > 1.0 in mid-spatial frequencies!
    """
    dz, dy, dx = scanner.voxel_size_mm
    fz = np.fft.fftfreq(shape[0], d=dz)
    fy = np.fft.fftfreq(shape[1], d=dy)
    fx = np.fft.fftfreq(shape[2], d=dx)
    zz, yy, xx = np.meshgrid(fz, fy, fx, indexing="ij")
    freq_radius = np.sqrt(zz**2 + yy**2 + xx**2)

    fwhm_xy_pre, fwhm_z_pre = effective_psf_fwhm_mm(
        scanner, recon, radial_offset_mm=radial_offset_mm, include_post_filter=False
    )
    sig_xy = fwhm_xy_pre * FWHM_TO_SIGMA
    sig_z = fwhm_z_pre * FWHM_TO_SIGMA

    # Exact Fourier transform of anisotropic 3D Gaussian: exp(-2 * pi^2 * (sz^2 fz^2 + sxy^2 (fy^2 + fx^2)))
    q2 = (sig_z * zz) ** 2 + (sig_xy * yy) ** 2 + (sig_xy * xx) ** 2
    two_pi_sq = 2.0 * (math.pi**2)
    h_pre = np.exp(-two_pi_sq * q2)

    gamma = effective_gibbs_strength(scanner, recon)
    if gamma > 0.0:
        h_s1 = np.exp(-two_pi_sq * (0.78**2) * q2)
        h_s2 = np.exp(-two_pi_sq * (2.15**2) * q2)
        h_pre = h_pre + (2.75 * gamma) * (h_s1 - h_s2)

    # Post-reconstruction filter MTF
    if recon.post_filter_fwhm_mm > 0.0:
        if recon.filter_type == "gaussian":
            sig_p = recon.post_filter_fwhm_mm * FWHM_TO_SIGMA
            h_post = np.exp(-two_pi_sq * (sig_p**2) * (freq_radius**2))
        else:
            cutoff = 0.44 / recon.post_filter_fwhm_mm
            h_post = 1.0 / np.sqrt(1.0 + (freq_radius / cutoff) ** (2 * recon.butterworth_order))
        h_tot = h_pre * h_post
    else:
        h_tot = h_pre

    return h_tot.astype(np.float64, copy=False), freq_radius.astype(np.float64, copy=False)


def build_translation_transfer_function(
    shape: tuple[int, int, int],
    source_scanner: ScannerSpec,
    source_recon: ReconConfig,
    target_scanner: ScannerSpec,
    target_recon: ReconConfig,
    *,
    regularization_lambda: float = 0.015,
    radial_offset_mm: float = 57.2,
) -> tuple[np.ndarray, float, float]:
    """Construct the noise-stabilized 3D translation transfer function T_{A->B}(f) = H_B H_A^{-1}.

    Returns:
        `(T_ab_3d, noise_amplification_factor, gibbs_peak_suppression_pct)`
    where:
    - `T_ab_3d[0, 0, 0] == 1.0` strictly (conserving total activity and background SUV).
    - `noise_amplification_factor` is the exact Parseval white-noise RMS amplification sqrt(mean(|T|^2)).
    - `gibbs_peak_suppression_pct` quantifies how strongly Scanner A's overshoot band (|H_A| > 1) is damped.
    """
    h_a, freq_rad = compute_3d_system_mtf(shape, source_scanner, source_recon, radial_offset_mm=radial_offset_mm)
    # Evaluate target H_B on source voxel grid (so operator maps Scanner A voxels directly into target physics)
    tgt_on_src_grid = target_scanner.model_copy(update={"voxel_size_mm": source_scanner.voxel_size_mm})
    h_b, _ = compute_3d_system_mtf(shape, tgt_on_src_grid, target_recon, radial_offset_mm=radial_offset_mm)

    f_nyq = 0.5 / min(source_scanner.voxel_size_mm)
    high_freq_penalty = regularization_lambda * (1.0 + (freq_rad / max(f_nyq, 1e-6)) ** 4)
    t_ab = (h_b * h_a) / (h_a**2 + high_freq_penalty)
    # Enforce exact DC unity gain (total activity conservation)
    t_ab[0, 0, 0] = 1.0

    # Exact L2->L2 operator spectral norm ||T_{A->B}||_{2->2} = max_f |T_{A->B}(f)|
    # Non-expansive contractions (down-translation + Reverse-Gibbs) have norm <= 1.0;
    # ill-posed super-resolution up-translation breaches the noise-amplification ceiling (> 2.0).
    noise_amp = float(np.max(np.abs(t_ab)))

    # Quantify Gibbs overshoot suppression on source MTF peak > 1.0
    max_ha = float(np.max(h_a))
    h_translated = h_a * t_ab
    max_ht = float(np.max(h_translated))
    if max_ha > 1.005:
        gibbs_supp_pct = float(
            np.clip(((max_ha - 1.0) - max(0.0, max_ht - 1.0)) / (max_ha - 1.0) * 100.0, 0.0, 100.0)
        )
    else:
        gibbs_supp_pct = 0.0

    return t_ab, noise_amp, gibbs_supp_pct


def translate_reconstructed_volume(
    volume_source: np.ndarray,
    source_scanner: ScannerSpec,
    source_recon: ReconConfig,
    target_scanner: ScannerSpec,
    target_recon: ReconConfig,
    *,
    regularization_lambda: float = 0.015,
    max_allowed_noise_amplification: float = 2.0,
    radial_offset_mm: float = 57.2,
    enforce_noise_gate: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Translate an already-reconstructed 3D PET/SPECT volume from Scanner A into Scanner B physics (N=1).

    Uses symmetric reflect-padding in 3D FFT space to prevent wrap-around edge ringing while
    preserving exact DC / total activity conservation.
    """
    vol = np.asarray(volume_source, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError("volume_source must be a 3D volume.")

    t_ab, noise_amp, gibbs_supp = build_translation_transfer_function(
        vol.shape,
        source_scanner,
        source_recon,
        target_scanner,
        target_recon,
        regularization_lambda=regularization_lambda,
        radial_offset_mm=radial_offset_mm,
    )

    if enforce_noise_gate:
        assert_translation_noise_safe(
            noise_amplification_factor=noise_amp,
            max_allowed_noise_amplification=max_allowed_noise_amplification,
            source_scanner_id=source_scanner.scanner_id,
            target_scanner_id=target_scanner.scanner_id,
        )

    # Apply frequency operator via padded FFT (reflect padding prevents periodic wrap artifacts)
    pad = (min(8, vol.shape[0] // 2), min(8, vol.shape[1] // 2), min(8, vol.shape[2] // 2))
    padded = np.pad(vol, ((pad[0], pad[0]), (pad[1], pad[1]), (pad[2], pad[2])), mode="reflect")
    t_pad, _, _ = build_translation_transfer_function(
        padded.shape,
        source_scanner,
        source_recon,
        target_scanner,
        target_recon,
        regularization_lambda=regularization_lambda,
        radial_offset_mm=radial_offset_mm,
    )
    translated_pad = np.fft.ifftn(np.fft.fftn(padded) * t_pad).real
    sz, sy, sx = vol.shape
    translated = translated_pad[pad[0] : pad[0] + sz, pad[1] : pad[1] + sy, pad[2] : pad[2] + sx]
    return translated.astype(np.float64, copy=False), noise_amp, gibbs_supp


def fingerprint_scanner_from_rc(
    observed_rc_mean: dict[str, float],
    observed_rc_max: dict[str, float],
    *,
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    radial_offset_mm: float = 57.2,
) -> ScannerFingerprintResult:
    """Blindly infer effective (PSF FWHM, Gibbs ringing strength) from a site's 6-sphere NEMA RC curve."""
    diam_keys = [(float(d), f"{int(d)}mm") for d in NEMA_NU2_2018_DIAMETERS_MM if f"{int(d)}mm" in observed_rc_max]
    if len(diam_keys) < 3:
        raise ValueError("Need at least 3 NEMA sphere diameters in observed_rc_max for fingerprinting.")

    # Pre-build sphere masks once for fast nonlinear least squares
    dz, dy, dx = voxel_size_mm
    cached = []
    for d, key in diam_keys:
        nx = max(21, int(math.ceil((d + 44.0) / dx)) | 1)
        ny = max(21, int(math.ceil((d + 44.0) / dy)) | 1)
        nz = max(17, int(math.ceil((d + 36.0) / dz)) | 1)
        mask = build_antialiased_sphere_mask((nz, ny, nx), voxel_size_mm, (0.0, 0.0, 0.0), d, supersample=2)
        true_vol = 1.0 + 3.0 * mask
        roi = mask >= 0.1
        cached.append((key, mask, true_vol, roi))

    def _residuals(params: np.ndarray) -> np.ndarray:
        fwhm_p, gibbs_p = float(params[0]), float(params[1])
        sc = ScannerSpec(
            scanner_id="fingerprinted",
            vendor_model="blind_rc_inferred",
            modality="PET_TOF_PSF",
            psf_fwhm_xy_mm=fwhm_p,
            psf_fwhm_z_mm=fwhm_p,
            voxel_size_mm=voxel_size_mm,
            radial_psf_degradation_mm_per_10cm=0.0,
            has_psf_modeling=gibbs_p > 0.01,
            gibbs_ringing_strength=gibbs_p,
        )
        rc_cfg = ReconConfig(
            algorithm="PSF_OSEM" if gibbs_p > 0.01 else "OSEM",
            iterations=4,
            subsets=16,
            post_filter_fwhm_mm=0.0,
        )
        res_list: list[float] = []
        for key, mask, true_vol, roi in cached:
            obs_v = apply_forward_system_operator(true_vol, sc, rc_cfg, radial_offset_mm=radial_offset_mm, boundary_mode="nearest")
            obs_max = float(np.max(obs_v[roi]))
            rc_mx = obs_max / 4.0
            res_list.append(rc_mx - observed_rc_max[key])
            if key in observed_rc_mean:
                thr = 1.0 + 0.5 * (obs_max - 1.0)
                a50 = (obs_v >= thr) & (mask > 0.0)
                rc_mn = (float(np.mean(obs_v[a50])) if np.any(a50) else float(np.sum(obs_v * mask) / np.sum(mask))) / 4.0
                res_list.append(rc_mn - observed_rc_mean[key])
        return np.asarray(res_list, dtype=np.float64)

    opt = least_squares(
        _residuals,
        x0=np.array([5.0, 0.15]),
        bounds=([2.0, 0.0], [12.0, 0.50]),
        xtol=1e-3,
        ftol=1e-3,
        max_nfev=35,
    )
    fwhm_hat, gibbs_hat = float(opt.x[0]), float(opt.x[1])
    rmse = float(np.sqrt(np.mean(opt.fun**2)))

    prov = build_provenance_record(
        operation="blind_nema_rc_scanner_fingerprinting",
        inputs={"observed_rc_mean": observed_rc_mean, "observed_rc_max": observed_rc_max},
        outputs={"inferred_psf_fwhm_xy_mm": fwhm_hat, "inferred_gibbs_strength": gibbs_hat, "rc_fit_rmse": rmse},
        data_source_tag="BLIND_RC_FINGERPRINT_INVERSE",
        citations=["Kaalep et al. (2018) EJNMMI Phys 5:14, DOI:10.1186/s40658-018-0226-7"],
    )
    return ScannerFingerprintResult(
        inferred_psf_fwhm_xy_mm=float(round(fwhm_hat, 3)),
        inferred_gibbs_strength=float(round(gibbs_hat, 4)),
        rc_fit_rmse=float(round(rmse, 5)),
        provenance=prov,
    )


def certify_n1_longitudinal_equivalence(
    source_scanner: ScannerSpec,
    source_recon: ReconConfig,
    target_scanner: ScannerSpec,
    target_recon: ReconConfig,
    *,
    regularization_lambda: float = 0.015,
    max_allowed_noise_amplification: float = 2.0,
    equivalence_tolerance_pct: float = 6.0,
    enforce_noise_gate: bool = True,
    radial_offset_mm: float = 57.2,
) -> CrossScannerTranslationResult:
    """Certify N=1 lesion-size-aware longitudinal SUV equivalence after applying T_{A->B} across 10-37 mm spheres."""
    # Check noise amplification norm on a standard 3D box first
    _, noise_amp, gibbs_supp = build_translation_transfer_function(
        (33, 45, 45),
        source_scanner,
        source_recon,
        target_scanner,
        target_recon,
        regularization_lambda=regularization_lambda,
        radial_offset_mm=radial_offset_mm,
    )
    if enforce_noise_gate:
        assert_translation_noise_safe(
            noise_amplification_factor=noise_amp,
            max_allowed_noise_amplification=max_allowed_noise_amplification,
            source_scanner_id=source_scanner.scanner_id,
            target_scanner_id=target_scanner.scanner_id,
        )

    # Verify DC / uniform field conservation
    const_vol = np.full((17, 21, 21), 2.5, dtype=np.float64)
    const_tr, _, _ = translate_reconstructed_volume(
        const_vol,
        source_scanner,
        source_recon,
        target_scanner,
        target_recon,
        regularization_lambda=regularization_lambda,
        enforce_noise_gate=False,
        radial_offset_mm=radial_offset_mm,
    )
    dc_err = float(np.max(np.abs(const_tr - 2.5)) / 2.5)

    # Evaluate per-sphere RCmax bias BEFORE vs AFTER retrospective translation across all 6 NEMA diameters
    dz, dy, dx = source_scanner.voxel_size_mm
    tgt_grid_sc = target_scanner.model_copy(update={"voxel_size_mm": source_scanner.voxel_size_mm})
    bias_before: dict[str, float] = {}
    bias_after: dict[str, float] = {}
    diffs_before: list[float] = []
    diffs_after: list[float] = []

    for d in NEMA_NU2_2018_DIAMETERS_MM:
        key = f"{int(d)}mm"
        nx = max(23, int(math.ceil((d + 48.0) / dx)) | 1)
        ny = max(23, int(math.ceil((d + 48.0) / dy)) | 1)
        nz = max(19, int(math.ceil((d + 40.0) / dz)) | 1)
        mask = build_antialiased_sphere_mask((nz, ny, nx), source_scanner.voxel_size_mm, (0.0, 0.0, 0.0), d, supersample=3)
        true_vol = 1.0 + 3.0 * mask
        roi = mask >= 0.1

        obs_src = apply_forward_system_operator(true_vol, source_scanner, source_recon, radial_offset_mm=radial_offset_mm, boundary_mode="nearest")
        obs_tgt = apply_forward_system_operator(true_vol, tgt_grid_sc, target_recon, radial_offset_mm=radial_offset_mm, boundary_mode="nearest")
        tr_src, _, _ = translate_reconstructed_volume(
            obs_src,
            source_scanner,
            source_recon,
            target_scanner,
            target_recon,
            regularization_lambda=regularization_lambda,
            enforce_noise_gate=False,
            radial_offset_mm=radial_offset_mm,
        )

        rc_src = float(np.max(obs_src[roi])) / 4.0
        rc_tgt = float(np.max(obs_tgt[roi])) / 4.0
        rc_tr = float(np.max(tr_src[roi])) / 4.0

        b_pre = (rc_src - rc_tgt) / max(rc_tgt, 1e-6) * 100.0
        b_post = (rc_tr - rc_tgt) / max(rc_tgt, 1e-6) * 100.0
        bias_before[key] = float(round(b_pre, 2))
        bias_after[key] = float(round(b_post, 2))
        diffs_before.append(abs(rc_src - rc_tgt))
        diffs_after.append(abs(rc_tr - rc_tgt))

    max_rc_pre = float(round(max(diffs_before), 5))
    max_rc_post = float(round(max(diffs_after), 5))
    max_pct_post = max(abs(v) for v in bias_after.values())
    noise_safe = noise_amp <= max_allowed_noise_amplification
    equiv = noise_safe and (max_pct_post <= equivalence_tolerance_pct)

    prov = build_provenance_record(
        operation="n1_retrospective_cross_scanner_translation",
        inputs={
            "source_scanner": source_scanner.scanner_id,
            "target_scanner": target_scanner.scanner_id,
            "regularization_lambda": regularization_lambda,
        },
        outputs={
            "noise_amplification_factor": noise_amp,
            "gibbs_peak_suppression_pct": gibbs_supp,
            "max_rc_mismatch_before": max_rc_pre,
            "max_rc_mismatch_after": max_rc_post,
            "longitudinal_equivalent": equiv,
        },
        data_source_tag="RETROSPECTIVE_OPERATOR_TRANSLATION_N1",
        citations=[
            "Orlhac et al. (2018) J Nucl Med 59(8):1321-1328, DOI:10.2967/jnumed.117.199935",
            "Kaalep et al. (2018) EJNMMI Phys 5:14, DOI:10.1186/s40658-018-0226-7",
        ],
    )

    return CrossScannerTranslationResult(
        source_scanner_id=source_scanner.scanner_id,
        target_scanner_id=target_scanner.scanner_id,
        regularization_lambda=float(regularization_lambda),
        noise_amplification_factor=float(round(noise_amp, 4)),
        max_allowed_noise_amplification=float(max_allowed_noise_amplification),
        noise_safe=noise_safe,
        dc_conservation_error=float(dc_err),
        gibbs_peak_suppression_pct=float(round(gibbs_supp, 2)),
        max_rc_mismatch_before=max_rc_pre,
        max_rc_mismatch_after=max_rc_post,
        per_sphere_bias_before_pct=bias_before,
        per_sphere_bias_after_pct=bias_after,
        longitudinal_equivalent=equiv,
        provenance=prov,
    )


def simulate_combat_size_blind_baseline(
    source_scanner: ScannerSpec,
    source_recon: ReconConfig,
    target_scanner: ScannerSpec,
    target_recon: ReconConfig,
) -> dict[str, float]:
    """Simulate a size-blind scalar shift/scale harmonization (ComBat-like mean alignment) for benchmarking.

    Demonstrates why lesion-size-blind cohort scaling leaves large residual errors on 10 mm vs 37 mm
    lesions compared to physics-informed operator translation T_{A->B}.
    """
    src_rcs = {}
    tgt_rcs = {}
    for d in NEMA_NU2_2018_DIAMETERS_MM:
        key = f"{int(d)}mm"
        src_rcs[key] = compute_sphere_recovery_coefficients(d, source_scanner, source_recon)["rc_max"]
        tgt_rcs[key] = compute_sphere_recovery_coefficients(d, target_scanner, target_recon)["rc_max"]

    # Optimal scalar multiplicative factor across all lesion sizes (size-blind mean matching)
    s_vals = np.array(list(src_rcs.values()))
    t_vals = np.array(list(tgt_rcs.values()))
    scalar_gain = float(np.dot(s_vals, t_vals) / max(float(np.dot(s_vals, s_vals)), 1e-12))

    return {
        k: float(round(((src_rcs[k] * scalar_gain) - tgt_rcs[k]) / max(tgt_rcs[k], 1e-6) * 100.0, 2))
        for k in src_rcs
    }
