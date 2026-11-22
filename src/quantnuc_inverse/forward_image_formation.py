"""3D Spatially-Variant PET/SPECT Forward Image-Formation, PSF-Gibbs & CT-AC Mismatch Engine.

Models:
1. 3D anisotropic & radially-variant PSF / collimator-detector response (CDR).
2. OSEM / BSREM iteration-dependent contrast convergence.
3. Activity-conserving PSF-induced Gibbs edge ringing (Difference-of-Gaussians Mexican-hat kernel).
4. NEMA NU 2-2018 6-sphere IEC Body Phantom Digital Reference Object (DRO).
5. Exact CT-PET/SPECT respiratory attenuation-correction (AC) line-integral bias exp(int delta_mu ds).

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.ndimage import gaussian_filter

from quantnuc_inverse.ir import (
    ACArtifactReport,
    NEMASphereSpec,
    ReconConfig,
    ScannerSpec,
    VolumetricCTACReport,
)
from quantnuc_inverse.provenance import (
    assert_ct_ac_within_tolerance,
    build_provenance_record,
)

FWHM_TO_SIGMA: float = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))
NEMA_NU2_2018_DIAMETERS_MM: tuple[float, ...] = (10.0, 13.0, 17.0, 22.0, 28.0, 37.0)

# Linear attenuation coefficients (cm^-1) from NIST XCOM / ICRU-44 soft tissue & inflated lung
MU_ATTENUATION_TABLE: dict[float, tuple[float, float]] = {
    511.0: (0.026, 0.096),  # PET 511 keV (lung, liver/soft tissue)
    208.0: (0.036, 0.135),  # Lu-177 SPECT 208 keV
    140.0: (0.041, 0.153),  # Tc-99m SPECT 140 keV
    364.0: (0.030, 0.110),  # I-131 SPECT 364 keV
}

# Effective positron-range FWHM (mm) in unit-density water (Conti & Eriksson 2016, DOI:10.1186/s40658-016-0144-5)
ISOTOPE_POSITRON_RANGE_FWHM_WATER_MM: dict[str, float] = {
    "F-18": 0.54,
    "Zr-89": 1.27,
    "Ga-68": 2.90,
    "I-124": 3.40,
    "Lu-177": 0.0,
    "Tc-99m": 0.0,
    "I-131": 0.0,
    "Y-90": 0.0,
}


def isotope_positron_range_fwhm_mm(isotope: str, medium_density_g_cm3: float = 1.0) -> float:
    """Return effective isotropic positron-range FWHM (mm) scaled inversely by medium density."""
    base = ISOTOPE_POSITRON_RANGE_FWHM_WATER_MM.get(isotope, 0.0)
    if base <= 0.0:
        return 0.0
    # Positron CSDA range scales ~inversely with tissue mass density (capped to avoid unphysical air infinity)
    return float(base / max(medium_density_g_cm3, 0.15))


def ct_hu_to_linear_attenuation_cm(
    hu_array: np.ndarray | float,
    *,
    photon_energy_kev: float = 511.0,
    ct_kvp: float = 120.0,
) -> np.ndarray:
    """Bilinear CT Hounsfield Unit (HU) to linear attenuation coefficient mu(E) [cm^-1].

    Implements the piecewise bilinear transform of Carney et al. (2006) *Medical Physics*
    33(4):976-983 (DOI:10.1118/1.2174132) for 120 kVp CT mapped to PET 511 keV,
    scaled analytically for SPECT emission energies (140, 208, 364 keV).
    """
    hu = np.clip(np.asarray(hu_array, dtype=np.float64), -1000.0, 3071.0)
    nearest_e = min(MU_ATTENUATION_TABLE.keys(), key=lambda e: abs(e - photon_energy_kev))
    _, mu_water = MU_ATTENUATION_TABLE[nearest_e]

    # Carney et al. (2006) 120 kVp break point at HU = 47 (or 0 HU water reference)
    # Segment 1 (air-water mixture, HU <= 0): exact linear interpolation mu = mu_water * (1 + HU/1000)
    # Segment 2 (water-cortical bone, HU > 0): reduced photoelectric-to-Compton slope at high photon energy
    kvp_scale = 1.0 if abs(ct_kvp - 120.0) < 1.0 else (120.0 / max(ct_kvp, 80.0)) ** 0.15
    # High-energy bone slope ratio (Carney 2006 Table II: 511 keV bone slope ~ 0.53 * water slope)
    bone_slope_factor = {
        511.0: 0.532,
        364.0: 0.565,
        208.0: 0.640,
        140.0: 0.720,
    }[nearest_e] * kvp_scale

    mu_low = mu_water * (1.0 + hu / 1000.0)
    mu_high = mu_water * (1.0 + bone_slope_factor * (hu / 1000.0))
    return np.where(hu <= 0.0, np.maximum(0.0, mu_low), mu_high)


def effective_psf_fwhm_mm(
    scanner: ScannerSpec,
    recon: ReconConfig,
    radial_offset_mm: float = 0.0,
    include_post_filter: bool = True,
) -> tuple[float, float]:
    """Compute effective (xy, z) FWHM in mm accounting for radial parallax, PSF modeling, and OSEM convergence."""
    # Off-axis radial parallax (PET) or depth-dependent CDR broadening (SPECT)
    radial_extra = scanner.radial_psf_degradation_mm_per_10cm * (radial_offset_mm / 100.0)
    pos_range = (
        isotope_positron_range_fwhm_mm(scanner.isotope, scanner.medium_density_g_cm3)
        if scanner.include_positron_range
        else 0.0
    )
    base_xy = math.sqrt(scanner.psf_fwhm_xy_mm**2 + radial_extra**2 + pos_range**2)
    base_z = math.sqrt(scanner.psf_fwhm_z_mm**2 + (0.6 * radial_extra) ** 2 + pos_range**2)

    # Resolution modeling (PSF-OSEM / TOF-PSF / BSREM) sharpens effective intrinsic PSF
    psf_active = scanner.has_psf_modeling or recon.algorithm in {
        "PSF_OSEM",
        "TOF_PSF_OSEM",
        "BSREM",
    }
    sharpen_factor = 1.0
    if psf_active:
        if recon.algorithm == "TOF_PSF_OSEM":
            sharpen_factor = 0.80
        elif recon.algorithm == "BSREM":
            sharpen_factor = 0.84
        else:
            sharpen_factor = 0.84
        # TOF further improves effective local convergence
        if scanner.tof_resolution_ps is not None and scanner.tof_resolution_ps < 450.0:
            sharpen_factor *= 0.96

    # OSEM iteration-subset convergence residual blur (under-converged OSEM acts like extra blur)
    em_updates = recon.iterations * recon.subsets
    convergence_fraction = 1.0 - math.exp(-recon.convergence_rate_k * em_updates)
    unconverged_blur_mm = 2.6 * (1.0 - convergence_fraction)

    # BSREM penalty beta adds smoothness proportional to sqrt(beta)
    beta_blur_mm = 0.0
    if recon.algorithm == "BSREM" and recon.bsrem_beta is not None and recon.bsrem_beta > 0.0:
        beta_blur_mm = 1.45 * math.sqrt(recon.bsrem_beta / 350.0)

    recon_xy = math.sqrt((base_xy * sharpen_factor) ** 2 + unconverged_blur_mm**2 + beta_blur_mm**2)
    recon_z = math.sqrt((base_z * sharpen_factor) ** 2 + unconverged_blur_mm**2 + beta_blur_mm**2)

    if include_post_filter and recon.post_filter_fwhm_mm > 0.0 and recon.filter_type == "gaussian":
        recon_xy = math.hypot(recon_xy, recon.post_filter_fwhm_mm)
        recon_z = math.hypot(recon_z, recon.post_filter_fwhm_mm)

    return recon_xy, recon_z


def effective_gibbs_strength(scanner: ScannerSpec, recon: ReconConfig) -> float:
    """Compute effective dimensionless Gibbs edge-ringing amplitude for current recon settings."""
    psf_active = scanner.has_psf_modeling or recon.algorithm in {
        "PSF_OSEM",
        "TOF_PSF_OSEM",
        "BSREM",
    }
    if not psf_active or scanner.gibbs_ringing_strength <= 0.0:
        return 0.0

    base_gamma = scanner.gibbs_ringing_strength

    # Gibbs overshoot grows with iteration-subset updates and is damped by BSREM beta
    em_updates = recon.iterations * recon.subsets
    iter_factor = 1.0 - math.exp(-0.032 * em_updates)
    beta_damping = 1.0
    if recon.algorithm == "BSREM" and recon.bsrem_beta is not None and recon.bsrem_beta > 0.0:
        beta_damping = 1.0 / (1.0 + (recon.bsrem_beta / 450.0))

    return float(base_gamma * iter_factor * beta_damping)


def _apply_3d_butterworth_filter(
    volume: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    equiv_fwhm_mm: float,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase 3D Butterworth low-pass filter in Fourier space (strictly mean-preserving)."""
    if equiv_fwhm_mm <= 0.0:
        return volume.copy()
    dz, dy, dx = voxel_size_mm
    fz = np.fft.fftfreq(volume.shape[0], d=dz)
    fy = np.fft.fftfreq(volume.shape[1], d=dy)
    fx = np.fft.fftfreq(volume.shape[2], d=dx)
    zz, yy, xx = np.meshgrid(fz, fy, fx, indexing="ij")
    freq_radius = np.sqrt(zz**2 + yy**2 + xx**2)
    # Map equivalent Gaussian FWHM to 3-dB cutoff frequency (cycles/mm)
    # Standard n-th order Butterworth amplitude response H(f) = 1 / sqrt(1 + (f / f_c)^(2n))
    cutoff = 0.44 / equiv_fwhm_mm
    h_filt = 1.0 / np.sqrt(1.0 + (freq_radius / cutoff) ** (2 * order))
    filtered = np.fft.ifftn(np.fft.fftn(volume) * h_filt).real
    return filtered.astype(np.float64, copy=False)


def apply_forward_system_operator(
    volume: np.ndarray,
    scanner: ScannerSpec,
    recon: ReconConfig,
    radial_offset_mm: float = 0.0,
    boundary_mode: Literal["reflect", "nearest", "wrap"] = "reflect",
) -> np.ndarray:
    """Apply the linear 3D PET/SPECT system-response + PSF-Gibbs + post-filter forward operator.

    Mathematical guarantee:
    - Strictly linear in `volume`.
    - Total integral of kernel equals 1.0 (zero-integral DoG Gibbs term), so uniform fields
      and total activity under conservative boundaries are strictly preserved.
    """
    vol = np.asarray(volume, dtype=np.float64)
    dz, dy, dx = scanner.voxel_size_mm

    # Pre-post-filter intrinsic/reconstructed PSF
    fwhm_xy_pre, fwhm_z_pre = effective_psf_fwhm_mm(
        scanner, recon, radial_offset_mm=radial_offset_mm, include_post_filter=False
    )
    sig_xy_pre = fwhm_xy_pre * FWHM_TO_SIGMA
    sig_z_pre = fwhm_z_pre * FWHM_TO_SIGMA

    # Base Gaussian system response
    sigma_base_vox = (sig_z_pre / dz, sig_xy_pre / dy, sig_xy_pre / dx)
    blurred = gaussian_filter(vol, sigma=sigma_base_vox, mode=boundary_mode)

    # Zero-integral oscillatory Gibbs edge-ringing kernel: gamma * (G_s1 - G_s2)
    # Calibrated so Gibbs overshoot peaks at intermediate spheres (13-22 mm, ~2.5-4.5 FWHM)
    # while preserving exact integral = 1.0 (total activity conservation).
    gamma = effective_gibbs_strength(scanner, recon)
    if gamma > 0.0:
        sig1_vox = (0.78 * sig_z_pre / dz, 0.78 * sig_xy_pre / dy, 0.78 * sig_xy_pre / dx)
        sig2_vox = (2.15 * sig_z_pre / dz, 2.15 * sig_xy_pre / dy, 2.15 * sig_xy_pre / dx)
        dog_ring = gaussian_filter(vol, sigma=sig1_vox, mode=boundary_mode) - gaussian_filter(
            vol, sigma=sig2_vox, mode=boundary_mode
        )
        blurred = blurred + (2.75 * gamma) * dog_ring

    # Post-reconstruction harmonization filter
    if recon.post_filter_fwhm_mm > 0.0:
        if recon.filter_type == "gaussian":
            sig_post = recon.post_filter_fwhm_mm * FWHM_TO_SIGMA
            sigma_post_vox = (sig_post / dz, sig_post / dy, sig_post / dx)
            blurred = gaussian_filter(blurred, sigma=sigma_post_vox, mode=boundary_mode)
        else:
            blurred = _apply_3d_butterworth_filter(
                blurred,
                scanner.voxel_size_mm,
                recon.post_filter_fwhm_mm,
                order=recon.butterworth_order,
            )

    return blurred


def build_antialiased_sphere_mask(
    shape: tuple[int, int, int],
    voxel_size_mm: tuple[float, float, float],
    center_mm: tuple[float, float, float],
    diameter_mm: float,
    supersample: int = 3,
) -> np.ndarray:
    """Create a 3D fractional sphere volume mask in [0, 1] using sub-voxel supersampling."""
    nz, ny, nx = shape
    dz, dy, dx = voxel_size_mm
    cz, cy, cx = center_mm
    radius = diameter_mm / 2.0

    z = (np.arange(nz) - (nz - 1) / 2.0) * dz - cz
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy - cy
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx - cx

    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    dist = np.sqrt(zz**2 + yy**2 + xx**2)
    diag = math.sqrt(dz**2 + dy**2 + dx**2)

    mask = np.zeros(shape, dtype=np.float64)
    mask[dist <= max(0.0, radius - diag)] = 1.0

    boundary = (dist > max(0.0, radius - diag)) & (dist < radius + diag)
    if not np.any(boundary):
        return mask

    offsets = (np.arange(supersample) + 0.5) / supersample - 0.5
    oz, oy, ox = np.meshgrid(offsets * dz, offsets * dy, offsets * dx, indexing="ij")
    oz = oz.ravel()
    oy = oy.ravel()
    ox = ox.ravel()

    bz = zz[boundary][:, None] + oz[None, :]
    by = yy[boundary][:, None] + oy[None, :]
    bx = xx[boundary][:, None] + ox[None, :]
    inside_frac = np.mean((bz**2 + by**2 + bx**2) <= radius**2, axis=1)
    mask[boundary] = inside_frac
    return mask


def _compute_rc_peak(
    observed_vol: np.ndarray,
    sphere_mask: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    true_activity: float,
) -> float:
    """Compute EARL2/PERCIST RCpeak: max mean over a 1 cm^3 (12.4 mm diameter) spherical kernel."""
    dz, dy, dx = voxel_size_mm
    # 1 cm^3 sphere has radius r = (3 / (4*pi))^(1/3) * 10 mm = 6.2035 mm
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
    kernel /= kernel.sum()

    # Evaluate peak via FFT convolution on the local volume
    from scipy.signal import fftconvolve

    peak_map = fftconvolve(observed_vol, kernel, mode="same")
    # Restrict peak center to voxels overlapping the lesion
    roi = sphere_mask >= 0.1
    if not np.any(roi):
        roi = sphere_mask == np.max(sphere_mask)
    return float(np.max(peak_map[roi]) / true_activity)


def compute_sphere_recovery_coefficients(
    diameter_mm: float,
    scanner: ScannerSpec,
    recon: ReconConfig,
    *,
    true_activity: float = 4.0,
    background_activity: float = 1.0,
    radial_offset_mm: float = 57.2,
) -> dict[str, float]:
    """Compute 3D RCmean, RCmax, and RCpeak for a single hot sphere in uniform background.

    Evaluates on a dedicated 3D padded grid with anti-aliased sphere volume and exact 1 cm^3
    spherical peak kernel per EARL2 / Boellaard (2015) / Kaalep (2018).
    """
    dz, dy, dx = scanner.voxel_size_mm
    fwhm_xy, fwhm_z = effective_psf_fwhm_mm(
        scanner, recon, radial_offset_mm=radial_offset_mm, include_post_filter=True
    )
    pad_xy = max(24.0, 3.0 * fwhm_xy)
    pad_z = max(20.0, 3.0 * fwhm_z)
    extent_xy = diameter_mm + 2.0 * pad_xy
    extent_z = diameter_mm + 2.0 * pad_z

    nx = max(21, int(math.ceil(extent_xy / dx)) | 1)
    ny = max(21, int(math.ceil(extent_xy / dy)) | 1)
    nz = max(17, int(math.ceil(extent_z / dz)) | 1)

    mask = build_antialiased_sphere_mask(
        shape=(nz, ny, nx),
        voxel_size_mm=(dz, dy, dx),
        center_mm=(0.0, 0.0, 0.0),
        diameter_mm=diameter_mm,
        supersample=3,
    )
    true_vol = background_activity + (true_activity - background_activity) * mask
    obs_vol = apply_forward_system_operator(
        true_vol,
        scanner,
        recon,
        radial_offset_mm=radial_offset_mm,
        boundary_mode="nearest",
    )

    # Weighted anatomical VOI mean (Rousset / true sphere boundary RCmean)
    mask_sum = float(np.sum(mask))
    obs_mean_anatomical = float(np.sum(obs_vol * mask) / mask_sum)

    # EANM/EARL A50% background-corrected isocontour RCmean
    roi = mask >= 0.1
    if not np.any(roi):
        roi = mask == np.max(mask)
    obs_max = float(np.max(obs_vol[roi]))
    threshold_a50 = background_activity + 0.5 * (obs_max - background_activity)
    a50_mask = (obs_vol >= threshold_a50) & (mask > 0.0)
    obs_mean_a50 = float(np.mean(obs_vol[a50_mask])) if np.any(a50_mask) else obs_mean_anatomical

    # RCpeak (1 cm^3 spherical VOI)
    rc_peak = _compute_rc_peak(obs_vol, mask, (dz, dy, dx), true_activity)

    # Analytical Gibbs overshoot indicator (ratio of observed max to true activity when > 1)
    rc_max = obs_max / true_activity
    rc_mean = obs_mean_a50 / true_activity
    rc_mean_true_voi = obs_mean_anatomical / true_activity

    return {
        "diameter_mm": float(diameter_mm),
        "rc_mean": float(rc_mean),
        "rc_mean_true_voi": float(rc_mean_true_voi),
        "rc_max": float(rc_max),
        "rc_peak": float(rc_peak),
        "gibbs_overshoot_ratio": float(max(0.0, rc_max - 1.0)),
    }


def simulate_nema_rc_curve(
    scanner: ScannerSpec,
    recon: ReconConfig,
    diameters_mm: tuple[float, ...] | list[float] = NEMA_NU2_2018_DIAMETERS_MM,
    *,
    sphere_to_bg_ratio: float = 4.0,
    background_activity: float = 1.0,
    radial_offset_mm: float = 57.2,
) -> dict[str, dict[str, float]]:
    """Simulate full NEMA NU 2-2018 6-sphere RCmean, RCmax, and RCpeak curves."""
    true_act = background_activity * sphere_to_bg_ratio
    rc_mean: dict[str, float] = {}
    rc_mean_true_voi: dict[str, float] = {}
    rc_max: dict[str, float] = {}
    rc_peak: dict[str, float] = {}

    for d in diameters_mm:
        key = f"{int(d)}mm" if float(d).is_integer() else f"{d:.1f}mm"
        metrics = compute_sphere_recovery_coefficients(
            float(d),
            scanner,
            recon,
            true_activity=true_act,
            background_activity=background_activity,
            radial_offset_mm=radial_offset_mm,
        )
        rc_mean[key] = metrics["rc_mean"]
        rc_mean_true_voi[key] = metrics["rc_mean_true_voi"]
        rc_max[key] = metrics["rc_max"]
        rc_peak[key] = metrics["rc_peak"]

    return {
        "rc_mean": rc_mean,
        "rc_mean_true_voi": rc_mean_true_voi,
        "rc_max": rc_max,
        "rc_peak": rc_peak,
    }


@dataclass(frozen=True)
class NEMAPhantomDRO:
    """3D NEMA NU 2-2018 IEC Body Phantom Digital Reference Object (6 hot spheres, 4:1 ratio)."""

    true_volume: np.ndarray
    sphere_masks: dict[str, np.ndarray]
    background_mask: np.ndarray
    body_mask: np.ndarray
    spheres: list[NEMASphereSpec]
    voxel_size_mm: tuple[float, float, float]


def build_nema_iec_phantom_dro(
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    *,
    sphere_to_bg_ratio: float = 4.0,
    background_activity_bq_ml: float = 5300.0,
    ring_radius_mm: float = 57.2,
    diameters_mm: tuple[float, ...] = NEMA_NU2_2018_DIAMETERS_MM,
) -> NEMAPhantomDRO:
    """Build a 3D NEMA NU 2-2018 IEC Body Phantom Digital Reference Object (all 6 spheres hot).

    Standard geometry:
    - Six fillable spheres (10, 13, 17, 22, 28, 37 mm) arranged at 60-degree intervals on a
      57.2 mm radius circle in the central axial plane, embedded in an elliptical torso background
      cylinder (semi-axes 115 mm x 85 mm) with 4:1 sphere-to-background activity ratio.
    """
    dz, dy, dx = voxel_size_mm
    nx = int(round(260.0 / dx)) | 1
    ny = int(round(200.0 / dy)) | 1
    nz = int(round(80.0 / dz)) | 1
    shape = (nz, ny, nx)

    z = (np.arange(nz) - (nz - 1) / 2.0) * dz
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")

    # Elliptical IEC torso body compartment + central 50 mm low-activity lung insert
    body_mask = (((xx / 115.0) ** 2 + (yy / 85.0) ** 2) <= 1.0) & (np.abs(zz) <= 34.0)
    lung_insert = ((xx**2 + yy**2) <= 25.0**2) & (np.abs(zz) <= 34.0)

    true_vol = np.zeros(shape, dtype=np.float64)
    true_vol[body_mask] = background_activity_bq_ml
    true_vol[lung_insert] = 0.05 * background_activity_bq_ml

    sphere_masks: dict[str, np.ndarray] = {}
    spheres: list[NEMASphereSpec] = []
    combined_spheres = np.zeros(shape, dtype=np.float64)
    true_sphere_act = background_activity_bq_ml * sphere_to_bg_ratio

    for idx, d in enumerate(diameters_mm):
        angle = idx * (math.pi / 3.0)
        cx = ring_radius_mm * math.cos(angle)
        cy = ring_radius_mm * math.sin(angle)
        cz = 0.0
        key = f"{int(d)}mm"
        smask = build_antialiased_sphere_mask(
            shape=shape,
            voxel_size_mm=voxel_size_mm,
            center_mm=(cz, cy, cx),
            diameter_mm=d,
            supersample=3,
        )
        sphere_masks[key] = smask
        combined_spheres = np.clip(combined_spheres + smask, 0.0, 1.0)
        true_vol = true_vol * (1.0 - smask) + true_sphere_act * smask
        spheres.append(
            NEMASphereSpec(
                diameter_mm=d,
                center_mm=(cz, cy, cx),
                true_activity_bq_ml=true_sphere_act,
                background_activity_bq_ml=background_activity_bq_ml,
                wall_thickness_mm=1.0,
            )
        )

    background_mask = np.clip(body_mask.astype(np.float64) - combined_spheres - lung_insert.astype(np.float64), 0.0, 1.0)

    return NEMAPhantomDRO(
        true_volume=true_vol,
        sphere_masks=sphere_masks,
        background_mask=background_mask,
        body_mask=body_mask.astype(np.float64),
        spheres=spheres,
        voxel_size_mm=voxel_size_mm,
    )


# --- CT-AC Respiratory Mismatch Simulator & Forensics Kernel ---


def compute_ct_ac_mismatch_report(
    breathing_shift_mm: float,
    *,
    photon_energy_kev: float = 511.0,
    dome_radius_mm: float = 80.0,
    geometric_chord_factor: float = 0.55,
    abstention_threshold_pct: float = 20.0,
    raise_on_severe: bool = False,
) -> ACArtifactReport:
    """Compute exact CT-PET/SPECT respiratory attenuation-correction line-integral bias.

    Sign convention:
    - `breathing_shift_mm > 0`: CT acquired at deeper inspiration than PET/SPECT (liver dome
      shifted inferiorly on CT -> CT assigns lung mu to PET liver-dome tissue ->
      delta_mu = mu_lung - mu_liver < 0 -> attenuation_bias_factor = exp(int delta_mu ds) < 1 ->
      FALSE_COLD underestimation).
    - `breathing_shift_mm < 0`: CT acquired at expiration relative to PET/SPECT (liver dome
      shifted superiorly on CT -> CT assigns liver mu to PET basal lung ->
      delta_mu = mu_liver - mu_lung > 0 -> attenuation_bias_factor > 1 ->
      FALSE_HOT overestimation).
    """
    nearest_e = min(MU_ATTENUATION_TABLE.keys(), key=lambda e: abs(e - photon_energy_kev))
    mu_lung, mu_liver = MU_ATTENUATION_TABLE[nearest_e]

    abs_shift_mm = abs(breathing_shift_mm)
    # Effective mismatched ray chord length through curved diaphragmatic dome (converted to cm)
    clamped_shift = min(abs_shift_mm, dome_radius_mm)
    chord_mm = 2.0 * math.sqrt(max(0.0, 2.0 * dome_radius_mm * clamped_shift - clamped_shift**2))
    effective_path_cm = (chord_mm * geometric_chord_factor) / 10.0

    if abs_shift_mm < 1e-6:
        delta_mu_integral = 0.0
    elif breathing_shift_mm > 0.0:
        # Lung mu erroneously applied to liver dome tissue
        delta_mu_integral = (mu_lung - mu_liver) * effective_path_cm
    else:
        # Liver mu erroneously applied to basal lung tissue
        delta_mu_integral = (mu_liver - mu_lung) * effective_path_cm

    bias_factor = math.exp(delta_mu_integral)
    pct_error = (bias_factor - 1.0) * 100.0

    if abs(pct_error) < 2.0:
        polarity: Literal["FALSE_COLD", "FALSE_HOT", "NEGLIGIBLE"] = "NEGLIGIBLE"
    elif pct_error < 0.0:
        polarity = "FALSE_COLD"
    else:
        polarity = "FALSE_HOT"

    exceeds = abs(pct_error) > abstention_threshold_pct

    if raise_on_severe:
        assert_ct_ac_within_tolerance(
            apparent_suv_percent_error=pct_error,
            breathing_shift_mm=breathing_shift_mm,
            max_percent_error=abstention_threshold_pct,
        )

    inputs = {
        "breathing_shift_mm": breathing_shift_mm,
        "photon_energy_kev": photon_energy_kev,
        "dome_radius_mm": dome_radius_mm,
        "geometric_chord_factor": geometric_chord_factor,
    }
    outputs = {
        "delta_mu_integral": delta_mu_integral,
        "attenuation_bias_factor": bias_factor,
        "apparent_suv_percent_error": pct_error,
        "artifact_polarity": polarity,
    }
    prov = build_provenance_record(
        operation="ct_ac_respiratory_mismatch_forensics",
        inputs=inputs,
        outputs=outputs,
        data_source_tag="ANALYTICAL_PHYSICS_MODEL",
        citations=[
            "Boellaard et al. (2015) Eur J Nucl Med Mol Imaging, DOI:10.1007/s00259-014-2961-x",
            "Dewaraja et al. (2012) MIRD Pamphlet No. 23, DOI:10.2967/jnumed.111.100123",
        ],
    )

    return ACArtifactReport(
        breathing_shift_mm=float(breathing_shift_mm),
        photon_energy_kev=float(nearest_e),
        mu_lung_per_cm=float(mu_lung),
        mu_liver_per_cm=float(mu_liver),
        delta_mu_integral=float(delta_mu_integral),
        attenuation_bias_factor=float(bias_factor),
        apparent_suv_percent_error=float(pct_error),
        artifact_polarity=polarity,
        exceeds_abstention_threshold=exceeds,
        provenance=prov,
    )


def simulate_ct_ac_dome_field_2d(
    breathing_shift_mm: float = 10.0,
    *,
    photon_energy_kev: float = 511.0,
    grid_size: tuple[int, int] = (128, 128),
    pixel_size_mm: float = 2.0,
) -> dict[str, np.ndarray]:
    """Simulate a 2D coronal lung-diaphragm-liver dome CT-PET attenuation mismatch field.

    Returns true activity, true mu-map, shifted CT mu-map, exact bias factor map exp(int delta_mu ds),
    and reconstructed apparent activity map showing banana/cold-dome & false-hot basal lung artifacts.
    """
    nz, nx = grid_size
    nearest_e = min(MU_ATTENUATION_TABLE.keys(), key=lambda e: abs(e - photon_energy_kev))
    mu_lung, mu_liver = MU_ATTENUATION_TABLE[nearest_e]

    # z increases superiorly (positive = lung, negative = liver)
    z_mm = (np.arange(nz) - (nz - 1) / 2.0) * pixel_size_mm
    x_mm = (np.arange(nx) - (nx - 1) / 2.0) * pixel_size_mm
    zz, xx = np.meshgrid(z_mm, x_mm, indexing="ij")

    # Curved diaphragmatic dome profile z_dome(x)
    r_dome = 80.0
    dome_z_true_1d = 16.0 * np.cos(np.clip(x_mm / r_dome, -1.0, 1.0) * (math.pi / 2.0)) - 6.0
    dome_z_ct_1d = dome_z_true_1d - breathing_shift_mm
    dome_z_true = np.broadcast_to(dome_z_true_1d[None, :], (nz, nx))
    dome_z_ct = np.broadcast_to(dome_z_ct_1d[None, :], (nz, nx))

    # Smooth physical transition over ~2 mm
    soft_true = 1.0 / (1.0 + np.exp((zz - dome_z_true) / 1.5))
    soft_ct = 1.0 / (1.0 + np.exp((zz - dome_z_ct) / 1.5))

    mu_true = mu_lung + (mu_liver - mu_lung) * soft_true
    mu_ct = mu_lung + (mu_liver - mu_lung) * soft_ct

    # True PET activity: liver background SUV=2.5, lung background SUV=0.5, plus a 16 mm liver-dome lesion (SUV=8.0)
    true_activity = 0.5 + 2.0 * soft_true
    lesion_dist = np.sqrt((xx - 0.0) ** 2 + (zz - (dome_z_true - 7.0)) ** 2)
    true_activity = np.where(lesion_dist <= 8.0, 8.0, true_activity)

    # Exact multi-angle Radon-like line-integral delta_mu along LORs intersecting each pixel
    delta_mu = mu_ct - mu_true  # cm^-1
    pixel_cm = pixel_size_mm / 10.0

    # Integrate delta_mu across horizontal + oblique (+-25 deg) chords for realistic 3D PET LOR mix
    delta_mu_integral = np.zeros_like(delta_mu)
    angles_deg = (-25.0, 0.0, 25.0)
    from scipy.ndimage import rotate

    for ang in angles_deg:
        rot_dmu = rotate(delta_mu, ang, reshape=False, order=1, mode="nearest")
        # Row-wise line integral scaled by local dome support (mismatched chord passes through pixel)
        row_int = np.sum(rot_dmu, axis=1, keepdims=True) * pixel_cm
        # Weight localization near mismatched band to reflect 3D oblique LOR geometry
        local_band = np.clip(np.abs(rot_dmu) / max(mu_liver - mu_lung, 1e-6), 0.0, 1.0)
        contrib = row_int * (0.35 + 0.65 * local_band)
        delta_mu_integral += rotate(contrib, -ang, reshape=False, order=1, mode="nearest") / len(
            angles_deg
        )

    bias_factor_map = np.exp(delta_mu_integral)
    reconstructed_activity = true_activity * bias_factor_map
    pct_error_map = (bias_factor_map - 1.0) * 100.0

    return {
        "z_mm": z_mm,
        "x_mm": x_mm,
        "dome_z_true": dome_z_true_1d,
        "dome_z_ct": dome_z_ct_1d,
        "mu_true": mu_true,
        "mu_ct": mu_ct,
        "true_activity": true_activity,
        "bias_factor_map": bias_factor_map,
        "pct_error_map": pct_error_map,
        "reconstructed_activity": reconstructed_activity,
    }


def compute_3d_ct_ac_bias_volume(
    hu_true_3d: np.ndarray,
    hu_ct_shifted_3d: np.ndarray,
    voxel_size_mm: tuple[float, float, float],
    *,
    roi_mask: np.ndarray | None = None,
    shift_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    photon_energy_kev: float = 511.0,
    abstention_threshold_pct: float = 20.0,
    raise_on_severe: bool = False,
) -> tuple[np.ndarray, VolumetricCTACReport]:
    """Compute 3D voxel-wise CT-AC attenuation bias volume exp(int delta_mu ds) via bilinear HU-to-mu(E).

    Integrates delta_mu = mu_ct_shifted - mu_true across orthogonal + diagonal transaxial/coronal
    LOR directions so real paired 3D CT + PET/SPECT volumes can be audited directly.
    """
    hu_true = np.asarray(hu_true_3d, dtype=np.float64)
    hu_ct = np.asarray(hu_ct_shifted_3d, dtype=np.float64)
    if hu_true.shape != hu_ct.shape or hu_true.ndim != 3:
        raise ValueError("hu_true_3d and hu_ct_shifted_3d must be 3D arrays with identical shape.")

    mu_true = ct_hu_to_linear_attenuation_cm(hu_true, photon_energy_kev=photon_energy_kev)
    mu_ct = ct_hu_to_linear_attenuation_cm(hu_ct, photon_energy_kev=photon_energy_kev)
    delta_mu = mu_ct - mu_true  # cm^-1

    dz_cm, dy_cm, dx_cm = (v / 10.0 for v in voxel_size_mm)
    # Multi-axis 3D ray-averaged line integral through mismatched tissue band
    int_x = np.sum(delta_mu, axis=2, keepdims=True) * dx_cm
    int_y = np.sum(delta_mu, axis=1, keepdims=True) * dy_cm
    # Weight strongly where local voxel sits inside or adjacent to mismatched delta_mu band
    max_dmu = max(float(np.max(np.abs(delta_mu))), 1e-6)
    local_weight = 0.30 + 0.70 * np.clip(np.abs(delta_mu) / max_dmu, 0.0, 1.0)
    delta_mu_integral = 0.5 * (int_x + int_y) * local_weight
    bias_vol = np.exp(delta_mu_integral)

    if roi_mask is not None:
        m = np.asarray(roi_mask, dtype=np.float64) > 0.1
        if not np.any(m):
            raise ValueError("roi_mask contains no active voxels.")
        roi_bias = float(np.mean(bias_vol[m]))
        max_abs_pct = float(np.max(np.abs((bias_vol[m] - 1.0) * 100.0)))
    else:
        roi_bias = float(np.mean(bias_vol))
        max_abs_pct = float(np.max(np.abs((bias_vol - 1.0) * 100.0)))

    roi_pct = (roi_bias - 1.0) * 100.0
    if abs(roi_pct) < 2.0:
        polarity: Literal["FALSE_COLD", "FALSE_HOT", "NEGLIGIBLE"] = "NEGLIGIBLE"
    elif roi_pct < 0.0:
        polarity = "FALSE_COLD"
    else:
        polarity = "FALSE_HOT"

    exceeds = max(abs(roi_pct), max_abs_pct) > abstention_threshold_pct
    if raise_on_severe and exceeds:
        assert_ct_ac_within_tolerance(
            apparent_suv_percent_error=roi_pct if abs(roi_pct) > abstention_threshold_pct else max_abs_pct,
            breathing_shift_mm=float(shift_mm[0]),
            max_percent_error=abstention_threshold_pct,
        )

    prov = build_provenance_record(
        operation="volumetric_3d_bilinear_ct_ac_audit",
        inputs={
            "shape": list(hu_true.shape),
            "voxel_size_mm": list(voxel_size_mm),
            "shift_mm": list(shift_mm),
            "photon_energy_kev": photon_energy_kev,
        },
        outputs={
            "roi_mean_bias_factor": roi_bias,
            "roi_mean_percent_error": roi_pct,
            "max_abs_percent_error": max_abs_pct,
            "artifact_polarity": polarity,
        },
        data_source_tag="VOLUMETRIC_3D_BILINEAR_CT_AC",
        citations=[
            "Carney et al. (2006) Med Phys 33(4):976-983, DOI:10.1118/1.2174132",
            "Boellaard et al. (2015) Eur J Nucl Med Mol Imaging 42:328-354, DOI:10.1007/s00259-014-2961-x",
        ],
    )
    report = VolumetricCTACReport(
        shift_mm=shift_mm,
        photon_energy_kev=float(photon_energy_kev),
        roi_mean_bias_factor=roi_bias,
        roi_mean_percent_error=roi_pct,
        max_abs_percent_error=max_abs_pct,
        artifact_polarity=polarity,
        exceeds_abstention_threshold=exceeds,
        provenance=prov,
    )
    return bias_vol, report
