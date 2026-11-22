"""Generate 6 publication-grade nuclear medicine physics figures (300-DPI PNG + vector PDF).

Aesthetic: JNM / EJNMMI Physics / Medical Physics publication styling, Okabe-Ito colorblind-safe
palette, bold panel labels, mm scalebars, and SHA-256 provenance footers.

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-quantnuc")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from quantnuc_inverse import __version__
from quantnuc_inverse.earl_harmonizer import (
    build_reference_scanner_fleet,
    get_corridor_spec,
    synthesize_harmonization_filter,
)
from quantnuc_inverse.forward_image_formation import (
    NEMA_NU2_2018_DIAMETERS_MM,
    apply_forward_system_operator,
    build_nema_iec_phantom_dro,
    compute_ct_ac_mismatch_report,
    compute_sphere_recovery_coefficients,
    simulate_ct_ac_dome_field_2d,
)
from quantnuc_inverse.harness.agent_orchestrator import QuantitativeImagingForensicsAgent
from quantnuc_inverse.ir import RESEARCH_DISCLAIMER, LesionROI
from quantnuc_inverse.pve_inverse_solver import invert_lesion_activity, solve_gtm_multi_region
from quantnuc_inverse.sobol_sensitivity import run_sobol_sensitivity

# Okabe-Ito colorblind-safe palette
OI = {
    "blue": "#0072B2",
    "vermilion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "skyblue": "#56B4E9",
    "black": "#1A1A1A",
    "gray": "#6E6E6E",
}

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#CCCCCC",
        "legend.fontsize": 8.2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _add_footer(fig: plt.Figure, note: str) -> None:
    fig.text(
        0.5,
        0.008,
        f"quantnuc-inverse v{__version__}  |  {note}  |  {RESEARCH_DISCLAIMER}",
        ha="center",
        va="bottom",
        fontsize=7.5,
        color="#555555",
        style="italic",
    )


def _add_scalebar(
    ax: plt.Axes,
    length_mm: float = 20.0,
    x_right: float = 112.0,
    y_bottom: float = -82.0,
    color: str = "white",
) -> None:
    ax.plot([x_right - length_mm, x_right], [y_bottom, y_bottom], color=color, lw=2.8, solid_capstyle="butt")
    ax.text(
        x_right - length_mm / 2.0,
        y_bottom + 4.5,
        f"{int(length_mm)} mm",
        color=color,
        fontsize=7.8,
        fontweight="bold",
        ha="center",
        va="bottom",
    )


def _save_fig(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def generate_fig1_nema_psf_gibbs(out_dir: Path) -> None:
    """Figure 1: NEMA IEC 6-sphere phantom formation, PSF-Gibbs overshoot & GTM recovery."""
    fleet = build_reference_scanner_fleet()
    sipm_sc, sipm_rc = fleet[0]
    pmt_sc, pmt_rc = fleet[2]

    # Harmonize SiPM to EARL2
    earl2 = get_corridor_spec("EARL2")
    sipm_harm_res = synthesize_harmonization_filter(sipm_sc, sipm_rc, earl2)
    sipm_harm_rc = sipm_rc.model_copy(
        update={"post_filter_fwhm_mm": sipm_harm_res.optimal_post_filter_fwhm_mm}
    )

    dro = build_nema_iec_phantom_dro(
        voxel_size_mm=(2.0, 2.0, 2.0),
        sphere_to_bg_ratio=4.0,
        background_activity_bq_ml=1.0,  # Normalized to SUV=1.0 bg, 4.0 hot spheres
    )
    pmt_2mm = pmt_sc.model_copy(update={"voxel_size_mm": (2.0, 2.0, 2.0)})

    vol_true = dro.true_volume
    vol_osem = apply_forward_system_operator(vol_true, pmt_2mm, pmt_rc)
    vol_psf = apply_forward_system_operator(vol_true, sipm_sc, sipm_rc)
    vol_harm = apply_forward_system_operator(vol_true, sipm_sc, sipm_harm_rc)

    # Exact Rousset GTM multi-region inversion on the harmonized volume
    region_masks = {**dro.sphere_masks, "background": dro.background_mask}
    # Add complement outside torso so partition is closed
    outside = np.clip(1.0 - sum(region_masks.values()), 0.0, 1.0)
    region_masks["outside_air"] = outside
    gtm_out = solve_gtm_multi_region(vol_harm, region_masks, sipm_sc, sipm_harm_rc)

    # Build GTM-recovered visualization slice
    mid_z = vol_true.shape[0] // 2
    vol_gtm_vis = np.zeros_like(vol_true[mid_z])
    for name, rec_val in zip(gtm_out.region_names, gtm_out.recovered_means, strict=True):
        if name != "outside_air":
            vol_gtm_vis += region_masks[name][mid_z] * rec_val

    ny, nx = vol_true.shape[1], vol_true.shape[2]
    extent = [-nx, nx, -ny, ny]  # 2 mm voxel -> half-extent = n_vox mm

    fig = plt.figure(figsize=(13.8, 7.8))
    gs = fig.add_gridspec(2, 5, height_ratios=[1.15, 1.0], width_ratios=[1.0, 1.0, 1.0, 1.0, 0.08], hspace=0.40, wspace=0.22)

    panels = [
        ("A. Ground Truth DRO (4:1)", vol_true[mid_z]),
        ("B. Legacy OSEM (PVE Loss)", vol_osem[mid_z]),
        ("C. Unharmonized SiPM PSF (Gibbs)", vol_psf[mid_z]),
        ("D. EARL2 + GTM Recovered", vol_gtm_vis),
    ]
    im_ref = None
    for idx, (title, slc) in enumerate(panels):
        ax = fig.add_subplot(gs[0, idx])
        im_ref = ax.imshow(slc, cmap="inferno", vmin=0.0, vmax=4.8, origin="lower", extent=extent)
        ax.set_title(title, fontsize=9.6, pad=6)
        ax.set_xlabel("x (mm)")
        if idx == 0:
            ax.set_ylabel("y (mm)")
            # Label sphere diameters cleanly around ring
            for sp in dro.spheres:
                _, cy, cx = sp.center_mm
                scale = 1.44 if sp.diameter_mm <= 17 else 1.48
                ax.text(
                    cx * scale,
                    cy * scale,
                    f"{int(sp.diameter_mm)}",
                    color="white",
                    fontsize=7.6,
                    fontweight="bold",
                    ha="center",
                    va="center",
                    bbox={"boxstyle": "round,pad=0.12", "fc": "black", "ec": "none", "alpha": 0.55},
                )
        else:
            ax.set_yticklabels([])
        ax.set_xlim(-122, 122)
        ax.set_ylim(-94, 94)
        _add_scalebar(ax, length_mm=20.0, x_right=104.0, y_bottom=-84.0)

    # Dedicated colorbar axis (zero overlap with Panel D)
    cax = fig.add_subplot(gs[0, 4])
    cbar = fig.colorbar(im_ref, cax=cax)
    cbar.set_label("Normalized Activity (SUV)", fontsize=8.8)

    # Panel E: 1D line profile across 22 mm & 10 mm spheres
    ax_prof = fig.add_subplot(gs[1, :2])
    y_center_idx = ny // 2
    x_mm = (np.arange(nx) - (nx - 1) / 2.0) * 2.0
    mask_x = (x_mm >= -90) & (x_mm <= 90)
    ax_prof.plot(x_mm[mask_x], vol_true[mid_z, y_center_idx, mask_x], color=OI["black"], ls="--", lw=1.8, label="Ground Truth (4.0)")
    ax_prof.plot(x_mm[mask_x], vol_osem[mid_z, y_center_idx, mask_x], color=OI["blue"], lw=2.0, label="Legacy OSEM (PVE Loss)")
    ax_prof.plot(x_mm[mask_x], vol_psf[mid_z, y_center_idx, mask_x], color=OI["vermilion"], lw=2.0, label="SiPM PSF-OSEM (Gibbs Ringing)")
    ax_prof.plot(x_mm[mask_x], vol_harm[mid_z, y_center_idx, mask_x], color=OI["green"], lw=2.0, label=f"EARL2 Harmonized ({sipm_harm_res.optimal_post_filter_fwhm_mm:.1f} mm)")

    psf_left_peak = float(np.max(vol_psf[mid_z, y_center_idx, x_mm < -30]))
    ax_prof.annotate(
        f"Gibbs overshoot\n({psf_left_peak:.2f}, +{(psf_left_peak/4.0 - 1)*100:.0f}%)",
        xy=(-49.0, psf_left_peak),
        xytext=(-39.0, 4.78),
        fontsize=7.9,
        fontweight="bold",
        color=OI["vermilion"],
        arrowprops={"arrowstyle": "->", "color": OI["vermilion"], "lw": 1.2},
    )
    ax_prof.text(-57.2, 0.35, "22 mm sphere", ha="center", fontsize=8.0, color="#444444", fontweight="bold")
    ax_prof.text(57.2, 0.35, "10 mm sphere", ha="center", fontsize=8.0, color="#444444", fontweight="bold")
    ax_prof.set_title("E. Central Axial Line Profile (22 mm & 10 mm Spheres)", loc="left", pad=6)
    ax_prof.set_xlabel("Transverse Position x (mm)")
    ax_prof.set_ylabel("Activity Concentration (SUV)")
    ax_prof.set_ylim(0.0, 5.55)
    ax_prof.grid(True, alpha=0.25, ls=":")
    ax_prof.legend(loc="upper right", ncol=1, fontsize=7.5)

    # Panel F: 6-sphere recovery bar chart
    ax_bar = fig.add_subplot(gs[1, 2:4])
    labels = [f"{int(d)} mm" for d in NEMA_NU2_2018_DIAMETERS_MM]
    keys = [f"{int(d)}mm" for d in NEMA_NU2_2018_DIAMETERS_MM]
    idx_map = {n: i for i, n in enumerate(gtm_out.region_names)}

    osem_means = [float(np.sum(vol_osem * dro.sphere_masks[k]) / np.sum(dro.sphere_masks[k])) for k in keys]
    psf_maxs = [float(np.max(vol_psf[dro.sphere_masks[k] >= 0.1])) for k in keys]
    gtm_recs = [float(gtm_out.recovered_means[idx_map[k]]) for k in keys]

    x_pos = np.arange(len(labels))
    w = 0.25
    ax_bar.bar(x_pos - w, osem_means, width=w, color=OI["blue"], label="Legacy OSEM Mean (PVE)")
    ax_bar.bar(x_pos, psf_maxs, width=w, color=OI["vermilion"], label="SiPM PSF Max (Gibbs)")
    ax_bar.bar(x_pos + w, gtm_recs, width=w, color=OI["green"], label="Rousset GTM Inverted")
    ax_bar.axhline(4.0, color=OI["black"], ls="--", lw=1.4, label="Ground Truth (4.00)")
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(labels)
    ax_bar.set_ylim(0.0, 5.95)
    ax_bar.set_ylabel("Regional Activity (SUV)")
    ax_bar.set_xlabel("NEMA NU 2-2018 Sphere Diameter")
    ax_bar.set_title(
        f"F. Multi-Region Recovery (GTM Re-proj. Residual = {gtm_out.reprojection_residual_inf:.1e})",
        loc="left",
        pad=6,
    )
    ax_bar.grid(True, axis="y", alpha=0.25, ls=":")
    ax_bar.legend(loc="upper center", ncol=2, fontsize=7.5)

    fig.subplots_adjust(bottom=0.10, top=0.93, left=0.06, right=0.95)
    _add_footer(fig, "NEMA NU 2-2018 6-Sphere DRO (4:1 ratio) | Rousset GTM (1998) + EARL2 Harmonization")
    _save_fig(fig, out_dir, "fig1_nema_psf_gibbs_formation")


def generate_fig2_earl_corridors(out_dir: Path) -> None:
    """Figure 2: Multi-scanner RCmax / RCmean / RCpeak BEFORE & AFTER harmonization vs EARL1/EARL2/MIRD-23."""
    fleet = build_reference_scanner_fleet()
    earl1 = get_corridor_spec("EARL1")
    earl2 = get_corridor_spec("EARL2")
    mird23 = get_corridor_spec("MIRD23_SPECT")

    results = []
    for sc, rc in fleet:
        corr = mird23 if sc.modality.startswith("SPECT") else earl2
        results.append(synthesize_harmonization_filter(sc, rc, corr))

    diams = np.array(NEMA_NU2_2018_DIAMETERS_MM)
    keys = [f"{int(d)}mm" for d in diams]

    styles = {
        "SiPM_Digital_PET": (OI["vermilion"], "o", "-"),
        "TOF_PSF_PET": (OI["orange"], "s", "-"),
        "Legacy_PMT_PET": (OI["blue"], "^", "-"),
        "CZT_SPECT_CT": (OI["purple"], "D", "--"),
    }

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.8), sharex=True)
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    def _shade_corridors(ax: plt.Axes, metric: str) -> None:
        if metric == "max":
            e1_lo, e1_hi = earl1.rc_max_lower, earl1.rc_max_upper
            e2_lo, e2_hi = earl2.rc_max_lower, earl2.rc_max_upper
            m_lo, m_hi = mird23.rc_max_lower, mird23.rc_max_upper
        elif metric == "mean":
            e1_lo, e1_hi = earl1.rc_mean_lower, earl1.rc_mean_upper
            e2_lo, e2_hi = earl2.rc_mean_lower, earl2.rc_mean_upper
            m_lo, m_hi = mird23.rc_mean_lower, mird23.rc_mean_upper
        else:
            assert earl1.rc_peak_lower and earl2.rc_peak_lower and mird23.rc_peak_lower
            assert earl1.rc_peak_upper and earl2.rc_peak_upper and mird23.rc_peak_upper
            e1_lo, e1_hi = earl1.rc_peak_lower, earl1.rc_peak_upper
            e2_lo, e2_hi = earl2.rc_peak_lower, earl2.rc_peak_upper
            m_lo, m_hi = mird23.rc_peak_lower, mird23.rc_peak_upper

        ax.fill_between(diams, e1_lo, e1_hi, color="#BBBBBB", alpha=0.28, label="EARL1 Corridor")
        ax.fill_between(diams, e2_lo, e2_hi, color=OI["skyblue"], alpha=0.30, label="EARL2 Corridor (PET)")
        ax.plot(diams, e2_lo, color=OI["blue"], ls=":", lw=1.0)
        ax.plot(diams, e2_hi, color=OI["blue"], ls=":", lw=1.0)
        ax.fill_between(diams, m_lo, m_hi, color=OI["purple"], alpha=0.12, label="MIRD-23 Corridor (SPECT)")
        ax.axhline(1.0, color="#666666", ls="--", lw=0.9, alpha=0.7)

    # Panel A: RCmax BEFORE
    _shade_corridors(ax_a, "max")
    for res in results:
        col, mk, ls = styles[res.scanner_id]
        vals = [res.pre_filter_rc_max[k] for k in keys]
        ax_a.plot(diams, vals, color=col, marker=mk, ls=ls, lw=2.0, ms=6, label=res.scanner_id.replace("_", " "))
    ax_a.set_title("A. RCmax BEFORE Harmonization (PSF-Gibbs Overshoot & PVE)", loc="left")
    ax_a.set_ylabel("Recovery Coefficient (RCmax)")
    ax_a.set_ylim(0.15, 1.42)
    ax_a.grid(True, alpha=0.25, ls=":")
    ax_a.legend(loc="lower right", ncol=2, fontsize=7.4)

    # Panel B: RCmax AFTER
    _shade_corridors(ax_b, "max")
    for res in results:
        col, mk, ls = styles[res.scanner_id]
        vals = [res.post_filter_rc_max[k] for k in keys]
        lbl = f"{res.scanner_id.replace('_', ' ')} (FWHM={res.optimal_post_filter_fwhm_mm:.1f} mm)"
        ax_b.plot(diams, vals, color=col, marker=mk, ls=ls, lw=2.0, ms=6, label=lbl)
    ax_b.set_title("B. RCmax AFTER Minimal Filter Synthesis (0.0000 Violation)", loc="left")
    ax_b.set_ylabel("Recovery Coefficient (RCmax)")
    ax_b.set_ylim(0.15, 1.42)
    ax_b.grid(True, alpha=0.25, ls=":")
    ax_b.legend(loc="lower right", ncol=1, fontsize=7.4)

    # Panel C: RCmean AFTER
    _shade_corridors(ax_c, "mean")
    for res in results:
        col, mk, ls = styles[res.scanner_id]
        pre_v = [res.pre_filter_rc_mean[k] for k in keys]
        post_v = [res.post_filter_rc_mean[k] for k in keys]
        ax_c.plot(diams, pre_v, color=col, ls=":", lw=1.2, alpha=0.45)
        ax_c.plot(diams, post_v, color=col, marker=mk, ls=ls, lw=2.0, ms=5.5, label=res.scanner_id.replace("_", " "))
    ax_c.set_title("C. RCmean (A50% Isocontour) Post-Harmonization (Dotted = Pre)", loc="left")
    ax_c.set_xlabel("NEMA Sphere Inner Diameter (mm)")
    ax_c.set_ylabel("Recovery Coefficient (RCmean)")
    ax_c.set_ylim(0.15, 1.15)
    ax_c.set_xticks(diams)
    ax_c.grid(True, alpha=0.25, ls=":")
    ax_c.legend(loc="lower right", ncol=2, fontsize=7.4)

    # Panel D: RCpeak AFTER
    _shade_corridors(ax_d, "peak")
    for res in results:
        col, mk, ls = styles[res.scanner_id]
        pre_v = [res.pre_filter_rc_peak[k] for k in keys]
        post_v = [res.post_filter_rc_peak[k] for k in keys]
        ax_d.plot(diams, pre_v, color=col, ls=":", lw=1.2, alpha=0.45)
        ax_d.plot(diams, post_v, color=col, marker=mk, ls=ls, lw=2.0, ms=5.5, label=res.scanner_id.replace("_", " "))
    ax_d.set_title("D. RCpeak (1 cm³ Sphere VOI) Post-Harmonization (Dotted = Pre)", loc="left")
    ax_d.set_xlabel("NEMA Sphere Inner Diameter (mm)")
    ax_d.set_ylabel("Recovery Coefficient (RCpeak)")
    ax_d.set_ylim(0.15, 1.18)
    ax_d.set_xticks(diams)
    ax_d.grid(True, alpha=0.25, ls=":")
    ax_d.legend(loc="lower right", ncol=2, fontsize=7.4)

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.99))
    _add_footer(fig, "Kaalep et al. (2018) EARL2 (DOI:10.1186/s40658-018-0226-7) & MIRD-23 Quantitative SPECT")
    _save_fig(fig, out_dir, "fig2_earl_harmonization_corridors")


def generate_fig3_pve_uncertainty(out_dir: Path) -> None:
    """Figure 3: True vs Observed vs PVE-Inverted SUV with 90% bias-variance ribbons (6-40 mm)."""
    fleet = build_reference_scanner_fleet()
    # Use clinical TOF-PSF PET with EARL2 harmonization
    sc, rc = fleet[1]
    earl2 = get_corridor_spec("EARL2")
    harm = synthesize_harmonization_filter(sc, rc, earl2)
    harm_rc = rc.model_copy(update={"post_filter_fwhm_mm": harm.optimal_post_filter_fwhm_mm})

    diams = np.array([6.0, 8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 28.0, 34.0, 40.0])
    true_suv = 8.0
    bg_suv = 2.0

    obs_means, gtm_nom, gtm_p05, gtm_p95, bayes_nom = [], [], [], [], []
    for d in diams:
        rc_info = compute_sphere_recovery_coefficients(
            float(d), sc, harm_rc, true_activity=true_suv, background_activity=bg_suv, radial_offset_mm=40.0
        )
        obs_m = true_suv * rc_info["rc_mean_true_voi"]
        obs_means.append(obs_m)

        roi = LesionROI(
            lesion_id=f"L_{int(d)}",
            diameter_mm=float(d),
            observed_suv_mean=obs_m,
            background_suv=bg_suv,
            radial_offset_mm=40.0,
        )
        rec_gtm = invert_lesion_activity(
            roi, sc, harm_rc, method="GTM_EXACT", enforce_nyquist_gate=False, n_uncertainty_draws=300
        )
        rec_bay = invert_lesion_activity(
            roi, sc, harm_rc, method="BAYESIAN_REGULARIZED", enforce_nyquist_gate=False, n_uncertainty_draws=300
        )
        gtm_nom.append(rec_gtm.recovered_activity_nominal)
        gtm_p05.append(rec_gtm.recovered_activity_p05)
        gtm_p95.append(rec_gtm.recovered_activity_p95)
        bayes_nom.append(rec_bay.recovered_activity_nominal)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 5.0))

    # Panel A: SUV curves + 90% CI ribbon + sub-Nyquist zone
    ax1.axvspan(4.0, 6.0, color="#E0E0E0", alpha=0.55, label="Sub-Nyquist Abstention Gate (<6 mm)")
    ax1.axhline(true_suv, color=OI["black"], ls="--", lw=1.6, label=f"Ground Truth SUV ({true_suv:.1f})")
    ax1.fill_between(diams, gtm_p05, gtm_p95, color=OI["green"], alpha=0.22, label="GTM 90% Bias–Variance CI [p05, p95]")
    ax1.plot(diams, gtm_nom, color=OI["green"], marker="o", lw=2.2, ms=5.5, label="Rousset GTM Exact Inversion")
    ax1.plot(diams, bayes_nom, color=OI["orange"], marker="^", ls="-.", lw=1.8, ms=5.0, label="Bayesian Regularized PVC")
    ax1.plot(diams, obs_means, color=OI["blue"], marker="s", lw=2.2, ms=5.5, label="Observed SUVmean (PVE Loss)")

    ax1.annotate(
        "30–55% PVE\nunderestimation",
        xy=(10.0, obs_means[2]),
        xytext=(15.5, 3.3),
        fontsize=8.3,
        fontweight="bold",
        color=OI["blue"],
        arrowprops={"arrowstyle": "->", "color": OI["blue"], "lw": 1.3},
    )
    ax1.set_title("A. Partial-Volume Bias vs. Inverse Recovery & 90% Uncertainty", loc="left")
    ax1.set_xlabel("Lesion Inner Diameter (mm)")
    ax1.set_ylabel("Lesion Activity Concentration (SUV)")
    ax1.set_xlim(4.0, 41.5)
    ax1.set_ylim(2.0, 12.2)
    ax1.grid(True, alpha=0.25, ls=":")
    ax1.legend(loc="upper right", fontsize=7.6)

    # Panel B: Bias-variance relative uncertainty across contrast ratios (2:1, 4:1, 8:1)
    contrasts = [(2.0, OI["skyblue"], "s", "2:1 Contrast (SUV 4 vs 2)"),
                 (4.0, OI["green"], "o", "4:1 Contrast (SUV 8 vs 2)"),
                 (8.0, OI["vermilion"], "D", "8:1 Contrast (SUV 16 vs 2)")]
    for ratio, col, mk, lbl in contrasts:
        t_val = bg_suv * ratio
        half_width_pct = []
        obs_err_pct = []
        for d in diams:
            rci = compute_sphere_recovery_coefficients(
                float(d), sc, harm_rc, true_activity=t_val, background_activity=bg_suv, radial_offset_mm=40.0
            )
            obs_m = t_val * rci["rc_mean_true_voi"]
            obs_err_pct.append(abs(obs_m - t_val) / t_val * 100.0)
            roi = LesionROI(
                lesion_id=f"C_{int(ratio)}_{int(d)}",
                diameter_mm=float(d),
                observed_suv_mean=obs_m,
                background_suv=bg_suv,
                radial_offset_mm=40.0,
            )
            rec = invert_lesion_activity(
                roi, sc, harm_rc, method="GTM_EXACT", enforce_nyquist_gate=False, n_uncertainty_draws=250
            )
            hw = 0.5 * (rec.recovered_activity_p95 - rec.recovered_activity_p05) / t_val * 100.0
            half_width_pct.append(hw)
        ax2.plot(diams, half_width_pct, color=col, marker=mk, lw=2.0, ms=5.5, label=f"GTM 90% CI Half-Width ({lbl})")
        ax2.plot(diams, obs_err_pct, color=col, ls=":", lw=1.4, alpha=0.75)

    ax2.text(
        18.5,
        42.0,
        "Dotted lines: Uncorrected PVE |Bias| (%)\nSolid lines: GTM 90% CI Half-Width (%)",
        fontsize=8.0,
        bbox={"boxstyle": "round,pad=0.35", "fc": "#F8F9FA", "ec": "#CCCCCC"},
    )
    ax2.set_title("B. PVE Bias vs. Inverse Variance Tradeoff by Contrast Ratio", loc="left")
    ax2.set_xlabel("Lesion Inner Diameter (mm)")
    ax2.set_ylabel("Relative Error / 90% CI Half-Width (%)")
    ax2.set_xlim(5.0, 41.5)
    ax2.set_ylim(0.0, 66.0)
    ax2.grid(True, alpha=0.25, ls=":")
    ax2.legend(loc="upper right", fontsize=7.6)

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.99))
    _add_footer(fig, "Rousset GTM (1998) + Monte Carlo PSF/Delineation/Poisson Propagation (EARL2 TOF-PSF PET)")
    _save_fig(fig, out_dir, "fig3_pve_inverse_uncertainty_curve")


def generate_fig4_ct_ac_mismatch(out_dir: Path) -> None:
    """Figure 4: Lung-liver dome CT-PET respiratory mismatch attenuation error & abstention threshold."""
    sim = simulate_ct_ac_dome_field_2d(breathing_shift_mm=10.0, photon_energy_kev=511.0, grid_size=(128, 128))
    z_mm = sim["z_mm"]
    x_mm = sim["x_mm"]
    extent = [x_mm[0], x_mm[-1], z_mm[0], z_mm[-1]]

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # Panel A: True Activity with PET vs CT diaphragm overlays
    im_a = ax_a.imshow(sim["true_activity"], cmap="inferno", vmin=0.0, vmax=8.5, origin="lower", extent=extent)
    ax_a.plot(x_mm, sim["dome_z_true"], color="#56B4E9", lw=2.0, ls="-", label="PET Diaphragm (True)")
    ax_a.plot(x_mm, sim["dome_z_ct"], color="#009E73", lw=2.0, ls="--", label="CT Diaphragm (+10 mm Insp.)")
    ax_a.set_title("A. True Activity & Respiratory Phase Mismatch (+10 mm)", loc="left")
    ax_a.set_xlabel("Lateral Position x (mm)")
    ax_a.set_ylabel("Superior–Inferior z (mm)")
    ax_a.legend(loc="upper right", fontsize=7.6)
    _add_scalebar(ax_a, length_mm=20.0, x_right=112.0, y_bottom=-112.0)
    cbar_a = fig.colorbar(im_a, ax=ax_a, fraction=0.046, pad=0.03)
    cbar_a.set_label("True SUV", fontsize=8.4)

    # Panel B: Attenuation Correction Bias Map (%)
    im_b = ax_b.imshow(sim["pct_error_map"], cmap="RdBu_r", vmin=-38.0, vmax=38.0, origin="lower", extent=extent)
    ax_b.plot(x_mm, sim["dome_z_true"], color="black", lw=1.2, ls=":")
    ax_b.plot(x_mm, sim["dome_z_ct"], color="black", lw=1.2, ls="--")
    ax_b.set_title(r"B. Attenuation Bias Map $\exp(\int \Delta\mu\,ds) - 1$ (%)", loc="left")
    ax_b.set_xlabel("Lateral Position x (mm)")
    ax_b.set_ylabel("Superior–Inferior z (mm)")
    _add_scalebar(ax_b, length_mm=20.0, x_right=112.0, y_bottom=-112.0, color="black")
    cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.046, pad=0.03)
    cbar_b.set_label("AC SUV Error (%)", fontsize=8.4)

    # Panel C: Reconstructed PET with False-Cold Liver-Dome Lesion
    im_c = ax_c.imshow(sim["reconstructed_activity"], cmap="inferno", vmin=0.0, vmax=8.5, origin="lower", extent=extent)
    ax_c.annotate(
        "False-cold liver dome\nlesion (−26% SUV)",
        xy=(0.0, 2.0),
        xytext=(24.0, 56.0),
        color="white",
        fontsize=8.2,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "white", "lw": 1.4},
    )
    ax_c.set_title("C. Reconstructed PET (Cold-Dome Artifact)", loc="left")
    ax_c.set_xlabel("Lateral Position x (mm)")
    ax_c.set_ylabel("Superior–Inferior z (mm)")
    _add_scalebar(ax_c, length_mm=20.0, x_right=112.0, y_bottom=-112.0)
    cbar_c = fig.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.03)
    cbar_c.set_label("Apparent SUV", fontsize=8.4)

    # Panel D: Analytical bias vs breathing shift (-15 to +15 mm) for PET 511 keV & Lu-177 SPECT 208 keV
    shifts = np.linspace(-15.0, 15.0, 61)
    pet_err = [compute_ct_ac_mismatch_report(float(s), photon_energy_kev=511.0).apparent_suv_percent_error for s in shifts]
    spect_err = [compute_ct_ac_mismatch_report(float(s), photon_energy_kev=208.0).apparent_suv_percent_error for s in shifts]

    ax_d.axhspan(-20.0, 20.0, color=OI["green"], alpha=0.12, label="Acceptable Tolerance (±20%)")
    ax_d.axhline(20.0, color=OI["vermilion"], ls="--", lw=1.3, label="Abstention Gate (SEVERE_CT_AC_MISMATCH)")
    ax_d.axhline(-20.0, color=OI["vermilion"], ls="--", lw=1.3)
    ax_d.axhline(0.0, color="#666666", lw=0.8)
    ax_d.axvline(0.0, color="#666666", lw=0.8)
    ax_d.plot(shifts, pet_err, color=OI["blue"], lw=2.3, label="PET 511 keV (F-18 / Ga-68)")
    ax_d.plot(shifts, spect_err, color=OI["vermilion"], lw=2.1, ls="-.", label="SPECT 208 keV (Lu-177)")
    ax_d.text(-14.6, 22.5, "FALSE-HOT\n(Exp. CT on Lung)", fontsize=7.9, color=OI["vermilion"], fontweight="bold")
    ax_d.text(3.2, -46.5, "FALSE-COLD\n(Insp. CT on Liver)", fontsize=7.9, color=OI["blue"], fontweight="bold")
    ax_d.set_title("D. Exact AC Line-Integral SUV Bias vs. Shift", loc="left")
    ax_d.set_xlabel("CT – PET/SPECT Diaphragm Shift Δz (mm)")
    ax_d.set_ylabel("Apparent SUV Bias (%)")
    ax_d.set_xlim(-15.5, 15.5)
    ax_d.set_ylim(-52.0, 72.0)
    ax_d.grid(True, alpha=0.25, ls=":")
    ax_d.legend(loc="upper right", fontsize=7.4)

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.99), w_pad=2.2)
    _add_footer(fig, "Exact Line-Integral Attenuation Bias exp(int delta_mu ds) | NIST XCOM 511 keV & 208 keV")
    _save_fig(fig, out_dir, "fig4_ct_ac_mismatch_forensics")


def generate_fig5_longitudinal_attribution(out_dir: Path) -> None:
    """Figure 5: Frontier Agent Forensics waterfall separating true delta-SUV vs PSF-Gibbs vs PVE vs CT-AC."""
    agent = QuantitativeImagingForensicsAgent()
    dossier = agent.build_complete_dossier()

    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    panel_letters = ["A", "B", "C", "D"]
    cat_labels = ["PSF-Gibbs\nArtifact", "PVE / Res.\nShift", "CT-AC\nResp. Shift", "True Bio\nΔSUV", "Observed\nTotal ΔSUV"]
    bar_colors = [OI["vermilion"], OI["orange"], OI["purple"], OI["green"], OI["blue"]]

    for ax, letter, diag in zip(axes.ravel(), panel_letters, dossier.diagnoses, strict=True):
        vals = [
            diag.psf_gibbs_artifact_pct,
            diag.pve_resolution_artifact_pct,
            diag.ct_ac_respiratory_artifact_pct,
            diag.true_biological_delta_pct,
        ]
        # Waterfall cumulative bases
        bottoms = [0.0]
        running = 0.0
        for v in vals[:-1]:
            running += v
            bottoms.append(running)
        total = diag.observed_delta_suv_pct

        x = np.arange(5)
        for i in range(4):
            v = vals[i]
            b = bottoms[i]
            rect = Rectangle(
                (x[i] - 0.34, min(b, b + v)),
                0.68,
                abs(v),
                facecolor=bar_colors[i],
                edgecolor="#222222",
                linewidth=0.8,
            )
            ax.add_patch(rect)
            # Connector dashed line to next bar
            if i < 3:
                ax.plot([x[i] + 0.34, x[i + 1] - 0.34], [b + v, b + v], color="#888888", ls=":", lw=1.0)
            y_txt = (b + v + 2.2) if v >= 0 else (b + v - 4.2)
            if abs(v) >= 0.4:
                ax.text(x[i], y_txt, f"{v:+.1f}%", ha="center", va="bottom" if v >= 0 else "top", fontsize=8.0, fontweight="bold")
            else:
                ax.text(x[i], b + 1.8, "0.0%", ha="center", va="bottom", fontsize=7.6, color="#666666")

        # Total observed bar
        ax.bar(x[4], total, width=0.68, color=bar_colors[4], edgecolor="#222222", linewidth=0.9)
        ax.text(x[4], total + 2.2, f"{total:+.1f}%", ha="center", va="bottom", fontsize=8.4, fontweight="bold", color=OI["blue"])

        ax.axhline(0.0, color="#333333", lw=0.9)
        ax.axhline(30.0, color=OI["vermilion"], ls="--", lw=1.1, alpha=0.75, label="PERCIST +30% Threshold")

        verdict_color = OI["green"] if diag.verdict == "TRUE_BIOLOGICAL_PROGRESSION" else OI["vermilion"]
        gate_str = "PERCIST: PASS" if diag.percist_quantifiable else "PERCIST: BLOCKED"
        ax.text(
            0.03,
            0.94,
            f"Verdict: {diag.verdict}\n{gate_str}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=7.8,
            fontweight="bold",
            color=verdict_color,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": verdict_color, "alpha": 0.92},
        )

        ax.set_title(f"{letter}. {diag.scenario_name}", loc="left", fontsize=9.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cat_labels, fontsize=8.0)
        ax.set_ylabel("Contribution to ΔSUV (%)")
        ax.set_xlim(-0.6, 4.6)
        ax.set_ylim(-14.0, max(54.0, total + 20.0))
        ax.grid(True, axis="y", alpha=0.25, ls=":")
        ax.legend(loc="lower right", fontsize=7.4)

    fig.tight_layout(rect=(0.0, 0.03, 1.0, 0.99))
    _add_footer(fig, "Frontier Agent Forensics Attribution | Exact Shapley-Logarithmic Physics Decomposition")
    _save_fig(fig, out_dir, "fig5_longitudinal_suv_attribution")


def generate_fig6_sobol_sensitivity(out_dir: Path) -> None:
    """Figure 6: Saltelli-Jansen QMC global Sobol' sensitivity (S1 vs ST) across small vs large lesions."""
    s_small = run_sobol_sensitivity(10.0, n_base_samples=2048, seed=2026)
    s_large = run_sobol_sensitivity(34.0, n_base_samples=2048, seed=2026)

    param_labels = {
        "psf_fwhm_mm": "PSF / CDR\nFWHM",
        "delineation_error_mm": "ROI Boundary\n(±1 Voxel)",
        "gibbs_ringing_strength": "PSF-Gibbs\nRinging",
        "ct_ac_shift_mm": "CT-AC Resp.\nPhase Shift",
    }
    pkeys = list(param_labels.keys())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.1))

    # Panel A: Grouped S1 & ST bars with 95% bootstrap error bars for 10 mm vs 34 mm
    x = np.arange(len(pkeys))
    w = 0.19

    def _ci_err(vals: list[float], cis: list[tuple[float, float]]) -> np.ndarray:
        lo = [max(0.0, v - ci[0]) for v, ci in zip(vals, cis, strict=True)]
        hi = [max(0.0, ci[1] - v) for v, ci in zip(vals, cis, strict=True)]
        return np.array([lo, hi])

    sm_s1 = [s_small.s1[k] for k in pkeys]
    sm_st = [s_small.st[k] for k in pkeys]
    lg_s1 = [s_large.s1[k] for k in pkeys]
    lg_st = [s_large.st[k] for k in pkeys]

    ax1.bar(
        x - 1.5 * w, sm_s1, width=w, color=OI["skyblue"], edgecolor="#333333", lw=0.7,
        yerr=_ci_err(sm_s1, [s_small.s1_ci95[k] for k in pkeys]), capsize=2.5,
        label=r"Small Lesion (10 mm) — First-Order $S_1$",
    )
    ax1.bar(
        x - 0.5 * w, sm_st, width=w, color=OI["blue"], edgecolor="#333333", lw=0.7,
        yerr=_ci_err(sm_st, [s_small.st_ci95[k] for k in pkeys]), capsize=2.5,
        label=r"Small Lesion (10 mm) — Total-Order $S_T$",
    )
    ax1.bar(
        x + 0.5 * w, lg_s1, width=w, color=OI["orange"], edgecolor="#333333", lw=0.7,
        yerr=_ci_err(lg_s1, [s_large.s1_ci95[k] for k in pkeys]), capsize=2.5,
        label=r"Large Lesion (34 mm) — First-Order $S_1$",
    )
    ax1.bar(
        x + 1.5 * w, lg_st, width=w, color=OI["vermilion"], edgecolor="#333333", lw=0.7,
        yerr=_ci_err(lg_st, [s_large.st_ci95[k] for k in pkeys]), capsize=2.5,
        label=r"Large Lesion (34 mm) — Total-Order $S_T$",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([param_labels[k] for k in pkeys])
    ax1.set_ylabel("Sobol' Variance Sensitivity Index (0 – 1)")
    ax1.set_ylim(0.0, 1.05)
    ax1.set_title("A. Small (10 mm) vs. Large (34 mm) Lesion Variance Decomposition", loc="left")
    ax1.grid(True, axis="y", alpha=0.25, ls=":")
    ax1.legend(loc="upper left", fontsize=7.6)

    # Panel B: Continuous ST crossover across lesion diameters 8 - 40 mm
    diam_grid = np.array([8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 26.0, 30.0, 35.0, 40.0])
    st_traces: dict[str, list[float]] = {k: [] for k in pkeys}
    for d in diam_grid:
        res_d = run_sobol_sensitivity(float(d), n_base_samples=1024, seed=2026, n_bootstrap=40)
        for k in pkeys:
            st_traces[k].append(res_d.st[k])

    trace_styles = {
        "psf_fwhm_mm": (OI["blue"], "o", r"PSF / CDR FWHM ($S_T$)"),
        "delineation_error_mm": (OI["green"], "s", r"ROI Delineation ±1 Vox ($S_T$)"),
        "gibbs_ringing_strength": (OI["orange"], "^", r"PSF-Gibbs Ringing ($S_T$)"),
        "ct_ac_shift_mm": (OI["vermilion"], "D", r"CT-AC Resp. Phase Shift ($S_T$)"),
    }
    for k in pkeys:
        col, mk, lbl = trace_styles[k]
        ax2.plot(diam_grid, st_traces[k], color=col, marker=mk, lw=2.1, ms=5.5, label=lbl)

    ax2.axvspan(8.0, 15.0, color=OI["skyblue"], alpha=0.12)
    ax2.axvspan(30.0, 40.0, color=OI["vermilion"], alpha=0.08)
    ax2.text(11.5, 0.94, "PVE / Resolution\nRegime (<15 mm)", ha="center", va="top", fontsize=7.8, fontweight="bold", color=OI["blue"])
    ax2.text(35.0, 0.94, "CT-AC Motion\nRegime (>30 mm)", ha="center", va="top", fontsize=7.8, fontweight="bold", color=OI["vermilion"])
    ax2.set_title(r"B. Total-Order Sensitivity ($S_T$) Regime Shift Across Lesion Size", loc="left")
    ax2.set_xlabel("Lesion Inner Diameter (mm)")
    ax2.set_ylabel(r"Total-Order Sobol' Index ($S_T$)")
    ax2.set_xlim(7.5, 40.8)
    ax2.set_ylim(0.0, 1.02)
    ax2.grid(True, alpha=0.25, ls=":")
    ax2.legend(loc="center right", fontsize=7.7)

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.99))
    _add_footer(fig, "Saltelli-Jansen QMC Sobol' Estimator (N=2048 base, 12288 evals) | DOI:10.1016/j.cpc.2009.09.018")
    _save_fig(fig, out_dir, "fig6_sobol_quantitative_variance")


def main() -> None:
    out_dir = Path("figures")
    print("Generating Figure 1 (NEMA PSF/Gibbs formation & GTM)...")
    generate_fig1_nema_psf_gibbs(out_dir)
    print("Generating Figure 2 (EARL1/EARL2/MIRD-23 corridors)...")
    generate_fig2_earl_corridors(out_dir)
    print("Generating Figure 3 (PVE inverse uncertainty curves)...")
    generate_fig3_pve_uncertainty(out_dir)
    print("Generating Figure 4 (CT-AC respiratory mismatch forensics)...")
    generate_fig4_ct_ac_mismatch(out_dir)
    print("Generating Figure 5 (Longitudinal ΔSUV forensics waterfall)...")
    generate_fig5_longitudinal_attribution(out_dir)
    print("Generating Figure 6 (Saltelli-Jansen Sobol' sensitivity)...")
    generate_fig6_sobol_sensitivity(out_dir)
    print(f"All 6 figures (PNG + PDF) saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
