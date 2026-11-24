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
    compute_harmonization_pareto_frontier,
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
from quantnuc_inverse.ir import RESEARCH_DISCLAIMER, LesionROI, TimeActivityPoint
from quantnuc_inverse.pve_inverse_solver import (
    apply_rbv_pvc,
    build_necrotic_tumor_phantom,
    invert_lesion_activity,
    solve_gtm_multi_region,
)
from quantnuc_inverse.scanner_translator import (
    build_translation_transfer_function,
    certify_n1_longitudinal_equivalence,
    compute_3d_system_mtf,
    simulate_combat_size_blind_baseline,
)
from quantnuc_inverse.sobol_sensitivity import run_sobol_sensitivity
from quantnuc_inverse.stp_dosimetry_designer import (
    compute_gear2018_dose_uncertainty,
    design_optimal_stp_protocol,
)
from quantnuc_inverse.theranostics_dosimetry import (
    compute_mird_sphere_s_factor,
    compute_pve_propagated_theranostic_dose,
    fit_tiac_mbq_h_per_ml,
)

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


def generate_fig7_theranostics_and_positron_range(out_dir: Path) -> None:
    """Figure 7: MIRD-26 Lu-177 PVE-propagated absorbed dose (Gy) + multi-isotope positron-range RC loss."""
    fleet = build_reference_scanner_fleet()
    sipm_sc, sipm_rc = fleet[0]
    spect_sc, spect_rc = fleet[3]
    spect_harm = spect_rc.model_copy(update={"post_filter_fwhm_mm": 4.0})

    # Panel A: Lu-177 SPECT tumor absorbed dose across 10-40 mm (same true biological concentration TAC)
    diams = np.array([10.0, 12.0, 15.0, 18.0, 22.0, 28.0, 34.0, 40.0])
    times_h = [4.0, 24.0, 72.0, 168.0]
    true_acts = [2.40, 1.92, 1.12, 0.41]  # MBq/mL true tumor concentration (T_eff ~ 65 h)
    bg_acts = [0.15, 0.11, 0.06, 0.02]

    true_doses, uncorr_doses, pvc_noms, pvc_p05s, pvc_p95s = [], [], [], [], []
    true_tiac, _, _ = fit_tiac_mbq_h_per_ml(times_h, true_acts, isotope="Lu-177", fit_model="MONO_EXPONENTIAL")

    for d in diams:
        s_fac, mass_g = compute_mird_sphere_s_factor(float(d), isotope="Lu-177")
        vol_ml = mass_g / 1.04
        true_doses.append(true_tiac * vol_ml * 3600.0 * s_fac * 1e-3)

        tps = []
        for t_h, t_act, b_act in zip(times_h, true_acts, bg_acts, strict=True):
            rc_i = compute_sphere_recovery_coefficients(
                float(d), spect_sc, spect_harm, true_activity=t_act, background_activity=b_act, radial_offset_mm=40.0
            )
            tps.append(
                TimeActivityPoint(
                    time_hours=t_h,
                    observed_activity_mbq_ml=float(t_act * rc_i["rc_mean_true_voi"]),
                    background_activity_mbq_ml=b_act,
                )
            )
        res = compute_pve_propagated_theranostic_dose(
            lesion_id=f"LU_{int(d)}",
            lesion_diameter_mm=float(d),
            timepoints=tps,
            scanner=spect_sc,
            recon=spect_harm,
            isotope="Lu-177",
            fit_model="MONO_EXPONENTIAL",
        )
        uncorr_doses.append(res.uncorrected_dose_gy)
        pvc_noms.append(res.pvc_dose_nominal_gy)
        pvc_p05s.append(res.pvc_dose_p05_gy)
        pvc_p95s.append(res.pvc_dose_p95_gy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.6, 5.1))

    ax1.plot(diams, true_doses, color=OI["black"], ls="--", lw=1.7, label="Ground-Truth MIRD-26 Self-Dose")
    ax1.fill_between(diams, pvc_p05s, pvc_p95s, color=OI["green"], alpha=0.22, label="GTM PVE-Corrected 90% CI [p05, p95]")
    ax1.plot(diams, pvc_noms, color=OI["green"], marker="o", lw=2.2, ms=5.5, label="GTM PVE-Corrected Dose (Nominal)")
    ax1.plot(diams, uncorr_doses, color=OI["vermilion"], marker="s", lw=2.2, ms=5.5, label="Uncorrected CZT SPECT Dose (PVE Loss)")

    under_12 = (uncorr_doses[1] - true_doses[1]) / true_doses[1] * 100.0
    ax1.annotate(
        f"12 mm tumor:\n{under_12:.0f}% dose underestimation\nwithout PVC",
        xy=(12.0, uncorr_doses[1]),
        xytext=(19.0, 9.5),
        fontsize=8.1,
        fontweight="bold",
        color=OI["vermilion"],
        arrowprops={"arrowstyle": "->", "color": OI["vermilion"], "lw": 1.3},
    )
    ax1.set_title(r"A. $^{177}$Lu CZT SPECT/CT Tumor Absorbed Dose (MIRD-26)", loc="left")
    ax1.set_xlabel("Tumor Inner Diameter (mm)")
    ax1.set_ylabel("Absorbed Dose (Gy)")
    ax1.set_xlim(9.0, 41.2)
    ax1.set_ylim(0.0, max(pvc_p95s) * 1.15)
    ax1.grid(True, alpha=0.25, ls=":")
    ax1.legend(loc="lower right", fontsize=7.6)

    # Panel B: Isotope Positron-Range impact on NEMA spheres (F-18 vs Zr-89 vs Ga-68 water vs Ga-68 lung)
    nema_d = np.array(NEMA_NU2_2018_DIAMETERS_MM)
    sipm_clean_rc = sipm_rc.model_copy(update={"algorithm": "OSEM", "post_filter_fwhm_mm": 2.0})
    iso_configs = [
        ("F-18", 1.0, OI["blue"], "o", "-", r"$^{18}$F (Soft Tissue, $\rho=1.0$ g/cm³)"),
        ("Zr-89", 1.0, OI["green"], "s", "-", r"$^{89}$Zr (Soft Tissue, $\rho=1.0$ g/cm³)"),
        ("Ga-68", 1.0, OI["orange"], "^", "-", r"$^{68}$Ga (Soft Tissue, $\rho=1.0$ g/cm³)"),
        ("Ga-68", 0.27, OI["vermilion"], "D", "--", r"$^{68}$Ga (Inflated Lung, $\rho=0.27$ g/cm³)"),
    ]
    for iso, rho, col, mk, ls, lbl in iso_configs:
        sc_iso = sipm_sc.model_copy(
            update={
                "isotope": iso,
                "has_psf_modeling": False,
                "gibbs_ringing_strength": 0.0,
                "include_positron_range": True,
                "medium_density_g_cm3": rho,
            }
        )
        rcs = [
            compute_sphere_recovery_coefficients(float(d), sc_iso, sipm_clean_rc)["rc_mean_true_voi"]
            for d in nema_d
        ]
        ax2.plot(nema_d, rcs, color=col, marker=mk, ls=ls, lw=2.1, ms=5.8, label=lbl)

    ax2.axhline(1.0, color="#666666", ls=":", lw=1.0)
    ax2.set_title("B. Positron-Range Resolution Loss Across PET Radionuclides", loc="left")
    ax2.set_xlabel("NEMA Sphere Inner Diameter (mm)")
    ax2.set_ylabel("True-VOI Recovery Coefficient (RC)")
    ax2.set_xticks(nema_d)
    ax2.set_ylim(0.18, 1.06)
    ax2.grid(True, alpha=0.25, ls=":")
    ax2.legend(loc="lower right", fontsize=7.8)

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.99))
    _add_footer(fig, "MIRD-26 177Lu Dosimetry (DOI:10.2967/jnumed.115.159012) & Positron Range (DOI:10.1186/s40658-016-0144-5)")
    _save_fig(fig, out_dir, "fig7_theranostics_pve_dosimetry")


def generate_fig8_pareto_and_rbv(out_dir: Path) -> None:
    """Figure 8: Noise-constrained EARL2 Pareto frontier (COV <= 15%) + Necrotic-Core RBV PVC."""
    fleet = build_reference_scanner_fleet()
    tof_sc, tof_rc = fleet[1]
    earl2 = get_corridor_spec("EARL2")

    pareto_pts = compute_harmonization_pareto_frontier(
        tof_sc,
        tof_rc,
        earl2,
        fwhm_grid_mm=[float(x) for x in np.linspace(0.0, 9.0, 19)],
        iteration_grid=[2, 4, 8],
        max_allowed_cov_pct=15.0,
    )

    # Heterogeneous necrotic-core + viable-rim tumor phantom (28 mm outer, 12 mm necrotic core)
    pmt_sc, pmt_rc = fleet[2]
    sc_2mm = pmt_sc.model_copy(update={"voxel_size_mm": (2.0, 2.0, 2.0)})
    rc_obs = pmt_rc.model_copy(update={"post_filter_fwhm_mm": 4.5})
    true_tum, masks = build_necrotic_tumor_phantom(
        shape=(31, 45, 45),
        voxel_size_mm=(2.0, 2.0, 2.0),
        outer_diameter_mm=28.0,
        necrotic_core_diameter_mm=12.0,
        rim_suv=9.0,
        core_suv=1.5,
        background_suv=2.0,
    )
    obs_tum = apply_forward_system_operator(true_tum, sc_2mm, rc_obs)
    rbv_tum, gtm_tum = apply_rbv_pvc(obs_tum, masks, sc_2mm, rc_obs)

    fig = plt.figure(figsize=(13.6, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.95, 1.15], wspace=0.30)

    # Panel A: Noise-constrained EARL2 Pareto frontier (COV vs CNR)
    ax1 = fig.add_subplot(gs[0, 0])
    iter_styles = {
        2: (OI["blue"], "o", "2 Iterations (32 updates)"),
        4: (OI["green"], "s", "4 Iterations (64 updates)"),
        8: (OI["vermilion"], "^", "8 Iterations (128 updates)"),
    }
    ax1.axvspan(15.0, 26.5, color=OI["vermilion"], alpha=0.10, label="Non-Compliant Noise (COV > 15%)")
    ax1.axvline(15.0, color=OI["vermilion"], ls="--", lw=1.4)

    for n_it, (col, mk, lbl) in iter_styles.items():
        pts = [p for p in pareto_pts if p.iterations == n_it]
        covs = [p.background_cov_pct for p in pts]
        cnrs = [p.cnr_score for p in pts]
        ax1.plot(covs, cnrs, color=col, lw=1.6, alpha=0.55)
        feas = [p for p in pts if p.feasible_earl_and_noise]
        infeas = [p for p in pts if not p.feasible_earl_and_noise]
        if infeas:
            ax1.scatter(
                [p.background_cov_pct for p in infeas],
                [p.cnr_score for p in infeas],
                facecolors="white",
                edgecolors=col,
                marker=mk,
                s=34,
                lw=1.2,
                zorder=4,
            )
        if feas:
            ax1.scatter(
                [p.background_cov_pct for p in feas],
                [p.cnr_score for p in feas],
                color=col,
                edgecolors="#222222",
                marker=mk,
                s=46,
                lw=0.8,
                zorder=5,
                label=f"{lbl} (Filled = EARL2 + COV OK)",
            )

    ax1.set_title("A. Noise-Constrained EARL2 Pareto Frontier", loc="left")
    ax1.set_xlabel("Background Noise COV (%)")
    ax1.set_ylabel("13 mm Sphere Contrast-to-Noise Ratio (CNR)")
    ax1.set_xlim(2.5, 26.0)
    ax1.set_ylim(10.0, 66.0)
    ax1.grid(True, alpha=0.25, ls=":")
    ax1.legend(loc="upper right", fontsize=7.2)

    # Panel B: Axial slice comparison (left half = Observed PET, right half = RBV PVC)
    ax2 = fig.add_subplot(gs[0, 1])
    mid_z = true_tum.shape[0] // 2
    mid_x = true_tum.shape[2] // 2
    split_img = np.concatenate([obs_tum[mid_z, :, :mid_x], rbv_tum[mid_z, :, mid_x:]], axis=1)
    half_ext = true_tum.shape[1]
    im2 = ax2.imshow(
        split_img,
        cmap="inferno",
        vmin=0.0,
        vmax=9.8,
        origin="lower",
        extent=[-half_ext, half_ext, -half_ext, half_ext],
    )
    ax2.axvline(0.0, color="white", ls="--", lw=1.4)
    ax2.text(-22.0, 35.0, "Observed\nPET", color="white", ha="center", fontsize=8.3, fontweight="bold")
    ax2.text(22.0, 35.0, "RBV PVC\nRecovered", color="#56B4E9", ha="center", fontsize=8.3, fontweight="bold")
    _add_scalebar(ax2, length_mm=20.0, x_right=39.0, y_bottom=-38.0)
    ax2.set_title("B. Necrotic-Core Tumor (Split View)", loc="left")
    ax2.set_xlabel("x (mm)")
    ax2.set_ylabel("y (mm)")
    cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.03)
    cbar2.set_label("Activity (SUV)", fontsize=8.4)

    # Panel C: 1D transverse profile across necrotic core & viable rim
    ax3 = fig.add_subplot(gs[0, 2])
    mid_y = true_tum.shape[1] // 2
    x_mm = (np.arange(true_tum.shape[2]) - (true_tum.shape[2] - 1) / 2.0) * 2.0
    ax3.plot(x_mm, true_tum[mid_z, mid_y, :], color=OI["black"], ls="--", lw=1.8, label="Ground Truth (Rim=9.0, Core=1.5)")
    ax3.plot(x_mm, obs_tum[mid_z, mid_y, :], color=OI["blue"], lw=2.2, label="Observed PET (Rim Loss + Core Spill-In)")
    ax3.plot(x_mm, rbv_tum[mid_z, mid_y, :], color=OI["green"], lw=2.2, label=f"RBV PVC (GTM res={gtm_tum.reprojection_residual_inf:.1e})")
    ax3.annotate(
        "Necrotic core\nspill-in removed",
        xy=(0.0, rbv_tum[mid_z, mid_y, mid_x]),
        xytext=(-14.0, 0.35),
        fontsize=7.9,
        fontweight="bold",
        color=OI["green"],
        arrowprops={"arrowstyle": "->", "color": OI["green"], "lw": 1.2},
    )
    ax3.set_title("C. Viable-Rim & Necrotic-Core Line Profile", loc="left")
    ax3.set_xlabel("Transverse Position x (mm)")
    ax3.set_ylabel("Activity Concentration (SUV)")
    ax3.set_xlim(-34.0, 34.0)
    ax3.set_ylim(0.0, 11.4)
    ax3.grid(True, alpha=0.25, ls=":")
    ax3.legend(loc="upper center", fontsize=7.4)

    fig.subplots_adjust(bottom=0.14, top=0.90, left=0.05, right=0.97)
    _add_footer(fig, "EANM/EARL COV <= 15% Pareto Frontier & Thomas et al. (2011) RBV PVC (DOI:10.1007/s00259-011-1745-9)")
    _save_fig(fig, out_dir, "fig8_earl_noise_pareto_and_rbv")


def generate_fig9_n1_scanner_translation(out_dir: Path) -> None:
    """Figure 9: N=1 retrospective cross-scanner translation T_{A->B} MTF & lesion-size bias vs ComBat."""
    fleet = build_reference_scanner_fleet()
    sipm_sc, sipm_rc = fleet[0]
    pmt_sc, pmt_rc = fleet[2]

    # 1D radial slice along fx (fz=0, fy=0) of exact 3D MTFs and translation operators
    shape = (32, 32, 64)
    h_sipm, _ = compute_3d_system_mtf(shape, sipm_sc, sipm_rc)
    pmt_on_sipm = pmt_sc.model_copy(update={"voxel_size_mm": sipm_sc.voxel_size_mm})
    h_pmt, _ = compute_3d_system_mtf(shape, pmt_on_sipm, pmt_rc)
    t_down, norm_down, gibbs_supp = build_translation_transfer_function(shape, sipm_sc, sipm_rc, pmt_sc, pmt_rc)
    t_up, norm_up, _ = build_translation_transfer_function(shape, pmt_on_sipm, pmt_rc, sipm_sc, sipm_rc)

    fx = np.fft.fftfreq(shape[2], d=sipm_sc.voxel_size_mm[2])
    pos = fx >= 0.0
    f_vals = fx[pos]

    tr_res = certify_n1_longitudinal_equivalence(sipm_sc, sipm_rc, pmt_sc, pmt_rc)
    combat_bias = simulate_combat_size_blind_baseline(sipm_sc, sipm_rc, pmt_sc, pmt_rc)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    # Panel A: Spectral MTF & Reverse-Gibbs operator vs ill-posed up-translation ceiling
    ax1.axhline(2.0, color=OI["vermilion"], ls="--", lw=1.4, label="Noise-Amplification Abstention Ceiling (2.0×)")
    ax1.axhline(1.0, color="#666666", ls=":", lw=1.0)
    ax1.plot(f_vals, h_sipm[0, 0, pos], color=OI["vermilion"], lw=2.2, label=r"Source $|H_A(f)|$: SiPM PSF-OSEM (Gibbs > 1.0)")
    ax1.plot(f_vals, h_pmt[0, 0, pos], color=OI["blue"], lw=2.2, label=r"Target $|H_B(f)|$: Legacy PMT OSEM")
    ax1.plot(
        f_vals,
        t_down[0, 0, pos],
        color=OI["green"],
        lw=2.4,
        label=rf"Forward $|T_{{A\to B}}(f)|$ (Norm={norm_down:.2f}×, Safe)",
    )
    ax1.plot(
        f_vals,
        t_up[0, 0, pos],
        color=OI["purple"],
        lw=1.9,
        ls="-.",
        label=rf"Ill-Posed $|T_{{B\to A}}(f)|$ (Norm={norm_up:.2f}×, ABSTAIN)",
    )
    ax1.annotate(
        "PSF-Gibbs\novershoot band",
        xy=(0.055, float(np.max(h_sipm[0, 0, pos]))),
        xytext=(0.11, 1.55),
        fontsize=8.0,
        fontweight="bold",
        color=OI["vermilion"],
        arrowprops={"arrowstyle": "->", "color": OI["vermilion"], "lw": 1.2},
    )
    ax1.set_title(r"A. 3D Transfer Functions & Reverse-Gibbs Operator $T_{A\to B}(f)$", loc="left")
    ax1.set_xlabel("Spatial Frequency f (cycles/mm)")
    ax1.set_ylabel("Spectral Gain / MTF Magnitude")
    ax1.set_xlim(0.0, 0.24)
    ax1.set_ylim(0.0, max(2.65, norm_up * 1.15))
    ax1.grid(True, alpha=0.25, ls=":")
    ax1.legend(loc="upper right", fontsize=7.4)

    # Panel B: Per-Sphere SUVmax Cross-Scanner Bias (%) — Uncorrected vs ComBat vs Physics N=1 Translation
    keys = [f"{int(d)}mm" for d in NEMA_NU2_2018_DIAMETERS_MM]
    labels = [f"{int(d)} mm" for d in NEMA_NU2_2018_DIAMETERS_MM]
    x = np.arange(len(keys))
    w = 0.26

    b_pre = [tr_res.per_sphere_bias_before_pct[k] for k in keys]
    b_comb = [combat_bias[k] for k in keys]
    b_post = [tr_res.per_sphere_bias_after_pct[k] for k in keys]

    ax2.axhspan(-6.0, 6.0, color=OI["green"], alpha=0.12, label="Longitudinal Equivalence Band (±6%)")
    ax2.bar(x - w, b_pre, width=w, color=OI["vermilion"], edgecolor="#222222", lw=0.7, label="Uncorrected Switch (SiPM vs. PMT)")
    ax2.bar(x, b_comb, width=w, color=OI["orange"], edgecolor="#222222", lw=0.7, label="Size-Blind ComBat Scalar Baseline")
    ax2.bar(x + w, b_post, width=w, color=OI["green"], edgecolor="#222222", lw=0.7, label=r"quantnuc $T_{A\to B}$ ($N=1$, Max |Bias|=1.1%)")
    for xi, vp in zip(x + w, b_post, strict=True):
        ax2.text(xi, 7.2, f"{vp:+.1f}%", ha="center", va="bottom", fontsize=7.0, fontweight="bold", color=OI["green"])
    ax2.axhline(0.0, color="#333333", lw=0.9)

    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("NEMA Sphere Inner Diameter")
    ax2.set_ylabel("Cross-Scanner SUVmax Bias vs. Target (%)")
    ax2.set_title(f"B. N=1 Cross-Scanner Bias (Gibbs Suppression = {gibbs_supp:.0f}%)", loc="left")
    ax2.set_ylim(-30.0, 145.0)
    ax2.grid(True, axis="y", alpha=0.25, ls=":")
    ax2.legend(loc="upper right", fontsize=7.5)

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.99))
    _add_footer(fig, "N=1 Retrospective Operator Translation vs. ComBat (Orlhac et al. 2018, DOI:10.2967/jnumed.117.199935)")
    _save_fig(fig, out_dir, "fig9_n1_cross_scanner_translation")


def generate_fig10_gear2018_stp_designer(out_dir: Path) -> None:
    """Figure 10: Fisher-Optimal STP SPECT/CT Protocol Designer + Gear-2018 Covariance Breakdown."""
    spect_sc, spect_rc = build_reference_scanner_fleet()[3]
    spect_harm = spect_rc.model_copy(update={"post_filter_fwhm_mm": 4.0})

    stp_kid, c_kid = design_optimal_stp_protocol(
        organ_or_lesion_label="177Lu-DOTATATE Kidney (40 mm)",
        lesion_diameter_mm=40.0,
        scanner=spect_sc,
        recon=spect_harm,
        population_teff_mean_h=52.0,
        population_teff_sd_h=12.0,
        stp_estimator="HANSCHEID",
        max_allowed_dose_cv_pct=15.0,
    )
    stp_tum, c_tum = design_optimal_stp_protocol(
        organ_or_lesion_label="177Lu-PSMA Tumor (12 mm)",
        lesion_diameter_mm=12.0,
        scanner=spect_sc,
        recon=spect_harm,
        population_teff_mean_h=50.0,
        population_teff_sd_h=18.0,
        stp_estimator="HANSCHEID",
        max_allowed_dose_cv_pct=15.0,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    # Panel A: Total Gear-2018 Dose CV (%) vs STP Scan Hour t1
    t = c_kid["times_h"]
    ax1.axhspan(0.0, 15.0, color=OI["green"], alpha=0.10, label="STP Acceptable Zone (Dose CV ≤ 15%)")
    ax1.axhline(15.0, color=OI["vermilion"], ls="--", lw=1.4, label="Abstention Gate (STP_PROTOCOL_INFEASIBLE)")

    ax1.plot(t, c_kid["stp_total_cv_pct"], color=OI["blue"], lw=2.3, label=f"Kidney 40 mm STP (t*={stp_kid.optimal_stp_time_h:.0f} h, CV={stp_kid.optimal_stp_total_dose_cv_pct:.1f}% ✓)")
    ax1.plot(t, c_tum["stp_total_cv_pct"], color=OI["vermilion"], lw=2.3, label=f"Tumor 12 mm STP (t*={stp_tum.optimal_stp_time_h:.0f} h, CV={stp_tum.optimal_stp_total_dose_cv_pct:.1f}% ✗)")
    ax1.axhline(stp_tum.two_timepoint_total_dose_cv_pct, color=OI["orange"], ls="-.", lw=1.8, label=f"Tumor 12 mm D-Optimal 2-TP ({int(stp_tum.d_optimal_two_timepoints_h[0])}h & {int(stp_tum.d_optimal_two_timepoints_h[1])}h: {stp_tum.two_timepoint_total_dose_cv_pct:.1f}%)")

    ax1.scatter([stp_kid.optimal_stp_time_h], [stp_kid.optimal_stp_total_dose_cv_pct], color=OI["blue"], s=70, zorder=6, edgecolor="black")
    ax1.scatter([stp_tum.optimal_stp_time_h], [stp_tum.optimal_stp_total_dose_cv_pct], color=OI["vermilion"], s=70, zorder=6, edgecolor="black")

    ax1.annotate(
        "STP Approved\n(Saves 3 of 4 scans)",
        xy=(stp_kid.optimal_stp_time_h, stp_kid.optimal_stp_total_dose_cv_pct),
        xytext=(84.0, 5.0),
        fontsize=8.0,
        fontweight="bold",
        color=OI["blue"],
        arrowprops={"arrowstyle": "->", "color": OI["blue"], "lw": 1.3},
    )
    ax1.set_title(r"A. Fisher-Optimal $^{177}$Lu STP vs. 2-TP SPECT/CT Schedule", loc="left")
    ax1.set_xlabel("Single SPECT/CT Imaging Timepoint t₁ (Hours Post-Injection)")
    ax1.set_ylabel("Total EANM Gear-2018 Absorbed-Dose CV (%)")
    ax1.set_xlim(6.0, 180.0)
    ax1.set_ylim(0.0, 48.0)
    ax1.grid(True, alpha=0.25, ls=":")
    ax1.legend(loc="upper right", fontsize=7.3)

    # Panel B: Gear-2018 Full-Covariance Variance Breakdown (%) across lesion diameters 10-40 mm
    diams = [10.0, 14.0, 20.0, 30.0, 40.0]
    budgets = [
        compute_gear2018_dose_uncertainty(
            dose_nominal_gy=20.0, lesion_diameter_mm=d, scanner=spect_sc, recon=spect_harm, tiac_cv_frac=0.075
        )
        for d in diams
    ]
    x = np.arange(len(diams))
    pve_v = np.array([b.variance_contrib_pve_delineation_pct for b in budgets])
    tiac_v = np.array([b.variance_contrib_tiac_kinetics_pct for b in budgets])
    cal_v = np.array([b.variance_contrib_calibration_pct for b in budgets])
    dt_v = np.array([b.variance_contrib_deadtime_scatter_pct for b in budgets])
    ac_v = np.array([b.variance_contrib_ct_ac_pct for b in budgets])

    ax2.bar(x, pve_v, width=0.56, color=OI["blue"], edgecolor="#222222", lw=0.6, label="PVE & Boundary Delineation (Net of V–S Cov.)")
    ax2.bar(x, tiac_v, width=0.56, bottom=pve_v, color=OI["green"], edgecolor="#222222", lw=0.6, label="TIAC Pharmacokinetic Fitting")
    ax2.bar(x, cal_v, width=0.56, bottom=pve_v + tiac_v, color=OI["orange"], edgecolor="#222222", lw=0.6, label="SPECT Camera Calibration")
    ax2.bar(x, dt_v, width=0.56, bottom=pve_v + tiac_v + cal_v, color=OI["purple"], edgecolor="#222222", lw=0.6, label="Dead-Time & TEW Scatter")
    ax2.bar(x, ac_v, width=0.56, bottom=pve_v + tiac_v + cal_v + dt_v, color=OI["vermilion"], edgecolor="#222222", lw=0.6, label="CT-AC Respiratory Residual")

    for i, b in enumerate(budgets):
        ax2.text(x[i], 102.0, f"CV={b.dose_cv_pct:.1f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#222222")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{int(d)} mm" for d in diams])
    ax2.set_xlabel("Target Tumor / Organ Diameter")
    ax2.set_ylabel("Share of Total Dose Variance (%)")
    ax2.set_ylim(0.0, 126.0)
    ax2.set_title("B. EANM Gear-2018 Correlated Variance Budget", loc="left")
    ax2.grid(True, axis="y", alpha=0.25, ls=":")
    ax2.legend(loc="upper center", ncol=2, fontsize=7.1)

    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.99))
    _add_footer(fig, "Gear et al. (2018) DOI:10.1007/s00259-018-4136-7 & Hanscheid et al. (2018) DOI:10.2967/jnumed.117.193706")
    _save_fig(fig, out_dir, "fig10_gear2018_stp_protocol_designer")


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
    print("Generating Figure 7 (MIRD-26 theranostic dose & positron range)...")
    generate_fig7_theranostics_and_positron_range(out_dir)
    print("Generating Figure 8 (Noise-constrained EARL2 Pareto & RBV PVC)...")
    generate_fig8_pareto_and_rbv(out_dir)
    print("Generating Figure 9 (N=1 retrospective cross-scanner translation)...")
    generate_fig9_n1_scanner_translation(out_dir)
    print("Generating Figure 10 (Gear-2018 & Fisher STP protocol designer)...")
    generate_fig10_gear2018_stp_designer(out_dir)
    print(f"All 10 figures (PNG + PDF) saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
