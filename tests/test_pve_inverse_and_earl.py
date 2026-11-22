"""Tests for Rousset GTM exact inversion, Muller-Gartner PVC, Bayesian PVC, and EARL/MIRD harmonization."""

from __future__ import annotations

import numpy as np
import pytest

from quantnuc_inverse.earl_harmonizer import (
    build_reference_scanner_fleet,
    get_corridor_spec,
    synthesize_harmonization_filter,
)
from quantnuc_inverse.forward_image_formation import (
    apply_forward_system_operator,
    build_nema_iec_phantom_dro,
    compute_sphere_recovery_coefficients,
)
from quantnuc_inverse.ir import AbstentionCode, AbstentionError, LesionROI, ReconConfig, ScannerSpec
from quantnuc_inverse.pve_inverse_solver import (
    apply_muller_gartner_pvc,
    invert_lesion_activity,
    solve_gtm_multi_region,
)


def test_gtm_exact_multi_region_reprojection_invariant_and_conservation() -> None:
    fleet = build_reference_scanner_fleet()
    sipm_sc, sipm_rc = fleet[0]
    rc_harm = sipm_rc.model_copy(update={"post_filter_fwhm_mm": 5.5})

    dro = build_nema_iec_phantom_dro(
        voxel_size_mm=(2.5, 2.5, 2.5),
        sphere_to_bg_ratio=4.0,
        background_activity_bq_ml=2.0,
    )
    obs_vol = apply_forward_system_operator(dro.true_volume, sipm_sc, rc_harm)

    masks = {**dro.sphere_masks, "background": dro.background_mask}
    outside = np.clip(1.0 - sum(masks.values()), 0.0, 1.0)
    masks["outside_air"] = outside

    gtm = solve_gtm_multi_region(obs_vol, masks, sipm_sc, rc_harm)
    # Formal invariant: forward re-projection residual <= 1e-6
    assert gtm.reprojection_residual_inf <= 1e-6
    # Formal invariant: total activity conservation on closed partition
    assert gtm.total_activity_conservation_error <= 1e-6

    # Verify all 6 spheres recover true activity (8.0) accurately (noise-free DRO)
    rec_map = dict(zip(gtm.region_names, gtm.recovered_means, strict=True))
    for k in ("10mm", "13mm", "17mm", "22mm", "28mm", "37mm"):
        assert abs(rec_map[k] - 8.0) < 0.05


def test_muller_gartner_voxelwise_pvc_recovers_sphere_activity() -> None:
    fleet = build_reference_scanner_fleet()
    pmt_sc, pmt_rc = fleet[2]
    sc = pmt_sc.model_copy(update={"voxel_size_mm": (2.0, 2.0, 2.0)})

    dro = build_nema_iec_phantom_dro(
        voxel_size_mm=(2.0, 2.0, 2.0),
        sphere_to_bg_ratio=4.0,
        background_activity_bq_ml=2.0,
    )
    obs = apply_forward_system_operator(dro.true_volume, sc, pmt_rc)
    tgt_mask = dro.sphere_masks["17mm"]
    bg_mask = 1.0 - tgt_mask
    # Restrict test to a single isolated sphere true volume for pure MG verification
    true_single = 2.0 + 6.0 * tgt_mask
    obs_single = apply_forward_system_operator(true_single, sc, pmt_rc)
    mg_vol = apply_muller_gartner_pvc(obs_single, tgt_mask, bg_mask, 2.0, sc, pmt_rc)
    core = tgt_mask >= 0.8
    assert abs(float(np.mean(mg_vol[core])) - 8.0) < 0.05
    assert obs.shape == mg_vol.shape


def test_lesion_inversion_methods_uncertainty_intervals_and_nyquist_abstention() -> None:
    fleet = build_reference_scanner_fleet()
    sc, rc = fleet[1]
    rc_harm = rc.model_copy(update={"post_filter_fwhm_mm": 5.0})

    true_suv = 9.0
    bg_suv = 2.0
    rci = compute_sphere_recovery_coefficients(12.0, sc, rc_harm, true_activity=true_suv, background_activity=bg_suv, radial_offset_mm=0.0)
    obs_mean = true_suv * rci["rc_mean_true_voi"]

    roi = LesionROI(lesion_id="L12", diameter_mm=12.0, observed_suv_mean=obs_mean, background_suv=bg_suv)

    res_gtm = invert_lesion_activity(roi, sc, rc_harm, method="GTM_EXACT")
    assert res_gtm.forward_reprojection_residual <= 1e-6
    assert abs(res_gtm.recovered_activity_nominal - true_suv) < 0.05
    assert res_gtm.recovered_activity_p05 < res_gtm.recovered_activity_nominal < res_gtm.recovered_activity_p95

    res_mg = invert_lesion_activity(roi, sc, rc_harm, method="MULLER_GARTNER")
    assert res_mg.forward_reprojection_residual <= 1e-6
    assert abs(res_mg.recovered_activity_nominal - true_suv) < 0.05

    res_bay = invert_lesion_activity(roi, sc, rc_harm, method="BAYESIAN_REGULARIZED")
    assert res_bay.recovered_activity_p05 < res_bay.recovered_activity_nominal < res_bay.recovered_activity_p95
    assert res_bay.recovered_activity_nominal <= res_gtm.recovered_activity_nominal + 1e-6

    # Hard abstention on sub-Nyquist lesion (2.5 mm on 2.0 mm voxels / 5+ mm FWHM)
    tiny_roi = LesionROI(lesion_id="TINY", diameter_mm=2.5, observed_suv_mean=2.6, background_suv=2.0)
    with pytest.raises(AbstentionError) as exc_info:
        invert_lesion_activity(tiny_roi, sc, rc_harm, enforce_nyquist_gate=True)
    assert exc_info.value.code == AbstentionCode.SUB_NYQUIST_LESION_UNRECOVERABLE


def test_earl_fleet_harmonization_all_four_scanners_feasible() -> None:
    fleet = build_reference_scanner_fleet()
    earl2 = get_corridor_spec("EARL2")
    mird23 = get_corridor_spec("MIRD23_SPECT")

    for sc, rc in fleet:
        corr = mird23 if sc.modality.startswith("SPECT") else earl2
        res = synthesize_harmonization_filter(sc, rc, corr, raise_on_infeasible=True)
        assert res.feasible is True
        assert res.max_corridor_violation_after <= 1e-4
        assert res.max_corridor_violation_after <= res.max_corridor_violation_before + 1e-9
        assert 0.0 < res.sharpness_retention_ratio <= 1.0


def test_earl1_qiba_and_butterworth_harmonization() -> None:
    fleet = build_reference_scanner_fleet()
    sipm_sc, sipm_rc = fleet[0]

    for std_name in ("EARL1", "QIBA_FDG_PET"):
        corr = get_corridor_spec(std_name)  # type: ignore[arg-type]
        res = synthesize_harmonization_filter(sipm_sc, sipm_rc, corr, filter_type="gaussian")
        assert res.feasible is True
        assert res.max_corridor_violation_after <= 1e-4

    # Butterworth filter synthesis
    earl2 = get_corridor_spec("EARL2")
    res_bw = synthesize_harmonization_filter(sipm_sc, sipm_rc, earl2, filter_type="butterworth")
    assert res_bw.feasible is True
    assert res_bw.filter_type == "butterworth"
    assert res_bw.butterworth_cutoff_cycles_per_mm is not None and res_bw.butterworth_cutoff_cycles_per_mm > 0.0


def test_earl_corridor_infeasible_raises_abstention_error() -> None:
    # Ultra-coarse legacy scanner (11 mm FWHM) cannot meet high-resolution EARL2 lower limits
    coarse_sc = ScannerSpec(
        scanner_id="Coarse_Legacy_NaI",
        vendor_model="Coarse 11mm Scanner",
        modality="SPECT_NAI",
        psf_fwhm_xy_mm=11.0,
        psf_fwhm_z_mm=11.0,
        voxel_size_mm=(3.0, 3.0, 3.0),
        has_psf_modeling=False,
        gibbs_ringing_strength=0.0,
    )
    coarse_rc = ReconConfig(algorithm="OSEM", iterations=3, subsets=8, post_filter_fwhm_mm=0.0)
    earl2 = get_corridor_spec("EARL2")

    with pytest.raises(AbstentionError) as exc_info:
        synthesize_harmonization_filter(coarse_sc, coarse_rc, earl2, raise_on_infeasible=True)
    assert exc_info.value.code == AbstentionCode.EARL_CORRIDOR_INFEASIBLE
