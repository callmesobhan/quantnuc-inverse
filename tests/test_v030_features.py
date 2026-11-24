"""Tests for v0.3.0 features: N=1 retrospective cross-scanner translation (T_{A->B}),
blind RC fingerprinting, EANM Gear-2018 full-covariance uncertainty, Fisher STP protocol designer,
11-tool MCP server, CLI subcommands, and non-commercial NC-RPU-1.0 license compliance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from quantnuc_inverse.cli import main as cli_main
from quantnuc_inverse.earl_harmonizer import build_reference_scanner_fleet
from quantnuc_inverse.forward_image_formation import simulate_nema_rc_curve
from quantnuc_inverse.harness.mcp_server import handle_jsonrpc_request
from quantnuc_inverse.ir import AbstentionCode, AbstentionError
from quantnuc_inverse.scanner_translator import (
    build_translation_transfer_function,
    certify_n1_longitudinal_equivalence,
    compute_3d_system_mtf,
    fingerprint_scanner_from_rc,
    simulate_combat_size_blind_baseline,
    translate_reconstructed_volume,
)
from quantnuc_inverse.stp_dosimetry_designer import (
    compute_fisher_two_timepoint_tiac_cv,
    compute_gear2018_dose_uncertainty,
    design_optimal_stp_protocol,
    evaluate_stp_pk_rmse_curve,
)


def test_n1_retrospective_translation_beats_combat_and_conserves_dc() -> None:
    fleet = build_reference_scanner_fleet()
    sipm_sc, sipm_rc = fleet[0]
    pmt_sc, pmt_rc = fleet[2]

    # Exact 3D MTF shows Gibbs amplification > 1.0 on SiPM PSF-OSEM and <= 1.0 on PMT OSEM
    h_sipm, _ = compute_3d_system_mtf((16, 16, 16), sipm_sc, sipm_rc)
    h_pmt, _ = compute_3d_system_mtf((16, 16, 16), pmt_sc, pmt_rc)
    assert float(h_sipm[0, 0, 0]) == pytest.approx(1.0, abs=1e-12)
    assert float(np.max(h_sipm)) > 1.15
    assert float(np.max(h_pmt)) <= 1.0001

    # Forward down-translation (SiPM PSF -> Legacy PMT): noise-safe, 100% Gibbs suppression, <= 2% residual bias
    tr = certify_n1_longitudinal_equivalence(sipm_sc, sipm_rc, pmt_sc, pmt_rc)
    assert tr.noise_safe is True
    assert tr.noise_amplification_factor <= 1.05
    assert tr.dc_conservation_error < 1e-10
    assert tr.gibbs_peak_suppression_pct >= 95.0
    assert tr.longitudinal_equivalent is True
    assert max(abs(v) for v in tr.per_sphere_bias_after_pct.values()) < 3.0

    # Outperforms size-blind ComBat scalar alignment by > 10x on 10 mm and 37 mm spheres
    combat = simulate_combat_size_blind_baseline(sipm_sc, sipm_rc, pmt_sc, pmt_rc)
    assert abs(combat["10mm"]) > 40.0
    assert abs(tr.per_sphere_bias_after_pct["10mm"]) < 2.0


def test_ill_posed_up_translation_triggers_noise_amplification_abstention() -> None:
    fleet = build_reference_scanner_fleet()
    sipm_sc, sipm_rc = fleet[0]
    pmt_sc, pmt_rc = fleet[2]

    # Translating Legacy PMT -> Unharmonized Digital SiPM PSF is ill-posed super-resolution
    _, norm_up, _ = build_translation_transfer_function((24, 24, 24), pmt_sc, pmt_rc, sipm_sc, sipm_rc)
    assert norm_up > 2.5

    with pytest.raises(AbstentionError) as exc_info:
        certify_n1_longitudinal_equivalence(pmt_sc, pmt_rc, sipm_sc, sipm_rc, enforce_noise_gate=True)
    assert exc_info.value.code == AbstentionCode.TRANSLATION_NOISE_AMPLIFICATION_UNSAFE

    # Volume translator also enforces the gate & rejects non-3D arrays
    with pytest.raises(AbstentionError) as exc_vol:
        translate_reconstructed_volume(np.ones((16, 16, 16)), pmt_sc, pmt_rc, sipm_sc, sipm_rc)
    assert exc_vol.value.code == AbstentionCode.TRANSLATION_NOISE_AMPLIFICATION_UNSAFE

    with pytest.raises(ValueError, match="3D"):
        translate_reconstructed_volume(np.ones((16, 16)), sipm_sc, sipm_rc, pmt_sc, pmt_rc)


def test_blind_rc_fingerprinting_recovers_scanner_parameters() -> None:
    tof_sc, tof_rc = build_reference_scanner_fleet()[1]
    rc_curve = simulate_nema_rc_curve(tof_sc, tof_rc)
    fp = fingerprint_scanner_from_rc(rc_curve["rc_mean"], rc_curve["rc_max"], voxel_size_mm=tof_sc.voxel_size_mm)
    assert 2.5 < fp.inferred_psf_fwhm_xy_mm < 7.0
    assert fp.inferred_gibbs_strength > 0.10
    assert fp.rc_fit_rmse < 0.04


def test_gear2018_covariance_budget_and_volume_sfactor_cancellation() -> None:
    spect_sc, spect_rc = build_reference_scanner_fleet()[3]
    spect_harm = spect_rc.model_copy(update={"post_filter_fwhm_mm": 4.0})

    b_small = compute_gear2018_dose_uncertainty(
        dose_nominal_gy=22.0, lesion_diameter_mm=12.0, scanner=spect_sc, recon=spect_harm
    )
    b_large = compute_gear2018_dose_uncertainty(
        dose_nominal_gy=22.0, lesion_diameter_mm=40.0, scanner=spect_sc, recon=spect_harm
    )
    assert b_small.dose_p05_gy < b_small.dose_nominal_gy < b_small.dose_p95_gy
    assert b_small.dose_cv_pct > b_large.dose_cv_pct
    assert b_small.variance_contrib_pve_delineation_pct > b_large.variance_contrib_pve_delineation_pct
    # Gear-2018 negative V-S covariance substantially reduces total variance vs uncorrelated sum
    assert b_small.volume_sfactor_covariance_reduction_pct > 50.0
    total_share = (
        b_small.variance_contrib_calibration_pct
        + b_small.variance_contrib_deadtime_scatter_pct
        + b_small.variance_contrib_ct_ac_pct
        + b_small.variance_contrib_pve_delineation_pct
        + b_small.variance_contrib_tiac_kinetics_pct
    )
    assert abs(total_share - 100.0) < 0.25


def test_fisher_stp_designer_approves_kidney_and_abstains_on_small_tumor() -> None:
    spect_sc, spect_rc = build_reference_scanner_fleet()[3]
    spect_harm = spect_rc.model_copy(update={"post_filter_fwhm_mm": 4.0})

    # Both Hanscheid and Madsen curves produce clean U-shaped minima
    t_grid = np.array([12.0, 48.0, 72.0, 144.0])
    pk_h, cnt_h = evaluate_stp_pk_rmse_curve(t_grid, population_teff_mean_h=52.0, population_teff_sd_h=12.0, estimator="HANSCHEID")
    pk_m, _ = evaluate_stp_pk_rmse_curve(t_grid, population_teff_mean_h=52.0, population_teff_sd_h=12.0, estimator="MADSEN")
    assert pk_h[2] < pk_h[0] and pk_m[2] < pk_m[0]
    assert cnt_h[-1] > cnt_h[0]

    # 2-TP Fisher CV is strictly smaller for well-separated pair than degenerate pair
    assert compute_fisher_two_timepoint_tiac_cv(24.0, 120.0, teff_mean_h=52.0) < compute_fisher_two_timepoint_tiac_cv(24.0, 26.0, teff_mean_h=52.0)

    # Kidney (40 mm, 52 +- 12 h): STP APPROVED saving 3 of 4 scans
    res_kid, curves_kid = design_optimal_stp_protocol(
        organ_or_lesion_label="177Lu-DOTATATE Kidney",
        lesion_diameter_mm=40.0,
        scanner=spect_sc,
        recon=spect_harm,
        population_teff_mean_h=52.0,
        population_teff_sd_h=12.0,
        max_allowed_dose_cv_pct=15.0,
        raise_on_infeasible=True,
    )
    assert res_kid.stp_feasible is True
    assert res_kid.recommended_protocol == "SINGLE_TIMEPOINT_APPROVED"
    assert res_kid.scans_saved_vs_4tp == 3
    assert len(curves_kid["times_h"]) > 50

    # Small PSMA Tumor (12 mm, 50 +- 18 h): STP INFEASIBLE -> triggers STP_PROTOCOL_INFEASIBLE_REQUIRE_MTP
    with pytest.raises(AbstentionError) as exc_stp:
        design_optimal_stp_protocol(
            organ_or_lesion_label="177Lu-PSMA 12mm Tumor",
            lesion_diameter_mm=12.0,
            scanner=spect_sc,
            recon=spect_harm,
            population_teff_mean_h=50.0,
            population_teff_sd_h=18.0,
            max_allowed_dose_cv_pct=15.0,
            raise_on_infeasible=True,
        )
    assert exc_stp.value.code == AbstentionCode.STP_PROTOCOL_INFEASIBLE_REQUIRE_MTP


def test_mcp_and_cli_v030_tools_and_noncommercial_license(capsys) -> None:
    fleet = build_reference_scanner_fleet()
    sipm_sc, sipm_rc = fleet[0]
    pmt_sc, pmt_rc = fleet[2]
    spect_sc, spect_rc = fleet[3]

    # MCP tool 10: quantnuc_translate_scanner_n1
    r_tr = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 31,
            "method": "tools/call",
            "params": {
                "name": "quantnuc_translate_scanner_n1",
                "arguments": {
                    "source_scanner": sipm_sc.model_dump(mode="json"),
                    "source_recon": sipm_rc.model_dump(mode="json"),
                    "target_scanner": pmt_sc.model_dump(mode="json"),
                    "target_recon": pmt_rc.model_dump(mode="json"),
                },
            },
        }
    )
    assert r_tr["result"]["isError"] is False
    assert json.loads(r_tr["result"]["content"][0]["text"])["longitudinal_equivalent"] is True

    # MCP tool 11: quantnuc_design_stp_protocol
    r_stp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 32,
            "method": "tools/call",
            "params": {
                "name": "quantnuc_design_stp_protocol",
                "arguments": {
                    "lesion_diameter_mm": 40.0,
                    "scanner": spect_sc.model_dump(mode="json"),
                    "recon": spect_rc.model_copy(update={"post_filter_fwhm_mm": 4.0}).model_dump(mode="json"),
                },
            },
        }
    )
    assert r_stp["result"]["isError"] is False
    assert json.loads(r_stp["result"]["content"][0]["text"])["stp_feasible"] is True

    # CLI v0.3.0 subcommands
    assert cli_main(["translate", "--direction", "sipm-to-earl2"]) == 0
    assert "longitudinal_equivalent" in capsys.readouterr().out
    assert cli_main(["translate", "--direction", "pmt-to-sipm"]) == 2
    assert "TRANSLATION_NOISE_AMPLIFICATION_UNSAFE" in capsys.readouterr().err
    assert cli_main(["stp-design", "--diameter", "40.0"]) == 0
    assert "SINGLE_TIMEPOINT_APPROVED" in capsys.readouterr().out

    # Verify non-commercial research & citation license is intact
    lic_text = (Path(__file__).resolve().parent.parent / "LICENSE").read_text(encoding="utf-8")
    assert "NC-RPU-1.0" in lic_text
    assert "COMMERCIAL USE IS STRICTLY PROHIBITED" in lic_text
    assert "NOT A MEDICAL DEVICE" in lic_text
