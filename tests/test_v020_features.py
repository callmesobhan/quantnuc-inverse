"""Tests for v0.2.0 features: positron range, bilinear 3D CT-AC, RBV PVC, noise-constrained EARL,
MIRD-26 theranostics dosimetry, 3D multi-slice DICOM series + CT resampling, CLI, and 9-tool MCP stdio.
"""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest

from quantnuc_inverse.cli import main as cli_main
from quantnuc_inverse.dicom_io import (
    build_synthetic_dicom_series,
    load_ct_dicom_series_3d,
    load_dicom_series_3d,
    resample_ct_to_pet_grid,
)
from quantnuc_inverse.earl_harmonizer import (
    build_reference_scanner_fleet,
    compute_harmonization_pareto_frontier,
    estimate_background_cov_pct,
    get_corridor_spec,
    synthesize_harmonization_filter,
)
from quantnuc_inverse.forward_image_formation import (
    apply_forward_system_operator,
    compute_3d_ct_ac_bias_volume,
    compute_sphere_recovery_coefficients,
    ct_hu_to_linear_attenuation_cm,
    effective_psf_fwhm_mm,
    isotope_positron_range_fwhm_mm,
)
from quantnuc_inverse.harness import mcp_server
from quantnuc_inverse.ir import AbstentionCode, AbstentionError, TimeActivityPoint
from quantnuc_inverse.pve_inverse_solver import apply_rbv_pvc, build_necrotic_tumor_phantom
from quantnuc_inverse.theranostics_dosimetry import (
    compute_mird_sphere_s_factor,
    compute_pve_propagated_theranostic_dose,
    fit_tiac_mbq_h_per_ml,
)


def test_positron_range_broadening_isotopes_and_lung_density() -> None:
    assert isotope_positron_range_fwhm_mm("Lu-177") == 0.0
    r_f18 = isotope_positron_range_fwhm_mm("F-18", 1.0)
    r_zr89 = isotope_positron_range_fwhm_mm("Zr-89", 1.0)
    r_ga68_water = isotope_positron_range_fwhm_mm("Ga-68", 1.0)
    r_ga68_lung = isotope_positron_range_fwhm_mm("Ga-68", 0.27)
    assert 0.0 < r_f18 < r_zr89 < r_ga68_water < r_ga68_lung

    sc_base, rc_base = build_reference_scanner_fleet()[0]
    sc_f18 = sc_base.model_copy(update={"isotope": "F-18", "include_positron_range": True, "medium_density_g_cm3": 1.0})
    sc_ga_lung = sc_base.model_copy(update={"isotope": "Ga-68", "include_positron_range": True, "medium_density_g_cm3": 0.27})
    f18_xy, _ = effective_psf_fwhm_mm(sc_f18, rc_base)
    ga_lung_xy, _ = effective_psf_fwhm_mm(sc_ga_lung, rc_base)
    assert ga_lung_xy > f18_xy + 3.0


def test_bilinear_ct_hu_to_mu_and_3d_volumetric_ct_ac_audit() -> None:
    mus = ct_hu_to_linear_attenuation_cm(np.array([-1000.0, -720.0, 0.0, 1000.0]), photon_energy_kev=511.0)
    assert abs(mus[0] - 0.0) < 1e-9
    assert 0.022 < mus[1] < 0.032  # inflated lung
    assert abs(mus[2] - 0.096) < 1e-6  # water at 511 keV
    assert 0.135 < mus[3] < 0.160  # cortical bone bilinear slope

    # 3D lung-liver slab shifted by +10 mm (inspiration CT -> lung mu on liver -> FALSE_COLD)
    nz, ny, nx = 20, 24, 24
    vox = (2.5, 2.5, 2.5)
    z_mm = (np.arange(nz) - (nz - 1) / 2.0) * vox[0]
    zz = z_mm[:, None, None]
    hu_true = np.broadcast_to(np.where(zz > 0.0, -720.0, 50.0), (nz, ny, nx)).copy()
    hu_ct = np.broadcast_to(np.where(zz > -10.0, -720.0, 50.0), (nz, ny, nx)).copy()
    roi = np.zeros((nz, ny, nx), dtype=np.float64)
    roi[7:11, 8:16, 8:16] = 1.0

    bias_vol, report = compute_3d_ct_ac_bias_volume(
        hu_true, hu_ct, vox, roi_mask=roi, shift_mm=(10.0, 0.0, 0.0), photon_energy_kev=511.0
    )
    assert bias_vol.shape == (nz, ny, nx)
    assert report.artifact_polarity == "FALSE_COLD"
    assert report.roi_mean_percent_error < -15.0

    # Severe shift triggers SEVERE_CT_AC_MISMATCH
    with pytest.raises(AbstentionError) as exc_info:
        compute_3d_ct_ac_bias_volume(
            hu_true, hu_ct, vox, roi_mask=roi, shift_mm=(10.0, 0.0, 0.0), raise_on_severe=True
        )
    assert exc_info.value.code == AbstentionCode.SEVERE_CT_AC_MISMATCH


def test_rbv_pvc_recovers_heterogeneous_necrotic_core_and_viable_rim() -> None:
    sc, rc = build_reference_scanner_fleet()[2]
    sc2 = sc.model_copy(update={"voxel_size_mm": (2.0, 2.0, 2.0)})
    rc_obs = rc.model_copy(update={"post_filter_fwhm_mm": 4.0})

    true_vol, masks = build_necrotic_tumor_phantom(
        shape=(25, 35, 35),
        voxel_size_mm=(2.0, 2.0, 2.0),
        outer_diameter_mm=28.0,
        necrotic_core_diameter_mm=12.0,
        rim_suv=9.0,
        core_suv=1.5,
        background_suv=2.0,
    )
    obs_vol = apply_forward_system_operator(true_vol, sc2, rc_obs)
    rbv_vol, gtm_out = apply_rbv_pvc(obs_vol, masks, sc2, rc_obs)

    assert gtm_out.reprojection_residual_inf <= 1e-6
    core_mask = masks["necrotic_core"] > 0.8
    rim_mask = masks["viable_rim"] > 0.8
    # Uncorrected observed PET suffers rim loss and necrotic core spill-in
    assert float(np.mean(obs_vol[core_mask])) > 2.5
    assert float(np.mean(obs_vol[rim_mask])) < 7.5
    # RBV PVC restores both viable rim (~9.0) and necrotic core (~1.5)
    assert abs(float(np.mean(rbv_vol[rim_mask])) - 9.0) < 0.25
    assert abs(float(np.mean(rbv_vol[core_mask])) - 1.5) < 0.25


def test_noise_constrained_earl_harmonization_and_pareto_frontier() -> None:
    sc, rc = build_reference_scanner_fleet()[1]
    earl2 = get_corridor_spec("EARL2")

    cov_unfiltered = estimate_background_cov_pct(sc, rc.model_copy(update={"iterations": 8, "post_filter_fwhm_mm": 0.0}))
    cov_filtered = estimate_background_cov_pct(sc, rc.model_copy(update={"iterations": 8, "post_filter_fwhm_mm": 6.0}))
    assert cov_unfiltered > 15.0
    assert cov_filtered < 12.0
    assert cov_filtered < cov_unfiltered * 0.5

    res = synthesize_harmonization_filter(
        sc, rc, earl2, max_allowed_cov_pct=15.0, enforce_noise_constraint=True
    )
    assert res.feasible is True
    assert res.noise_compliant is True
    assert res.background_cov_pct <= 15.0
    assert res.cnr_score > 10.0

    pareto = compute_harmonization_pareto_frontier(
        sc, rc, earl2, fwhm_grid_mm=[0.0, 3.0, 6.0, 8.0], iteration_grid=[2, 4]
    )
    assert len(pareto) == 8
    assert any(p.feasible_earl_and_noise for p in pareto)


def test_mird26_theranostics_dosimetry_pve_propagation_and_tiac_abstention() -> None:
    # S-factor decreases as sphere mass grows; Y-90 has stronger self-dose than Lu-177 at 20 mm
    s_lu_12, m12 = compute_mird_sphere_s_factor(12.0, "Lu-177")
    s_lu_30, m30 = compute_mird_sphere_s_factor(30.0, "Lu-177")
    s_y90_20, _ = compute_mird_sphere_s_factor(20.0, "Y-90")
    s_lu_20, _ = compute_mird_sphere_s_factor(20.0, "Lu-177")
    assert s_lu_12 > s_lu_30 and m30 > m12
    assert s_y90_20 > s_lu_20

    # Bi-exponential uptake + clearance fitting
    t_bi = [1.0, 6.0, 24.0, 72.0, 168.0]
    a_bi = [0.8, 2.2, 2.0, 1.1, 0.35]
    tiac_bi, teff_bi, model_bi = fit_tiac_mbq_h_per_ml(t_bi, a_bi, isotope="Lu-177", fit_model="AUTO")
    assert tiac_bi > 100.0 and 10.0 < teff_bi <= 160.0
    assert model_bi in {"BI_EXPONENTIAL", "MONO_EXPONENTIAL"}

    # End-to-end Lu-177 SPECT PVE dose propagation on a 12 mm tumor
    spect_sc, spect_rc = build_reference_scanner_fleet()[3]
    spect_harm = spect_rc.model_copy(update={"post_filter_fwhm_mm": 4.0})
    times = [4.0, 24.0, 72.0, 168.0]
    true_acts = [2.4, 1.9, 1.1, 0.4]
    bg_acts = [0.15, 0.11, 0.06, 0.02]
    tps = []
    for t_h, ta, ba in zip(times, true_acts, bg_acts, strict=True):
        rci = compute_sphere_recovery_coefficients(12.0, spect_sc, spect_harm, true_activity=ta, background_activity=ba, radial_offset_mm=40.0)
        tps.append(TimeActivityPoint(time_hours=t_h, observed_activity_mbq_ml=ta * rci["rc_mean_true_voi"], background_activity_mbq_ml=ba))

    dose_res = compute_pve_propagated_theranostic_dose(
        lesion_id="LU_12",
        lesion_diameter_mm=12.0,
        timepoints=tps,
        scanner=spect_sc,
        recon=spect_harm,
        isotope="Lu-177",
    )
    assert dose_res.pvc_dose_p05_gy < dose_res.pvc_dose_nominal_gy < dose_res.pvc_dose_p95_gy
    assert dose_res.uncorrected_dose_gy < dose_res.pvc_dose_nominal_gy * 0.70
    assert dose_res.pve_dose_underestimation_pct < -30.0

    # Hard abstention when < 2 timepoints or non-monotonic timestamps
    with pytest.raises(AbstentionError) as exc1:
        fit_tiac_mbq_h_per_ml([24.0], [1.5], isotope="Lu-177")
    assert exc1.value.code == AbstentionCode.INSUFFICIENT_TIMEPOINTS_FOR_TIAC

    with pytest.raises(AbstentionError) as exc2:
        fit_tiac_mbq_h_per_ml([24.0, 12.0], [1.5, 1.8], isotope="Lu-177")
    assert exc2.value.code == AbstentionCode.INSUFFICIENT_TIMEPOINTS_FOR_TIAC


def test_3d_multislice_dicom_series_and_ct_to_pet_resampling(tmp_path) -> None:
    pet_vol = np.zeros((6, 12, 12), dtype=np.float64)
    for z in range(6):
        pet_vol[z] = 4000.0 + z * 1500.0

    pet_dir = tmp_path / "pet_series"
    written = build_synthetic_dicom_series(pet_vol, pet_dir, modality="PT", voxel_size_mm=(3.0, 2.0, 2.0))
    assert len(written) == 6

    loaded = load_dicom_series_3d(pet_dir)
    assert loaded.activity_bq_ml.shape == (6, 12, 12)
    assert math.isclose(loaded.voxel_size_mm[0], 3.0, rel_tol=1e-6)
    assert np.allclose(loaded.activity_bq_ml, pet_vol, rtol=1e-3, atol=2.0)

    # Empty directory raises UNCALIBRATED_SUV_METADATA
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(AbstentionError) as exc_empty:
        load_dicom_series_3d(empty_dir)
    assert exc_empty.value.code == AbstentionCode.UNCALIBRATED_SUV_METADATA

    # Paired 3D CT series + resampling to PET grid with shift
    ct_vol = np.full((12, 24, 24), -720.0, dtype=np.float64)
    ct_vol[:6, :, :] = 40.0
    ct_dir = tmp_path / "ct_series"
    build_synthetic_dicom_series(ct_vol, ct_dir, modality="CT", voxel_size_mm=(1.5, 1.0, 1.0))
    loaded_ct, ct_vox = load_ct_dicom_series_3d(ct_dir)
    assert loaded_ct.shape == (12, 24, 24)
    assert math.isclose(ct_vox[0], 1.5, rel_tol=1e-6)

    resampled = resample_ct_to_pet_grid(loaded_ct, ct_vox, (6, 12, 12), (3.0, 2.0, 2.0), shift_mm=(0.0, 0.0, 0.0))
    assert resampled.shape == (6, 12, 12)
    assert float(np.mean(resampled[0])) > 0.0
    assert float(np.mean(resampled[-1])) < -500.0


def test_mcp_new_tools_and_stdio_main_loop(monkeypatch) -> None:
    spect_sc, spect_rc = build_reference_scanner_fleet()[3]
    tps = [
        {"time_hours": 4.0, "observed_activity_mbq_ml": 1.8, "background_activity_mbq_ml": 0.1},
        {"time_hours": 72.0, "observed_activity_mbq_ml": 0.9, "background_activity_mbq_ml": 0.05},
    ]
    r_dose = mcp_server.handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "tools/call",
            "params": {
                "name": "quantnuc_compute_theranostic_dose",
                "arguments": {
                    "lesion_diameter_mm": 15.0,
                    "timepoints": tps,
                    "scanner": spect_sc.model_dump(mode="json"),
                    "recon": spect_rc.model_dump(mode="json"),
                    "isotope": "Lu-177",
                },
            },
        }
    )
    assert r_dose["result"]["isError"] is False
    assert json.loads(r_dose["result"]["content"][0]["text"])["pvc_dose_nominal_gy"] > 0.0

    r_ac3d = mcp_server.handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {"name": "quantnuc_audit_3d_ct_ac_volume", "arguments": {"breathing_shift_mm": 10.0}},
        }
    )
    assert r_ac3d["result"]["isError"] is False
    assert json.loads(r_ac3d["result"]["content"][0]["text"])["artifact_polarity"] == "FALSE_COLD"

    r_dos = mcp_server.handle_jsonrpc_request(
        {"jsonrpc": "2.0", "id": 23, "method": "tools/call", "params": {"name": "quantnuc_generate_physics_dossier", "arguments": {}}}
    )
    assert r_dos["result"]["isError"] is False
    assert json.loads(r_dos["result"]["content"][0]["text"])["audit_report"]["overall_passed"] is True

    # Test live stdio main() loop: blank line, invalid JSON, notification (no id), and valid ping
    stdin_payload = '\n{bad-json\n{"jsonrpc":"2.0","method":"notifications/initialized"}\n{"jsonrpc":"2.0","id":7,"method":"ping"}\n'
    fake_in = io.StringIO(stdin_payload)
    fake_out = io.StringIO()
    monkeypatch.setattr(mcp_server.sys, "stdin", fake_in)
    monkeypatch.setattr(mcp_server.sys, "stdout", fake_out)
    mcp_server.main()
    lines = [json.loads(line) for line in fake_out.getvalue().strip().splitlines()]
    assert len(lines) == 2
    assert lines[0]["error"]["code"] == -32700
    assert lines[1]["id"] == 7 and lines[1]["result"] == {}


def test_cli_all_subcommands_and_abstention_exit_code(tmp_path, capsys) -> None:
    assert cli_main(["audit-ac", "--shift-mm", "8.0"]) == 0
    assert "FALSE_COLD" in capsys.readouterr().out

    assert cli_main(["pvc", "--diameter", "14.0", "--observed-suv", "6.0", "--background-suv", "2.0"]) == 0
    assert "recovered_activity_nominal" in capsys.readouterr().out

    # Sub-Nyquist lesion returns exit code 2 with structured abstention payload on stderr
    assert cli_main(["pvc", "--diameter", "2.0", "--observed-suv", "2.5"]) == 2
    assert "SUB_NYQUIST_LESION_UNRECOVERABLE" in capsys.readouterr().err

    assert cli_main(["dose", "--diameter", "15.0", "--isotope", "Lu-177"]) == 0
    assert "pvc_dose_nominal_gy" in capsys.readouterr().out

    assert cli_main(["harmonize", "--pet-standard", "EARL2", "--noise-constrained"]) == 0
    assert "SiPM_Digital_PET" in capsys.readouterr().out

    assert cli_main(["dossier", "--output-dir", str(tmp_path / "cli_dos")]) == 0
    assert "audit_passed" in capsys.readouterr().out
