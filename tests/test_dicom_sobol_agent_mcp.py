"""Tests for DICOM PET/SPECT/CT I/O, Sobol' sensitivity, Frontier Agent Orchestrator, and MCP Server."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from quantnuc_inverse.dicom_io import (
    build_synthetic_ct_dicom,
    build_synthetic_nucmed_dicom,
    compute_janmahasatian_lbm_kg,
    load_ct_hu_slice,
    load_quantitative_nucmed_dicom,
)
from quantnuc_inverse.earl_harmonizer import build_reference_scanner_fleet
from quantnuc_inverse.harness.agent_orchestrator import QuantitativeImagingForensicsAgent
from quantnuc_inverse.harness.mcp_server import handle_jsonrpc_request
from quantnuc_inverse.ir import AbstentionCode, AbstentionError, LesionROI
from quantnuc_inverse.sobol_sensitivity import run_sobol_sensitivity


def test_pet_dicom_suvbw_suvlbm_roundtrip_accuracy(tmp_path) -> None:
    activity = np.array([[5000.0, 12000.0], [20000.0, 35000.0]], dtype=np.float64)
    dose_bq = 3.0e8
    half_life_s = 6586.2  # F-18
    weight_kg = 75.0
    height_m = 1.78
    dt_s = 3600.0  # 09:00:00 -> 10:00:00

    dcm_path = build_synthetic_nucmed_dicom(
        activity,
        tmp_path / "pet_roundtrip.dcm",
        modality="PT",
        injected_dose_bq=dose_bq,
        injection_time_hhmmss="090000",
        scan_time_hhmmss="100000",
        half_life_s=half_life_s,
        patient_weight_kg=weight_kg,
        patient_size_m=height_m,
        patient_sex="M",
        decay_correction="START",
        units="BQML",
    )
    loaded = load_quantitative_nucmed_dicom(dcm_path)
    expected_decayed_dose = dose_bq * (2.0 ** (-dt_s / half_life_s))
    expected_suvbw = activity * (weight_kg * 1000.0) / expected_decayed_dose
    lbm = compute_janmahasatian_lbm_kg(weight_kg, height_m, "M")
    expected_suvlbm = activity * (lbm * 1000.0) / expected_decayed_dose

    assert math.isclose(loaded.decay_corrected_dose_bq, expected_decayed_dose, rel_tol=1e-9)
    assert np.allclose(loaded.activity_bq_ml, activity, rtol=1e-3, atol=2.0)
    assert np.allclose(loaded.suvbw, expected_suvbw, rtol=1e-3)
    assert np.allclose(loaded.suvlbm, expected_suvlbm, rtol=1e-3)


def test_dicom_admin_decay_and_spect_and_ct_roundtrip(tmp_path) -> None:
    activity = np.full((8, 8), 10000.0, dtype=np.float64)
    p_admin = build_synthetic_nucmed_dicom(
        activity,
        tmp_path / "pet_admin.dcm",
        modality="PT",
        decay_correction="ADMIN",
        injected_dose_bq=2.5e8,
        injection_time_hhmmss="080000",
        scan_time_hhmmss="090000",
    )
    loaded_admin = load_quantitative_nucmed_dicom(p_admin)
    assert math.isclose(loaded_admin.decay_corrected_dose_bq, 2.5e8, rel_tol=1e-9)

    # SPECT NM multiframe + CT HU round-trip
    spect_vol = np.full((4, 8, 8), 8000.0, dtype=np.float64)
    p_nm = build_synthetic_nucmed_dicom(spect_vol, tmp_path / "spect_nm.dcm", modality="NM", half_life_s=574300.8)
    loaded_nm = load_quantitative_nucmed_dicom(p_nm)
    assert loaded_nm.modality == "NM"
    assert loaded_nm.activity_bq_ml.shape == (4, 8, 8)

    hu = np.array([[-1000.0, -720.0], [0.0, 450.0]], dtype=np.float64)
    p_ct = build_synthetic_ct_dicom(hu, tmp_path / "ct_slice.dcm")
    loaded_hu, vox = load_ct_hu_slice(p_ct)
    assert np.allclose(loaded_hu, hu, atol=1.0)
    assert vox == (2.0, 2.0, 2.0)


def test_dicom_abstention_gates_missing_decay_and_uncalibrated_units(tmp_path) -> None:
    arr = np.full((4, 4), 5000.0)

    p_nodecay = build_synthetic_nucmed_dicom(arr, tmp_path / "no_decay.dcm", decay_correction="NONE")
    with pytest.raises(AbstentionError) as exc1:
        load_quantitative_nucmed_dicom(p_nodecay)
    assert exc1.value.code == AbstentionCode.MISSING_DECAY_CORRECTION

    p_cnts = build_synthetic_nucmed_dicom(arr, tmp_path / "cnts.dcm", units="CNTS")
    with pytest.raises(AbstentionError) as exc2:
        load_quantitative_nucmed_dicom(p_cnts)
    assert exc2.value.code == AbstentionCode.UNCALIBRATED_SUV_METADATA

    p_zero_wt = build_synthetic_nucmed_dicom(arr, tmp_path / "zero_wt.dcm", patient_weight_kg=0.0)
    with pytest.raises(AbstentionError) as exc3:
        load_quantitative_nucmed_dicom(p_zero_wt)
    assert exc3.value.code == AbstentionCode.UNCALIBRATED_SUV_METADATA


def test_sobol_sensitivity_bounds_and_lesion_size_regime_shift() -> None:
    s_small = run_sobol_sensitivity(10.0, n_base_samples=512, seed=101, n_bootstrap=60)
    s_large = run_sobol_sensitivity(34.0, n_base_samples=512, seed=101, n_bootstrap=60)

    for res in (s_small, s_large):
        assert res.total_evaluations == 512 * 6
        for k in res.parameter_names:
            assert 0.0 <= res.s1[k] <= res.st[k] <= 1.0
            lo, hi = res.st_ci95[k]
            assert 0.0 <= lo <= hi <= 1.0
        assert sum(res.st.values()) >= 0.90

    # Physical regime shift: PSF + delineation dominate more on 10 mm; CT-AC dominates more on 34 mm
    assert s_small.st["psf_fwhm_mm"] > s_large.st["psf_fwhm_mm"]
    assert s_large.st["ct_ac_shift_mm"] > s_small.st["ct_ac_shift_mm"]

    with pytest.raises(ValueError, match="power of 2"):
        run_sobol_sensitivity(15.0, n_base_samples=100)


def test_agent_orchestrator_dossier_and_critic_verifier_audit(tmp_path) -> None:
    agent = QuantitativeImagingForensicsAgent()
    dossier = agent.build_complete_dossier()

    assert dossier.audit_report is not None
    assert dossier.audit_report.overall_passed is True
    assert dossier.audit_report.ledger_integrity_verified is True
    assert len(dossier.fleet_harmonization) == 4
    assert len(dossier.diagnoses) == 4

    verdicts = [d.verdict for d in dossier.diagnoses]
    assert "TRUE_BIOLOGICAL_PROGRESSION" in verdicts
    assert "CT_AC_RESPIRATORY_ARTIFACT" in verdicts
    assert any(v in {"SCANNER_PSF_GIBBS_ARTIFACT", "PVE_DELINEATION_ARTIFACT"} for v in verdicts)

    md_path, json_path = dossier.write_bundle(tmp_path)
    assert md_path.exists() and json_path.exists()
    md_text = md_path.read_text(encoding="utf-8")
    assert "NOT A MEDICAL DEVICE" in md_text
    assert "PROVENANCE_SHA256_CHAIN" in md_text
    parsed = json.loads(json_path.read_text(encoding="utf-8"))
    assert parsed["provenance_ledger"]["integrity_verified"] is True


def test_mcp_server_jsonrpc_protocol_all_tools_and_abstention() -> None:
    # initialize & ping & tools/list
    init_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert init_resp["result"]["serverInfo"]["name"] == "quantnuc-inverse"

    ping_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "ping"})
    assert ping_resp["result"] == {}

    list_resp = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 3, "method": "tools/list"})
    tools = list_resp["result"]["tools"]
    names = {t["name"] for t in tools}
    assert len(tools) == 11
    assert names == {
        "quantnuc_forward_nema_rc",
        "quantnuc_invert_pve_activity",
        "quantnuc_synthesize_earl_filter",
        "quantnuc_audit_ct_ac_mismatch",
        "quantnuc_diagnose_suv_shift",
        "quantnuc_run_sobol_sensitivity",
        "quantnuc_compute_theranostic_dose",
        "quantnuc_audit_3d_ct_ac_volume",
        "quantnuc_generate_physics_dossier",
        "quantnuc_translate_scanner_n1",
        "quantnuc_design_stp_protocol",
    }

    fleet = build_reference_scanner_fleet()
    sc0, rc0 = fleet[0]
    sc_dict = sc0.model_dump(mode="json")
    rc_dict = rc0.model_dump(mode="json")

    # 1. forward_nema_rc
    r_fwd = handle_jsonrpc_request(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "quantnuc_forward_nema_rc", "arguments": {"scanner": sc_dict, "recon": rc_dict}}}
    )
    assert r_fwd["result"]["isError"] is False
    assert "10mm" in json.loads(r_fwd["result"]["content"][0]["text"])["rc_max"]

    # 2. invert_pve_activity
    lesion_dict = LesionROI(lesion_id="MCP_L1", diameter_mm=15.0, observed_suv_mean=6.0, background_suv=2.0).model_dump(mode="json")
    r_pve = handle_jsonrpc_request(
        {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "quantnuc_invert_pve_activity", "arguments": {"lesion": lesion_dict, "scanner": sc_dict, "recon": rc_dict}}}
    )
    assert r_pve["result"]["isError"] is False
    assert json.loads(r_pve["result"]["content"][0]["text"])["forward_reprojection_residual"] <= 1e-6

    # 3. synthesize_earl_filter
    r_earl = handle_jsonrpc_request(
        {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "quantnuc_synthesize_earl_filter", "arguments": {"scanner": sc_dict, "recon": rc_dict, "standard_name": "EARL2"}}}
    )
    assert r_earl["result"]["isError"] is False
    assert json.loads(r_earl["result"]["content"][0]["text"])["feasible"] is True

    # 4. audit_ct_ac_mismatch (normal + abstention trigger)
    r_ac_ok = handle_jsonrpc_request(
        {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "quantnuc_audit_ct_ac_mismatch", "arguments": {"breathing_shift_mm": 5.0}}}
    )
    assert r_ac_ok["result"]["isError"] is False

    r_ac_abs = handle_jsonrpc_request(
        {"jsonrpc": "2.0", "id": 8, "method": "tools/call", "params": {"name": "quantnuc_audit_ct_ac_mismatch", "arguments": {"breathing_shift_mm": 13.0, "raise_on_severe": True}}}
    )
    assert r_ac_abs["result"]["isError"] is True
    abs_payload = json.loads(r_ac_abs["result"]["content"][0]["text"])
    assert abs_payload["abstained"] is True
    assert abs_payload["code"] == "SEVERE_CT_AC_MISMATCH"

    # 5. diagnose_suv_shift
    r_diag = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "quantnuc_diagnose_suv_shift",
                "arguments": {
                    "baseline_scanner": sc_dict,
                    "baseline_recon": rc_dict,
                    "baseline_lesion": lesion_dict,
                    "followup_scanner": sc_dict,
                    "followup_recon": rc_dict,
                    "followup_lesion": {**lesion_dict, "observed_suv_mean": 8.5},
                },
            },
        }
    )
    assert r_diag["result"]["isError"] is False
    assert json.loads(r_diag["result"]["content"][0]["text"])["verdict"] == "TRUE_BIOLOGICAL_PROGRESSION"

    # 6. sobol_sensitivity
    r_sob = handle_jsonrpc_request(
        {"jsonrpc": "2.0", "id": 10, "method": "tools/call", "params": {"name": "quantnuc_run_sobol_sensitivity", "arguments": {"lesion_diameter_mm": 12.0, "n_base_samples": 128}}}
    )
    assert r_sob["result"]["isError"] is False
    assert "psf_fwhm_mm" in json.loads(r_sob["result"]["content"][0]["text"])["st"]

    # Unknown tool & unknown method return JSON-RPC -32601
    r_unk = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 11, "method": "tools/call", "params": {"name": "no_such_tool"}})
    assert r_unk["error"]["code"] == -32601
    r_meth = handle_jsonrpc_request({"jsonrpc": "2.0", "id": 12, "method": "unknown/method"})
    assert r_meth["error"]["code"] == -32601
