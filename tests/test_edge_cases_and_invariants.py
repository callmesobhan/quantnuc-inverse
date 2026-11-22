"""Edge-case, physical-invariant, abstention-gate, and end-to-end sample-staging tests."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pydicom
import pytest

from quantnuc_inverse.dicom_io import (
    build_synthetic_nucmed_dicom,
    compute_janmahasatian_lbm_kg,
    load_quantitative_nucmed_dicom,
)
from quantnuc_inverse.earl_harmonizer import (
    build_reference_scanner_fleet,
    get_corridor_spec,
    synthesize_harmonization_filter,
)
from quantnuc_inverse.forward_image_formation import (
    apply_forward_system_operator,
    compute_ct_ac_mismatch_report,
    effective_gibbs_strength,
    effective_psf_fwhm_mm,
)
from quantnuc_inverse.harness.agent_orchestrator import QuantitativeImagingForensicsAgent
from quantnuc_inverse.harness.mcp_server import handle_jsonrpc_request
from quantnuc_inverse.ir import (
    AbstentionCode,
    AbstentionError,
    EARLCorridorSpec,
    LesionROI,
    ReconConfig,
)
from quantnuc_inverse.provenance import compute_sha256
from quantnuc_inverse.pve_inverse_solver import (
    apply_muller_gartner_pvc,
    invert_lesion_activity,
    solve_gtm_multi_region,
)


def test_bsrem_beta_damps_gibbs_and_broadens_psf() -> None:
    sc = build_reference_scanner_fleet()[0][0]
    rc_low_beta = ReconConfig(algorithm="BSREM", iterations=4, subsets=16, post_filter_fwhm_mm=0.0, bsrem_beta=100.0)
    rc_high_beta = ReconConfig(algorithm="BSREM", iterations=4, subsets=16, post_filter_fwhm_mm=0.0, bsrem_beta=900.0)

    g_low = effective_gibbs_strength(sc, rc_low_beta)
    g_high = effective_gibbs_strength(sc, rc_high_beta)
    assert g_high < g_low

    f_low, _ = effective_psf_fwhm_mm(sc, rc_low_beta)
    f_high, _ = effective_psf_fwhm_mm(sc, rc_high_beta)
    assert f_high > f_low


def test_ct_ac_mismatch_monotonic_across_photon_energies() -> None:
    # Lower photon energy -> larger mu_liver - mu_lung -> larger attenuation bias at same 10 mm shift
    energies = [140.0, 208.0, 364.0, 511.0]
    abs_errs = [
        abs(compute_ct_ac_mismatch_report(10.0, photon_energy_kev=e).apparent_suv_percent_error)
        for e in energies
    ]
    assert all(abs_errs[i] > abs_errs[i + 1] for i in range(len(abs_errs) - 1))


def test_gtm_tikhonov_regularization_and_invalid_method_guard() -> None:
    sc, rc = build_reference_scanner_fleet()[2]
    sc2 = sc.model_copy(update={"voxel_size_mm": (2.0, 2.0, 2.0)})
    shape = (17, 17, 17)
    r1 = np.zeros(shape, dtype=np.float64)
    r1[7:10, 7:10, 7:10] = 1.0
    r2 = 1.0 - r1
    true_vol = 2.0 * r2 + 8.0 * r1
    obs_vol = apply_forward_system_operator(true_vol, sc2, rc)

    out_exact = solve_gtm_multi_region(obs_vol, {"lesion": r1, "bg": r2}, sc2, rc, regularization_lambda=0.0)
    out_reg = solve_gtm_multi_region(obs_vol, {"lesion": r1, "bg": r2}, sc2, rc, regularization_lambda=0.05)
    assert out_exact.reprojection_residual_inf <= 1e-6
    assert out_reg.recovered_means[0] < out_exact.recovered_means[0]

    roi = LesionROI(lesion_id="BAD", diameter_mm=15.0, observed_suv_mean=5.0, background_suv=2.0)
    with pytest.raises(ValueError, match="Unsupported PVC method"):
        invert_lesion_activity(roi, sc2, rc, method="UNKNOWN")


def test_muller_gartner_masks_out_low_support_background() -> None:
    sc, rc = build_reference_scanner_fleet()[2]
    sc2 = sc.model_copy(update={"voxel_size_mm": (2.0, 2.0, 2.0)})
    shape = (21, 21, 21)
    tgt = np.zeros(shape, dtype=np.float64)
    tgt[7:14, 7:14, 7:14] = 1.0  # 14 mm cube so center support well exceeds 0.35
    bg = 1.0 - tgt
    obs = apply_forward_system_operator(2.0 * bg + 6.0 * tgt, sc2, rc)
    mg = apply_muller_gartner_pvc(obs, tgt, bg, 2.0, sc2, rc, support_threshold=0.35)
    # Far corner voxels have support < 0.35 and must be zeroed
    assert mg[0, 0, 0] == 0.0
    assert abs(mg[10, 10, 10] - 6.0) < 0.15


def test_corridor_spec_rejects_inverted_bounds_and_bad_peak() -> None:
    with pytest.raises(ValueError, match="rc_mean_lower"):
        EARLCorridorSpec(
            standard_name="EARL1",
            sphere_diameters_mm=[10.0],
            rc_mean_lower=[0.7],
            rc_mean_upper=[0.4],
            rc_max_lower=[0.5],
            rc_max_upper=[0.9],
            reference_citation="test",
        )
    with pytest.raises(ValueError, match="rc_max_lower"):
        EARLCorridorSpec(
            standard_name="EARL1",
            sphere_diameters_mm=[10.0],
            rc_mean_lower=[0.3],
            rc_mean_upper=[0.5],
            rc_max_lower=[0.95],
            rc_max_upper=[0.5],
            reference_citation="test",
        )
    with pytest.raises(ValueError, match="rc_peak_lower"):
        EARLCorridorSpec(
            standard_name="EARL2",
            sphere_diameters_mm=[10.0, 13.0],
            rc_mean_lower=[0.3, 0.5],
            rc_mean_upper=[0.6, 0.8],
            rc_max_lower=[0.4, 0.7],
            rc_max_upper=[0.8, 1.1],
            rc_peak_lower=[0.3],
            rc_peak_upper=[0.6, 0.8],
            reference_citation="test",
        )


def test_compute_sha256_handles_numpy_nan_inf_and_bools() -> None:
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    digest = compute_sha256({"arr": arr, "flag": True, "nan": float("nan"), "inf": float("inf")})
    assert len(digest) == 64
    assert digest == compute_sha256({"inf": float("inf"), "flag": True, "nan": float("nan"), "arr": arr.copy()})


def test_dicom_female_lbm_and_header_abstention_edge_cases(tmp_path) -> None:
    lbm_m = compute_janmahasatian_lbm_kg(70.0, 1.75, "M")
    lbm_f = compute_janmahasatian_lbm_kg(70.0, 1.75, "F")
    assert 40.0 < lbm_f < lbm_m < 65.0

    # Negative elapsed time (scan before injection) raises UNCALIBRATED_SUV_METADATA
    p_neg = build_synthetic_nucmed_dicom(
        np.full((4, 4), 5000.0),
        tmp_path / "neg_time.dcm",
        injection_time_hhmmss="110000",
        scan_time_hhmmss="100000",
    )
    with pytest.raises(AbstentionError) as exc_neg:
        load_quantitative_nucmed_dicom(p_neg)
    assert exc_neg.value.code == AbstentionCode.UNCALIBRATED_SUV_METADATA

    # Wrong modality (CT loaded as PET/SPECT) raises UNCALIBRATED_SUV_METADATA
    ds = pydicom.dcmread(str(p_neg))
    ds.Modality = "MR"
    p_mr = tmp_path / "wrong_mod.dcm"
    ds.save_as(str(p_mr))
    with pytest.raises(AbstentionError) as exc_mr:
        load_quantitative_nucmed_dicom(p_mr)
    assert exc_mr.value.code == AbstentionCode.UNCALIBRATED_SUV_METADATA

    # Missing RadiopharmaceuticalInformationSequence
    ds2 = pydicom.dcmread(str(p_neg))
    del ds2.RadiopharmaceuticalInformationSequence
    p_norad = tmp_path / "no_rph.dcm"
    ds2.save_as(str(p_norad))
    with pytest.raises(AbstentionError) as exc_rph:
        load_quantitative_nucmed_dicom(p_norad)
    assert exc_rph.value.code == AbstentionCode.UNCALIBRATED_SUV_METADATA


def test_agent_verdicts_response_stable_and_sub_nyquist_abstention() -> None:
    agent = QuantitativeImagingForensicsAgent()
    sc, rc = build_reference_scanner_fleet()[0]
    harm_rc = rc.model_copy(update={"post_filter_fwhm_mm": 6.4})

    # Biological metabolic response (-45%)
    d_resp = agent.diagnose_longitudinal_suv_shift(
        scenario_name="Metabolic Response",
        baseline_scanner=sc,
        baseline_recon=harm_rc,
        baseline_lesion=LesionROI(lesion_id="R1", diameter_mm=20.0, observed_suv_mean=9.0, observed_suv_max=10.5, background_suv=2.0),
        followup_scanner=sc,
        followup_recon=harm_rc,
        followup_lesion=LesionROI(lesion_id="R2", diameter_mm=20.0, observed_suv_mean=5.0, observed_suv_max=5.8, background_suv=2.0),
    )
    assert d_resp.verdict == "BIOLOGICAL_RESPONSE"

    # Stable disease (+4%)
    d_stab = agent.diagnose_longitudinal_suv_shift(
        scenario_name="Metabolic Stability",
        baseline_scanner=sc,
        baseline_recon=harm_rc,
        baseline_lesion=LesionROI(lesion_id="S1", diameter_mm=20.0, observed_suv_mean=7.0, observed_suv_max=8.0, background_suv=2.0),
        followup_scanner=sc,
        followup_recon=harm_rc,
        followup_lesion=LesionROI(lesion_id="S2", diameter_mm=20.0, observed_suv_mean=7.3, observed_suv_max=8.3, background_suv=2.0),
    )
    assert d_stab.verdict == "STABLE_DISEASE"

    # Sub-Nyquist lesion triggers ABSTAIN_UNQUANTIFIABLE
    d_tiny = agent.diagnose_longitudinal_suv_shift(
        scenario_name="Sub-Nyquist Nodule",
        baseline_scanner=sc,
        baseline_recon=harm_rc,
        baseline_lesion=LesionROI(lesion_id="T1", diameter_mm=2.2, observed_suv_mean=2.8, observed_suv_max=3.1, background_suv=2.0),
        followup_scanner=sc,
        followup_recon=harm_rc,
        followup_lesion=LesionROI(lesion_id="T2", diameter_mm=2.2, observed_suv_mean=3.5, observed_suv_max=4.0, background_suv=2.0),
    )
    assert d_tiny.verdict == "ABSTAIN_UNQUANTIFIABLE"
    assert d_tiny.percist_quantifiable is False


def test_mcp_server_invalid_params_error_code() -> None:
    resp = handle_jsonrpc_request(
        {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "tools/call",
            "params": {"name": "quantnuc_audit_ct_ac_mismatch", "arguments": {"breathing_shift_mm": "not-a-number"}},
        }
    )
    assert resp["error"]["code"] == -32602


def test_earl_bsrem_beta_guidance_present_for_modern_pet() -> None:
    sc, rc = build_reference_scanner_fleet()[0]
    res = synthesize_harmonization_filter(sc, rc, get_corridor_spec("EARL2"))
    assert res.recommended_bsrem_beta is not None and res.recommended_bsrem_beta > 200.0


def test_staged_nema_dro_manifest_hashes_and_dicom_load(tmp_path) -> None:
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parent.parent / "scripts" / "fetch_public_nucmed_sample.py"
    spec = importlib.util.spec_from_file_location("fetch_public_nucmed_sample", script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    staged_dir = mod.stage_public_nucmed_samples(tmp_path / "dro")
    manifest = json.loads((staged_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert len(manifest["files"]) >= 2

    for entry in manifest["files"]:
        fpath = staged_dir / entry["file"]
        assert fpath.exists()
        digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
        assert digest == entry["sha256"]

    pet_vol = load_quantitative_nucmed_dicom(staged_dir / "nema_nu2_2018_dro_pt.dcm")
    assert pet_vol.modality == "PT"
    assert float(np.max(pet_vol.suvbw)) > 0.0
    assert (staged_dir / "LICENSE").exists()
    assert (staged_dir / "ATTRIBUTION.md").exists()


def test_forward_operator_zero_gibbs_when_psf_disabled() -> None:
    pmt_sc, pmt_rc = build_reference_scanner_fleet()[2]
    assert effective_gibbs_strength(pmt_sc, pmt_rc) == 0.0
