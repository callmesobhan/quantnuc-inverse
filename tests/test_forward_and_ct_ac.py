"""Tests for IR validation, provenance ledger, forward image formation, and CT-AC mismatch forensics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from quantnuc_inverse.earl_harmonizer import build_reference_scanner_fleet
from quantnuc_inverse.forward_image_formation import (
    apply_forward_system_operator,
    build_nema_iec_phantom_dro,
    compute_ct_ac_mismatch_report,
    compute_sphere_recovery_coefficients,
    effective_psf_fwhm_mm,
    simulate_ct_ac_dome_field_2d,
    simulate_nema_rc_curve,
)
from quantnuc_inverse.ir import (
    AbstentionCode,
    AbstentionError,
    EARLCorridorSpec,
    ScannerSpec,
)
from quantnuc_inverse.provenance import ProvenanceLedger, build_provenance_record, compute_sha256


def test_ir_validation_rejects_invalid_voxel_and_corridor() -> None:
    with pytest.raises(ValueError, match="voxel_size_mm"):
        ScannerSpec(
            scanner_id="bad",
            vendor_model="bad",
            modality="PET_SIPM",
            psf_fwhm_xy_mm=4.0,
            psf_fwhm_z_mm=4.0,
            voxel_size_mm=(0.0, 2.0, 2.0),
        )

    with pytest.raises(ValueError, match="length must match"):
        EARLCorridorSpec(
            standard_name="EARL2",
            sphere_diameters_mm=[10.0, 13.0],
            rc_mean_lower=[0.4],
            rc_mean_upper=[0.6, 0.8],
            rc_max_lower=[0.5, 0.8],
            rc_max_upper=[0.9, 1.2],
            reference_citation="test",
        )


def test_provenance_ledger_determinism_and_chain_integrity(tmp_path) -> None:
    h1 = compute_sha256({"b": 1.0, "a": [2, 3]})
    h2 = compute_sha256({"a": [2, 3], "b": 1.0})
    assert h1 == h2 and len(h1) == 64

    ledger = ProvenanceLedger()
    r1 = build_provenance_record("op1", {"x": 1}, {"y": 2})
    r2 = build_provenance_record("op2", {"x": 2}, {"y": 4})
    ledger.append(r1)
    ledger.append(r2)
    assert ledger.verify_integrity() is True
    assert len(ledger.records) == 2

    out_json = ledger.write_json(tmp_path / "ledger.json")
    assert out_json.exists()

    # Tampering detection
    ledger._chain_hashes[0] = "0" * 64
    assert ledger.verify_integrity() is False


def test_forward_operator_linearity_and_activity_conservation() -> None:
    fleet = build_reference_scanner_fleet()
    sipm_sc, sipm_rc = fleet[0]
    shape = (25, 25, 25)
    rng = np.random.default_rng(7)
    vol_a = rng.uniform(0.0, 5.0, size=shape)
    vol_b = rng.uniform(0.0, 5.0, size=shape)

    # Test with PSF-Gibbs active + Gaussian post-filter
    rc_gauss = sipm_rc.model_copy(update={"post_filter_fwhm_mm": 3.5, "filter_type": "gaussian"})
    out_ab = apply_forward_system_operator(2.0 * vol_a + 3.0 * vol_b, sipm_sc, rc_gauss)
    out_lin = 2.0 * apply_forward_system_operator(vol_a, sipm_sc, rc_gauss) + 3.0 * apply_forward_system_operator(vol_b, sipm_sc, rc_gauss)
    assert np.max(np.abs(out_ab - out_lin)) < 1e-12

    # Uniform constant field must be preserved exactly even with Gibbs DoG kernel
    const_vol = np.full(shape, 3.75, dtype=np.float64)
    out_const = apply_forward_system_operator(const_vol, sipm_sc, rc_gauss)
    assert np.max(np.abs(out_const - 3.75)) < 1e-12

    # Total activity conservation on interior impulse (zero boundary leakage)
    impulse = np.zeros(shape, dtype=np.float64)
    impulse[12, 12, 12] = 1000.0
    out_imp = apply_forward_system_operator(impulse, sipm_sc, rc_gauss, boundary_mode="wrap")
    assert abs(float(np.sum(out_imp)) - 1000.0) / 1000.0 < 1e-10

    # Butterworth post-filter also preserves total mean
    rc_bw = sipm_rc.model_copy(update={"post_filter_fwhm_mm": 4.5, "filter_type": "butterworth"})
    out_bw = apply_forward_system_operator(impulse, sipm_sc, rc_bw, boundary_mode="wrap")
    assert abs(float(np.sum(out_bw)) - 1000.0) / 1000.0 < 1e-10


def test_radial_psf_degradation_and_gibbs_overshoot_physics() -> None:
    fleet = build_reference_scanner_fleet()
    sipm_sc, sipm_rc = fleet[0]
    pmt_sc, pmt_rc = fleet[2]

    fwhm_center, _ = effective_psf_fwhm_mm(sipm_sc, sipm_rc, radial_offset_mm=0.0)
    fwhm_edge, _ = effective_psf_fwhm_mm(sipm_sc, sipm_rc, radial_offset_mm=150.0)
    assert fwhm_edge > fwhm_center

    # Unharmonized SiPM PSF-OSEM exhibits Gibbs overshoot (RCmax > 1.0) on 13-22 mm spheres
    rc_psf = compute_sphere_recovery_coefficients(17.0, sipm_sc, sipm_rc)
    assert rc_psf["rc_max"] > 1.05
    assert rc_psf["gibbs_overshoot_ratio"] > 0.05

    # Legacy OSEM (no PSF) suffers partial volume underestimation on 10 mm sphere (RCmean < 0.70)
    rc_pmt_10 = compute_sphere_recovery_coefficients(10.0, pmt_sc, pmt_rc)
    assert rc_pmt_10["rc_mean_true_voi"] < 0.70


def test_nema_dro_geometry_and_rc_curve() -> None:
    dro = build_nema_iec_phantom_dro(voxel_size_mm=(2.0, 2.0, 2.0), sphere_to_bg_ratio=4.0, background_activity_bq_ml=5000.0)
    assert len(dro.spheres) == 6
    assert [s.diameter_mm for s in dro.spheres] == [10.0, 13.0, 17.0, 22.0, 28.0, 37.0]
    assert np.isclose(np.max(dro.true_volume), 20000.0, rtol=1e-3)

    fleet = build_reference_scanner_fleet()
    curve = simulate_nema_rc_curve(fleet[2][0], fleet[2][1])
    # Without Gibbs ringing, RCmean increases monotonically with sphere diameter
    vals = [curve["rc_mean_true_voi"][f"{int(d)}mm"] for d in (10, 13, 17, 22, 28, 37)]
    assert all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))


def test_ct_ac_mismatch_exact_exponential_physics_and_abstention() -> None:
    rep_zero = compute_ct_ac_mismatch_report(0.0, photon_energy_kev=511.0)
    assert rep_zero.artifact_polarity == "NEGLIGIBLE"
    assert abs(rep_zero.attenuation_bias_factor - 1.0) < 1e-12

    rep_cold = compute_ct_ac_mismatch_report(10.0, photon_energy_kev=511.0)
    assert rep_cold.artifact_polarity == "FALSE_COLD"
    assert rep_cold.attenuation_bias_factor < 0.80
    assert math.isclose(rep_cold.attenuation_bias_factor, math.exp(rep_cold.delta_mu_integral), rel_tol=1e-12)
    assert rep_cold.exceeds_abstention_threshold is True

    rep_hot = compute_ct_ac_mismatch_report(-10.0, photon_energy_kev=511.0)
    assert rep_hot.artifact_polarity == "FALSE_HOT"
    assert rep_hot.attenuation_bias_factor > 1.25

    # Lower photon energy (208 keV Lu-177 SPECT) has larger delta_mu -> larger attenuation error
    rep_spect = compute_ct_ac_mismatch_report(10.0, photon_energy_kev=208.0)
    assert abs(rep_spect.apparent_suv_percent_error) > abs(rep_cold.apparent_suv_percent_error)

    with pytest.raises(AbstentionError) as exc_info:
        compute_ct_ac_mismatch_report(12.0, photon_energy_kev=511.0, raise_on_severe=True)
    assert exc_info.value.code == AbstentionCode.SEVERE_CT_AC_MISMATCH
    assert exc_info.value.to_dict()["abstained"] is True

    # 2D coronal dome field produces realistic bias map
    field = simulate_ct_ac_dome_field_2d(breathing_shift_mm=10.0, grid_size=(64, 64))
    assert field["pct_error_map"].shape == (64, 64)
    assert float(np.min(field["pct_error_map"])) < -15.0
