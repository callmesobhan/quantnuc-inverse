"""Frontier Agent Harness: Autonomous Quantitative Imaging Forensics Physicist & Critic-Verifier Audit.

Capabilities:
1. Longitudinal delta-SUV hypothesis falsification: separates true biological progression
   from scanner-switch PSF-Gibbs overshoot, PVE diameter/delineation bias, and CT-AC
   respiratory phase mismatch.
2. Multi-vendor 4-scanner fleet harmonization (SiPM PET, TOF-PSF PET, Legacy PMT PET, CZT SPECT)
   into EARL2 / QIBA / MIRD-23 corridors.
3. Independent Critic-Verifier Audit gating PERCIST 1.0 / Deauville / theranostic dosimetry
   quantifiability, emitting an EANM/EARL/QIBA Quantitative Physics Dossier (Markdown + JSON ledger).

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from quantnuc_inverse.earl_harmonizer import (
    CorridorName,
    build_reference_scanner_fleet,
    get_corridor_spec,
    synthesize_harmonization_filter,
)
from quantnuc_inverse.forward_image_formation import (
    compute_ct_ac_mismatch_report,
    compute_sphere_recovery_coefficients,
)
from quantnuc_inverse.ir import (
    RESEARCH_DISCLAIMER,
    AbstentionError,
    HarmonizationFilterResult,
    LesionROI,
    ProvenanceRecord,
    ReconConfig,
    ScannerSpec,
)
from quantnuc_inverse.provenance import ProvenanceLedger, build_provenance_record
from quantnuc_inverse.pve_inverse_solver import invert_lesion_activity

AttributionVerdict = Literal[
    "TRUE_BIOLOGICAL_PROGRESSION",
    "SCANNER_PSF_GIBBS_ARTIFACT",
    "CT_AC_RESPIRATORY_ARTIFACT",
    "PVE_DELINEATION_ARTIFACT",
    "STABLE_DISEASE",
    "BIOLOGICAL_RESPONSE",
    "ABSTAIN_UNQUANTIFIABLE",
]


class LongitudinalSUVDiagnosis(BaseModel):
    """Structured mechanistic decomposition of an observed longitudinal delta-SUV shift."""

    model_config = ConfigDict(frozen=True)

    scenario_name: str
    baseline_observed_suv: float
    followup_observed_suv: float
    observed_delta_suv_pct: float
    # Additive attribution components (in % of baseline observed SUV) summing to observed_delta_suv_pct
    psf_gibbs_artifact_pct: float
    pve_resolution_artifact_pct: float
    ct_ac_respiratory_artifact_pct: float
    true_biological_delta_pct: float
    # Recovered true activity (after removing AC + PVE + Gibbs biases)
    baseline_recovered_suv: float
    followup_recovered_suv: float
    followup_recovered_p05: float
    followup_recovered_p95: float
    verdict: AttributionVerdict
    percist_quantifiable: bool
    falsified_hypotheses: list[str]
    dominant_mechanism: str
    recommendation: str
    provenance: ProvenanceRecord


class CriticAuditCheck(BaseModel):
    """Single independent Critic-Verifier audit invariant check."""

    model_config = ConfigDict(frozen=True)

    check_id: str
    passed: bool
    metric_value: str
    threshold: str
    rationale: str


class CriticAuditReport(BaseModel):
    """Independent Critic-Verifier Audit verdict gating downstream clinical/trial quantifiability."""

    model_config = ConfigDict(frozen=True)

    overall_passed: bool
    quantifiability_gate: Literal["PASS_QUANTIFIABLE", "PASS_WITH_CAUTION", "ABSTAIN_BLOCKED"]
    checks: list[CriticAuditCheck]
    ledger_head_sha256: str
    ledger_integrity_verified: bool
    disclaimer: str = Field(default=RESEARCH_DISCLAIMER)


@dataclass
class QuantitativePhysicsDossier:
    """Complete EANM/EARL/QIBA Quantitative Physics Dossier produced by the Frontier Agent."""

    title: str
    diagnoses: list[LongitudinalSUVDiagnosis] = field(default_factory=list)
    fleet_harmonization: list[HarmonizationFilterResult] = field(default_factory=list)
    audit_report: CriticAuditReport | None = None
    ledger: ProvenanceLedger = field(default_factory=ProvenanceLedger)

    def to_markdown(self) -> str:
        """Render a publication/trial-grade EANM/EARL/QIBA Quantitative Physics Dossier."""
        lines: list[str] = [
            f"# {self.title}",
            "",
            f"> **Notice:** {RESEARCH_DISCLAIMER}.",
            "",
            "## 1. Multi-Vendor Fleet Harmonization Summary (EARL2 / MIRD-23)",
            "",
            "| Scanner ID | Target Standard | Feasible | Optimal Post-Filter (mm) | Rec. Iters | Max Viol. Before | Max Viol. After | Sharpness Retention |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
        for h in self.fleet_harmonization:
            lines.append(
                f"| `{h.scanner_id}` | `{h.standard_name}` | {'YES' if h.feasible else 'NO'} | "
                f"{h.optimal_post_filter_fwhm_mm:.2f} ({h.filter_type}) | {h.recommended_iterations} | "
                f"{h.max_corridor_violation_before:.4f} | {h.max_corridor_violation_after:.4f} | "
                f"{h.sharpness_retention_ratio:.2%} |"
            )

        lines.extend(
            [
                "",
                "## 2. Longitudinal ΔSUV Forensics & Hypothesis Falsification",
                "",
                "| Scenario | Obs ΔSUV (%) | PSF-Gibbs (%) | PVE/Res (%) | CT-AC (%) | True Bio ΔSUV (%) | Verdict | PERCIST Gate |",
                "|---|---:|---:|---:|---:|---:|---|---|",
            ]
        )
        for d in self.diagnoses:
            lines.append(
                f"| {d.scenario_name} | {d.observed_delta_suv_pct:+.1f}% | "
                f"{d.psf_gibbs_artifact_pct:+.1f}% | {d.pve_resolution_artifact_pct:+.1f}% | "
                f"{d.ct_ac_respiratory_artifact_pct:+.1f}% | {d.true_biological_delta_pct:+.1f}% | "
                f"`{d.verdict}` | {'PASS' if d.percist_quantifiable else 'BLOCKED'} |"
            )

        if self.audit_report is not None:
            lines.extend(
                [
                    "",
                    "## 3. Independent Critic-Verifier Audit",
                    "",
                    f"- **Overall Gate:** `{self.audit_report.quantifiability_gate}`",
                    f"- **Provenance Ledger Integrity:** `{self.audit_report.ledger_integrity_verified}`",
                    f"- **Ledger Head SHA-256:** `{self.audit_report.ledger_head_sha256}`",
                    "",
                    "| Check ID | Status | Measured | Threshold | Rationale |",
                    "|---|---|---|---|---|",
                ]
            )
            for chk in self.audit_report.checks:
                lines.append(
                    f"| `{chk.check_id}` | {'PASS' if chk.passed else 'FAIL'} | "
                    f"{chk.metric_value} | {chk.threshold} | {chk.rationale} |"
                )

        lines.append("")
        return "\n".join(lines)

    def write_bundle(self, output_dir: str | Path, stem: str = "quantitative_physics_dossier") -> tuple[Path, Path]:
        """Write both the Markdown dossier and SHA-256 JSON provenance ledger."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        md_path = out / f"{stem}.md"
        json_path = out / f"{stem}_ledger.json"
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        payload = {
            "title": self.title,
            "disclaimer": RESEARCH_DISCLAIMER,
            "diagnoses": [d.model_dump(mode="json") for d in self.diagnoses],
            "fleet_harmonization": [h.model_dump(mode="json") for h in self.fleet_harmonization],
            "audit_report": self.audit_report.model_dump(mode="json") if self.audit_report else None,
            "provenance_ledger": self.ledger.to_dict(),
        }
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return md_path, json_path


class QuantitativeImagingForensicsAgent:
    """Autonomous Quantitative Imaging Forensics Physicist orchestrating inverse & audit pipelines."""

    def __init__(self) -> None:
        self.ledger = ProvenanceLedger()

    def diagnose_longitudinal_suv_shift(
        self,
        *,
        scenario_name: str,
        baseline_scanner: ScannerSpec,
        baseline_recon: ReconConfig,
        baseline_lesion: LesionROI,
        followup_scanner: ScannerSpec,
        followup_recon: ReconConfig,
        followup_lesion: LesionROI,
        use_suv_max_metric: bool = True,
        photon_energy_kev: float = 511.0,
    ) -> LongitudinalSUVDiagnosis:
        """Falsify competing physical vs. biological hypotheses for an observed longitudinal delta-SUV."""
        bl_obs = (
            baseline_lesion.observed_suv_max
            if (use_suv_max_metric and baseline_lesion.observed_suv_max is not None)
            else baseline_lesion.observed_suv_mean
        )
        fu_obs = (
            followup_lesion.observed_suv_max
            if (use_suv_max_metric and followup_lesion.observed_suv_max is not None)
            else followup_lesion.observed_suv_mean
        )
        obs_delta_pct = ((fu_obs - bl_obs) / max(bl_obs, 1e-6)) * 100.0

        # 1. Exact CT-AC respiratory mismatch bias at baseline and follow-up
        ac_bl = compute_ct_ac_mismatch_report(
            baseline_lesion.ct_pet_shift_mm,
            photon_energy_kev=photon_energy_kev,
            raise_on_severe=False,
        )
        ac_fu = compute_ct_ac_mismatch_report(
            followup_lesion.ct_pet_shift_mm,
            photon_energy_kev=photon_energy_kev,
            raise_on_severe=False,
        )
        self.ledger.append(ac_bl.provenance)
        self.ledger.append(ac_fu.provenance)

        # De-bias observed SUVs for CT-AC mismatch before PVE/Gibbs inversion
        bl_ac_corrected_mean = baseline_lesion.observed_suv_mean / ac_bl.attenuation_bias_factor
        fu_ac_corrected_mean = followup_lesion.observed_suv_mean / ac_fu.attenuation_bias_factor

        bl_clean_lesion = baseline_lesion.model_copy(
            update={"observed_suv_mean": bl_ac_corrected_mean}
        )
        fu_clean_lesion = followup_lesion.model_copy(
            update={"observed_suv_mean": fu_ac_corrected_mean}
        )

        # 2. Invert PVE + Gibbs using Rousset GTM
        abstained = False
        try:
            rec_bl = invert_lesion_activity(
                bl_clean_lesion, baseline_scanner, baseline_recon, method="GTM_EXACT"
            )
            rec_fu = invert_lesion_activity(
                fu_clean_lesion, followup_scanner, followup_recon, method="GTM_EXACT"
            )
            self.ledger.append(rec_bl.provenance)
            self.ledger.append(rec_fu.provenance)
        except AbstentionError:
            abstained = True
            rec_bl = invert_lesion_activity(
                bl_clean_lesion,
                baseline_scanner,
                baseline_recon,
                method="BAYESIAN_REGULARIZED",
                enforce_nyquist_gate=False,
            )
            rec_fu = invert_lesion_activity(
                fu_clean_lesion,
                followup_scanner,
                followup_recon,
                method="BAYESIAN_REGULARIZED",
                enforce_nyquist_gate=False,
            )

        # 3. Isolate pure PSF-Gibbs overshoot vs. non-Gibbs PVE resolution shift via counterfactual forward simulation
        # Zero ONLY `gibbs_ringing_strength` while keeping intrinsic PSF / sharpening identical
        fu_scanner_no_gibbs = followup_scanner.model_copy(update={"gibbs_ringing_strength": 0.0})
        bl_scanner_no_gibbs = baseline_scanner.model_copy(update={"gibbs_ringing_strength": 0.0})

        rc_bl_full = compute_sphere_recovery_coefficients(
            baseline_lesion.diameter_mm,
            baseline_scanner,
            baseline_recon,
            radial_offset_mm=baseline_lesion.radial_offset_mm,
        )
        rc_bl_nogibbs = compute_sphere_recovery_coefficients(
            baseline_lesion.diameter_mm,
            bl_scanner_no_gibbs,
            baseline_recon,
            radial_offset_mm=baseline_lesion.radial_offset_mm,
        )
        rc_fu_full = compute_sphere_recovery_coefficients(
            followup_lesion.diameter_mm,
            followup_scanner,
            followup_recon,
            radial_offset_mm=followup_lesion.radial_offset_mm,
        )
        rc_fu_nogibbs = compute_sphere_recovery_coefficients(
            followup_lesion.diameter_mm,
            fu_scanner_no_gibbs,
            followup_recon,
            radial_offset_mm=followup_lesion.radial_offset_mm,
        )

        metric_key = "rc_max" if use_suv_max_metric else "rc_mean"
        # Multiplicative physical bias factors relative to baseline
        gibbs_ratio_bl = rc_bl_full[metric_key] / max(rc_bl_nogibbs[metric_key], 1e-4)
        gibbs_ratio_fu = rc_fu_full[metric_key] / max(rc_fu_nogibbs[metric_key], 1e-4)
        rel_gibbs_mult = gibbs_ratio_fu / max(gibbs_ratio_bl, 1e-4)

        rel_pve_mult = rc_fu_nogibbs[metric_key] / max(rc_bl_nogibbs[metric_key], 1e-4)
        rel_ac_mult = ac_fu.attenuation_bias_factor / max(ac_bl.attenuation_bias_factor, 1e-4)

        # True biological ratio remaining after dividing out physical multipliers
        obs_ratio = fu_obs / max(bl_obs, 1e-6)
        bio_ratio = obs_ratio / max(rel_gibbs_mult * rel_pve_mult * rel_ac_mult, 1e-6)

        # Logarithmic Shapley-exact additive decomposition of observed_delta_suv_pct so components sum identically
        import math

        log_tot = math.log(max(obs_ratio, 1e-6))
        log_gibbs = math.log(max(rel_gibbs_mult, 1e-6))
        log_pve = math.log(max(rel_pve_mult, 1e-6))
        log_ac = math.log(max(rel_ac_mult, 1e-6))
        log_bio = math.log(max(bio_ratio, 1e-6))

        if abs(log_tot) > 1e-6:
            scale = obs_delta_pct / log_tot
            gibbs_pct = log_gibbs * scale
            pve_pct = log_pve * scale
            ac_pct = log_ac * scale
            bio_pct = log_bio * scale
        else:
            # Exact first-order % decomposition when net change ~ 0 (canceling artifacts!)
            gibbs_pct = (rel_gibbs_mult - 1.0) * 100.0
            pve_pct = (rel_pve_mult - 1.0) * 100.0
            ac_pct = (rel_ac_mult - 1.0) * 100.0
            bio_pct = obs_delta_pct - (gibbs_pct + pve_pct + ac_pct)

        # Hypothesis falsification & clinical verdict
        falsified: list[str] = []
        if abs(gibbs_pct) < 5.0:
            falsified.append("H1_PSF_GIBBS_OVERSHOOT (<5% contribution)")
        if abs(pve_pct) < 5.0:
            falsified.append("H2_PVE_RESOLUTION_BIAS (<5% contribution)")
        if abs(ac_pct) < 5.0:
            falsified.append("H3_CT_AC_RESPIRATORY_MISMATCH (<5% contribution)")
        if abs(bio_pct) < 15.0:
            falsified.append("H4_TRUE_BIOLOGICAL_CHANGE (<15% after physical inversion)")

        severe_ac = ac_bl.exceeds_abstention_threshold or ac_fu.exceeds_abstention_threshold
        percist_ok = (not abstained) and (not severe_ac) and (abs(gibbs_pct) < 12.0)

        components = {
            "PSF-Gibbs Overshoot": abs(gibbs_pct),
            "PVE / Resolution Shift": abs(pve_pct),
            "CT-AC Respiratory Mismatch": abs(ac_pct),
            "True Biological Change": abs(bio_pct),
        }
        dominant = max(components, key=lambda k: components[k])

        if abstained:
            verdict: AttributionVerdict = "ABSTAIN_UNQUANTIFIABLE"
            recommendation = "Lesion is sub-Nyquist for current scanner voxel/PSF; abstain from PERCIST SUV adjudication."
        elif severe_ac and abs(ac_pct) >= max(abs(bio_pct), abs(gibbs_pct)):
            verdict = "CT_AC_RESPIRATORY_ARTIFACT"
            recommendation = (
                "Apparent SUV shift is dominated by CT-PET respiratory AC mismatch; "
                "repeat respiratory-gated/cine-CT AC or evaluate non-attenuation-corrected (NAC) PET."
            )
        elif abs(gibbs_pct) >= 12.0 and abs(bio_pct) < 20.0:
            verdict = "SCANNER_PSF_GIBBS_ARTIFACT"
            recommendation = (
                "Apparent +SUV inflation is an unharmonized PSF-OSEM Gibbs artifact from scanner switch; "
                "apply EARL2 Gaussian harmonization filter before longitudinal PERCIST scoring."
            )
        elif abs(pve_pct) >= 15.0 and abs(bio_pct) < 20.0:
            verdict = "PVE_DELINEATION_ARTIFACT"
            recommendation = (
                "Observed SUV shift is driven by spatial-resolution / PVE diameter bias; "
                "use GTM partial-volume correction or harmonized reconstruction."
            )
        elif bio_pct >= 25.0:
            verdict = "TRUE_BIOLOGICAL_PROGRESSION"
            recommendation = (
                "True biological SUV progression persists after accounting for PSF-Gibbs, PVE, and CT-AC physics."
            )
        elif bio_pct <= -25.0:
            verdict = "BIOLOGICAL_RESPONSE"
            recommendation = (
                "True metabolic response confirmed after physical bias inversion."
            )
        else:
            verdict = "STABLE_DISEASE"
            recommendation = (
                "No significant biological SUV change after physical bias removal (metabolically stable)."
            )

        prov = build_provenance_record(
            operation="agent_longitudinal_suv_forensics",
            inputs={
                "scenario_name": scenario_name,
                "baseline_scanner": baseline_scanner.scanner_id,
                "followup_scanner": followup_scanner.scanner_id,
                "baseline_obs": bl_obs,
                "followup_obs": fu_obs,
            },
            outputs={
                "observed_delta_suv_pct": obs_delta_pct,
                "psf_gibbs_artifact_pct": gibbs_pct,
                "pve_resolution_artifact_pct": pve_pct,
                "ct_ac_respiratory_artifact_pct": ac_pct,
                "true_biological_delta_pct": bio_pct,
                "verdict": verdict,
            },
            data_source_tag="AGENT_FORENSICS_PHYSICS_DECOMPOSITION",
            citations=[
                "Kaalep et al. (2018) EJNMMI Phys 5:14, DOI:10.1186/s40658-018-0226-7",
                "Boellaard et al. (2015) Eur J Nucl Med Mol Imaging 42:328-354, DOI:10.1007/s00259-014-2961-x",
                "Rousset, Ma, & Evans (1998) J Nucl Med 39(5):904-911, PMID:9591599",
            ],
        )
        self.ledger.append(prov)

        return LongitudinalSUVDiagnosis(
            scenario_name=scenario_name,
            baseline_observed_suv=float(bl_obs),
            followup_observed_suv=float(fu_obs),
            observed_delta_suv_pct=float(round(obs_delta_pct, 2)),
            psf_gibbs_artifact_pct=float(round(gibbs_pct, 2)),
            pve_resolution_artifact_pct=float(round(pve_pct, 2)),
            ct_ac_respiratory_artifact_pct=float(round(ac_pct, 2)),
            true_biological_delta_pct=float(round(bio_pct, 2)),
            baseline_recovered_suv=float(round(rec_bl.recovered_activity_nominal, 3)),
            followup_recovered_suv=float(round(rec_fu.recovered_activity_nominal, 3)),
            followup_recovered_p05=float(round(rec_fu.recovered_activity_p05, 3)),
            followup_recovered_p95=float(round(rec_fu.recovered_activity_p95, 3)),
            verdict=verdict,
            percist_quantifiable=percist_ok,
            falsified_hypotheses=falsified,
            dominant_mechanism=dominant,
            recommendation=recommendation,
            provenance=prov,
        )

    def harmonize_multivendor_fleet(
        self,
        fleet: list[tuple[ScannerSpec, ReconConfig]] | None = None,
        pet_standard: CorridorName = "EARL2",
        spect_standard: CorridorName = "MIRD23_SPECT",
    ) -> list[HarmonizationFilterResult]:
        """Harmonize a multi-vendor PET + SPECT fleet into shared EARL2/QIBA and MIRD-23 corridors."""
        target_fleet = fleet if fleet is not None else build_reference_scanner_fleet()
        pet_corridor = get_corridor_spec(pet_standard)
        spect_corridor = get_corridor_spec(spect_standard)

        results: list[HarmonizationFilterResult] = []
        for scanner, recon in target_fleet:
            corr = spect_corridor if scanner.modality.startswith("SPECT") else pet_corridor
            res = synthesize_harmonization_filter(scanner, recon, corr, raise_on_infeasible=False)
            self.ledger.append(res.provenance)
            results.append(res)
        return results

    def run_critic_verifier_audit(
        self,
        diagnoses: list[LongitudinalSUVDiagnosis],
        fleet_results: list[HarmonizationFilterResult],
        *,
        max_reprojection_residual: float = 1e-6,
    ) -> CriticAuditReport:
        """Run independent Critic-Verifier Audit over all physics invariants and provenance."""
        checks: list[CriticAuditCheck] = []

        # 1. Provenance Ledger SHA-256 chain integrity
        ledger_ok = self.ledger.verify_integrity()
        checks.append(
            CriticAuditCheck(
                check_id="PROVENANCE_SHA256_CHAIN",
                passed=ledger_ok,
                metric_value=f"{len(self.ledger.records)} records verified={ledger_ok}",
                threshold="True",
                rationale="Every calculation must link deterministically in the SHA-256 provenance ledger.",
            )
        )

        # 2. Fleet harmonization convergence
        n_feasible = sum(1 for r in fleet_results if r.feasible)
        max_post_viol = max((r.max_corridor_violation_after for r in fleet_results), default=0.0)
        fleet_ok = (len(fleet_results) > 0) and (n_feasible == len(fleet_results)) and (max_post_viol <= 1e-3)
        checks.append(
            CriticAuditCheck(
                check_id="FLEET_CORRIDOR_CONVERGENCE",
                passed=fleet_ok,
                metric_value=f"{n_feasible}/{len(fleet_results)} feasible (max_viol={max_post_viol:.5f})",
                threshold="100% feasible, max_viol <= 0.001",
                rationale="All fleet scanners must satisfy their target EARL2/QIBA/MIRD-23 RC corridors.",
            )
        )

        # 3. Additive attribution conservation (sum of decomposed components == observed delta-SUV)
        max_attrib_err = 0.0
        for d in diagnoses:
            s = (
                d.psf_gibbs_artifact_pct
                + d.pve_resolution_artifact_pct
                + d.ct_ac_respiratory_artifact_pct
                + d.true_biological_delta_pct
            )
            max_attrib_err = max(max_attrib_err, abs(s - d.observed_delta_suv_pct))
        attrib_ok = max_attrib_err <= 0.15
        checks.append(
            CriticAuditCheck(
                check_id="DELTA_SUV_ATTRIBUTION_CONSERVATION",
                passed=attrib_ok,
                metric_value=f"max_err={max_attrib_err:.4f}%",
                threshold="<= 0.15%",
                rationale="Decomposed physical + biological components must sum to observed delta-SUV.",
            )
        )

        # 4. GTM re-projection accuracy spot-check on reference lesion
        fleet = build_reference_scanner_fleet()
        sc0, rc0 = fleet[0]
        gtm_check = invert_lesion_activity(
            LesionROI(lesion_id="AUDIT_REF", diameter_mm=17.0, observed_suv_mean=6.5, background_suv=2.0),
            sc0,
            rc0,
            method="GTM_EXACT",
        )
        gtm_ok = gtm_check.forward_reprojection_residual <= max_reprojection_residual
        checks.append(
            CriticAuditCheck(
                check_id="GTM_EXACT_REPROJECTION_INVARIANT",
                passed=gtm_ok,
                metric_value=f"residual={gtm_check.forward_reprojection_residual:.2e}",
                threshold=f"<= {max_reprojection_residual:.1e}",
                rationale="Forward re-projection of GTM-inverted activity must recover observed mean.",
            )
        )

        # 5. Quantifiability gate across diagnosed scenarios
        any_blocked = any(not d.percist_quantifiable for d in diagnoses)
        all_passed = all(c.passed for c in checks)
        if not all_passed:
            gate: Literal["PASS_QUANTIFIABLE", "PASS_WITH_CAUTION", "ABSTAIN_BLOCKED"] = "ABSTAIN_BLOCKED"
        elif any_blocked:
            gate = "PASS_WITH_CAUTION"
        else:
            gate = "PASS_QUANTIFIABLE"

        return CriticAuditReport(
            overall_passed=all_passed,
            quantifiability_gate=gate,
            checks=checks,
            ledger_head_sha256=self.ledger.head_sha256,
            ledger_integrity_verified=ledger_ok,
        )

    def build_complete_dossier(
        self,
        title: str = "EANM / EARL2 / QIBA Quantitative Nuclear Medicine Physics Dossier",
    ) -> QuantitativePhysicsDossier:
        """Run end-to-end fleet harmonization, 4 clinical forensics scenarios, and Critic-Verifier Audit."""
        fleet = build_reference_scanner_fleet()
        fleet_results = self.harmonize_multivendor_fleet(fleet)

        sipm_sc, sipm_rc = fleet[0]
        pmt_sc, pmt_rc = fleet[2]

        # Build harmonized recon for SiPM
        sipm_harm_rc = sipm_rc.model_copy(
            update={"post_filter_fwhm_mm": fleet_results[0].optimal_post_filter_fwhm_mm}
        )

        # Forward-consistent clinical scenarios (ground truth SUV=8.0, bg=2.0 -> true bio ~0% for artifacts, +40% for progression)
        # Scenario A: Same 14 mm lesion (true SUV=8.0), switched from standard OSEM (no PSF) -> unharmonized SiPM PSF-OSEM (+30%+ Gibbs pseudo-progression!)
        sipm_no_psf_sc = sipm_sc.model_copy(
            update={"scanner_id": "SiPM_Standard_OSEM", "has_psf_modeling": False, "gibbs_ringing_strength": 0.0}
        )
        sipm_no_psf_rc = ReconConfig(algorithm="OSEM", iterations=4, subsets=16, post_filter_fwhm_mm=3.5)
        rc_a_bl = compute_sphere_recovery_coefficients(14.0, sipm_no_psf_sc, sipm_no_psf_rc, true_activity=8.0, background_activity=2.0)
        rc_a_fu = compute_sphere_recovery_coefficients(14.0, sipm_sc, sipm_rc, true_activity=8.0, background_activity=2.0)
        # Scenario B: True biological progression (true SUV 8.0 -> 11.2, +40%) on EARL2-harmonized SiPM
        rc_b_bl = compute_sphere_recovery_coefficients(18.0, sipm_sc, sipm_harm_rc, true_activity=8.0, background_activity=2.0)
        rc_b_fu = compute_sphere_recovery_coefficients(18.0, sipm_sc, sipm_harm_rc, true_activity=11.2, background_activity=2.0)
        # Scenario C: CT-AC respiratory shift (-11 mm basal lung -> false-hot), stable true SUV=8.0
        rc_c_bl = compute_sphere_recovery_coefficients(16.0, sipm_sc, sipm_harm_rc, true_activity=8.0, background_activity=1.8)
        ac_c_fu = compute_ct_ac_mismatch_report(-11.0, photon_energy_kev=511.0)
        # Scenario D: Small-lesion anatomical growth (10 mm -> 16 mm) with stable intrinsic SUV=8.0 on Legacy PMT
        rc_d_bl = compute_sphere_recovery_coefficients(10.0, pmt_sc, pmt_rc, true_activity=8.0, background_activity=2.0)
        rc_d_fu = compute_sphere_recovery_coefficients(16.0, pmt_sc, pmt_rc, true_activity=8.0, background_activity=2.0)

        diagnoses = [
            self.diagnose_longitudinal_suv_shift(
                scenario_name="Scanner Switch (Standard OSEM -> SiPM PSF-OSEM)",
                baseline_scanner=sipm_no_psf_sc,
                baseline_recon=sipm_no_psf_rc,
                baseline_lesion=LesionROI(
                    lesion_id="S1_BL",
                    diameter_mm=14.0,
                    observed_suv_mean=round(8.0 * rc_a_bl["rc_mean_true_voi"], 3),
                    observed_suv_max=round(8.0 * rc_a_bl["rc_max"], 3),
                    background_suv=2.0,
                ),
                followup_scanner=sipm_sc,
                followup_recon=sipm_rc,
                followup_lesion=LesionROI(
                    lesion_id="S1_FU",
                    diameter_mm=14.0,
                    observed_suv_mean=round(8.0 * rc_a_fu["rc_mean_true_voi"], 3),
                    observed_suv_max=round(8.0 * rc_a_fu["rc_max"], 3),
                    background_suv=2.0,
                ),
            ),
            self.diagnose_longitudinal_suv_shift(
                scenario_name="True Progression (EARL2-Harmonized SiPM)",
                baseline_scanner=sipm_sc,
                baseline_recon=sipm_harm_rc,
                baseline_lesion=LesionROI(
                    lesion_id="S2_BL",
                    diameter_mm=18.0,
                    observed_suv_mean=round(8.0 * rc_b_bl["rc_mean_true_voi"], 3),
                    observed_suv_max=round(8.0 * rc_b_bl["rc_max"], 3),
                    background_suv=2.0,
                ),
                followup_scanner=sipm_sc,
                followup_recon=sipm_harm_rc,
                followup_lesion=LesionROI(
                    lesion_id="S2_FU",
                    diameter_mm=18.0,
                    observed_suv_mean=round(11.2 * rc_b_fu["rc_mean_true_voi"], 3),
                    observed_suv_max=round(11.2 * rc_b_fu["rc_max"], 3),
                    background_suv=2.0,
                ),
            ),
            self.diagnose_longitudinal_suv_shift(
                scenario_name="CT-AC Respiratory Shift (-11 mm Basal Lung)",
                baseline_scanner=sipm_sc,
                baseline_recon=sipm_harm_rc,
                baseline_lesion=LesionROI(
                    lesion_id="S3_BL",
                    diameter_mm=16.0,
                    observed_suv_mean=round(8.0 * rc_c_bl["rc_mean_true_voi"], 3),
                    observed_suv_max=round(8.0 * rc_c_bl["rc_max"], 3),
                    background_suv=1.8,
                    is_near_lung_diaphragm=True,
                    ct_pet_shift_mm=0.0,
                ),
                followup_scanner=sipm_sc,
                followup_recon=sipm_harm_rc,
                followup_lesion=LesionROI(
                    lesion_id="S3_FU",
                    diameter_mm=16.0,
                    observed_suv_mean=round(8.0 * rc_c_bl["rc_mean_true_voi"] * ac_c_fu.attenuation_bias_factor, 3),
                    observed_suv_max=round(8.0 * rc_c_bl["rc_max"] * ac_c_fu.attenuation_bias_factor, 3),
                    background_suv=1.8,
                    is_near_lung_diaphragm=True,
                    ct_pet_shift_mm=-11.0,
                ),
            ),
            self.diagnose_longitudinal_suv_shift(
                scenario_name="PVE Diameter Bias (10 mm -> 16 mm Lesion)",
                baseline_scanner=pmt_sc,
                baseline_recon=pmt_rc,
                baseline_lesion=LesionROI(
                    lesion_id="S4_BL",
                    diameter_mm=10.0,
                    observed_suv_mean=round(8.0 * rc_d_bl["rc_mean_true_voi"], 3),
                    observed_suv_max=round(8.0 * rc_d_bl["rc_max"], 3),
                    background_suv=2.0,
                ),
                followup_scanner=pmt_sc,
                followup_recon=pmt_rc,
                followup_lesion=LesionROI(
                    lesion_id="S4_FU",
                    diameter_mm=16.0,
                    observed_suv_mean=round(8.0 * rc_d_fu["rc_mean_true_voi"], 3),
                    observed_suv_max=round(8.0 * rc_d_fu["rc_max"], 3),
                    background_suv=2.0,
                ),
            ),
        ]

        audit = self.run_critic_verifier_audit(diagnoses, fleet_results)
        return QuantitativePhysicsDossier(
            title=title,
            diagnoses=diagnoses,
            fleet_harmonization=fleet_results,
            audit_report=audit,
            ledger=self.ledger,
        )
