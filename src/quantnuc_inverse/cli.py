"""Ergonomic command-line interface (`quantnuc`) for quantitative PET/SPECT inverse physics workflows.

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import argparse
import json
import sys

from quantnuc_inverse import __version__
from quantnuc_inverse.earl_harmonizer import (
    build_reference_scanner_fleet,
    get_corridor_spec,
    synthesize_harmonization_filter,
)
from quantnuc_inverse.forward_image_formation import compute_ct_ac_mismatch_report
from quantnuc_inverse.harness.agent_orchestrator import QuantitativeImagingForensicsAgent
from quantnuc_inverse.ir import RESEARCH_DISCLAIMER, AbstentionError, LesionROI, TimeActivityPoint
from quantnuc_inverse.pve_inverse_solver import invert_lesion_activity
from quantnuc_inverse.theranostics_dosimetry import compute_pve_propagated_theranostic_dose


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quantnuc",
        description=f"quantnuc-inverse v{__version__} — Quantitative PET/SPECT Physics CLI ({RESEARCH_DISCLAIMER})",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_harm = sub.add_parser("harmonize", help="Harmonize the 4-scanner reference fleet into EARL/QIBA/MIRD corridors.")
    p_harm.add_argument("--pet-standard", default="EARL2", choices=["EARL1", "EARL2", "QIBA_FDG_PET"])
    p_harm.add_argument("--noise-constrained", action="store_true", help="Enforce EANM/EARL COV <= 15% constraint.")

    p_pvc = sub.add_parser("pvc", help="Invert partial-volume-corrupted lesion SUV via Rousset GTM / MG / Bayesian PVC.")
    p_pvc.add_argument("--diameter", type=float, required=True, help="Lesion inner diameter in mm.")
    p_pvc.add_argument("--observed-suv", type=float, required=True, help="Observed ROI mean SUV.")
    p_pvc.add_argument("--background-suv", type=float, default=2.0, help="Local background SUV.")
    p_pvc.add_argument("--method", default="GTM_EXACT", choices=["GTM_EXACT", "MULLER_GARTNER", "BAYESIAN_REGULARIZED"])

    p_ac = sub.add_parser("audit-ac", help="Audit CT-PET/SPECT respiratory AC mismatch bias.")
    p_ac.add_argument("--shift-mm", type=float, required=True, help="CT-PET diaphragm shift in mm (+ inspiration, - expiration).")
    p_ac.add_argument("--energy-kev", type=float, default=511.0, help="Emission photon energy in keV (511, 208, 140, 364).")

    p_dose = sub.add_parser("dose", help="Run MIRD-23/26 PVE-propagated theranostic absorbed-dose benchmark.")
    p_dose.add_argument("--diameter", type=float, default=14.0, help="Tumor diameter in mm.")
    p_dose.add_argument("--isotope", default="Lu-177", choices=["Lu-177", "I-131", "Y-90"])

    p_dos = sub.add_parser("dossier", help="Run full Frontier Agent Forensics & Critic-Verifier Audit dossier.")
    p_dos.add_argument("--output-dir", default="outputs/dossier", help="Directory for Markdown + JSON provenance ledger.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "harmonize":
            fleet = build_reference_scanner_fleet()
            pet_corr = get_corridor_spec(args.pet_standard)
            spect_corr = get_corridor_spec("MIRD23_SPECT")
            out = []
            for sc, rc in fleet:
                corr = spect_corr if sc.modality.startswith("SPECT") else pet_corr
                res = synthesize_harmonization_filter(
                    sc, rc, corr, enforce_noise_constraint=bool(args.noise_constrained)
                )
                out.append(res.model_dump(mode="json"))
            print(json.dumps(out, indent=2))
            return 0

        if args.command == "pvc":
            sc, rc = build_reference_scanner_fleet()[0]
            harm_rc = rc.model_copy(update={"post_filter_fwhm_mm": 6.4})
            roi = LesionROI(
                lesion_id="CLI_LESION",
                diameter_mm=args.diameter,
                observed_suv_mean=args.observed_suv,
                background_suv=args.background_suv,
            )
            res = invert_lesion_activity(roi, sc, harm_rc, method=args.method)
            print(res.model_dump_json(indent=2))
            return 0

        if args.command == "audit-ac":
            rep = compute_ct_ac_mismatch_report(args.shift_mm, photon_energy_kev=args.energy_kev)
            print(rep.model_dump_json(indent=2))
            return 0

        if args.command == "dose":
            spect_sc, spect_rc = build_reference_scanner_fleet()[3]
            tps = [
                TimeActivityPoint(time_hours=4.0, observed_activity_mbq_ml=1.85, background_activity_mbq_ml=0.12),
                TimeActivityPoint(time_hours=24.0, observed_activity_mbq_ml=1.60, background_activity_mbq_ml=0.09),
                TimeActivityPoint(time_hours=72.0, observed_activity_mbq_ml=0.95, background_activity_mbq_ml=0.05),
                TimeActivityPoint(time_hours=168.0, observed_activity_mbq_ml=0.38, background_activity_mbq_ml=0.02),
            ]
            res = compute_pve_propagated_theranostic_dose(
                lesion_id="CLI_THERA",
                lesion_diameter_mm=args.diameter,
                timepoints=tps,
                scanner=spect_sc,
                recon=spect_rc.model_copy(update={"post_filter_fwhm_mm": 5.0}),
                isotope=args.isotope,
            )
            print(res.model_dump_json(indent=2))
            return 0

        if args.command == "dossier":
            agent = QuantitativeImagingForensicsAgent()
            dossier = agent.build_complete_dossier()
            md_path, json_path = dossier.write_bundle(args.output_dir)
            print(
                json.dumps(
                    {
                        "markdown": str(md_path),
                        "ledger_json": str(json_path),
                        "audit_passed": dossier.audit_report.overall_passed if dossier.audit_report else False,
                        "gate": dossier.audit_report.quantifiability_gate if dossier.audit_report else None,
                    },
                    indent=2,
                )
            )
            return 0
    except AbstentionError as exc:
        print(json.dumps(exc.to_dict(), indent=2), file=sys.stderr)
        return 2

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
