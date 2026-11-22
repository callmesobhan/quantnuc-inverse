"""JSON-RPC 2.0 stdio MCP Server exposing 6 provenance-gated nuclear medicine physics tools.

Tools exposed:
1. `quantnuc_forward_nema_rc`
2. `quantnuc_invert_pve_activity`
3. `quantnuc_synthesize_earl_filter`
4. `quantnuc_audit_ct_ac_mismatch`
5. `quantnuc_diagnose_suv_shift`
6. `quantnuc_run_sobol_sensitivity`

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from quantnuc_inverse import __version__
from quantnuc_inverse.earl_harmonizer import (
    get_corridor_spec,
    synthesize_harmonization_filter,
)
from quantnuc_inverse.forward_image_formation import (
    compute_ct_ac_mismatch_report,
    simulate_nema_rc_curve,
)
from quantnuc_inverse.harness.agent_orchestrator import QuantitativeImagingForensicsAgent
from quantnuc_inverse.ir import (
    RESEARCH_DISCLAIMER,
    AbstentionError,
    LesionROI,
    ReconConfig,
    ScannerSpec,
)
from quantnuc_inverse.provenance import build_provenance_record
from quantnuc_inverse.pve_inverse_solver import invert_lesion_activity
from quantnuc_inverse.sobol_sensitivity import run_sobol_sensitivity

PROTOCOL_VERSION = "2024-11-05"

TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "quantnuc_forward_nema_rc",
        "description": "Simulate 3D NEMA NU 2-2018 6-sphere Recovery Coefficients (RCmean, RCmax, RCpeak) including PSF-Gibbs ringing.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scanner": {"type": "object"},
                "recon": {"type": "object"},
                "sphere_to_bg_ratio": {"type": "number", "default": 4.0},
            },
            "required": ["scanner", "recon"],
        },
    },
    {
        "name": "quantnuc_invert_pve_activity",
        "description": "Invert true lesion activity from observed SUV using Rousset GTM, Muller-Gartner, or Bayesian PVC with p05/nominal/p95 CI.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lesion": {"type": "object"},
                "scanner": {"type": "object"},
                "recon": {"type": "object"},
                "method": {"type": "string", "enum": ["GTM_EXACT", "MULLER_GARTNER", "BAYESIAN_REGULARIZED"], "default": "GTM_EXACT"},
            },
            "required": ["lesion", "scanner", "recon"],
        },
    },
    {
        "name": "quantnuc_synthesize_earl_filter",
        "description": "Synthesize minimal post-reconstruction Gaussian/Butterworth harmonization filter for EARL1, EARL2, QIBA, or MIRD23_SPECT.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scanner": {"type": "object"},
                "recon": {"type": "object"},
                "standard_name": {"type": "string", "enum": ["EARL1", "EARL2", "QIBA_FDG_PET", "MIRD23_SPECT"], "default": "EARL2"},
                "filter_type": {"type": "string", "enum": ["gaussian", "butterworth"], "default": "gaussian"},
                "raise_on_infeasible": {"type": "boolean", "default": False},
            },
            "required": ["scanner", "recon"],
        },
    },
    {
        "name": "quantnuc_audit_ct_ac_mismatch",
        "description": "Compute exact CT-PET/SPECT respiratory attenuation-correction line-integral bias exp(int delta_mu ds) and abstention status.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "breathing_shift_mm": {"type": "number"},
                "photon_energy_kev": {"type": "number", "default": 511.0},
                "raise_on_severe": {"type": "boolean", "default": False},
            },
            "required": ["breathing_shift_mm"],
        },
    },
    {
        "name": "quantnuc_diagnose_suv_shift",
        "description": "Decompose a longitudinal delta-SUV shift into PSF-Gibbs, PVE resolution, CT-AC respiratory artifact, and true biological change.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "scenario_name": {"type": "string", "default": "Longitudinal Lesion Audit"},
                "baseline_scanner": {"type": "object"},
                "baseline_recon": {"type": "object"},
                "baseline_lesion": {"type": "object"},
                "followup_scanner": {"type": "object"},
                "followup_recon": {"type": "object"},
                "followup_lesion": {"type": "object"},
            },
            "required": [
                "baseline_scanner",
                "baseline_recon",
                "baseline_lesion",
                "followup_scanner",
                "followup_recon",
                "followup_lesion",
            ],
        },
    },
    {
        "name": "quantnuc_run_sobol_sensitivity",
        "description": "Run Saltelli-Jansen QMC global Sobol' sensitivity analysis (S1, ST, interactions) on quantitative SUV variance.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "lesion_diameter_mm": {"type": "number"},
                "true_suv": {"type": "number", "default": 8.0},
                "background_suv": {"type": "number", "default": 2.0},
                "n_base_samples": {"type": "integer", "default": 512},
            },
            "required": ["lesion_diameter_mm"],
        },
    },
]


def _dispatch_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute one provenance-gated quantnuc-inverse physics tool."""
    if name == "quantnuc_forward_nema_rc":
        scanner = ScannerSpec.model_validate(arguments["scanner"])
        recon = ReconConfig.model_validate(arguments["recon"])
        ratio = float(arguments.get("sphere_to_bg_ratio", 4.0))
        curves = simulate_nema_rc_curve(scanner, recon, sphere_to_bg_ratio=ratio)
        prov = build_provenance_record(
            operation="mcp_forward_nema_rc",
            inputs={"scanner": scanner, "recon": recon, "sphere_to_bg_ratio": ratio},
            outputs=curves,
            data_source_tag="SYNTHETIC_NEMA_NU2_2018_DRO",
            citations=["NEMA NU 2-2018 IEC Body Phantom"],
        )
        return {**curves, "provenance": prov.model_dump(mode="json")}

    if name == "quantnuc_invert_pve_activity":
        lesion = LesionROI.model_validate(arguments["lesion"])
        scanner = ScannerSpec.model_validate(arguments["scanner"])
        recon = ReconConfig.model_validate(arguments["recon"])
        method = str(arguments.get("method", "GTM_EXACT"))
        res = invert_lesion_activity(lesion, scanner, recon, method=method)
        return res.model_dump(mode="json")

    if name == "quantnuc_synthesize_earl_filter":
        scanner = ScannerSpec.model_validate(arguments["scanner"])
        recon = ReconConfig.model_validate(arguments["recon"])
        std = arguments.get("standard_name", "EARL2")
        corridor = get_corridor_spec(std)
        res = synthesize_harmonization_filter(
            scanner,
            recon,
            corridor,
            filter_type=arguments.get("filter_type", "gaussian"),
            raise_on_infeasible=bool(arguments.get("raise_on_infeasible", False)),
        )
        return res.model_dump(mode="json")

    if name == "quantnuc_audit_ct_ac_mismatch":
        rep = compute_ct_ac_mismatch_report(
            float(arguments["breathing_shift_mm"]),
            photon_energy_kev=float(arguments.get("photon_energy_kev", 511.0)),
            raise_on_severe=bool(arguments.get("raise_on_severe", False)),
        )
        return rep.model_dump(mode="json")

    if name == "quantnuc_diagnose_suv_shift":
        agent = QuantitativeImagingForensicsAgent()
        diag = agent.diagnose_longitudinal_suv_shift(
            scenario_name=str(arguments.get("scenario_name", "Longitudinal Lesion Audit")),
            baseline_scanner=ScannerSpec.model_validate(arguments["baseline_scanner"]),
            baseline_recon=ReconConfig.model_validate(arguments["baseline_recon"]),
            baseline_lesion=LesionROI.model_validate(arguments["baseline_lesion"]),
            followup_scanner=ScannerSpec.model_validate(arguments["followup_scanner"]),
            followup_recon=ReconConfig.model_validate(arguments["followup_recon"]),
            followup_lesion=LesionROI.model_validate(arguments["followup_lesion"]),
        )
        return diag.model_dump(mode="json")

    if name == "quantnuc_run_sobol_sensitivity":
        res = run_sobol_sensitivity(
            float(arguments["lesion_diameter_mm"]),
            true_suv=float(arguments.get("true_suv", 8.0)),
            background_suv=float(arguments.get("background_suv", 2.0)),
            n_base_samples=int(arguments.get("n_base_samples", 512)),
        )
        return {
            "lesion_diameter_mm": res.lesion_diameter_mm,
            "n_base_samples": res.n_base_samples,
            "total_evaluations": res.total_evaluations,
            "s1": res.s1,
            "st": res.st,
            "interactions": res.interactions,
            "s1_ci95": res.s1_ci95,
            "st_ci95": res.st_ci95,
            "output_mean_suv": res.output_mean_suv,
            "output_std_suv": res.output_std_suv,
            "provenance": res.provenance.model_dump(mode="json"),
        }

    raise KeyError(f"Unknown MCP tool: {name}")


def handle_jsonrpc_request(request: dict[str, Any]) -> dict[str, Any]:
    """Process a single JSON-RPC 2.0 MCP request dictionary and return a JSON-RPC 2.0 response."""
    req_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "quantnuc-inverse",
                    "version": __version__,
                    "disclaimer": RESEARCH_DISCLAIMER,
                },
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": req_id, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOL_DEFINITIONS}}

    if method == "tools/call":
        tool_name = params.get("name", "")
        arguments = params.get("arguments") or {}
        try:
            payload = _dispatch_tool(tool_name, arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
                    "isError": False,
                },
            }
        except AbstentionError as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(exc.to_dict(), indent=2)}],
                    "isError": True,
                },
            }
        except KeyError as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": str(exc)},
            }
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": f"Invalid tool parameters or execution error: {exc}"},
            }

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


def main() -> None:
    """Run line-delimited JSON-RPC 2.0 stdio MCP server loop."""
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            err = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            sys.stdout.write(json.dumps(err) + "\n")
            sys.stdout.flush()
            continue

        # Notifications without an `id` do not receive a response per JSON-RPC 2.0
        if "id" not in req:
            continue

        resp = handle_jsonrpc_request(req)
        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
