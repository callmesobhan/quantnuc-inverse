# EANM / EARL2 / QIBA Quantitative Nuclear Medicine Physics Dossier

> **Notice:** Research Software — NOT A MEDICAL DEVICE.

## 1. Multi-Vendor Fleet Harmonization Summary (EARL2 / MIRD-23)

| Scanner ID | Target Standard | Feasible | Optimal Post-Filter (mm) | Rec. Iters | Max Viol. Before | Max Viol. After | Sharpness Retention |
|---|---|---:|---:|---:|---:|---:|---:|
| `SiPM_Digital_PET` | `EARL2` | YES | 6.40 (gaussian) | 4 | 0.4031 | 0.0000 | 38.78% |
| `TOF_PSF_PET` | `EARL2` | YES | 6.02 (gaussian) | 4 | 0.3478 | 0.0000 | 49.14% |
| `Legacy_PMT_PET` | `EARL2` | YES | 0.00 (gaussian) | 2 | 0.1394 | 0.0000 | 100.00% |
| `CZT_SPECT_CT` | `MIRD23_SPECT` | YES | 10.12 (gaussian) | 6 | 0.3095 | 0.0000 | 52.54% |

## 2. Longitudinal ΔSUV Forensics & Hypothesis Falsification

| Scenario | Obs ΔSUV (%) | PSF-Gibbs (%) | PVE/Res (%) | CT-AC (%) | True Bio ΔSUV (%) | Verdict | PERCIST Gate |
|---|---:|---:|---:|---:|---:|---|---|
| Scanner Switch (Standard OSEM -> SiPM PSF-OSEM) | +15.6% | +14.3% | +1.3% | +0.0% | +0.0% | `SCANNER_PSF_GIBBS_ARTIFACT` | BLOCKED |
| True Progression (EARL2-Harmonized SiPM) | +40.5% | +0.0% | +0.0% | +0.0% | +40.5% | `TRUE_BIOLOGICAL_PROGRESSION` | PASS |
| CT-AC Respiratory Shift (-11 mm Basal Lung) | +36.6% | +0.0% | +0.0% | +36.6% | +0.0% | `CT_AC_RESPIRATORY_ARTIFACT` | BLOCKED |
| PVE Diameter Bias (10 mm -> 16 mm Lesion) | +58.8% | +0.0% | +58.7% | +0.0% | +0.1% | `PVE_DELINEATION_ARTIFACT` | PASS |

## 3. Independent Critic-Verifier Audit

- **Overall Gate:** `PASS_WITH_CAUTION`
- **Provenance Ledger Integrity:** `True`
- **Ledger Head SHA-256:** `7e6ab2894960d3ab08caca77a912c4a155a846d30cece33d096e7adaa25b5b5d`

| Check ID | Status | Measured | Threshold | Rationale |
|---|---|---|---|---|
| `PROVENANCE_SHA256_CHAIN` | PASS | 24 records verified=True | True | Every calculation must link deterministically in the SHA-256 provenance ledger. |
| `FLEET_CORRIDOR_CONVERGENCE` | PASS | 4/4 feasible (max_viol=0.00000) | 100% feasible, max_viol <= 0.001 | All fleet scanners must satisfy their target EARL2/QIBA/MIRD-23 RC corridors. |
| `DELTA_SUV_ATTRIBUTION_CONSERVATION` | PASS | max_err=0.0000% | <= 0.15% | Decomposed physical + biological components must sum to observed delta-SUV. |
| `GTM_EXACT_REPROJECTION_INVARIANT` | PASS | residual=0.00e+00 | <= 1.0e-06 | Forward re-projection of GTM-inverted activity must recover observed mean. |
