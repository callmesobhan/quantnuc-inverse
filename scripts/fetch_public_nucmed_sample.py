"""Build / Stage Open-License NEMA NU 2-2018 6-Sphere Digital Reference Object (DRO) & Public Sample Metadata.

Generates:
1. Calibrated NEMA NU 2-2018 6-sphere IEC Body Phantom PET (`PT`) and paired CT (`CT`) DICOM files
   (4:1 sphere-to-background ratio, all 6 spheres hot: 10, 13, 17, 22, 28, 37 mm; clearly tagged
   `SYNTHETIC_NEMA_NU2_2018_DRO` in provenance).
2. Copies pydicom's built-in open-license (MIT) CT sample reference (`ct_small.dcm`) if installed
   with its verbatim MIT license notice.
3. Writes `MANIFEST.json` (SHA-256 digests of every staged file), `LICENSE`, and `ATTRIBUTION.md`.

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from quantnuc_inverse.dicom_io import (
    build_synthetic_ct_dicom,
    build_synthetic_nucmed_dicom,
)
from quantnuc_inverse.forward_image_formation import build_nema_iec_phantom_dro
from quantnuc_inverse.ir import RESEARCH_DISCLAIMER

PYDICOM_MIT_LICENSE = """MIT License (for pydicom bundled sample test dataset)

Copyright (c) 2008-2026 Darcy Mason and pydicom contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def stage_public_nucmed_samples(output_dir: str | Path = "data/sample_nema_dro") -> Path:
    """Create reproducible NEMA NU 2-2018 DRO DICOMs and stage open-license sample metadata."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Exact NEMA NU 2-2018 6-sphere IEC Body Phantom DRO (central axial slice PET + CT)
    dro = build_nema_iec_phantom_dro(
        voxel_size_mm=(2.0, 2.0, 2.0),
        sphere_to_bg_ratio=4.0,
        background_activity_bq_ml=5300.0,
    )
    mid_z = dro.true_volume.shape[0] // 2
    pet_slice = dro.true_volume[mid_z]
    pet_path = build_synthetic_nucmed_dicom(
        pet_slice,
        out / "nema_nu2_2018_dro_pt.dcm",
        modality="PT",
        voxel_size_mm=(2.0, 2.0, 2.0),
        injected_dose_bq=3.0e8,
        injection_time_hhmmss="090000",
        scan_time_hhmmss="100000",
        half_life_s=6586.2,
        patient_weight_kg=70.0,
        decay_correction="START",
        units="BQML",
        patient_id="NEMA_NU2_2018_DRO",
    )

    # Paired CT attenuation slice in HU (air=-1000, water/PMMA body=0, lung insert=-720)
    hu_slice = np.full_like(pet_slice, -1000.0)
    hu_slice[dro.body_mask[mid_z] > 0.5] = 0.0
    # Central 50 mm lung insert
    ny, nx = pet_slice.shape
    y = (np.arange(ny) - (ny - 1) / 2.0) * 2.0
    x = (np.arange(nx) - (nx - 1) / 2.0) * 2.0
    yy, xx = np.meshgrid(y, x, indexing="ij")
    hu_slice[(xx**2 + yy**2) <= 25.0**2] = -720.0
    ct_path = build_synthetic_ct_dicom(
        hu_slice,
        out / "nema_nu2_2018_dro_ct.dcm",
        voxel_size_mm=(2.0, 2.0, 2.0),
        patient_id="NEMA_NU2_2018_DRO",
    )

    staged_files: list[dict[str, str]] = [
        {
            "file": pet_path.name,
            "data_source_tag": "SYNTHETIC_NEMA_NU2_2018_DRO",
            "license": "Apache-2.0",
            "description": "NEMA NU 2-2018 6-sphere (10-37 mm, 4:1 ratio) IEC body phantom PET DRO slice (BQML, START decay-corrected).",
            "sha256": _sha256_file(pet_path),
        },
        {
            "file": ct_path.name,
            "data_source_tag": "SYNTHETIC_NEMA_NU2_2018_DRO",
            "license": "Apache-2.0",
            "description": "Paired NEMA NU 2-2018 IEC body phantom CT attenuation map slice in HU.",
            "sha256": _sha256_file(ct_path),
        },
    ]

    # 2. Stage pydicom built-in open-license (MIT) CT sample if available in environment
    try:
        from pydicom import examples

        src_ct = Path(examples.get_path("ct"))
        if src_ct.exists():
            dst_ct = out / "pydicom_public_ct_sample.dcm"
            shutil.copy2(src_ct, dst_ct)
            staged_files.append(
                {
                    "file": dst_ct.name,
                    "data_source_tag": "UPSTREAM_PYDICOM_OPEN_TEST_DATA",
                    "license": "MIT",
                    "upstream_source": "pydicom.examples.ct (https://github.com/pydicom/pydicom)",
                    "description": "Publicly distributed small CT DICOM sample bundled with pydicom (MIT License).",
                    "sha256": _sha256_file(dst_ct),
                }
            )
    except Exception:
        pass

    # 3. Verbatim upstream LICENSE & ATTRIBUTION.md & SHA-256 MANIFEST.json
    (out / "LICENSE").write_text(PYDICOM_MIT_LICENSE, encoding="utf-8")
    attribution = f"""# Data Attribution & Provenance

> **Notice:** {RESEARCH_DISCLAIMER}.

## Staged Datasets
1. **NEMA NU 2-2018 6-Sphere IEC Body Phantom Digital Reference Object (`nema_nu2_2018_dro_pt.dcm`, `nema_nu2_2018_dro_ct.dcm`)**
   - **Tag:** `SYNTHETIC_NEMA_NU2_2018_DRO`
   - **Geometry:** NEMA NU 2-2018 standard (all 6 hot spheres: 10, 13, 17, 22, 28, 37 mm; 57.2 mm ring radius; 4:1 sphere-to-background ratio; 50 mm central lung insert).
   - **License:** Apache-2.0 (generated deterministically by `quantnuc-inverse`).
2. **Bundled Open-License CT DICOM Sample (`pydicom_public_ct_sample.dcm`)**
   - **Tag:** `UPSTREAM_PYDICOM_OPEN_TEST_DATA`
   - **Upstream:** `pydicom` (`pydicom.examples.ct`, https://github.com/pydicom/pydicom)
   - **License:** MIT License (verbatim copy included in `LICENSE` in this directory).
"""
    (out / "ATTRIBUTION.md").write_text(attribution, encoding="utf-8")

    manifest = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "disclaimer": RESEARCH_DISCLAIMER,
        "files": staged_files,
    }
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return out


if __name__ == "__main__":
    staged = stage_public_nucmed_samples()
    print(f"Staged NEMA DRO & open-license sample bundle in: {staged.resolve()}")
