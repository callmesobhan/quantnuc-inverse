"""Real pydicom PET (`PT`), SPECT (`NM`), and CT DICOM Loader & Synthetic Round-Trip Builder.

Verifies quantitative nuclear medicine DICOM headers per EANM v2.0 / QIBA FDG-PET Profile:
- `RadiopharmaceuticalInformationSequence` (`RadionuclideTotalDose`, `RadiopharmaceuticalStartTime`,
  `RadionuclideHalfLife`)
- `DecayCorrection` (`START` / `ADMIN`; raises `AbstentionError(MISSING_DECAY_CORRECTION)` if invalid)
- `Units` (`BQML`), `RescaleSlope`, `RescaleIntercept`, `PatientWeight`, `PatientSize`, `PatientSex`
- Exact decay-corrected SUVbw (g/mL) and Janmahasatian SUVlbm (lean body mass, PERCIST 1.0) calculation.

Research Software — NOT A MEDICAL DEVICE.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.sequence import Sequence
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from quantnuc_inverse.ir import AbstentionCode, AbstentionError, ProvenanceRecord
from quantnuc_inverse.provenance import (
    assert_calibrated_suv_metadata,
    assert_decay_corrected,
    build_provenance_record,
)

# Standard SOP Class UIDs
PET_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.128"
NM_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.20"
CT_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.2"


def _parse_dicom_tm_seconds(tm_str: str | None) -> float:
    """Parse DICOM Time (TM: HHMMSS[.ffffff]) into seconds since midnight."""
    if not tm_str:
        raise AbstentionError(
            AbstentionCode.UNCALIBRATED_SUV_METADATA,
            "Missing DICOM timestamp required for radiopharmaceutical decay correction.",
            details={"time_value": tm_str},
        )
    s = str(tm_str).strip()
    if len(s) < 6:
        raise AbstentionError(
            AbstentionCode.UNCALIBRATED_SUV_METADATA,
            f"Malformed DICOM TM timestamp '{s}' (expected HHMMSS[.ffffff]).",
            details={"time_value": s},
        )
    hh = int(s[0:2])
    mm = int(s[2:4])
    ss = float(s[4:])
    return hh * 3600.0 + mm * 60.0 + ss


def compute_janmahasatian_lbm_kg(
    weight_kg: float,
    height_m: float,
    sex: str = "M",
) -> float:
    """Compute Lean Body Mass (kg) via Janmahasatian et al. (2005) formula recommended by PERCIST/EANM."""
    bmi = weight_kg / max(height_m**2, 0.25)
    if sex.strip().upper().startswith("F"):
        return (9270.0 * weight_kg) / (8780.0 + 244.0 * bmi)
    return (9270.0 * weight_kg) / (6680.0 + 216.0 * bmi)


@dataclass(frozen=True)
class QuantDicomVolume:
    """Loaded and quantitatively calibrated PET/SPECT activity and SUV volume."""

    modality: str
    activity_bq_ml: np.ndarray
    suvbw: np.ndarray
    suvlbm: np.ndarray
    voxel_size_mm: tuple[float, float, float]
    injected_dose_bq: float
    decay_corrected_dose_bq: float
    decay_elapsed_s: float
    half_life_s: float
    patient_weight_kg: float
    patient_lbm_kg: float
    decay_correction_tag: str
    units: str
    provenance: ProvenanceRecord


def load_quantitative_nucmed_dicom(
    path: str | Path,
    *,
    data_source_tag: str = "CLINICAL_OR_PHANTOM_DICOM",
) -> QuantDicomVolume:
    """Load a PET (`PT`) or SPECT (`NM`) DICOM file, verify calibration/decay gates, and compute SUVbw/SUVlbm."""
    dcm_path = Path(path)
    ds = pydicom.dcmread(str(dcm_path))
    modality = str(getattr(ds, "Modality", "")).strip().upper()
    if modality not in {"PT", "NM"}:
        raise AbstentionError(
            AbstentionCode.UNCALIBRATED_SUV_METADATA,
            f"Expected PET ('PT') or SPECT ('NM') DICOM modality; got '{modality}'.",
            details={"modality": modality, "path": str(dcm_path)},
        )

    decay_corr = getattr(ds, "DecayCorrection", None)
    assert_decay_corrected(str(decay_corr) if decay_corr is not None else None, modality=modality)

    units = getattr(ds, "Units", "BQML" if modality == "NM" else None)
    slope = float(getattr(ds, "RescaleSlope", 0.0)) if hasattr(ds, "RescaleSlope") else None
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    weight_kg = float(getattr(ds, "PatientWeight", 0.0)) if hasattr(ds, "PatientWeight") else None
    height_m = float(getattr(ds, "PatientSize", 1.70)) if hasattr(ds, "PatientSize") else 1.70
    sex = str(getattr(ds, "PatientSex", "M"))

    rph_seq = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
    if not rph_seq or len(rph_seq) == 0:
        raise AbstentionError(
            AbstentionCode.UNCALIBRATED_SUV_METADATA,
            "Missing RadiopharmaceuticalInformationSequence in DICOM header.",
            details={"path": str(dcm_path)},
        )
    rph = rph_seq[0]
    injected_dose_bq = (
        float(getattr(rph, "RadionuclideTotalDose", 0.0))
        if hasattr(rph, "RadionuclideTotalDose")
        else None
    )
    half_life_s = (
        float(getattr(rph, "RadionuclideHalfLife", 0.0))
        if hasattr(rph, "RadionuclideHalfLife")
        else None
    )

    assert_calibrated_suv_metadata(
        units=str(units) if units is not None else None,
        patient_weight_kg=weight_kg,
        injected_dose_bq=injected_dose_bq,
        half_life_s=half_life_s,
        rescale_slope=slope,
    )
    assert weight_kg is not None
    assert injected_dose_bq is not None
    assert half_life_s is not None
    assert slope is not None

    inj_time_str = getattr(rph, "RadiopharmaceuticalStartTime", None)
    scan_time_str = getattr(ds, "SeriesTime", None) or getattr(ds, "AcquisitionTime", None)
    t_inj = _parse_dicom_tm_seconds(inj_time_str)
    t_scan = _parse_dicom_tm_seconds(scan_time_str)
    dt_s = t_scan - t_inj
    if dt_s < 0.0:
        raise AbstentionError(
            AbstentionCode.UNCALIBRATED_SUV_METADATA,
            f"Negative injection-to-scan elapsed time ({dt_s:.1f} s) in DICOM timestamps.",
            details={"injection_time": inj_time_str, "scan_time": scan_time_str},
        )

    decay_tag = str(decay_corr or "START").strip().upper()
    if decay_tag == "ADMIN":
        # Image pixel values already decay-corrected back to injection time
        decay_corrected_dose_bq = injected_dose_bq
    else:
        # Standard PET DecayCorrection == "START": decay injected dose to SeriesTime
        decay_corrected_dose_bq = injected_dose_bq * (2.0 ** (-dt_s / half_life_s))

    raw_pixels = ds.pixel_array.astype(np.float64)
    activity_bq_ml = raw_pixels * slope + intercept

    # SUVbw = Activity [Bq/mL] * PatientWeight [g] / DecayCorrectedDose [Bq]
    suvbw = activity_bq_ml * (weight_kg * 1000.0) / decay_corrected_dose_bq
    lbm_kg = compute_janmahasatian_lbm_kg(weight_kg, height_m, sex=sex)
    suvlbm = activity_bq_ml * (lbm_kg * 1000.0) / decay_corrected_dose_bq

    ps = getattr(ds, "PixelSpacing", [2.0, 2.0])
    dz = float(getattr(ds, "SliceThickness", 2.0))
    voxel_size = (dz, float(ps[0]), float(ps[1]))

    prov = build_provenance_record(
        operation="dicom_quantitative_suv_load",
        inputs={
            "path": dcm_path.name,
            "modality": modality,
            "decay_correction": decay_tag,
            "units": str(units),
            "injected_dose_bq": injected_dose_bq,
            "dt_s": dt_s,
            "weight_kg": weight_kg,
        },
        outputs={
            "shape": list(activity_bq_ml.shape),
            "max_suvbw": float(np.max(suvbw)),
            "mean_suvbw": float(np.mean(suvbw)),
            "decay_corrected_dose_bq": decay_corrected_dose_bq,
        },
        data_source_tag=data_source_tag,
        citations=["Boellaard et al. (2015) Eur J Nucl Med Mol Imaging 42:328-354, DOI:10.1007/s00259-014-2961-x"],
    )

    return QuantDicomVolume(
        modality=modality,
        activity_bq_ml=activity_bq_ml,
        suvbw=suvbw,
        suvlbm=suvlbm,
        voxel_size_mm=voxel_size,
        injected_dose_bq=injected_dose_bq,
        decay_corrected_dose_bq=decay_corrected_dose_bq,
        decay_elapsed_s=dt_s,
        half_life_s=half_life_s,
        patient_weight_kg=weight_kg,
        patient_lbm_kg=lbm_kg,
        decay_correction_tag=decay_tag,
        units=str(units),
        provenance=prov,
    )


def build_synthetic_nucmed_dicom(
    activity_bq_ml: np.ndarray,
    output_path: str | Path,
    *,
    modality: Literal["PT", "NM"] = "PT",
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    injected_dose_bq: float = 3.0e8,
    injection_time_hhmmss: str = "090000",
    scan_time_hhmmss: str = "100000",
    half_life_s: float = 6586.2,
    patient_weight_kg: float = 70.0,
    patient_size_m: float = 1.75,
    patient_sex: str = "M",
    decay_correction: str = "START",
    units: str = "BQML",
    patient_id: str = "NEMA_NU2_2018_DRO",
    series_description: str = "quantnuc-inverse synthetic NEMA DRO",
) -> Path:
    """Create a compliant, near-lossless 16-bit signed PET (`PT`) or SPECT (`NM`) DICOM file."""
    arr = np.asarray(activity_bq_ml, dtype=np.float64)
    if arr.ndim == 3:
        # Store central axial slice for single-file DICOM or full multi-frame
        is_multiframe = True
        rows, cols = arr.shape[1], arr.shape[2]
        n_frames = arr.shape[0]
    elif arr.ndim == 2:
        is_multiframe = False
        rows, cols = arr.shape[0], arr.shape[1]
        n_frames = 1
    else:
        raise ValueError("activity_bq_ml must be a 2D or 3D array.")

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    span = max(max_val - min_val, 1e-6)
    # Maximize 15-bit integer dynamic range (0..32000) for high-precision round-trip (<0.005% error)
    slope = span / 32000.0
    intercept = min_val
    quantized = np.rint((arr - intercept) / slope).clip(0, 32000).astype(np.uint16)

    sop_class = PET_IMAGE_STORAGE_UID if modality == "PT" else NM_IMAGE_STORAGE_UID
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = sop_class
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = FileDataset(str(out), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = sop_class
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = modality
    ds.PatientID = patient_id
    ds.PatientName = f"SYNTHETIC^{patient_id}"
    ds.PatientWeight = patient_weight_kg
    ds.PatientSize = patient_size_m
    ds.PatientSex = patient_sex
    ds.SeriesDescription = series_description
    ds.SeriesTime = scan_time_hhmmss
    ds.AcquisitionTime = scan_time_hhmmss
    ds.StudyDate = "20260101"
    ds.SeriesDate = "20260101"
    ds.Units = units
    ds.DecayCorrection = decay_correction
    ds.RescaleSlope = float(slope)
    ds.RescaleIntercept = float(intercept)
    ds.PixelSpacing = [float(voxel_size_mm[1]), float(voxel_size_mm[2])]
    ds.SliceThickness = float(voxel_size_mm[0])

    rph = Dataset()
    rph.Radiopharmaceutical = "Fluorodeoxyglucose F^18" if modality == "PT" else "Lutetium Lu^177"
    rph.RadiopharmaceuticalStartTime = injection_time_hhmmss
    rph.RadionuclideTotalDose = float(injected_dose_bq)
    rph.RadionuclideHalfLife = float(half_life_s)
    ds.RadiopharmaceuticalInformationSequence = Sequence([rph])

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows = int(rows)
    ds.Columns = int(cols)
    if is_multiframe:
        ds.NumberOfFrames = int(n_frames)
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = quantized.tobytes()

    ds.save_as(str(out), enforce_file_format=True)
    return out


def build_synthetic_ct_dicom(
    hu_array: np.ndarray,
    output_path: str | Path,
    *,
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    patient_id: str = "NEMA_NU2_2018_DRO",
) -> Path:
    """Create a compliant paired CT attenuation-map DICOM file in Hounsfield Units (HU)."""
    arr = np.asarray(hu_array, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("hu_array must be a 2D slice array.")

    slope = 1.0
    intercept = -1024.0
    stored = np.rint(arr - intercept).clip(0, 65535).astype(np.uint16)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = CT_IMAGE_STORAGE_UID
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ds = FileDataset(str(out), {}, file_meta=file_meta, preamble=b"\0" * 128)
    ds.SOPClassUID = CT_IMAGE_STORAGE_UID if CT_IMAGE_STORAGE_UID else SecondaryCaptureImageStorage
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Modality = "CT"
    ds.PatientID = patient_id
    ds.PatientName = f"SYNTHETIC^{patient_id}"
    ds.RescaleSlope = slope
    ds.RescaleIntercept = intercept
    ds.PixelSpacing = [float(voxel_size_mm[1]), float(voxel_size_mm[2])]
    ds.SliceThickness = float(voxel_size_mm[0])
    ds.Rows = int(arr.shape[0])
    ds.Columns = int(arr.shape[1])
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.PixelData = stored.tobytes()

    ds.save_as(str(out), enforce_file_format=True)
    return out


def load_ct_hu_slice(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a CT DICOM slice and convert stored pixels to Hounsfield Units (HU)."""
    ds = pydicom.dcmread(str(path))
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    hu = ds.pixel_array.astype(np.float64) * slope + intercept
    ps = getattr(ds, "PixelSpacing", [2.0, 2.0])
    dz = float(getattr(ds, "SliceThickness", 2.0))
    return hu, (dz, float(ps[0]), float(ps[1]))
