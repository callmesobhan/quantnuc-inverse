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


def build_synthetic_dicom_series(
    volume_3d: np.ndarray,
    output_dir: str | Path,
    *,
    modality: Literal["PT", "NM", "CT"] = "PT",
    voxel_size_mm: tuple[float, float, float] = (2.0, 2.0, 2.0),
    z_origin_mm: float = 0.0,
    injected_dose_bq: float = 3.0e8,
    injection_time_hhmmss: str = "090000",
    scan_time_hhmmss: str = "100000",
    half_life_s: float = 6586.2,
    patient_weight_kg: float = 70.0,
    decay_correction: str = "START",
    units: str = "BQML",
    patient_id: str = "SERIES_3D_DRO",
) -> list[Path]:
    """Write a compliant multi-slice 3D DICOM series directory with ImagePositionPatient z-ordering."""
    vol = np.asarray(volume_3d, dtype=np.float64)
    if vol.ndim != 3:
        raise ValueError("volume_3d must be a 3D array of shape (nz, ny, nx).")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    series_uid = generate_uid()
    study_uid = generate_uid()
    dz, dy, dx = voxel_size_mm
    written: list[Path] = []

    for z_idx in range(vol.shape[0]):
        fpath = out_dir / f"{modality.lower()}_slice_{z_idx:04d}.dcm"
        if modality == "CT":
            build_synthetic_ct_dicom(vol[z_idx], fpath, voxel_size_mm=voxel_size_mm, patient_id=patient_id)
        else:
            build_synthetic_nucmed_dicom(
                vol[z_idx],
                fpath,
                modality=modality,
                voxel_size_mm=voxel_size_mm,
                injected_dose_bq=injected_dose_bq,
                injection_time_hhmmss=injection_time_hhmmss,
                scan_time_hhmmss=scan_time_hhmmss,
                half_life_s=half_life_s,
                patient_weight_kg=patient_weight_kg,
                decay_correction=decay_correction,
                units=units,
                patient_id=patient_id,
            )
        ds = pydicom.dcmread(str(fpath))
        ds.StudyInstanceUID = study_uid
        ds.SeriesInstanceUID = series_uid
        ds.InstanceNumber = int(z_idx + 1)
        ds.ImagePositionPatient = [0.0, 0.0, float(z_origin_mm + z_idx * dz)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.save_as(str(fpath), enforce_file_format=True)
        written.append(fpath)

    return written


def load_dicom_series_3d(
    series_dir: str | Path,
    *,
    data_source_tag: str = "CLINICAL_OR_PHANTOM_DICOM_SERIES",
) -> QuantDicomVolume:
    """Load a multi-slice 3D PET (`PT`) or SPECT (`NM`) DICOM directory sorted by ImagePositionPatient[2]."""
    sdir = Path(series_dir)
    dcm_files = sorted(p for p in sdir.glob("*.dcm") if p.is_file())
    if not dcm_files:
        raise AbstentionError(
            AbstentionCode.UNCALIBRATED_SUV_METADATA,
            f"No .dcm slice files found in directory '{sdir}'.",
            details={"series_dir": str(sdir)},
        )

    indexed_slices: list[tuple[float, Path]] = []
    for fpath in dcm_files:
        ds = pydicom.dcmread(str(fpath), stop_before_pixels=True)
        ipp = getattr(ds, "ImagePositionPatient", None)
        z_pos = float(ipp[2]) if (ipp is not None and len(ipp) >= 3) else float(getattr(ds, "InstanceNumber", 0))
        indexed_slices.append((z_pos, fpath))
    indexed_slices.sort(key=lambda item: item[0])

    loaded_slices = [load_quantitative_nucmed_dicom(p, data_source_tag=data_source_tag) for _, p in indexed_slices]
    ref = loaded_slices[0]

    act_3d = np.stack([s.activity_bq_ml for s in loaded_slices], axis=0)
    suvbw_3d = np.stack([s.suvbw for s in loaded_slices], axis=0)
    suvlbm_3d = np.stack([s.suvlbm for s in loaded_slices], axis=0)

    if len(indexed_slices) >= 2:
        dz_est = float(np.median(np.diff([z for z, _ in indexed_slices])))
        dz_val = dz_est if dz_est > 0.0 else ref.voxel_size_mm[0]
    else:
        dz_val = ref.voxel_size_mm[0]
    vox = (dz_val, ref.voxel_size_mm[1], ref.voxel_size_mm[2])

    prov = build_provenance_record(
        operation="dicom_3d_series_quantitative_load",
        inputs={"series_dir": sdir.name, "n_slices": len(loaded_slices), "modality": ref.modality},
        outputs={"shape": list(act_3d.shape), "voxel_size_mm": list(vox), "max_suvbw": float(np.max(suvbw_3d))},
        data_source_tag=data_source_tag,
        citations=["Boellaard et al. (2015) Eur J Nucl Med Mol Imaging 42:328-354, DOI:10.1007/s00259-014-2961-x"],
    )

    return QuantDicomVolume(
        modality=ref.modality,
        activity_bq_ml=act_3d,
        suvbw=suvbw_3d,
        suvlbm=suvlbm_3d,
        voxel_size_mm=vox,
        injected_dose_bq=ref.injected_dose_bq,
        decay_corrected_dose_bq=ref.decay_corrected_dose_bq,
        decay_elapsed_s=ref.decay_elapsed_s,
        half_life_s=ref.half_life_s,
        patient_weight_kg=ref.patient_weight_kg,
        patient_lbm_kg=ref.patient_lbm_kg,
        decay_correction_tag=ref.decay_correction_tag,
        units=ref.units,
        provenance=prov,
    )


def load_ct_dicom_series_3d(series_dir: str | Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a multi-slice 3D CT DICOM series directory sorted by ImagePositionPatient[2]."""
    sdir = Path(series_dir)
    dcm_files = sorted(p for p in sdir.glob("*.dcm") if p.is_file())
    if not dcm_files:
        raise ValueError(f"No .dcm CT slices found in '{sdir}'.")

    indexed: list[tuple[float, Path]] = []
    for p in dcm_files:
        ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        ipp = getattr(ds, "ImagePositionPatient", None)
        z_pos = float(ipp[2]) if (ipp is not None and len(ipp) >= 3) else float(getattr(ds, "InstanceNumber", 0))
        indexed.append((z_pos, p))
    indexed.sort(key=lambda item: item[0])

    slices, vox_ref = [], (2.0, 2.0, 2.0)
    for _, p in indexed:
        hu_slc, vox_ref = load_ct_hu_slice(p)
        slices.append(hu_slc)
    dz_val = float(np.median(np.diff([z for z, _ in indexed]))) if len(indexed) >= 2 else vox_ref[0]
    if dz_val <= 0.0:
        dz_val = vox_ref[0]
    return np.stack(slices, axis=0), (dz_val, vox_ref[1], vox_ref[2])


def resample_ct_to_pet_grid(
    ct_hu_3d: np.ndarray,
    ct_voxel_size_mm: tuple[float, float, float],
    pet_shape: tuple[int, int, int],
    pet_voxel_size_mm: tuple[float, float, float],
    *,
    shift_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Resample a 3D CT HU volume onto a PET/SPECT voxel grid with optional 3D shift (dz, dy, dx) in mm."""
    from scipy.ndimage import affine_transform

    ct = np.asarray(ct_hu_3d, dtype=np.float64)
    if ct.ndim != 3:
        raise ValueError("ct_hu_3d must be a 3D volume.")

    # Map output PET index i_pet -> physical coord relative to center -> input CT index i_ct
    scale = np.array(
        [pet_voxel_size_mm[k] / ct_voxel_size_mm[k] for k in range(3)],
        dtype=np.float64,
    )
    pet_center = np.array([(pet_shape[k] - 1) / 2.0 for k in range(3)], dtype=np.float64)
    ct_center = np.array([(ct.shape[k] - 1) / 2.0 for k in range(3)], dtype=np.float64)
    shift_ct_vox = np.array([shift_mm[k] / ct_voxel_size_mm[k] for k in range(3)], dtype=np.float64)

    offset = ct_center - scale * pet_center + shift_ct_vox
    resampled = affine_transform(
        ct,
        matrix=np.diag(scale),
        offset=offset,
        output_shape=pet_shape,
        order=1,
        mode="constant",
        cval=-1000.0,
    )
    return resampled.astype(np.float64, copy=False)
