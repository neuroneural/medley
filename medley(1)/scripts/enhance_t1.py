#!/usr/bin/env python3
"""Tissue-guided T1 intensity enhancement for Medley."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def load_intensity(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D image, got {data.shape}: {path}")
    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data, image


def load_labels(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    image = nib.load(str(path))
    data = np.rint(image.get_fdata()).astype(np.int32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D label map, got {data.shape}: {path}")
    return data, image


def same_geometry(a: nib.Nifti1Image, b: nib.Nifti1Image) -> bool:
    return a.shape == b.shape and np.allclose(a.affine, b.affine, atol=1e-4)


def save_float(data: np.ndarray, reference: nib.Nifti1Image, path: Path) -> None:
    output = np.asarray(data, dtype=np.float32)
    header = reference.header.copy()
    header.set_data_dtype(np.float32)
    header.set_slope_inter(1.0, 0.0)
    header["cal_min"] = float(np.min(output))
    header["cal_max"] = float(np.max(output))
    image = nib.Nifti1Image(output, reference.affine.copy(), header)
    _, qcode = reference.get_qform(coded=True)
    _, scode = reference.get_sform(coded=True)
    image.set_qform(reference.affine, int(qcode) if qcode else 1)
    image.set_sform(reference.affine, int(scode) if scode else 1)
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(image, str(path))


def normalize_archived_formula(data: np.ndarray) -> np.ndarray:
    brain = data > 0
    if not np.any(brain):
        raise ValueError("Input image has no positive voxels")
    mean = float(np.mean(data[brain]))
    std = float(np.std(data[brain]))
    if std == 0:
        raise ValueError("Input image has zero standard deviation in brain")

    z = (data - mean) / std
    shifted = z - float(np.min(z))
    denom = float(np.max(shifted))
    if denom == 0:
        raise ValueError("Normalized image has zero range")
    scaled = shifted / denom * 255.0
    scaled[data == 0] = 0.0
    return scaled.astype(np.float32)


def make_template(tissue: np.ndarray, normalized: np.ndarray, modality: str) -> np.ndarray:
    mask = tissue != 0
    if not np.any(mask):
        raise ValueError("Tissue map contains no nonzero voxels")
    mean = float(np.mean(normalized[mask]))
    if modality == "T1":
        targets = {1: mean * 1.0, 2: mean * 1.9, 3: mean * 2.5}
    else:
        targets = {1: mean * 4.0, 2: mean * 2.0, 3: mean * 1.0}

    template = np.zeros_like(normalized, dtype=np.float32)
    for label, value in targets.items():
        template[tissue == label] = value
    return template


def enhance(template: np.ndarray, normalized: np.ndarray, factor: float) -> np.ndarray:
    output = normalized + template
    missing_tissue = (template == 0) & (normalized != 0)
    output[missing_tissue] += factor * normalized[missing_tissue]
    return output.astype(np.float32)


def process_modality(
    subject: Path,
    modality: str,
    image_name: str,
    tissue_name: str,
    norm_name: str,
    template_name: str,
    enhanced_name: str,
    factor: float,
    overwrite: bool,
    required: bool,
) -> None:
    image_path = subject / image_name
    tissue_path = subject / tissue_name
    outputs = [subject / norm_name, subject / template_name, subject / enhanced_name]

    if not image_path.exists():
        if required:
            raise FileNotFoundError(image_path)
        print(f"SKIP {modality}: missing {image_path.name}")
        return
    if not tissue_path.exists():
        if required:
            raise FileNotFoundError(tissue_path)
        print(f"SKIP {modality}: missing {tissue_path.name}")
        return
    if all(path.exists() for path in outputs) and not overwrite:
        enhanced_dtype = nib.load(str(outputs[2])).get_data_dtype()
        if np.dtype(enhanced_dtype) == np.dtype(np.uint8):
            raise RuntimeError(
                f"Existing {outputs[2].name} is uint8 and may contain wrapped "
                "intensities from the archived script. Rerun with --overwrite."
            )
        print(f"SKIP {modality}: safe outputs already exist")
        return

    intensity, intensity_img = load_intensity(image_path)
    tissue, tissue_img = load_labels(tissue_path)
    if not same_geometry(intensity_img, tissue_img):
        raise ValueError(
            f"Geometry mismatch between {image_path.name} and {tissue_path.name}"
        )

    normalized = normalize_archived_formula(intensity)
    template = make_template(tissue, normalized, modality)
    enhanced = enhance(template, normalized, factor)

    save_float(normalized, intensity_img, outputs[0])
    save_float(template, intensity_img, outputs[1])
    save_float(enhanced, intensity_img, outputs[2])
    print(
        f"DONE {modality}: {enhanced_name}; range="
        f"[{float(enhanced.min()):.3f}, {float(enhanced.max()):.3f}]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--with-t2", action="store_true")
    args = parser.parse_args()

    if not args.subject_dir.is_dir():
        raise SystemExit(f"Subject directory does not exist: {args.subject_dir}")

    process_modality(
        args.subject_dir,
        modality="T1",
        image_name="T1.nii.gz",
        tissue_name="T1-skullstripped-tissue.nii.gz",
        norm_name="T1-norm-15.nii.gz",
        template_name="tissue-T1-17.nii.gz",
        enhanced_name="T1_enhanced-18.nii.gz",
        factor=2.5,
        overwrite=args.overwrite,
        required=True,
    )

    if args.with_t2:
        process_modality(
            args.subject_dir,
            modality="T2",
            image_name="T2.nii.gz",
            tissue_name="T2-skullstripped-tissue.nii.gz",
            norm_name="T2-norm-15.nii.gz",
            template_name="tissue-T2-15.nii.gz",
            enhanced_name="T2_enhanced-15.nii.gz",
            factor=1.0,
            overwrite=args.overwrite,
            required=False,
        )


if __name__ == "__main__":
    main()
