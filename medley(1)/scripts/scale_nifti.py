#!/usr/bin/env python3
"""Center-preserving NIfTI scaling for Medley images and labels."""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform


def parse_scale(value: str) -> float:
    try:
        parts = value.strip().split("/")
        if len(parts) == 1:
            scale = Fraction(parts[0])
        elif len(parts) == 2:
            scale = Fraction(parts[0]) / Fraction(parts[1])
        else:
            raise ValueError
        if scale <= 0:
            raise ValueError
        return float(scale)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(f"Invalid positive scale: {value}") from exc


def scale_image(input_path: Path, output_path: Path, scale: float, order: int) -> None:
    image = nib.load(str(input_path))
    data = image.get_fdata(dtype=np.float32)
    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got {data.shape}: {input_path}")
    if not np.any(np.isfinite(data)):
        raise ValueError(f"No finite data in {input_path}")

    original_min = float(np.nanmin(data))
    original_max = float(np.nanmax(data))
    shape = np.asarray(data.shape, dtype=np.float64)
    center = (shape - 1.0) / 2.0
    matrix = np.eye(3, dtype=np.float64) / scale
    offset = center - matrix @ center

    result = affine_transform(
        data,
        matrix=matrix,
        offset=offset,
        output_shape=data.shape,
        order=order,
        mode="constant",
        cval=0.0,
        prefilter=(order > 1),
    )
    result = np.nan_to_num(result, nan=0.0, posinf=original_max, neginf=original_min)

    header = image.header.copy()
    if order == 0:
        output = np.rint(result).astype(np.int32)
        header.set_data_dtype(np.int32)
    else:
        output = np.clip(result, original_min, original_max).astype(np.float32)
        header.set_data_dtype(np.float32)

    header.set_slope_inter(1.0, 0.0)
    header["cal_min"] = float(np.min(output))
    header["cal_max"] = float(np.max(output))
    out_img = nib.Nifti1Image(output, image.affine.copy(), header)
    _, qcode = image.get_qform(coded=True)
    _, scode = image.get_sform(coded=True)
    out_img.set_qform(image.affine, int(qcode) if qcode else 1)
    out_img.set_sform(image.affine, int(scode) if scode else 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--scale", required=True, type=parse_scale)
    parser.add_argument("--order", required=True, type=int, choices=range(0, 6))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Missing input: {args.input}")
    if args.output.exists() and not args.overwrite:
        print(f"EXISTS: {args.output}")
        return
    scale_image(args.input, args.output, args.scale, args.order)
    print(f"SAVED: {args.output}")


if __name__ == "__main__":
    main()
