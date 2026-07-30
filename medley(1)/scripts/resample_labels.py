#!/usr/bin/env python3
"""Nearest-neighbor label resampling to a reference NIfTI grid."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("label", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        print(f"EXISTS: {args.output}")
        return
    label_img = nib.load(str(args.label))
    reference_img = nib.load(str(args.reference))

    if label_img.shape == reference_img.shape and np.allclose(
        label_img.affine, reference_img.affine, atol=1e-4
    ):
        data = np.rint(label_img.get_fdata()).astype(np.int32)
    else:
        resampled = resample_from_to(label_img, reference_img, order=0, mode="constant", cval=0)
        data = np.rint(resampled.get_fdata()).astype(np.int32)

    header = reference_img.header.copy()
    header.set_data_dtype(np.int32)
    header.set_slope_inter(1.0, 0.0)
    out_img = nib.Nifti1Image(data, reference_img.affine.copy(), header)
    _, qcode = reference_img.get_qform(coded=True)
    _, scode = reference_img.get_sform(coded=True)
    out_img.set_qform(reference_img.affine, int(qcode) if qcode else 1)
    out_img.set_sform(reference_img.affine, int(scode) if scode else 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(args.output))
    print(f"SAVED: {args.output}")


if __name__ == "__main__":
    main()
