#!/usr/bin/env python3
"""Validate shape and affine consistency of Medley input volumes."""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject-dir", type=Path, required=True)
    parser.add_argument("--files", nargs="+", required=True)
    args = parser.parse_args()

    images = []
    for name in args.files:
        path = args.subject_dir / name
        if not path.exists():
            raise SystemExit(f"MISSING: {path}")
        img = nib.load(str(path))
        images.append((name, img))

    ref_name, ref = images[0]
    failures = []
    for name, img in images[1:]:
        if img.shape != ref.shape:
            failures.append(f"shape {name}={img.shape}, {ref_name}={ref.shape}")
        if not np.allclose(img.affine, ref.affine, atol=1e-4):
            failures.append(f"affine mismatch: {name} vs {ref_name}")

    if failures:
        raise SystemExit("GEOMETRY ERROR: " + "; ".join(failures))
    print(f"VALID: {args.subject_dir} ({len(images)} aligned files)")


if __name__ == "__main__":
    main()
