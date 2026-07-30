"""Label-map input/output and editing utilities for Medley.

This module provides label-safe NIfTI loading and saving, deterministic label
editing, and optional CuPy acceleration with a NumPy fallback.
"""

import nibabel as nib
import numpy as np

try:
    import cupy as cp
    HAS_CUPY = cp.cuda.runtime.getDeviceCount() > 0
    if not HAS_CUPY:
        cp = None
except Exception:  # allows CPU execution without a usable CUDA device
    cp = None
    HAS_CUPY = False


def get_xp(array):
    """Return cupy for CuPy arrays, otherwise numpy."""
    if HAS_CUPY:
        return cp.get_array_module(array)
    return np


def to_numpy(data):
    """Convert CuPy array to NumPy array when needed."""
    if HAS_CUPY and isinstance(data, cp.ndarray):
        return cp.asnumpy(data)
    return np.asarray(data)


def loadNii(input_path):
    """
    Load a NIfTI label map safely.

    Matches editData2test.py behavior by rounding get_fdata() and converting to int32.
    Returns a CuPy array when CuPy is available, otherwise a NumPy array.
    """
    img = nib.load(input_path)
    data = np.rint(img.get_fdata()).astype(np.int32)

    if HAS_CUPY:
        data = cp.asarray(data)

    affine = img.affine
    header = img.header.copy()
    return data, affine, header


def saveNii(data, affine, header, output_path):
    """
    Save a NIfTI label map safely as uint16.

    Matches editData2test.py behavior and prevents accidental float-label output.
    """
    data = to_numpy(data)
    data = np.rint(data).astype(np.uint16)

    header = header.copy()
    nii_image = nib.Nifti1Image(data, affine, header)
    nii_image.set_data_dtype(np.uint16)
    nib.save(nii_image, output_path)


def editLabels(data, changes):
    """
    Sequential label editing.

    Safe only when new labels do not overlap with old labels.
    For tissue grouping, use group17ToTissue() instead.

    Accepted change formats:
      (begin, end, new): replace begin <= label < end with new
      (value, new): replace label == value with new

    Any other format raises ValueError, matching editData2test.py.
    """
    xp = get_xp(data)
    data = xp.array(data, copy=True)

    for change in changes:
        if len(change) == 3:
            begin, end, new = change
            data = xp.where((data >= begin) & (data < end), new, data)

        elif len(change) == 2:
            value, new = change
            data = xp.where(data == value, new, data)

        else:
            raise ValueError(f"Bad change format: {change}")

    return data


def nonzeroToLabel(data, new):
    """
    Convert every nonzero voxel in a mask to one label, preserving zero background.

    Use this instead of editLabels(data, [(new,)]).
    Example: edited_mask = nonzeroToLabel(mask_data, 24)
    """
    xp = get_xp(data)
    data = xp.array(data, copy=True)
    return xp.where(data != 0, new, data)


# Backward-compatible alias with a more explicit name for masks.
maskToLabel = nonzeroToLabel


def group17ToTissue(data):
    """
    Single-pass grouping for converted 0-17 labels.

    Input:
      1 = Cerebral WM
      2 = Cerebral Cortex / GM
      3 = Lateral Ventricle
      4 = Inferior Lateral Ventricle
      5 = Cerebellum WM
      6 = Cerebellum Cortex
      7 = Thalamus
      8 = Caudate
      9 = Putamen
      10 = Pallidum
      11 = 3rd Ventricle
      12 = 4th Ventricle
      13 = Brainstem
      14 = Hippocampus
      15 = Amygdala
      16 = Accumbens
      17 = VentralDC

    Output:
      0 = ignored background / CSF / ventricles
      1 = white matter
      2 = gray matter
      4 = brainstem / other
    """
    xp = get_xp(data)
    grouped = xp.zeros_like(data, dtype=xp.uint8)

    # White matter
    grouped[xp.isin(data, xp.asarray([1, 5]))] = 1

    # Gray matter
    grouped[xp.isin(data, xp.asarray([2, 6, 7, 8, 9, 10, 14, 15, 16, 17]))] = 2

    # Brainstem / other
    grouped[data == 13] = 4

    # CSF / ventricles stay 0:
    # 3, 4, 11, 12 -> 0

    return grouped


def printLabels(data, name="labels"):
    """Print unique labels and voxel counts."""
    data_np = to_numpy(data)
    vals, counts = np.unique(data_np, return_counts=True)
    print(f"\n{name}")
    for v, c in zip(vals, counts):
        print(f"  label {int(v):2d}: {c}")


def editLabelValue(data1, data2, value, left, right):
    xp = get_xp(data1)
    data2 = xp.asarray(data2) if xp is not np else np.asarray(data2)

    data = xp.where(((data1 == value) & (data2 < 41)), left, data1)
    data = xp.where(((data == value) & (data2 >= 41)), right, data)
    return data


def editLabelValueX(data, x, value, left, right):
    xp = get_xp(data)
    data = xp.array(data, copy=True)

    X = xp.arange(data.shape[0])[:, None, None]
    condition = data == value

    data = xp.where(condition & (X > x), left, data)
    data = xp.where(condition & (X <= x), right, data)
    return data


def editLabelValueYZ(data, y, z, value, new1, new2):
    xp = get_xp(data)
    data = xp.array(data, copy=True)

    Y = xp.arange(data.shape[1])[None, :, None]
    Z = xp.arange(data.shape[2])[None, None, :]

    condition = data == value
    mask = (Y >= y) & (Z >= z)

    data = xp.where(condition & mask, new2, data)
    data = xp.where(condition & (~mask), new1, data)
    return data
