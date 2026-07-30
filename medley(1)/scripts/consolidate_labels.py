#!/usr/bin/env python3
"""Medley anatomy-aware label consolidation (pipeline stages 4-5).

This script does not perform tissue-guided intensity stabilization or
center-preserving enlargement. It consumes the tissue map, brain mask, and
candidate segmentations produced by those earlier pipeline stages and applies
deterministic source-priority, boundary-repair, leakage-suppression, and
topology-cleanup rules.
"""

import os
import glob
import argparse
import numpy as np

USING_CUPY = False
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("No CUDA device available")
    USING_CUPY = True
except Exception:
    import numpy as cp
    if not hasattr(cp, "asnumpy"):
        cp.asnumpy = np.asarray

import label_utils as eD


# FreeSurfer/SynthSeg anatomical labels used by the deterministic source rules.
# The names are included here so that the implementation can be reported in the
# manuscript without relying on unexplained numeric codes.
CEREBELLUM_LABELS = (7, 8, 46, 47)
ASEG_PRIORITY_LATERAL_VENTRICLE_LABELS = (4, 5, 43, 44)
# These ventricular labels use SynthSeg plus the iBEAT CSF check. The lateral
# and inferior lateral ventricles are intentionally excluded because FreeSurfer
# aseg gave better Dice for these structures and therefore retains priority.
CSF_FILTERED_VENTRICLE_LABELS = (14, 15, 72)
SYNTHSEG_PRIORITY_LABELS = (
    # CSF-filtered ventricles and brainstem; lateral ventricle pairs omitted
    14, 15, 72, 16,
    # Thalamus, caudate, putamen, pallidum
    10, 11, 12, 13, 49, 50, 51, 52,
    # Hippocampus, amygdala, ventral diencephalon
    17, 18, 28, 53, 54, 60,
)
ARTIFACT_LABELS = (501, 517)

# ---------- Morphology helpers (GPU-first with CPU fallback) ----------

def _ball(radius: int):
    """3D spherical structuring element (NumPy), used by both SciPy and CuPy backends."""
    import numpy as np
    r = int(radius)
    if r <= 0:
        return np.ones((1, 1, 1), dtype=bool)
    zz, yy, xx = np.ogrid[-r:r+1, -r:r+1, -r:r+1]
    return (xx*xx + yy*yy + zz*zz) <= (r*r)

def _ndimage_backend():
    """
    Returns (backend_name, ndimage_module).
    Prefers cupyx.scipy.ndimage; falls back to scipy.ndimage.
    """
    if USING_CUPY:
        try:
            from cupyx.scipy import ndimage as ndi
            return "cupyx", ndi
        except Exception:
            pass
    from scipy import ndimage as ndi
    return "scipy", ndi

def _to_cpu(x):
    import numpy as np
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x).astype(bool)
    return np.asarray(x).astype(bool)

def _to_gpu(x):
    import numpy as np
    if isinstance(x, cp.ndarray):
        return x
    return cp.asarray(np.asarray(x))

def binary_dilation(x, radius=1):
    backend, ndi = _ndimage_backend()
    st = _ball(radius)
    if backend == "cupyx":
        return ndi.binary_dilation(x.astype(bool), structure=cp.asarray(st))
    y = ndi.binary_dilation(_to_cpu(x), structure=st)
    return _to_gpu(y)

def binary_erosion(x, radius=1):
    backend, ndi = _ndimage_backend()
    st = _ball(radius)
    if backend == "cupyx":
        return ndi.binary_erosion(x.astype(bool), structure=cp.asarray(st))
    y = ndi.binary_erosion(_to_cpu(x), structure=st)
    return _to_gpu(y)

def binary_closing(x, radius=2):
    backend, ndi = _ndimage_backend()
    st = _ball(radius)
    if backend == "cupyx":
        return ndi.binary_closing(x.astype(bool), structure=cp.asarray(st))
    y = ndi.binary_closing(_to_cpu(x), structure=st)
    return _to_gpu(y)

def binary_fill_holes(x):
    backend, ndi = _ndimage_backend()
    if backend == "cupyx":
        return ndi.binary_fill_holes(x.astype(bool))
    y = ndi.binary_fill_holes(_to_cpu(x))
    return _to_gpu(y)

def constrained_region_grow(seed, allowed, steps=2, dilate_radius=1):
    """
    Cautious growth: iteratively dilate seed and keep only voxels within 'allowed'.
    Implements: grow from intersection into union.
    """
    grown = seed.astype(bool)
    allowed = allowed.astype(bool)
    for _ in range(int(steps)):
        grown = grown | (binary_dilation(grown, radius=dilate_radius) & allowed)
    return grown

def smooth_envelope(mask, close_radius=3, erode_radius=0):
    """
    Smooth morphological envelope Ω:
      - closing to smooth boundaries and bridge small gaps
      - fill holes
      - optional slight erosion if you want extra conservative trimming
    """
    env = binary_closing(mask.astype(bool), radius=close_radius)
    env = binary_fill_holes(env)
    if erode_radius and int(erode_radius) > 0:
        env = binary_erosion(env, radius=int(erode_radius))
    return env


def collapse_parcellation_ranges(label_data, brain_mask):
    """Collapse standard FreeSurfer cortical/WM parcellations to aseg classes.

    This implements the manuscript's "range collapsing" step while retaining
    left/right hemisphere identity:
      left cortex  -> Left-Cerebral-Cortex (3)
      right cortex -> Right-Cerebral-Cortex (42)
      left WM      -> Left-Cerebral-White-Matter (2)
      right WM     -> Right-Cerebral-White-Matter (41)
    """
    bm = brain_mask != 0
    rules = (
        (1000, 1999, 3),
        (2000, 2999, 42),
        (3000, 3999, 2),
        (4000, 4999, 41),
    )
    for low, high, target in rules:
        condition = bm & (label_data >= low) & (label_data <= high)
        label_data = apply_coordinates_mask(
            label_data, condition, cp.full_like(label_data, target)
        )
    return label_data


def remove_small_label_islands(label_data, brain_mask, labels, min_size=20):
    """Remove small 26-connected components for selected major-tissue labels.

    Cleanup is deliberately limited to cerebral GM/WM labels. Applying a
    global component-size rule to every aseg class can erase small legitimate
    infant structures.
    """
    if int(min_size) <= 0:
        return label_data

    import numpy as np
    from scipy import ndimage as ndi

    arr = cp.asnumpy(label_data)
    bm = cp.asnumpy(brain_mask != 0)
    structure = np.ones((3, 3, 3), dtype=bool)
    removed_total = 0

    for value in labels:
        components, n_components = ndi.label((arr == int(value)) & bm,
                                              structure=structure)
        if n_components == 0:
            continue
        sizes = np.bincount(components.ravel())
        small_ids = np.flatnonzero((sizes < int(min_size)) &
                                   (np.arange(sizes.size) != 0))
        if small_ids.size == 0:
            continue
        remove = np.isin(components, small_ids)
        removed_total += int(remove.sum())
        arr[remove] = 0

    if removed_total:
        print(f"Removed {removed_total} voxels in small GM/WM islands "
              f"(< {int(min_size)} voxels).")
    return cp.asarray(arr)

# ---------- Basic helpers ----------

def apply_coordinates_mask(target, condition, source):
    idx = cp.where(condition)
    if idx[0].size > 0:
        print(f"Applied edits to {idx[0].size} voxels.")
        target[idx] = source[idx]
    return target

def find_yz(data, value):
    """
    Find minimum y and z coordinates for a label value.
    If the label does not exist, return (None, None).
    """
    indices = cp.where(data == value)
    if indices[0].size == 0:
        return None, None
    y = int(cp.asnumpy(cp.min(indices[1])))
    z = int(cp.asnumpy(cp.min(indices[2])))
    return y, z


def has_voxels(mask):
    """Return True if a CuPy boolean mask contains at least one voxel."""
    return bool(cp.asnumpy(cp.any(mask)))


def voxel_count(mask):
    """Safe Python int voxel count for debug printing."""
    return int(cp.asnumpy(cp.sum(mask)))

# ---------- Optional nonbrain base builder (only if you have a real labelmap) ----------

def edit_nonbrain(input_mask, input_label, output_path):
    """
    Create withoutBrain.nii.gz as a starting label volume:
      - edits mask_csf using your rule
      - edits label_data using your mapping
      - injects edited mask into label where mask != 0
    This ONLY makes sense if input_label is a real whole-head labelmap.
    """
    mask_data, _, _ = eD.loadNii(input_mask)
    label_data, affine, header = eD.loadNii(input_label)

    edited_mask = eD.nonzeroToLabel(mask_data, 24)  # explicit: all nonzero mask voxels -> 24
    changes = [(2, 15, 515), (17, 81, 515), (520, 515), (16, 512)]
    edited_label = eD.editLabels(label_data, changes)

    edited_label[edited_mask != 0] = edited_mask[edited_mask != 0]
    eD.saveNii(edited_label, affine, header, output_path)

# ---------- 2.5-style cortex merge using 3/42 ----------

def cortex_merge_masks_simple(fs_like, ss_like, brain_mask,
                             grow_steps=2, grow_radius=1,
                             env_close_radius=3, env_erode_radius=0,
                             seed_erode_radius=0,
                             debug=False):
    """
    Returns (cortex_L, cortex_R, fs_L, fs_R) boolean masks.

    FS-like cortex is defined by labels 3/42 in fs_like.
    SS cortex is defined by labels 3/42 in ss_like.

    Steps:
      - union/intersection between FS and SS cortex
      - if intersection is empty AND seed_erode_radius>0, seed = erode(union) to enable growth
      - cautious growth from seed into union (dilation constrained by union)
      - envelope Ω from union (closing + fill holes + optional erosion)
      - trim to brain_mask
    """
    fs = fs_like
    ss = ss_like
    bm = (brain_mask != 0)

    # cortex masks (simple)
    fs_L = (fs == 3)
    fs_R = (fs == 42)
    ss_L = (ss == 3)
    ss_R = (ss == 42)

    union_L = fs_L | ss_L
    union_R = fs_R | ss_R
    inter_L = fs_L & ss_L
    inter_R = fs_R & ss_R

    # Seed choice
    seed_L = inter_L
    seed_R = inter_R

    # If no overlap, optionally make a seed by eroding the union
    if (not has_voxels(seed_L)) and seed_erode_radius and int(seed_erode_radius) > 0:
        seed_L = binary_erosion(union_L, radius=int(seed_erode_radius))
    if (not has_voxels(seed_R)) and seed_erode_radius and int(seed_erode_radius) > 0:
        seed_R = binary_erosion(union_R, radius=int(seed_erode_radius))

    grown_L = constrained_region_grow(seed_L, union_L, steps=grow_steps, dilate_radius=grow_radius)
    grown_R = constrained_region_grow(seed_R, union_R, steps=grow_steps, dilate_radius=grow_radius)

    env_L = smooth_envelope(union_L, close_radius=env_close_radius, erode_radius=env_erode_radius)
    env_R = smooth_envelope(union_R, close_radius=env_close_radius, erode_radius=env_erode_radius)

    cortex_L = grown_L & env_L & bm
    cortex_R = grown_R & env_R & bm

    if debug:
        def _s(x): return voxel_count(x)
        print(f"[DEBUG cortex] L: fs={_s(fs_L)} ss={_s(ss_L)} union={_s(union_L)} inter={_s(inter_L)} seed={_s(seed_L)} grown={_s(grown_L)}")
        print(f"[DEBUG cortex] R: fs={_s(fs_R)} ss={_s(ss_R)} union={_s(union_R)} inter={_s(inter_R)} seed={_s(seed_R)} grown={_s(grown_R)}")

    return cortex_L, cortex_R, fs_L, fs_R

def cortex_gap_fill(data3, cortex_L, cortex_R, ss_like, brain_mask, tissue, aseg_like):
    """
    Optional “recover missing cortex”:
      - inside brain_mask
      - GM voxels (tissue==2)
      - where aseg is empty and data3 is empty
      - adjacent to existing cortex (1-voxel neighborhood)
      - SS hemisphere hint to choose side
    """
    bm = (brain_mask != 0)
    gm = (tissue == 2) if tissue is not None else bm

    safe_empty = (data3 == 0)
    safe_zone = bm & gm & safe_empty & (aseg_like == 0)

    near_L = binary_dilation(cortex_L, radius=1)
    near_R = binary_dilation(cortex_R, radius=1)

    ssL = (ss_like == 3)
    ssR = (ss_like == 42)

    fill_L = safe_zone & near_L & (ssL | (~ssR))
    fill_R = safe_zone & near_R & (ssR | (~ssL))

    data3 = apply_coordinates_mask(data3, fill_L, cp.full_like(data3, 3))
    data3 = apply_coordinates_mask(data3, (data3 == 0) & fill_R, cp.full_like(data3, 42))
    return data3

# ---------- Main per-subject pipeline ----------

def process_subject(dir_path, paths,
                    grow_steps=2, grow_radius=1,
                    env_close_radius=3, env_erode_radius=0,
                    seed_erode_radius=0,
                    island_min_size=20,
                    debug=False):
    print(f"\nProcessing: {os.path.basename(dir_path)}")
    full = {k: (None if v is None else os.path.join(dir_path, v)) for k, v in paths.items()}

    # Decide if we have a real whole-head labelmap
    label_path = full.get('label', None)
    has_labelmap = bool(label_path and os.path.exists(label_path) and (not os.path.basename(label_path).startswith("mask-csf")))

    # 1) Build intermediate base (withoutBrain) ONLY if labelmap exists
    if has_labelmap:
        try:
            edit_nonbrain(full['mask_csf'], label_path, full['out_no_brain'])
        except Exception as e:
            print(f"✘ edit_nonbrain error in {dir_path}: {e}")
            return
    else:
        print("⚠ No whole-head labelmap. Skipping edit_nonbrain; starting from SynthSeg inside brain mask (0 outside).")

    # 2) Load required inputs
    load_keys = ['aseg', 'mask', 'tissue', 'T1_ss']
    if has_labelmap:
        load_keys.append('out_no_brain')

    data = {}
    for k in load_keys:
        try:
            data[k] = eD.loadNii(full[k])
        except Exception as e:
            print(f"✘ Failed to load {full[k]}: {e}")
            return

    reference_key = 'mask'
    reference_shape = tuple(data[reference_key][0].shape)
    reference_affine = np.asarray(data[reference_key][1])
    for key in load_keys:
        shape = tuple(data[key][0].shape)
        affine = np.asarray(data[key][1])
        if shape != reference_shape:
            print(f"✘ Geometry mismatch: {key} shape {shape} != {reference_shape}")
            return
        if not np.allclose(affine, reference_affine, atol=1e-4):
            print(f"✘ Geometry mismatch: {key} affine differs from {reference_key}")
            return

    aseg = cp.asarray(data['aseg'][0])
    brain_mask = cp.asarray(data['mask'][0])
    tissue = cp.asarray(data['tissue'][0])
    ss = cp.asarray(data['T1_ss'][0])

    # Reference affine/header for saving
    affine, header = data['mask'][1], data['mask'][2]

    # 2.5) Initialize working label volume
    if has_labelmap:
        data3 = cp.asarray(data['out_no_brain'][0])
        affine, header = data['out_no_brain'][1], data['out_no_brain'][2]
    else:
        data3 = cp.zeros_like(aseg)
        data3 = apply_coordinates_mask(data3, (brain_mask != 0), ss)

    # 3) Inject nonzero FS-like labels inside the brain mask. Cerebellar and
    # selected ventricular labels are excluded because these are restored from
    # SynthSeg using separate anatomy/tissue-constrained rules below. Left and
    # right lateral and inferior lateral ventricles are NOT excluded: they
    # retain aseg priority, matching the version with the better Dice results.
    cerebellum_labels = cp.asarray(CEREBELLUM_LABELS)
    ventricle_labels = cp.asarray(CSF_FILTERED_VENTRICLE_LABELS)
    fs_excluded = cp.concatenate((cerebellum_labels, ventricle_labels))
    data3 = apply_coordinates_mask(
        data3,
        (brain_mask != 0) & (aseg != 0) & (~cp.isin(aseg, fs_excluded)),
        aseg
    )

    # 3.5) Convert the five FreeSurfer corpus-callosum subdivisions (251-255)
    # to hemisphere-specific cerebral white matter. SynthSeg supplies the
    # left/right assignment at the corresponding voxels (normally labels 2/41).
    # This is a deliberate label-harmonization rule, not extracranial handling.
    corpus_callosum = (
        (brain_mask != 0) & (aseg > 250) & (aseg < 256)
    )
    data3 = apply_coordinates_mask(data3, corpus_callosum, ss)

    # 4) Suppress leakage for the selected ventricular labels: discard labels
    # that disagree with the iBEAT CSF class, then restore tissue-consistent
    # ventricles from SynthSeg. Lateral and inferior lateral ventricles are
    # intentionally untouched and retain the aseg-first behavior above.
    leaking_ventricles = (
        (brain_mask != 0) & cp.isin(data3, ventricle_labels) & (tissue != 1)
    )
    data3 = apply_coordinates_mask(
        data3, leaking_ventricles, cp.zeros_like(data3)
    )
    synthseg_ventricles = (
        (brain_mask != 0) & cp.isin(ss, ventricle_labels) & (tissue == 1)
    )
    data3 = apply_coordinates_mask(data3, synthseg_ventricles, ss)

    # 5) 2.5-style cortex merge using simple 3/42 cortex in fs_like and ss_like
    cortex_L, cortex_R, fs_L, fs_R = cortex_merge_masks_simple(
        fs_like=aseg,
        ss_like=ss,
        brain_mask=brain_mask,
        grow_steps=grow_steps,
        grow_radius=grow_radius,
        env_close_radius=env_close_radius,
        env_erode_radius=env_erode_radius,
        seed_erode_radius=seed_erode_radius,
        debug=debug
    )

    # Write cortex as 3/42, but avoid overwriting nonzero aseg labels unless they are cortex or empty.
    # If your aseg does NOT contain 3/42 cortex, this effectively writes only where aseg==0.
    write_L = cortex_L & ((aseg == 0) | (aseg == 3))
    write_R = cortex_R & ((aseg == 0) | (aseg == 42))
    data3 = apply_coordinates_mask(data3, write_L, cp.full_like(data3, 3))
    data3 = apply_coordinates_mask(data3, write_R, cp.full_like(data3, 42))

    # 6) Collapse redundant cortical/WM parcellation ranges and remove known
    # artifact codes, as described in the manuscript.
    data3 = collapse_parcellation_ranges(data3, brain_mask)
    changes = [(value, 0) for value in ARTIFACT_LABELS]
    data3 = eD.editLabels(data3, changes)

    # 7) Restore cerebellum labels from SynthSeg inside brain
    data3 = apply_coordinates_mask(
        data3,
        (brain_mask != 0) & cp.isin(ss, cerebellum_labels),
        ss
    )

    # 8) Fill missing GM/WM using tissue map when aseg is empty (your original idea)
    data3 = apply_coordinates_mask(
        data3,
        (tissue == 2) & (aseg == 0) & cp.isin(ss, cp.array([3, 42])),
        ss
    )
    data3 = apply_coordinates_mask(
        data3,
        (tissue == 3) & (aseg == 0) & cp.isin(ss, cp.array([2, 41])),
        ss
    )

    # 9) Keep selected bilateral subcortical structures from SynthSeg. The
    # brain-mask condition prevents source labels from leaking into head tissue.
    # Listed ventricles remain subject to the CSF-consistency rule from step 4;
    # lateral and inferior lateral ventricles are absent from this priority list.
    important = cp.asarray(SYNTHSEG_PRIORITY_LABELS)
    nonventricular_important = important[~cp.isin(important, ventricle_labels)]
    data3 = apply_coordinates_mask(
        data3,
        (brain_mask != 0) & cp.isin(ss, nonventricular_important),
        ss
    )
    data3 = apply_coordinates_mask(data3, synthseg_ventricles, ss)

    # 10) Optional: recover tiny missing cortex bits using adjacency + tissue + SS hint
    data3 = cortex_gap_fill(data3, cortex_L, cortex_R, ss, brain_mask, tissue, aseg)

    # 11) Your targeted fixes (unchanged)
    # Left/right choroid plexus
    try:
        y1, z1 = find_yz(data3, 5)
        if y1 is not None:
            data3 = eD.editLabelValueYZ(data3, y1, z1, 31, 136, 137)
        else:
            print("⚠ Choroid-L fix skipped: label 5 not found")
    except Exception as e:
        print(f"⚠ Choroid-L fix skipped: {e}")

    try:
        y2, z2 = find_yz(data3, 44)
        if y2 is not None:
            data3 = eD.editLabelValueYZ(data3, y2, z2, 63, 163, 164)
        else:
            print("⚠ Choroid-R fix skipped: label 44 not found")
    except Exception as e:
        print(f"⚠ Choroid-R fix skipped: {e}")

    # Left/right WMH
    try:
        data3 = eD.editLabelValue(data3, ss, 77, 25, 57)
    except Exception as e:
        print(f"⚠ WMH fix skipped: {e}")

    # 12) Conservative topology cleanup: remove only small islands belonging to
    # the four major cerebral GM/WM labels. Hole filling is already applied to
    # the smooth cortex envelope in smooth_envelope().
    data3 = remove_small_label_islands(
        data3,
        brain_mask,
        labels=(2, 3, 41, 42),
        min_size=island_min_size,
    )

    # Save
    out_path = full['out_final']
    eD.saveNii(cp.asnumpy(data3), affine, header, out_path)
    print(f"✔ Saved: {out_path}")

# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser()
    location = ap.add_mutually_exclusive_group(required=True)
    location.add_argument("--base-dir", help="Folder containing subject subfolders")
    location.add_argument("--subject-dir", help="Process exactly one subject folder")

    ap.add_argument("--grow-steps", type=int, default=2, help="Cortex cautious growth steps")
    ap.add_argument("--grow-radius", type=int, default=1, help="Radius used per growth step")
    ap.add_argument("--env-close", type=int, default=3, help="Cortex envelope closing radius")
    ap.add_argument("--env-erode", type=int, default=0, help="Optional erosion after envelope (0=off)")
    ap.add_argument("--seed-erode", type=int, default=0,
                    help="If FS∩SS cortex is empty, erode union by this radius to create a seed (0=off, 1=recommended)")
    ap.add_argument("--island-min-size", type=int, default=20,
                    help="Remove GM/WM connected components smaller than this many voxels (0=off)")
    ap.add_argument("--debug", action="store_true", help="Print cortex voxel counts for union/inter/seed/grown")

    # Optional override of filenames
    ap.add_argument("--aseg-name", default="aseg1r.nii.gz",
                    help="FreeSurfer-derived aseg aligned to the working grid")
    ap.add_argument("--mask-csf-name", default="mask-csf.nii.gz",
                    help="CSF/head mask used only with --label-name")
    ap.add_argument("--mask-name", default="mask.nii.gz",
                    help="SynthStrip brain mask aligned to the working grid")
    ap.add_argument("--tissue-name", default="T1-skullstripped-tissue.nii.gz",
                    help="iBEAT tissue map: 1=CSF, 2=GM, 3=WM")
    ap.add_argument("--ss-name", default="T1-ss1rrf.nii.gz",
                    help="SynthSeg label map aligned to the working grid")
    ap.add_argument("--label-name", default=None,
                    help="Optional CHARM-based whole-head label map. If omitted, output is brain-only and edit_nonbrain is skipped.")
    ap.add_argument("--out-name", default="medley_segmentation.nii.gz")

    args = ap.parse_args()

    paths = {
        'aseg': args.aseg_name,
        'mask_csf': args.mask_csf_name,
        'mask': args.mask_name,
        'tissue': args.tissue_name,
        'T1_ss': args.ss_name,
        # Optional whole-head labelmap: provide a real file here if you have it
        'label': args.label_name,
        'out_no_brain': "withoutBrain.nii.gz",
        'out_final': args.out_name
    }

    if args.subject_dir:
        if not os.path.isdir(args.subject_dir):
            raise SystemExit(f"Subject folder not found: {args.subject_dir}")
        subj_dirs = [args.subject_dir]
    else:
        subj_dirs = sorted([p for p in glob.glob(os.path.join(args.base_dir, "*")) if os.path.isdir(p)])
        if not subj_dirs:
            raise SystemExit(f"No subject folders found under: {args.base_dir}")

    for dir_path in subj_dirs:
        try:
            process_subject(
                dir_path,
                paths,
                grow_steps=args.grow_steps,
                grow_radius=args.grow_radius,
                env_close_radius=args.env_close,
                env_erode_radius=args.env_erode,
                seed_erode_radius=args.seed_erode,
                island_min_size=args.island_min_size,
                debug=args.debug
            )
        except Exception as e:
            print(f"✘ Unexpected error in {dir_path}: {e}")

if __name__ == "__main__":
    main()
