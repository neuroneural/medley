#!/usr/bin/env bash
set -euo pipefail

stage="${1:?stage required}"
manifest="${2:?manifest required}"
code_dir="${3:?code directory required}"
ibeat_bind_root="${4:?iBEAT bind root required}"
ibeat_container_root="${5:-/InfantData}"
ibeat_sif="${6:-}"
fs_subjects_dir="${7:?FreeSurfer SUBJECTS_DIR required}"
freesurfer_home="${8:?FREESURFER_HOME required}"
simnibs_env_script="${9:-}"
charm_input_name="${10:-oT1e-18-big3.nii.gz}"
overwrite="${11:-0}"
final_name="${12:-medley_segmentation.nii.gz}"
use_charm="${13:-1}"
conda_base="${14:?Conda base required}"
conda_env="${15:-labelling}"
python_exe="${16:-}"
freesurfer_final="${17:-enhanced_enlarged}"
charm_forceqform="${18:-1}"
charm_forcerun="${19:-1}"
synthseg_final="${20:-raw_robust}"

case "$freesurfer_final" in
  enhanced|enhanced_enlarged) ;;
  *) echo "Invalid FreeSurfer final variant: $freesurfer_final" >&2; exit 2;;
esac
case "$synthseg_final" in
  raw_robust|enhanced_robust|raw_standard|enhanced_standard) ;;
  *) echo "Invalid SynthSeg final variant: $synthseg_final" >&2; exit 2;;
esac

activate_medley_python() {
  [[ -f "$conda_base/etc/profile.d/conda.sh" ]] || {
    echo "Conda initialization script not found: $conda_base/etc/profile.d/conda.sh" >&2
    exit 3
  }
  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"

  if [[ "$conda_env" == */* ]]; then
    [[ -d "$conda_env" ]] || {
      echo "Conda environment prefix does not exist: $conda_env" >&2
      exit 3
    }
    conda_env="$(cd "$conda_env" && pwd)"
  fi

  conda activate "$conda_env"

  if [[ "$conda_env" == */* && -x "$conda_env/bin/python" ]]; then
    python_exe="$conda_env/bin/python"
  elif [[ -z "$python_exe" || ! -x "$python_exe" ]]; then
    python_exe="$(command -v python)"
  fi

  "$python_exe" -c "import numpy, nibabel, scipy"
  export MEDLEY_PYTHON="$python_exe"
}

case "$stage" in
  gather_ibeat|preprocess|postprocess|consolidate)
    activate_medley_python
    ;;
esac

: "${SLURM_ARRAY_TASK_ID:?This worker must run as a SLURM array task}"
row="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$manifest")"
[[ -n "$row" ]] || { echo "No manifest row for task $SLURM_ARRAY_TASK_ID" >&2; exit 2; }
IFS=$'\t' read -r age age_name subject raw_t1 ibeat_subject_dir work_dir synthseg_dir charm_case_dir fs_id scale <<<"$row"
[[ -n "$scale" ]] || { echo "Malformed manifest row: $row" >&2; exit 2; }
charm_id="${age_name}_${subject}"

inverse_scale="$(awk -v value="$scale" 'BEGIN {
  if (value <= 0) {
    print "Scale must be positive" > "/dev/stderr"
    exit 1
  }
  printf "%.15g\n", 1.0 / value
}')"

ow=()
[[ "$overwrite" == "1" ]] && ow=(--overwrite)

log() { printf '[%s] %s/%s: %s\n' "$stage" "$age_name" "$subject" "$*"; }

atomic_copy() {
  local src="$1" dst="$2"
  [[ -f "$src" ]] || { echo "Missing source file: $src" >&2; exit 4; }
  if [[ -f "$dst" && "$overwrite" != "1" ]]; then
    log "EXISTS $dst"
    return
  fi
  mkdir -p "$(dirname "$dst")"
  local tmp="${dst}.tmp.${SLURM_JOB_ID:-$$}.${SLURM_ARRAY_TASK_ID:-0}"
  cp -p "$src" "$tmp"
  mv -f "$tmp" "$dst"
  log "COPIED $src -> $dst"
}

run_if_needed() {
  local output="$1"
  shift
  if [[ -e "$output" && "$overwrite" != "1" ]]; then
    log "EXISTS $output"
  else
    "$@"
  fi
}

fs_variant_subjects_dir() {
  local variant="${1:?FreeSurfer variant required}"
  case "$variant" in
    enhanced)
      printf '%s/enhanced/%s\n' "$fs_subjects_dir" "$age_name"
      ;;
    enhanced_enlarged)
      printf '%s/enhanced_enlarged/%s\n' "$fs_subjects_dir" "$age_name"
      ;;
    runtime)
      printf '%s/_runtime\n' "$fs_subjects_dir"
      ;;
    *)
      echo "Unknown FreeSurfer variant: $variant" >&2
      return 2
      ;;
  esac
}

find_fs_mri_dir() {
  local variant="${1:?FreeSurfer variant required}"
  local explicit legacy

  explicit="$(fs_variant_subjects_dir "$variant")/$fs_id/mri"
  if [[ -f "$explicit/aseg.mgz" ]]; then
    printf '%s\n' "$explicit"
    return 0
  fi

  # Backward-compatible reuse of the former enhanced+enlarged layout:
  #   <fs-root>/<age>/<subject>/mri
  if [[ "$variant" == "enhanced_enlarged" ]]; then
    legacy="$fs_subjects_dir/$age_name/$fs_id/mri"
    if [[ -f "$legacy/aseg.mgz" ]]; then
      printf '%s\n' "$legacy"
      return 0
    fi
  fi

  printf '%s\n' "$explicit"
  return 1
}

setup_freesurfer() {
  local variant="${1:-runtime}"
  local fs_module="${FREESURFER_MODULE:-freesurfer/7.4.1}"
  local fs_expect="${FREESURFER_EXPECT_VERSION:-7.4.1}"

  # Make `module` available in non-login batch shells.
  if ! command -v module >/dev/null 2>&1; then
    for _init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash \
                 /usr/share/lmod/lmod/init/bash /opt/apps/lmod/lmod/init/bash; do
      [[ -r "$_init" ]] && { set +u; source "$_init"; set -u; break; }
    done
  fi

  # Clear inherited FreeSurfer variables before loading the requested version.
  # Some modulefiles use prepend-path and can create invalid colon-separated
  # scalar values when these variables are already defined.
  unset FREESURFER_HOME FSFAST_HOME MNI_DIR SUBJECTS_DIR \
        MINC_BIN_DIR MINC_LIB_DIR MNI_DATAPATH FS_LICENSE || true

  if command -v module >/dev/null 2>&1; then
    module unload freesurfer 2>/dev/null || true
    set +u
    module load "$fs_module" 2>/dev/null \
      || echo "WARN: 'module load $fs_module' failed; falling back to FREESURFER_HOME" >&2
    set -u
  fi

  # Fallback to the explicit install path if the module system is unavailable.
  [[ -n "${FREESURFER_HOME:-}" ]] || export FREESURFER_HOME="$freesurfer_home"
  [[ -d "$FREESURFER_HOME" ]] || { echo "FREESURFER_HOME does not exist: $FREESURFER_HOME" >&2; exit 5; }

  # Source SetUpFreeSurfer.sh explicitly so required PATH, MNI, MINC, and Perl
  # settings are available in non-login batch shells.
  # SetUpFreeSurfer.sh is not errexit/nounset clean, so temporarily disable both.
  set +eu
  # shellcheck disable=SC1090
  source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
  set -eu

  # Set the FreeSurfer license path and a variant-specific SUBJECTS_DIR.
  export FS_LICENSE="${FS_LICENSE:-$FREESURFER_HOME/license.txt}"
  export SUBJECTS_DIR="$(fs_variant_subjects_dir "$variant")"
  mkdir -p "$SUBJECTS_DIR"

  local ver; ver="$(recon-all -version 2>&1 | head -n1 || true)"
  echo "FreeSurfer variant=$variant  module=$fs_module"
  echo "FREESURFER_HOME=$FREESURFER_HOME"
  echo "FS_LICENSE=$FS_LICENSE"
  echo "SUBJECTS_DIR=$SUBJECTS_DIR"
  echo "recon-all: $(command -v recon-all || true)  [$ver]"

  local _t
  for _t in recon-all mri_synthseg mri_synthstrip mri_label2vol; do
    command -v "$_t" >/dev/null || { echo "$_t not found after loading $fs_module" >&2; exit 5; }
  done

  # Guard: never silently run a different build (e.g. freesurfer-new dev).
  # Override with FREESURFER_EXPECT_VERSION="" if you really intend to.
  if [[ -n "$fs_expect" && "$ver" != *"$fs_expect"* && "$FREESURFER_HOME" != *"$fs_expect"* ]]; then
    echo "ERROR: expected FreeSurfer '$fs_expect' but loaded [$ver] at $FREESURFER_HOME" >&2
    echo "       fix FREESURFER_MODULE/FREESURFER_HOME, or override FREESURFER_EXPECT_VERSION=''" >&2
    exit 5
  fi
}

setup_fsl() {
  local fsl_module="${FSL_MODULE:-fsl}"

  # Make `module` available in non-login batch shells.
  if ! command -v module >/dev/null 2>&1; then
    for _init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash \
                 /usr/share/lmod/lmod/init/bash /opt/apps/lmod/lmod/init/bash; do
      [[ -r "$_init" ]] && { set +u; source "$_init"; set -u; break; }
    done
  fi

  # Prefer the cluster module, but allow an already configured FSL install.
  if ! command -v flirt >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
      set +u
      module load "$fsl_module"
      set -u
      hash -r
    fi
  fi

  command -v flirt >/dev/null 2>&1 || {
    echo "ERROR: flirt was not found. Set FSL_MODULE in config/site.env or configure FSL in PATH." >&2
    exit 5
  }

  echo "FSL module=$fsl_module"
  echo "FSLDIR=${FSLDIR:-not-set}"
  echo "flirt=$(command -v flirt)"
}

flirt_label_to_reference() {
  local moving="${1:?moving label required}"
  local reference="${2:?reference image required}"
  local output="${3:?output image required}"

  [[ -f "$moving" ]] || {
    echo "Missing FLIRT input: $moving" >&2
    exit 4
  }
  [[ -f "$reference" ]] || {
    echo "Missing FLIRT reference: $reference" >&2
    exit 4
  }

  run_if_needed "$output" \
    flirt \
      -in "$moving" \
      -ref "$reference" \
      -applyxfm \
      -usesqform \
      -interp nearestneighbour \
      -out "$output"
}

validate_and_cast_discrete_labels() {
  local label_file="${1:?label file required}"

  [[ -f "$label_file" ]] || {
    echo "Missing label file: $label_file" >&2
    exit 4
  }

  # Verify that no interpolation-created fractional labels are present, then
  # rewrite the NIfTI with an integer dtype while preserving affine/qform/sform.
  "$python_exe" - "$label_file" <<'PYLABEL'
import os
import sys
import tempfile

import nibabel as nib
import numpy as np

path = sys.argv[1]
img = nib.load(path)
data = np.asanyarray(img.dataobj)

if not np.isfinite(data).all():
    raise SystemExit(f"ERROR: non-finite values in label map: {path}")

rounded = np.rint(data)
max_deviation = float(np.max(np.abs(data - rounded))) if data.size else 0.0
if max_deviation > 1e-6:
    raise SystemExit(
        f"ERROR: non-integer values in label map {path}; "
        f"maximum deviation from integer={max_deviation:.9g}"
    )

# int32 safely covers FreeSurfer aseg label IDs.
integer_data = rounded.astype(np.int32, copy=False)
header = img.header.copy()
header.set_data_dtype(np.int32)

qform, qcode = img.get_qform(coded=True)
sform, scode = img.get_sform(coded=True)

folder = os.path.dirname(os.path.abspath(path))
fd, tmp = tempfile.mkstemp(prefix=".medley-label-", suffix=".nii.gz", dir=folder)
os.close(fd)
try:
    out = nib.Nifti1Image(integer_data, img.affine, header=header)
    if qform is not None:
        out.set_qform(qform, int(qcode))
    if sform is not None:
        out.set_sform(sform, int(scode))
    nib.save(out, tmp)
    os.replace(tmp, path)
finally:
    if os.path.exists(tmp):
        os.unlink(tmp)

labels = np.unique(integer_data)
if labels.size:
    print(
        f"Validated discrete labels: {path} "
        f"dtype=int32 count={labels.size} range={int(labels.min())}..{int(labels.max())}"
    )
else:
    print(f"Validated empty discrete label map: {path} dtype=int32")
PYLABEL
}

setup_simnibs() {
  # Preferred path: an explicit activation script (reproducible, version-pinned).
  # Fallback: a site 'module load simnibs' when no script is configured.
  if [[ -n "$simnibs_env_script" ]]; then
    [[ -f "$simnibs_env_script" ]] || {
      echo "SimNIBS activation script not found: $simnibs_env_script" >&2
      exit 5
    }
    export SIMNIBS_HOME="$(cd "$(dirname "$simnibs_env_script")/../.." && pwd)"
    simnibs_bin="$(dirname "$simnibs_env_script")"
    export PATH="$simnibs_bin:$PATH"

    set +u
    # shellcheck disable=SC1090
    source "$simnibs_env_script"
    set -u
    export PATH="$simnibs_bin:$PATH"
    hash -r
  else
    module load simnibs 2>/dev/null || true
    hash -r
  fi

  CHARM_BIN="$(command -v charm || true)"
  echo "SIMNIBS_HOME=${SIMNIBS_HOME:-not-set}"
  echo "SimNIBS activation=${simnibs_env_script:-<module load simnibs>}"
  echo "charm path=$CHARM_BIN"
  [[ -n "$CHARM_BIN" && -x "$CHARM_BIN" ]] || {
    echo "charm not found. Set SIMNIBS_ENV_SCRIPT in config/site.env, or ensure 'module load simnibs' provides charm." >&2
    exit 5
  }
  export CHARM_BIN
}

find_ibeat_subject_dir() {
  local base expected candidate
  expected="$ibeat_subject_dir"
  if [[ -f "$expected/T1.nii.gz" && -f "$expected/T1-skullstripped-tissue.nii.gz" ]]; then
    printf '%s\n' "$expected"
    return 0
  fi
  base="$(dirname "$(dirname "$ibeat_subject_dir")")"
  for candidate in \
    "$base/T1-${age}m/$subject" \
    "$base/T1-${age}/$subject" \
    "$base/T1-${age}mo/$subject"; do
    if [[ -f "$candidate/T1.nii.gz" && -f "$candidate/T1-skullstripped-tissue.nii.gz" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  while IFS= read -r candidate; do
    candidate="$(dirname "$candidate")"
    if [[ -f "$candidate/T1-skullstripped-tissue.nii.gz" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done < <(find "$base" -maxdepth 4 -type f -path "*/$subject/T1.nii.gz" | sort -V)
  return 1
}

find_charm_label() {
  local m2m="$charm_case_dir/m2m_${charm_id}" candidate
  for candidate in \
    "$m2m/label_prep/tissue_labeling_upsampled.nii.gz" \
    "$m2m/segmentation/labeling.nii.gz" \
    "$m2m/final_tissues.nii.gz" \
    "$charm_case_dir/m2m_head1c/segmentation/labeling.nii.gz"; do
    [[ -f "$candidate" ]] && { printf '%s\n' "$candidate"; return 0; }
  done
  find "$charm_case_dir" -maxdepth 6 -type f \
    \( -name 'tissue_labeling_upsampled.nii.gz' -o -name 'labeling.nii.gz' -o -name 'final_tissues.nii.gz' \) \
    | sort -V | head -n 1
}

log "scale=$scale inverse=$inverse_scale host=${HOSTNAME:-unknown}"

case "$stage" in
  ibeat)
    module load singularity
    [[ -f "$raw_t1" ]] || { echo "Missing raw T1: $raw_t1" >&2; exit 4; }
    [[ -f "$ibeat_sif" ]] || { echo "Missing iBEAT SIF: $ibeat_sif" >&2; exit 4; }
    case "$raw_t1" in
      "$ibeat_bind_root"/*) ;;
      *) echo "Raw T1 is outside --ibeat-bind-root: $raw_t1" >&2; exit 4;;
    esac
    host_out_root="$(dirname "$ibeat_subject_dir")"
    case "$host_out_root" in
      "$ibeat_bind_root"/*) ;;
      *) echo "iBEAT output is outside --ibeat-bind-root: $host_out_root" >&2; exit 4;;
    esac
    container_t1="$ibeat_container_root/${raw_t1#"$ibeat_bind_root"/}"
    container_out="$ibeat_container_root/${host_out_root#"$ibeat_bind_root"/}"
    if [[ -f "$ibeat_subject_dir/T1.nii.gz" && -f "$ibeat_subject_dir/T1-skullstripped-tissue.nii.gz" && "$overwrite" != "1" ]]; then
      log "iBEAT outputs already exist"
      exit 0
    fi
    mkdir -p "$host_out_root"
    if [[ "$overwrite" == "1" && -d "$ibeat_subject_dir" ]]; then
      rm -rf "$ibeat_subject_dir"
    fi
    singularity run --nv --bind "$ibeat_bind_root:$ibeat_container_root" "$ibeat_sif" \
      iBEAT --t1 "$container_t1" --age "$age" --out_dir "$container_out" \
      --sub_name "$subject" --skip_surface 0
    [[ -f "$ibeat_subject_dir/T1.nii.gz" ]] || { echo "iBEAT did not produce $ibeat_subject_dir/T1.nii.gz" >&2; exit 4; }
    [[ -f "$ibeat_subject_dir/T1-skullstripped-tissue.nii.gz" ]] || { echo "iBEAT did not produce the tissue map for $age_name/$subject" >&2; exit 4; }
    ;;

  gather_ibeat)
    src_dir="$(find_ibeat_subject_dir)" || {
      echo "Could not locate iBEAT T1/tissue outputs for $age_name/$subject" >&2
      exit 4
    }
    mkdir -p "$work_dir"
    atomic_copy "$src_dir/T1.nii.gz" "$work_dir/T1.nii.gz"
    atomic_copy "$src_dir/T1-skullstripped-tissue.nii.gz" "$work_dir/T1-skullstripped-tissue.nii.gz"
    if [[ -f "$src_dir/T2.nii.gz" ]]; then
      atomic_copy "$src_dir/T2.nii.gz" "$work_dir/T2.nii.gz"
    fi
    if [[ -f "$src_dir/T2-skullstripped-tissue.nii.gz" ]]; then
      atomic_copy "$src_dir/T2-skullstripped-tissue.nii.gz" "$work_dir/T2-skullstripped-tissue.nii.gz"
    fi
    "$python_exe" "$code_dir/validate_inputs.py" --subject-dir "$work_dir" --files \
      T1.nii.gz T1-skullstripped-tissue.nii.gz
    ;;

  preprocess)
    "$python_exe" "$code_dir/enhance_t1.py" --subject-dir "$work_dir" "${ow[@]}"
    "$python_exe" "$code_dir/scale_nifti.py" \
      "$work_dir/T1.nii.gz" "$work_dir/T1-big3.nii.gz" \
      --scale "$scale" --order 3 "${ow[@]}"
    "$python_exe" "$code_dir/scale_nifti.py" \
      "$work_dir/T1_enhanced-18.nii.gz" "$work_dir/oT1e-18-big3.nii.gz" \
      --scale "$scale" --order 3 "${ow[@]}"
    ;;

  synthseg)
    setup_freesurfer runtime
    mkdir -p "$synthseg_dir"
    threads="${SLURM_CPUS_PER_TASK:-8}"
    run_if_needed "$synthseg_dir/oT1e_ss1rr.nii.gz" \
      mri_synthseg --i "$work_dir/oT1e-18-big3.nii.gz" --o "$synthseg_dir/oT1e_ss1rr.nii.gz" --robust --threads "$threads" --cpu
    run_if_needed "$synthseg_dir/T1_ss1rr.nii.gz" \
      mri_synthseg --i "$work_dir/T1-big3.nii.gz" --o "$synthseg_dir/T1_ss1rr.nii.gz" --robust --threads "$threads" --cpu
    run_if_needed "$synthseg_dir/oT1e_ss2-1rr.nii.gz" \
      mri_synthseg --i "$work_dir/oT1e-18-big3.nii.gz" --o "$synthseg_dir/oT1e_ss2-1rr.nii.gz" --threads "$threads" --cpu
    run_if_needed "$synthseg_dir/T1_ss2-1rr.nii.gz" \
      mri_synthseg --i "$work_dir/T1-big3.nii.gz" --o "$synthseg_dir/T1_ss2-1rr.nii.gz" --threads "$threads" --cpu
    # Generate both SynthStrip masks on the native enhanced-T1 grid so no
    # inverse scaling is required during downstream consolidation.
    run_if_needed "$synthseg_dir/mask.nii.gz" \
      mri_synthstrip -i "$work_dir/T1_enhanced-18.nii.gz" -o "$synthseg_dir/stripped.nii.gz" -m "$synthseg_dir/mask.nii.gz" --no-csf
    run_if_needed "$synthseg_dir/mask-csf.nii.gz" \
      mri_synthstrip -i "$work_dir/T1_enhanced-18.nii.gz" -o "$synthseg_dir/stripped-csf.nii.gz" -m "$synthseg_dir/mask-csf.nii.gz"
    ;;

  freesurfer_enhanced)
    setup_freesurfer enhanced
    input="$work_dir/T1_enhanced-18.nii.gz"
    aseg="$SUBJECTS_DIR/$fs_id/mri/aseg.mgz"
    [[ -f "$input" ]] || { echo "Missing FreeSurfer enhanced input: $input" >&2; exit 4; }

    if [[ -f "$aseg" && "$overwrite" != "1" ]]; then
      log "Enhanced-only FreeSurfer aseg already exists"
    else
      if [[ "$overwrite" == "1" && -d "$SUBJECTS_DIR/$fs_id" ]]; then
        rm -rf "$SUBJECTS_DIR/$fs_id"
      fi
      recon-all -s "$fs_id" -i "$input" -all \
        -parallel -openmp "${SLURM_CPUS_PER_TASK:-16}"
    fi
    ;;

  freesurfer_enlarged)
    setup_freesurfer enhanced_enlarged
    input="$work_dir/oT1e-18-big3.nii.gz"
    explicit_aseg="$SUBJECTS_DIR/$fs_id/mri/aseg.mgz"
    legacy_aseg="$fs_subjects_dir/$age_name/$fs_id/mri/aseg.mgz"
    [[ -f "$input" ]] || { echo "Missing FreeSurfer enhanced+enlarged input: $input" >&2; exit 4; }

    if [[ -f "$explicit_aseg" && "$overwrite" != "1" ]]; then
      log "Enhanced+enlarged FreeSurfer aseg already exists"
    elif [[ -f "$legacy_aseg" && "$overwrite" != "1" ]]; then
      log "Reusing legacy enhanced+enlarged FreeSurfer result: $legacy_aseg"
    else
      if [[ "$overwrite" == "1" && -d "$SUBJECTS_DIR/$fs_id" ]]; then
        rm -rf "$SUBJECTS_DIR/$fs_id"
      fi
      recon-all -s "$fs_id" -i "$input" -all \
        -parallel -openmp "${SLURM_CPUS_PER_TASK:-16}"
    fi
    ;;

  charm)
    [[ "$use_charm" == "1" ]] || { log "CHARM disabled"; exit 0; }
    setup_simnibs
    input="$work_dir/$charm_input_name"
    [[ -f "$input" ]] || { echo "Missing CHARM input: $input" >&2; exit 4; }
    mkdir -p "$charm_case_dir"
    existing="$(find_charm_label || true)"
    if [[ -n "$existing" && -f "$existing" && "$overwrite" != "1" ]]; then
      log "CHARM label already exists: $existing"
      exit 0
    fi
    if [[ "$overwrite" == "1" ]]; then
      rm -rf "$charm_case_dir/m2m_${charm_id}"
      rm -f "$charm_case_dir/${charm_id}.msh"
    fi
    cd "$charm_case_dir"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

    charm_args=()
    [[ "$charm_forceqform" == "1" ]] && charm_args+=(--forceqform)
    [[ "$charm_forcerun" == "1" ]] && charm_args+=(--forcerun)

    echo "CHARM command: $CHARM_BIN $charm_id $input ${charm_args[*]}"
    "$CHARM_BIN" "$charm_id" "$input" "${charm_args[@]}"
    ;;

  gather_tools)
    mkdir -p "$work_dir"
    for name in oT1e_ss1rr.nii.gz T1_ss1rr.nii.gz oT1e_ss2-1rr.nii.gz T1_ss2-1rr.nii.gz \
                mask.nii.gz mask-csf.nii.gz; do
      atomic_copy "$synthseg_dir/$name" "$work_dir/$name"
    done

    # Gather every complete FreeSurfer variant available for this subject.
    # This pipeline uses rawavg.mgz as the mri_label2vol template. orig.mgz is
    # optional and is copied only for QC and provenance.
    fs_available=0

    fs_enhanced_mri="$(find_fs_mri_dir enhanced || true)"
    if [[ -f "$fs_enhanced_mri/aseg.mgz" && -f "$fs_enhanced_mri/rawavg.mgz" ]]; then
      atomic_copy "$fs_enhanced_mri/aseg.mgz" "$work_dir/fs_enhanced_aseg.mgz"
      atomic_copy "$fs_enhanced_mri/rawavg.mgz" "$work_dir/fs_enhanced_rawavg.mgz"
      [[ -f "$fs_enhanced_mri/orig.mgz" ]] && \
        atomic_copy "$fs_enhanced_mri/orig.mgz" "$work_dir/fs_enhanced_orig.mgz"
      fs_available=$((fs_available + 1))
      log "Available complete FreeSurfer variant: enhanced"
    elif [[ -f "$work_dir/fs_enhanced_aseg.mgz" && -f "$work_dir/fs_enhanced_rawavg.mgz" ]]; then
      fs_available=$((fs_available + 1))
      log "Reusing already gathered complete FreeSurfer variant: enhanced"
    else
      log "Complete FreeSurfer variant unavailable: enhanced"
    fi

    fs_enlarged_mri="$(find_fs_mri_dir enhanced_enlarged || true)"
    if [[ -f "$fs_enlarged_mri/aseg.mgz" && -f "$fs_enlarged_mri/rawavg.mgz" ]]; then
      atomic_copy "$fs_enlarged_mri/aseg.mgz" "$work_dir/fs_enlarged_aseg.mgz"
      atomic_copy "$fs_enlarged_mri/rawavg.mgz" "$work_dir/fs_enlarged_rawavg.mgz"
      [[ -f "$fs_enlarged_mri/orig.mgz" ]] && \
        atomic_copy "$fs_enlarged_mri/orig.mgz" "$work_dir/fs_enlarged_orig.mgz"
      fs_available=$((fs_available + 1))
      log "Available complete FreeSurfer variant: enhanced_enlarged"
    elif [[ -f "$work_dir/fs_enlarged_aseg.mgz" && -f "$work_dir/fs_enlarged_rawavg.mgz" ]]; then
      fs_available=$((fs_available + 1))
      log "Reusing already gathered complete FreeSurfer variant: enhanced_enlarged"
    else
      log "Complete FreeSurfer variant unavailable: enhanced_enlarged"
    fi

    if [[ "$fs_available" -lt 1 ]]; then
      echo "No complete FreeSurfer result found for $age_name/$subject." >&2
      echo "At least one variant must contain both aseg.mgz and rawavg.mgz:" >&2
      echo "  enhanced:          <FS root>/enhanced/$age_name/$fs_id/mri/{aseg,rawavg}.mgz" >&2
      echo "  enhanced_enlarged: <FS root>/enhanced_enlarged/$age_name/$fs_id/mri/{aseg,rawavg}.mgz" >&2
      exit 4
    fi

    if [[ "$use_charm" == "1" ]]; then
      charm_label="$(find_charm_label || true)"
      [[ -n "$charm_label" && -f "$charm_label" ]] || {
        echo "Could not find a CHARM label map under $charm_case_dir" >&2
        exit 4
      }
      atomic_copy "$charm_label" "$work_dir/charm_label_source.nii.gz"
    fi
    ;;

  postprocess)
    # FreeSurfer segmentations are discrete label maps. Convert each aseg.mgz
    # to its rawavg template grid with mri_label2vol, then apply nearest-neighbor
    # FLIRT for final grid harmonization.
    setup_freesurfer runtime
    setup_fsl

    native_ref="$work_dir/T1_enhanced-18.nii.gz"
    [[ -f "$native_ref" ]] || {
      echo "Missing FLIRT reference: $native_ref" >&2
      exit 4
    }

    # SynthSeg candidates were generated on enlarged images. Undo the
    # enlargement first, then harmonize each label map to the native enhanced
    # T1 grid with the original nearest-neighbor FLIRT command.
    declare -a pairs=(
      "oT1e_ss1rr.nii.gz:oT1e_ss1rr1.nii.gz:oT1e-ss1rrf.nii.gz"
      "T1_ss1rr.nii.gz:T1_ss1rr1.nii.gz:T1-ss1rrf.nii.gz"
      "oT1e_ss2-1rr.nii.gz:oT1e_ss2rr1.nii.gz:oT1e-ss2rrf.nii.gz"
      "T1_ss2-1rr.nii.gz:T1_ss2rr1.nii.gz:T1-ss2rrf.nii.gz"
    )
    for entry in "${pairs[@]}"; do
      IFS=: read -r src restored aligned <<<"$entry"
      "$python_exe" "$code_dir/scale_nifti.py" \
        "$work_dir/$src" "$work_dir/$restored" \
        --scale "$inverse_scale" --order 0 "${ow[@]}"
      flirt_label_to_reference \
        "$work_dir/$restored" \
        "$native_ref" \
        "$work_dir/$aligned"
      validate_and_cast_discrete_labels "$work_dir/$aligned"
    done

    # Process whichever FreeSurfer variants were gathered. At least one must
    # be available, but the two branches do not depend on each other.
    fs_processed=0

    if [[ -f "$work_dir/fs_enhanced_aseg.mgz" ]]; then
      [[ -f "$work_dir/fs_enhanced_rawavg.mgz" ]] || {
        echo "Missing enhanced FreeSurfer template: $work_dir/fs_enhanced_rawavg.mgz" >&2
        exit 4
      }

      # Map the discrete aseg labels from FreeSurfer conformed space to the
      # enhanced scan's rawavg template grid.
      run_if_needed "$work_dir/aseg_enhanced_native.nii.gz" \
        mri_label2vol \
          --seg "$work_dir/fs_enhanced_aseg.mgz" \
          --temp "$work_dir/fs_enhanced_rawavg.mgz" \
          --regheader "$work_dir/fs_enhanced_aseg.mgz" \
          --o "$work_dir/aseg_enhanced_native.nii.gz"
      validate_and_cast_discrete_labels "$work_dir/aseg_enhanced_native.nii.gz"

      flirt_label_to_reference \
        "$work_dir/aseg_enhanced_native.nii.gz" \
        "$native_ref" \
        "$work_dir/aseg_enhanced1r.nii.gz"
      validate_and_cast_discrete_labels "$work_dir/aseg_enhanced1r.nii.gz"

      fs_processed=$((fs_processed + 1))
      log "Postprocessed FreeSurfer variant: enhanced (mri_label2vol + FLIRT)"
    fi

    if [[ -f "$work_dir/fs_enlarged_aseg.mgz" ]]; then
      [[ -f "$work_dir/fs_enlarged_rawavg.mgz" ]] || {
        echo "Missing enhanced+enlarged FreeSurfer template: $work_dir/fs_enlarged_rawavg.mgz" >&2
        exit 4
      }

      # Return the enlarged branch to its rawavg/input geometry, undo the
      # age-specific enlargement, then harmonize to the native enhanced T1.
      run_if_needed "$work_dir/aseg_enhanced_enlarged_native.nii.gz" \
        mri_label2vol \
          --seg "$work_dir/fs_enlarged_aseg.mgz" \
          --temp "$work_dir/fs_enlarged_rawavg.mgz" \
          --regheader "$work_dir/fs_enlarged_aseg.mgz" \
          --o "$work_dir/aseg_enhanced_enlarged_native.nii.gz"
      validate_and_cast_discrete_labels "$work_dir/aseg_enhanced_enlarged_native.nii.gz"

      "$python_exe" "$code_dir/scale_nifti.py" \
        "$work_dir/aseg_enhanced_enlarged_native.nii.gz" \
        "$work_dir/aseg_enhanced_enlarged_unscaled.nii.gz" \
        --scale "$inverse_scale" --order 0 "${ow[@]}"
      validate_and_cast_discrete_labels "$work_dir/aseg_enhanced_enlarged_unscaled.nii.gz"

      flirt_label_to_reference \
        "$work_dir/aseg_enhanced_enlarged_unscaled.nii.gz" \
        "$native_ref" \
        "$work_dir/aseg_enhanced_enlarged1r.nii.gz"
      validate_and_cast_discrete_labels "$work_dir/aseg_enhanced_enlarged1r.nii.gz"

      fs_processed=$((fs_processed + 1))
      log "Postprocessed FreeSurfer variant: enhanced_enlarged (mri_label2vol + inverse scale + FLIRT)"
    fi

    if [[ "$fs_processed" -lt 1 ]]; then
      echo "No gathered FreeSurfer variant is available for postprocessing: $work_dir" >&2
      exit 4
    fi

    # CHARM also runs on the enlarged enhanced image. First place its labels on
    # that enlarged input grid, undo enlargement, then use FLIRT for the final
    # native-grid harmonization.
    if [[ "$use_charm" == "1" ]]; then
      flirt_label_to_reference \
        "$work_dir/charm_label_source.nii.gz" \
        "$work_dir/$charm_input_name" \
        "$work_dir/charm_label_enlarged_grid.nii.gz"
      validate_and_cast_discrete_labels "$work_dir/charm_label_enlarged_grid.nii.gz"

      "$python_exe" "$code_dir/scale_nifti.py" \
        "$work_dir/charm_label_enlarged_grid.nii.gz" \
        "$work_dir/charm_label_unscaled.nii.gz" \
        --scale "$inverse_scale" --order 0 "${ow[@]}"
      validate_and_cast_discrete_labels "$work_dir/charm_label_unscaled.nii.gz"

      flirt_label_to_reference \
        "$work_dir/charm_label_unscaled.nii.gz" \
        "$native_ref" \
        "$work_dir/charm_label_native.nii.gz"
      validate_and_cast_discrete_labels "$work_dir/charm_label_native.nii.gz"
    fi

    validate=(
      T1.nii.gz
      T1-skullstripped-tissue.nii.gz
      mask.nii.gz
      mask-csf.nii.gz
      T1-ss1rrf.nii.gz
    )
    [[ -f "$work_dir/aseg_enhanced1r.nii.gz" ]] && validate+=(aseg_enhanced1r.nii.gz)
    [[ -f "$work_dir/aseg_enhanced_enlarged1r.nii.gz" ]] && validate+=(aseg_enhanced_enlarged1r.nii.gz)
    [[ "$use_charm" == "1" ]] && validate+=(charm_label_native.nii.gz)
    "$python_exe" "$code_dir/validate_inputs.py" \
      --subject-dir "$work_dir" --files "${validate[@]}"
    ;;

  consolidate)
    if [[ -f "$work_dir/$final_name" && "$overwrite" != "1" ]]; then
      log "Final output already exists: $work_dir/$final_name"
      exit 0
    fi
    # FREESURFER_FINAL is a preference. When the preferred branch is absent,
    # automatically fall back to the other completed branch for this subject.
    case "$freesurfer_final" in
      enhanced)
        preferred_aseg_name="aseg_enhanced1r.nii.gz"
        fallback_aseg_name="aseg_enhanced_enlarged1r.nii.gz"
        fallback_variant="enhanced_enlarged"
        ;;
      enhanced_enlarged)
        preferred_aseg_name="aseg_enhanced_enlarged1r.nii.gz"
        fallback_aseg_name="aseg_enhanced1r.nii.gz"
        fallback_variant="enhanced"
        ;;
    esac

    selected_freesurfer_variant="$freesurfer_final"
    if [[ -f "$work_dir/$preferred_aseg_name" ]]; then
      final_aseg_name="$preferred_aseg_name"
    elif [[ -f "$work_dir/$fallback_aseg_name" ]]; then
      final_aseg_name="$fallback_aseg_name"
      selected_freesurfer_variant="$fallback_variant"
      log "Preferred FreeSurfer variant '$freesurfer_final' is unavailable; falling back to '$fallback_variant'"
    else
      echo "No postprocessed FreeSurfer aseg is available in $work_dir" >&2
      exit 4
    fi

    case "$synthseg_final" in
      raw_robust) final_synthseg_name="T1-ss1rrf.nii.gz" ;;
      enhanced_robust) final_synthseg_name="oT1e-ss1rrf.nii.gz" ;;
      raw_standard) final_synthseg_name="T1-ss2rrf.nii.gz" ;;
      enhanced_standard) final_synthseg_name="oT1e-ss2rrf.nii.gz" ;;
    esac

    log "FreeSurfer source=$selected_freesurfer_variant file=$final_aseg_name"
    log "SynthSeg source=$synthseg_final file=$final_synthseg_name"

    validate=(T1.nii.gz T1-skullstripped-tissue.nii.gz mask.nii.gz \
              "$final_synthseg_name" "$final_aseg_name")
    args=(
      --subject-dir "$work_dir"
      --aseg-name "$final_aseg_name"
      --mask-name mask.nii.gz
      --tissue-name T1-skullstripped-tissue.nii.gz
      --ss-name "$final_synthseg_name"
      --out-name "$final_name"
      --seed-erode 1
      --debug
    )
    if [[ "$use_charm" == "1" ]]; then
      validate+=(mask-csf.nii.gz charm_label_native.nii.gz)
      args+=(--mask-csf-name mask-csf.nii.gz --label-name charm_label_native.nii.gz)
    fi
    "$python_exe" "$code_dir/validate_inputs.py" --subject-dir "$work_dir" --files "${validate[@]}"
    "$python_exe" "$code_dir/consolidate_labels.py" "${args[@]}"
    ;;

  *)
    echo "Unknown stage: $stage" >&2
    exit 64
    ;;
esac
