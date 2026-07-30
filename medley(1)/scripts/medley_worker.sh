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
  if [[ -f "$explicit/aseg.mgz" && -f "$explicit/rawavg.mgz" ]]; then
    printf '%s\n' "$explicit"
    return 0
  fi

  # Backward-compatible reuse of the former enhanced+enlarged layout:
  #   <fs-root>/<age>/<subject>/mri
  if [[ "$variant" == "enhanced_enlarged" ]]; then
    legacy="$fs_subjects_dir/$age_name/$fs_id/mri"
    if [[ -f "$legacy/aseg.mgz" && -f "$legacy/rawavg.mgz" ]]; then
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

  # Clear scalar vars BEFORE loading: this modulefile uses prepend-path on
  # FREESURFER_HOME/SUBJECTS_DIR/MNI_DIR, so a value left by a login profile
  # (e.g. the personal freesurfer-new dev build) would concatenate into a
  # broken colon-list. Also drops an auto-loaded dev FreeSurfer.
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

  # The modulefile does NOT source SetUpFreeSurfer.sh nor set the MINC/mni
  # PATH + PERL5LIB that a full `recon-all -all` needs; complete it here.
  # SetUpFreeSurfer.sh is not errexit/nounset clean -> disable both.
  set +eu
  # shellcheck disable=SC1090
  source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
  set -eu

  # License (module does not set it) and per-variant output tree
  # (module points SUBJECTS_DIR at $modroot/subjects; override it).
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
  for _t in recon-all mri_synthseg mri_synthstrip; do
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
    # These two masks are intentionally generated on the native working grid,
    # so babyHead does not need to inverse-scale a skull-strip mask.
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

    fs_enhanced_mri="$(find_fs_mri_dir enhanced || true)"
    [[ -f "$fs_enhanced_mri/aseg.mgz" && -f "$fs_enhanced_mri/rawavg.mgz" ]] || {
      echo "Missing enhanced-only FreeSurfer outputs under: $fs_enhanced_mri" >&2
      exit 4
    }
    atomic_copy "$fs_enhanced_mri/aseg.mgz" "$work_dir/fs_enhanced_aseg.mgz"
    atomic_copy "$fs_enhanced_mri/rawavg.mgz" "$work_dir/fs_enhanced_rawavg.mgz"
    [[ -f "$fs_enhanced_mri/orig.mgz" ]] && \
      atomic_copy "$fs_enhanced_mri/orig.mgz" "$work_dir/fs_enhanced_orig.mgz"

    fs_enlarged_mri="$(find_fs_mri_dir enhanced_enlarged || true)"
    [[ -f "$fs_enlarged_mri/aseg.mgz" && -f "$fs_enlarged_mri/rawavg.mgz" ]] || {
      echo "Missing enhanced+enlarged FreeSurfer outputs under: $fs_enlarged_mri" >&2
      exit 4
    }
    atomic_copy "$fs_enlarged_mri/aseg.mgz" "$work_dir/fs_enlarged_aseg.mgz"
    atomic_copy "$fs_enlarged_mri/rawavg.mgz" "$work_dir/fs_enlarged_rawavg.mgz"
    [[ -f "$fs_enlarged_mri/orig.mgz" ]] && \
      atomic_copy "$fs_enlarged_mri/orig.mgz" "$work_dir/fs_enlarged_orig.mgz"

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
    setup_freesurfer runtime
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
      "$python_exe" "$code_dir/resample_labels.py" \
        "$work_dir/$restored" "$work_dir/T1.nii.gz" "$work_dir/$aligned" "${ow[@]}"
    done

    # FreeSurfer on enhanced-only T1: return from conformed space directly
    # to the native enhanced grid, which has the same geometry as T1.nii.gz.
    run_if_needed "$work_dir/aseg_enhanced_native.nii.gz" \
      mri_vol2vol --mov "$work_dir/fs_enhanced_aseg.mgz" \
        --targ "$work_dir/fs_enhanced_rawavg.mgz" \
        --regheader --o "$work_dir/aseg_enhanced_native.nii.gz" \
        --no-save-reg --interp nearest
    "$python_exe" "$code_dir/resample_labels.py" \
      "$work_dir/aseg_enhanced_native.nii.gz" "$work_dir/T1.nii.gz" \
      "$work_dir/aseg_enhanced1r.nii.gz" "${ow[@]}"

    # FreeSurfer on enhanced+enlarged T1: return to the enlarged native grid,
    # undo the age-specific enlargement, then align to the original T1 grid.
    run_if_needed "$work_dir/aseg_enhanced_enlarged_native.nii.gz" \
      mri_vol2vol --mov "$work_dir/fs_enlarged_aseg.mgz" \
        --targ "$work_dir/fs_enlarged_rawavg.mgz" \
        --regheader --o "$work_dir/aseg_enhanced_enlarged_native.nii.gz" \
        --no-save-reg --interp nearest
    "$python_exe" "$code_dir/scale_nifti.py" \
      "$work_dir/aseg_enhanced_enlarged_native.nii.gz" \
      "$work_dir/aseg_enhanced_enlarged_unscaled.nii.gz" \
      --scale "$inverse_scale" --order 0 "${ow[@]}"
    "$python_exe" "$code_dir/resample_labels.py" \
      "$work_dir/aseg_enhanced_enlarged_unscaled.nii.gz" "$work_dir/T1.nii.gz" \
      "$work_dir/aseg_enhanced_enlarged1r.nii.gz" "${ow[@]}"

    if [[ "$use_charm" == "1" ]]; then
      "$python_exe" "$code_dir/resample_labels.py" \
        "$work_dir/charm_label_source.nii.gz" "$work_dir/$charm_input_name" \
        "$work_dir/charm_label_enlarged_grid.nii.gz" "${ow[@]}"
      "$python_exe" "$code_dir/scale_nifti.py" \
        "$work_dir/charm_label_enlarged_grid.nii.gz" "$work_dir/charm_label_unscaled.nii.gz" \
        --scale "$inverse_scale" --order 0 "${ow[@]}"
      "$python_exe" "$code_dir/resample_labels.py" \
        "$work_dir/charm_label_unscaled.nii.gz" "$work_dir/T1.nii.gz" \
        "$work_dir/charm_label_native.nii.gz" "${ow[@]}"
    fi

    validate=(T1.nii.gz T1-skullstripped-tissue.nii.gz mask.nii.gz mask-csf.nii.gz T1-ss1rrf.nii.gz aseg_enhanced1r.nii.gz aseg_enhanced_enlarged1r.nii.gz)
    [[ "$use_charm" == "1" ]] && validate+=(charm_label_native.nii.gz)
    "$python_exe" "$code_dir/validate_inputs.py" --subject-dir "$work_dir" --files "${validate[@]}"
    ;;

  consolidate)
    if [[ -f "$work_dir/$final_name" && "$overwrite" != "1" ]]; then
      log "Final output already exists: $work_dir/$final_name"
      exit 0
    fi
    case "$freesurfer_final" in
      enhanced)
        final_aseg_name="aseg_enhanced1r.nii.gz"
        ;;
      enhanced_enlarged)
        final_aseg_name="aseg_enhanced_enlarged1r.nii.gz"
        ;;
    esac

    case "$synthseg_final" in
      raw_robust) final_synthseg_name="T1-ss1rrf.nii.gz" ;;
      enhanced_robust) final_synthseg_name="oT1e-ss1rrf.nii.gz" ;;
      raw_standard) final_synthseg_name="T1-ss2rrf.nii.gz" ;;
      enhanced_standard) final_synthseg_name="oT1e-ss2rrf.nii.gz" ;;
    esac

    log "FreeSurfer source=$freesurfer_final file=$final_aseg_name"
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
