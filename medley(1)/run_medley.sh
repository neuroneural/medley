#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Submit the full Medley workflow as subject-independent dependent SLURM arrays:
  iBEAT -> gather -> enhance/enlarge -> SynthSeg/two FreeSurfer variants/CHARM
  -> gather tool outputs -> inverse-scale/resample/register -> consolidation

Required:
  --raw-root DIR             Raw Baby Open Brains root with ses-1mo ... ses-8mo
  --work-root DIR            Working root that will contain AGE/sub-* folders
  --ibeat-sif FILE           iBEAT Singularity image
  --fs-subjects-dir DIR      FreeSurfer SUBJECTS_DIR
  --scale-map FILE           Two-column file: AGE SCALE, e.g. 1m 1.33

Usually needed:
  --ibeat-bind-root DIR      Host directory bound to /InfantData
                             (default: parent of --raw-root)
  --ibeat-output-root DIR    Host root for T1-AGE iBEAT outputs
                             (default: --raw-root)
  --synthseg-root DIR        Separate SynthSeg output root
                             (default: WORK_PARENT/tool_outputs/synthseg)
  --charm-root DIR           Separate CHARM output root
                             (default: WORK_PARENT/tool_outputs/charm)
  --simnibs-env-script FILE  Script to source before running charm
  --conda-env NAME_OR_PATH   Conda environment name or full prefix path
                             (default: labelling)
  --conda-base DIR           Conda installation root containing etc/profile.d/conda.sh
                             (auto-detected when omitted)
  --charm-input NAME         Input inside each work folder
                             (default: oT1e-18-big3.nii.gz)
  --charm-forceqform BOOL    Pass --forceqform to CHARM (default: 1)
  --charm-forcerun BOOL      Pass --forcerun to CHARM (default: 1)
  --synthseg-final MODE      SynthSeg map passed to final consolidation:
                             raw_robust, enhanced_robust, raw_standard,
                             or enhanced_standard (default: raw_robust)

Optional control:
  --ages SPEC                Default: 1-8; accepts 1-4,6,8
  --skip-ibeat               Reuse existing iBEAT outputs
  --skip-synthseg            Reuse existing SynthSeg output folders
  --skip-freesurfer          Do not run FreeSurfer; reuse at least one existing variant
  --freesurfer-final MODE    Preferred FreeSurfer aseg for final consolidation:
                             enhanced or enhanced_enlarged
                             automatically falls back if unavailable
                             (default: enhanced_enlarged)
  --skip-charm               Do not run or use CHARM whole-head labels
  --overwrite                Recompute/replace existing outputs
  --dry-run                  Build manifest and print sbatch commands only
  --run-dir DIR              Logs and manifest directory
  --final-name NAME          Default: medley_segmentation.nii.gz
  --email ADDRESS            SLURM mail address

Cluster settings:
  --account NAME             SLURM account
  --gpu-partition NAME       Partition for GPU stages
  --cpu-partition NAME       Partition for CPU stages
  --long-partition NAME      Partition for long-running stages
  --freesurfer-home DIR      FreeSurfer 7.4.1 installation directory

Example:
  ./run_medley.sh \
    --raw-root /path/to/raw/BabyOpen \
    --work-root /path/to/medley_work \
    --ibeat-bind-root /path/to/ibeat_root \
    --ibeat-sif /path/to/ibeat_v2.sif \
    --fs-subjects-dir /path/to/freesurfer_subjects \
    --scale-map ./config/babyopenbrains_scales.tsv \
    --account project_account \
    --gpu-partition gpu_partition \
    --cpu-partition cpu_partition \
    --long-partition long_partition \
    --freesurfer-home /path/to/freesurfer/7.4.1
EOF
}

original_cli=("$@")
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
code_dir="$repo_dir/scripts"
raw_root=""
work_root=""
ibeat_bind_root=""
ibeat_container_root="/InfantData"
ibeat_output_root=""
ibeat_sif=""
fs_subjects_dir=""
synthseg_root=""
charm_root=""
scale_map=""
simnibs_env_script=""
conda_env="labelling"
conda_base=""
python_exe=""
charm_input_name="oT1e-18-big3.nii.gz"
charm_forceqform=1
charm_forcerun=1
synthseg_final="raw_robust"
ages="1-8"
run_dir=""
final_name="medley_segmentation.nii.gz"
email=""
account=""
gpu_partition=""
cpu_partition=""
long_partition=""
freesurfer_home=""
freesurfer_final="enhanced_enlarged"
ibeat_parallel=5
gather_parallel=12
preprocess_parallel=8
synthseg_parallel=8
freesurfer_parallel=4
charm_parallel=4
postprocess_parallel=8
consolidate_parallel=4
skip_ibeat=0
skip_synthseg=0
skip_freesurfer=0
skip_charm=0
overwrite=0
dry_run=0

while (($#)); do
  case "$1" in
    --raw-root) raw_root="$2"; shift 2;;
    --work-root) work_root="$2"; shift 2;;
    --ibeat-bind-root) ibeat_bind_root="$2"; shift 2;;
    --ibeat-container-root) ibeat_container_root="$2"; shift 2;;
    --ibeat-output-root) ibeat_output_root="$2"; shift 2;;
    --ibeat-sif) ibeat_sif="$2"; shift 2;;
    --fs-subjects-dir) fs_subjects_dir="$2"; shift 2;;
    --synthseg-root) synthseg_root="$2"; shift 2;;
    --charm-root) charm_root="$2"; shift 2;;
    --scale-map) scale_map="$2"; shift 2;;
    --simnibs-env-script) simnibs_env_script="$2"; shift 2;;
    --conda-env) conda_env="$2"; shift 2;;
    --conda-base) conda_base="$2"; shift 2;;
    --charm-input) charm_input_name="$2"; shift 2;;
    --charm-forceqform) charm_forceqform="$2"; shift 2;;
    --charm-forcerun) charm_forcerun="$2"; shift 2;;
    --synthseg-final) synthseg_final="$2"; shift 2;;
    --ages) ages="$2"; shift 2;;
    --run-dir) run_dir="$2"; shift 2;;
    --final-name) final_name="$2"; shift 2;;
    --email) email="$2"; shift 2;;
    --account) account="$2"; shift 2;;
    --gpu-partition) gpu_partition="$2"; shift 2;;
    --cpu-partition) cpu_partition="$2"; shift 2;;
    --long-partition) long_partition="$2"; shift 2;;
    --freesurfer-home) freesurfer_home="$2"; shift 2;;
    --freesurfer-final) freesurfer_final="$2"; shift 2;;
    --ibeat-parallel) ibeat_parallel="$2"; shift 2;;
    --gather-parallel) gather_parallel="$2"; shift 2;;
    --preprocess-parallel) preprocess_parallel="$2"; shift 2;;
    --synthseg-parallel) synthseg_parallel="$2"; shift 2;;
    --freesurfer-parallel) freesurfer_parallel="$2"; shift 2;;
    --charm-parallel) charm_parallel="$2"; shift 2;;
    --postprocess-parallel) postprocess_parallel="$2"; shift 2;;
    --consolidate-parallel) consolidate_parallel="$2"; shift 2;;
    --skip-ibeat) skip_ibeat=1; shift;;
    --skip-synthseg) skip_synthseg=1; shift;;
    --skip-freesurfer) skip_freesurfer=1; shift;;
    --skip-charm) skip_charm=1; shift;;
    --overwrite) overwrite=1; shift;;
    --dry-run) dry_run=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 64;;
  esac
done

activate_medley_conda() {
  if [[ -z "$conda_base" ]]; then
    if [[ -n "${CONDA_EXE:-}" ]]; then
      conda_base="$(cd "$(dirname "$CONDA_EXE")/.." && pwd)"
    elif command -v conda >/dev/null 2>&1; then
      conda_base="$(conda info --base)"
    else
      for candidate in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" "$HOME/miniforge3"; do
        if [[ -f "$candidate/etc/profile.d/conda.sh" ]]; then
          conda_base="$candidate"
          break
        fi
      done
    fi
  fi

  [[ -n "$conda_base" && -f "$conda_base/etc/profile.d/conda.sh" ]] || {
    echo "Could not locate conda.sh. Pass --conda-base /path/to/conda." >&2
    exit 2
  }

  # shellcheck disable=SC1090
  source "$conda_base/etc/profile.d/conda.sh"

  # Prefix environments outside CONDA_BASE/envs should be activated by full path.
  if [[ "$conda_env" == */* ]]; then
    [[ -d "$conda_env" ]] || {
      echo "Conda environment prefix does not exist: $conda_env" >&2
      exit 2
    }
    conda_env="$(cd "$conda_env" && pwd)"
  fi

  conda activate "$conda_env"

  if [[ "$conda_env" == */* && -x "$conda_env/bin/python" ]]; then
    python_exe="$conda_env/bin/python"
  else
    python_exe="$(command -v python)"
  fi

  "$python_exe" - <<'PYENV'
import sys
import numpy
import nibabel
import scipy
print(f"Medley Python: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")
PYENV
}

activate_medley_conda

[[ -d "$raw_root" ]] || { echo "Invalid --raw-root: $raw_root" >&2; exit 2; }
[[ -n "$work_root" ]] || { echo "--work-root is required" >&2; exit 2; }
[[ -f "$ibeat_sif" || "$skip_ibeat" == "1" || "$dry_run" == "1" ]] || { echo "Invalid --ibeat-sif: $ibeat_sif" >&2; exit 2; }
[[ -n "$fs_subjects_dir" ]] || { echo "--fs-subjects-dir is required" >&2; exit 2; }
[[ -f "$scale_map" ]] || { echo "Invalid --scale-map: $scale_map" >&2; exit 2; }
[[ -n "$account" ]] || { echo "--account is required" >&2; exit 2; }
[[ -n "$gpu_partition" ]] || { echo "--gpu-partition is required" >&2; exit 2; }
[[ -n "$cpu_partition" ]] || { echo "--cpu-partition is required" >&2; exit 2; }
[[ -n "$long_partition" ]] || { echo "--long-partition is required" >&2; exit 2; }
[[ -n "$freesurfer_home" ]] || { echo "--freesurfer-home is required" >&2; exit 2; }
[[ -d "$freesurfer_home" || "$dry_run" == "1" ]] || { echo "Invalid --freesurfer-home: $freesurfer_home" >&2; exit 2; }

case "$freesurfer_final" in
  enhanced|enhanced_enlarged) ;;
  *)
    echo "--freesurfer-final must be enhanced or enhanced_enlarged" >&2
    exit 2
    ;;
esac

case "$synthseg_final" in
  raw_robust|enhanced_robust|raw_standard|enhanced_standard) ;;
  *)
    echo "--synthseg-final must be raw_robust, enhanced_robust, raw_standard, or enhanced_standard" >&2
    exit 2
    ;;
esac

if [[ "$skip_charm" != "1" ]]; then
  [[ -n "$simnibs_env_script" && -f "$simnibs_env_script" ]] || {
    echo "CHARM is enabled, but --simnibs-env-script is missing or invalid: $simnibs_env_script" >&2
    exit 2
  }
fi

if [[ -z "$ibeat_bind_root" ]]; then
  ibeat_bind_root="$(dirname "$raw_root")"
fi
if [[ -z "$ibeat_output_root" ]]; then
  ibeat_output_root="$raw_root"
fi
work_parent="$(dirname "$work_root")"
[[ -n "$synthseg_root" ]] || synthseg_root="$work_parent/tool_outputs/synthseg"
[[ -n "$charm_root" ]] || charm_root="$work_parent/tool_outputs/charm"
if [[ -z "$run_dir" ]]; then
  run_dir="$PWD/medley_run_$(date +%Y%m%d_%H%M%S)"
fi

for file in build_manifest.py medley_worker.sh preflight.sh enhance_t1.py scale_nifti.py \
            resample_labels.py validate_inputs.py consolidate_labels.py \
            label_utils.py check_status.py; do
  [[ -f "$code_dir/$file" ]] || { echo "Missing pipeline file: $code_dir/$file" >&2; exit 2; }
done

mkdir -p "$run_dir/logs" "$work_root" "$ibeat_output_root" "$fs_subjects_dir" "$synthseg_root" "$charm_root"
manifest="$run_dir/subjects.tsv"
provenance="$run_dir/provenance.txt"
{
  echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host=$(hostname)"
  echo "repository=$repo_dir"
  printf 'command='; printf '%q ' "$0" "${original_cli[@]}"; echo
  if command -v git >/dev/null 2>&1 && git -C "$repo_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "git_commit=$(git -C "$repo_dir" rev-parse HEAD)"
  else
    echo "git_commit=not-a-git-checkout"
  fi
  echo "freesurfer_final=$freesurfer_final"
  echo "synthseg_final=$synthseg_final"
  echo "charm_input=$charm_input_name"
  echo "simnibs_env_script=$simnibs_env_script"
  echo "scale_map=$scale_map"
  echo "script_sha256:"
  sha256sum "$repo_dir/run_medley.sh" "$code_dir"/*.sh "$code_dir"/*.py "$scale_map"
} > "$provenance"
cp -f "$scale_map" "$run_dir/scale_map.tsv"
"$python_exe" "$code_dir/build_manifest.py" \
  --raw-root "$raw_root" \
  --work-root "$work_root" \
  --ibeat-output-root "$ibeat_output_root" \
  --synthseg-root "$synthseg_root" \
  --charm-root "$charm_root" \
  --fs-subjects-dir "$fs_subjects_dir" \
  --scale-map "$scale_map" \
  --ages "$ages" \
  --output "$manifest"
count="$(wc -l < "$manifest")"
(( count > 0 )) || { echo "Manifest is empty" >&2; exit 2; }

# Do not leave Conda active while calling the cluster Slurm client.
# Workers activate the same environment only for stages that execute Python.
conda deactivate >/dev/null 2>&1 || true

mail_args=()
if [[ -n "$email" ]]; then
  mail_args=(--mail-user="$email" --mail-type=FAIL,END)
fi
common=(--account="$account" --nodes=1 --ntasks=1 --time=20:00:00 \
  --kill-on-invalid-dep=yes "${mail_args[@]}")
log_pattern="$run_dir/logs/%x-%A_%a.out"
err_pattern="$run_dir/logs/%x-%A_%a.err"
worker="$code_dir/medley_worker.sh"
use_charm=1
[[ "$skip_charm" == "1" ]] && use_charm=0
worker_args=(
  "$manifest" "$code_dir" "$ibeat_bind_root" "$ibeat_container_root" "$ibeat_sif"
  "$fs_subjects_dir" "$freesurfer_home" "$simnibs_env_script" "$charm_input_name"
  "$overwrite" "$final_name" "$use_charm" "$conda_base" "$conda_env" "$python_exe"
  "$freesurfer_final" "$charm_forceqform" "$charm_forcerun" "$synthseg_final"
)

submit() {
  local label="$1"
  shift
  if (( dry_run )); then
    printf 'sbatch ' >&2
    printf '%q ' "$@" >&2
    printf '\n' >&2
    echo "DRYRUN_${label}"
  else
    sbatch --parsable "$@"
  fi
}

dep_args() {
  local ids=()
  local id
  for id in "$@"; do
    [[ -n "$id" ]] && ids+=("$id")
  done
  if ((${#ids[@]})); then
    local joined
    joined="$(IFS=:; echo "${ids[*]}")"
    # All Medley stages are arrays over the same manifest. aftercorr makes
    # task N wait only for task N of each upstream array, so a failure for one
    # subject cannot block unrelated subjects.
    printf '%s\n' "--dependency=aftercorr:$joined"
  fi
}

ibeat_job=""
if [[ "$skip_ibeat" != "1" ]]; then
  ibeat_job="$(submit ibeat "${common[@]}" --partition="$gpu_partition" --gres=gpu:A40:1 \
    --cpus-per-task=8 --mem=32G --job-name=medley-ibeat --array="1-${count}%${ibeat_parallel}" \
    --output="$log_pattern" --error="$err_pattern" \
    "$worker" ibeat "${worker_args[@]}")"
fi

gather_ibeat_dep=()
[[ -n "$ibeat_job" ]] && gather_ibeat_dep=("$(dep_args "$ibeat_job")")
gather_ibeat_job="$(submit gather_ibeat "${common[@]}" "${gather_ibeat_dep[@]}" \
  --partition="$cpu_partition" --cpus-per-task=1 --mem=8G \
  --job-name=medley-gather-ibeat --array="1-${count}%${gather_parallel}" \
  --output="$log_pattern" --error="$err_pattern" \
  "$worker" gather_ibeat "${worker_args[@]}")"

preprocess_job="$(submit preprocess "${common[@]}" "$(dep_args "$gather_ibeat_job")" \
  --partition="$cpu_partition" --cpus-per-task=2 --mem=24G \
  --job-name=medley-pre --array="1-${count}%${preprocess_parallel}" \
  --output="$log_pattern" --error="$err_pattern" \
  "$worker" preprocess "${worker_args[@]}")"

tool_jobs=()
synthseg_job=""
if [[ "$skip_synthseg" != "1" ]]; then
  synthseg_job="$(submit synthseg "${common[@]}" "$(dep_args "$preprocess_job")" \
    --partition="$cpu_partition" --cpus-per-task=8 --mem=32G \
    --job-name=medley-synthseg --array="1-${count}%${synthseg_parallel}" \
    --output="$log_pattern" --error="$err_pattern" \
    "$worker" synthseg "${worker_args[@]}")"
  tool_jobs+=("$synthseg_job")
fi

freesurfer_enhanced_job=""
freesurfer_enlarged_job=""
if [[ "$skip_freesurfer" != "1" ]]; then
  freesurfer_enhanced_job="$(submit freesurfer_enhanced "${common[@]}" "$(dep_args "$preprocess_job")" \
    --partition="$long_partition" --cpus-per-task=8 --mem=32G \
    --job-name=medley-fs-enh --array="1-${count}%${freesurfer_parallel}" \
    --output="$log_pattern" --error="$err_pattern" \
    "$worker" freesurfer_enhanced "${worker_args[@]}")"

  freesurfer_enlarged_job="$(submit freesurfer_enlarged "${common[@]}" "$(dep_args "$preprocess_job")" \
    --partition="$long_partition" --cpus-per-task=8 --mem=32G \
    --job-name=medley-fs-enlarged --array="1-${count}%${freesurfer_parallel}" \
    --output="$log_pattern" --error="$err_pattern" \
    "$worker" freesurfer_enlarged "${worker_args[@]}")"

  tool_jobs+=("$freesurfer_enhanced_job" "$freesurfer_enlarged_job")
fi

charm_job=""
if [[ "$skip_charm" != "1" ]]; then
  charm_job="$(submit charm "${common[@]}" "$(dep_args "$preprocess_job")" \
    --partition="$long_partition" --cpus-per-task=8 --mem=32G \
    --job-name=medley-charm --array="1-${count}%${charm_parallel}" \
    --output="$log_pattern" --error="$err_pattern" \
    "$worker" charm "${worker_args[@]}")"
  tool_jobs+=("$charm_job")
fi

# If one or more tools are being reused, gather_tools validates their existing outputs.
# Each gather task waits only for the corresponding subject task from every tool
# submitted in this run. A missing/failing subject does not block other subjects.
gather_tools_dependencies=("$preprocess_job" "${tool_jobs[@]}")
gather_tools_job="$(submit gather_tools "${common[@]}" "$(dep_args "${gather_tools_dependencies[@]}")" \
  --partition="$cpu_partition" --cpus-per-task=1 --mem=12G \
  --job-name=medley-gather-tools --array="1-${count}%${gather_parallel}" \
  --output="$log_pattern" --error="$err_pattern" \
  "$worker" gather_tools "${worker_args[@]}")"

postprocess_job="$(submit postprocess "${common[@]}" "$(dep_args "$gather_tools_job")" \
  --partition="$cpu_partition" --cpus-per-task=4 --mem=16G \
  --job-name=medley-post --array="1-${count}%${postprocess_parallel}" \
  --output="$log_pattern" --error="$err_pattern" \
  "$worker" postprocess "${worker_args[@]}")"

consolidate_job="$(submit consolidate "${common[@]}" "$(dep_args "$postprocess_job")" \
  --partition="$gpu_partition" --gres=gpu:A40:1 --cpus-per-task=4 --mem=16G \
  --job-name=medley-consolidate --array="1-${count}%${consolidate_parallel}" \
  --output="$log_pattern" --error="$err_pattern" \
  "$worker" consolidate "${worker_args[@]}")"

cat <<EOF
Medley pipeline prepared for $count subjects.
Run directory:       $run_dir
Manifest:            $manifest
Working root:        $work_root
iBEAT output root:   $ibeat_output_root
SynthSeg output root:$synthseg_root
FreeSurfer root:     $fs_subjects_dir
FreeSurfer preferred:$freesurfer_final
SynthSeg final:      $synthseg_final
CHARM output root:   $charm_root
SimNIBS activation:  $simnibs_env_script
CHARM force qform:   $charm_forceqform
CHARM force rerun:   $charm_forcerun
Final output:        <work-root>/<age>/<subject>/$final_name
Conda environment:   $conda_env
Python executable:   $python_exe

iBEAT:               ${ibeat_job:-SKIPPED}
Gather iBEAT:        $gather_ibeat_job
Preprocess:          $preprocess_job
SynthSeg:            ${synthseg_job:-REUSED/SKIPPED}
FreeSurfer enhanced: ${freesurfer_enhanced_job:-REUSED/SKIPPED}
FreeSurfer enhanced+enlarged: ${freesurfer_enlarged_job:-REUSED/SKIPPED}
CHARM:               ${charm_job:-DISABLED/REUSED}
Gather tools:        $gather_tools_job
Postprocess:         $postprocess_job
Consolidation:       $consolidate_job

Check progress later with:
  "$python_exe" "$code_dir/check_status.py" "$manifest" --final-name "$final_name"
EOF
