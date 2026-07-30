#!/usr/bin/env bash
set -euo pipefail

EXAMPLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$EXAMPLE_DIR/.." && pwd)"
SITE_CONFIG="${MEDLEY_SITE_CONFIG:-$REPO_ROOT/config/site.env}"

if [[ ! -f "$SITE_CONFIG" ]]; then
  cat >&2 <<EOF
Missing site configuration: $SITE_CONFIG

Create it with:
  cp "$REPO_ROOT/config/site.env.example" "$REPO_ROOT/config/site.env"

Then edit the paths for your cluster. config/site.env is ignored by Git.
EOF
  exit 2
fi

set -a
# shellcheck disable=SC1090
source "$SITE_CONFIG"
set +a

required_vars=(
  RAW_ROOT WORK_ROOT IBEAT_BIND_ROOT IBEAT_OUTPUT_ROOT IBEAT_SIF
  FS_SUBJECTS_DIR SYNTHSEG_ROOT CHARM_ROOT CONDA_BASE CONDA_ENV
  ACCOUNT GPU_PARTITION CPU_PARTITION LONG_PARTITION FREESURFER_HOME
  SIMNIBS_ENV_SCRIPT
)
for name in "${required_vars[@]}"; do
  [[ -n "${!name:-}" ]] || {
    echo "Missing required setting in $SITE_CONFIG: $name" >&2
    exit 2
  }
done

"$REPO_ROOT/scripts/preflight.sh" --config "$SITE_CONFIG"

args=(
  --raw-root "$RAW_ROOT"
  --work-root "$WORK_ROOT"
  --ibeat-bind-root "$IBEAT_BIND_ROOT"
  --ibeat-output-root "$IBEAT_OUTPUT_ROOT"
  --ibeat-sif "$IBEAT_SIF"
  --fs-subjects-dir "$FS_SUBJECTS_DIR"
  --synthseg-root "$SYNTHSEG_ROOT"
  --charm-root "$CHARM_ROOT"
  --scale-map "$REPO_ROOT/config/babyopenbrains_scales.tsv"
  --conda-base "$CONDA_BASE"
  --conda-env "$CONDA_ENV"
  --simnibs-env-script "$SIMNIBS_ENV_SCRIPT"
  --charm-input "${CHARM_INPUT:-oT1e-18-big3.nii.gz}"
  --charm-forceqform "${CHARM_FORCE_QFORM:-1}"
  --charm-forcerun "${CHARM_FORCE_RUN:-1}"
  --freesurfer-final "${FREESURFER_FINAL:-enhanced_enlarged}"
  --synthseg-final "${SYNTHSEG_FINAL:-raw_robust}"
  --gpu-partition "$GPU_PARTITION"
  --cpu-partition "$CPU_PARTITION"
  --long-partition "$LONG_PARTITION"
  --freesurfer-home "$FREESURFER_HOME"
  --account "$ACCOUNT"
  --ages "${AGES:-1-8}"
  --ibeat-parallel "${IBEAT_PARALLEL:-5}"
  --gather-parallel "${GATHER_PARALLEL:-12}"
  --preprocess-parallel "${PREPROCESS_PARALLEL:-8}"
  --synthseg-parallel "${SYNTHSEG_PARALLEL:-8}"
  --freesurfer-parallel "${FREESURFER_PARALLEL:-4}"
  --charm-parallel "${CHARM_PARALLEL:-4}"
  --postprocess-parallel "${POSTPROCESS_PARALLEL:-8}"
  --consolidate-parallel "${CONSOLIDATE_PARALLEL:-4}"
)
[[ -n "${EMAIL:-}" ]] && args+=(--email "$EMAIL")

exec "$REPO_ROOT/run_medley.sh" "${args[@]}" "$@"
