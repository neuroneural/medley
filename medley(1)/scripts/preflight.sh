#!/usr/bin/env bash
set -euo pipefail

usage() { echo "Usage: $0 --config /path/to/config/site.env" >&2; }
config=""
while (($#)); do
  case "$1" in
    --config) config="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown argument: $1" >&2; usage; exit 64;;
  esac
done
[[ -f "$config" ]] || { echo "Configuration file not found: $config" >&2; exit 2; }

set -a
# shellcheck disable=SC1090
source "$config"
set +a

fail() { echo "PREFLIGHT ERROR: $*" >&2; exit 2; }
ok() { echo "PREFLIGHT OK: $*"; }

for directory in "$RAW_ROOT" "$IBEAT_BIND_ROOT" "$CONDA_BASE" "$CONDA_ENV" "$FREESURFER_HOME"; do
  [[ -d "$directory" ]] || fail "directory not found: $directory"
done
[[ -f "$IBEAT_SIF" ]] || fail "iBEAT container not found: $IBEAT_SIF"
[[ -f "$SIMNIBS_ENV_SCRIPT" ]] || fail "SimNIBS activation script not found: $SIMNIBS_ENV_SCRIPT"
[[ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]] || fail "conda.sh not found under $CONDA_BASE"
[[ -x "$CONDA_ENV/bin/python" ]] || fail "Python not found: $CONDA_ENV/bin/python"
[[ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]] || fail "SetUpFreeSurfer.sh not found"
[[ -f "$FREESURFER_HOME/license.txt" ]] || fail "FreeSurfer license.txt not found"

command -v sbatch >/dev/null || fail "sbatch is not available"
command -v sinfo >/dev/null || fail "sinfo is not available"
if ! command -v singularity >/dev/null 2>&1 && ! command -v apptainer >/dev/null 2>&1; then
  module load singularity 2>/dev/null || true
fi
command -v singularity >/dev/null || command -v apptainer >/dev/null || fail "Singularity/Apptainer is not available"
sinfo >/dev/null 2>&1 || fail "Slurm is not responding on this node"
ok "Slurm client"

"$CONDA_ENV/bin/python" - <<'PYCODE'
import sys
import numpy, scipy, nibabel
print(f"PREFLIGHT OK: Python {sys.version.split()[0]} at {sys.executable}")
PYCODE

# The login node may not provide /bin/tcsh even though qTRDGPU compute
# nodes do. Test FreeSurfer where the actual jobs will run.
echo "PREFLIGHT: testing FreeSurfer via module '${FREESURFER_MODULE:-freesurfer/7.4.1}' on a $LONG_PARTITION compute node"

srun \
  --partition="$LONG_PARTITION" \
  --account="$ACCOUNT" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=1 \
  --mem=2G \
  --time=00:05:00 \
  env MEDLEY_FS_HOME="$FREESURFER_HOME" \
      MEDLEY_FS_MODULE="${FREESURFER_MODULE:-freesurfer/7.4.1}" \
      MEDLEY_FS_EXPECT="${FREESURFER_EXPECT_VERSION:-7.4.1}" \
  bash -lc '
    set -uo pipefail
    if ! command -v module >/dev/null 2>&1; then
      for i in /etc/profile.d/modules.sh /usr/share/Modules/init/bash /usr/share/lmod/lmod/init/bash; do
        [[ -r "$i" ]] && { source "$i"; break; }
      done
    fi
    # Clear scalars before load (this modulefile uses prepend-path on them).
    unset FREESURFER_HOME FSFAST_HOME MNI_DIR SUBJECTS_DIR \
          MINC_BIN_DIR MINC_LIB_DIR MNI_DATAPATH FS_LICENSE || true
    if command -v module >/dev/null 2>&1; then
      module unload freesurfer 2>/dev/null || true
      module load "$MEDLEY_FS_MODULE" || { echo "module load $MEDLEY_FS_MODULE failed" >&2; exit 15; }
    fi
    [[ -n "${FREESURFER_HOME:-}" ]] || export FREESURFER_HOME="$MEDLEY_FS_HOME"
    echo "Node: $HOSTNAME"
    echo "module: $MEDLEY_FS_MODULE"
    echo "FREESURFER_HOME=$FREESURFER_HOME"

    [[ -f "$FREESURFER_HOME/SetUpFreeSurfer.sh" ]] || { echo "MISSING SetUpFreeSurfer.sh under FREESURFER_HOME" >&2; exit 11; }
    export FS_LICENSE="${FS_LICENSE:-$FREESURFER_HOME/license.txt}"
    [[ -f "$FS_LICENSE" ]] || { echo "MISSING license: $FS_LICENSE" >&2; exit 12; }
    export SUBJECTS_DIR="$(mktemp -d)"
    trap '\''rm -rf "$SUBJECTS_DIR"'\'' EXIT

    set +eu
    source "$FREESURFER_HOME/SetUpFreeSurfer.sh"
    src_rc=$?
    set -u
    [[ $src_rc -eq 0 ]] || { echo "SetUpFreeSurfer.sh returned $src_rc" >&2; exit 13; }

    miss=""
    for t in tcsh recon-all mri_synthseg mri_synthstrip; do
      command -v "$t" >/dev/null || miss="$miss $t"
    done
    [[ -z "$miss" ]] || { echo "MISSING on $HOSTNAME:$miss" >&2; exit 14; }

    ver="$(recon-all -version 2>&1 | head -n1)"
    echo "recon-all:      $(command -v recon-all)  [$ver]"
    echo "mri_synthseg:   $(command -v mri_synthseg)"
    echo "mri_synthstrip: $(command -v mri_synthstrip)"

    if [[ -n "$MEDLEY_FS_EXPECT" && "$ver" != *"$MEDLEY_FS_EXPECT"* && "$FREESURFER_HOME" != *"$MEDLEY_FS_EXPECT"* ]]; then
      echo "VERSION MISMATCH: expected $MEDLEY_FS_EXPECT, got [$ver] at $FREESURFER_HOME" >&2
      exit 16
    fi
  ' \
  || fail "FreeSurfer module check failed on $LONG_PARTITION. Exit codes: 11=SetUp script missing, 12=license missing, 13=source error, 14=tool(s) missing, 15=module load failed, 16=version mismatch. See the line above."
ok "FreeSurfer, SynthSeg, and SynthStrip on $LONG_PARTITION"

(
  set -eo pipefail
  simnibs_bin="$(dirname "$SIMNIBS_ENV_SCRIPT")"
  export PATH="$simnibs_bin:$PATH"
  set +u
  # shellcheck disable=SC1090
  source "$SIMNIBS_ENV_SCRIPT"
  set -u
  export PATH="$simnibs_bin:$PATH"
  hash -r
  command -v charm >/dev/null
  echo "PREFLIGHT OK: CHARM at $(command -v charm)"
) || fail "SimNIBS CHARM initialization failed"

for age in {1..8}; do
  [[ -d "$RAW_ROOT/ses-${age}mo" ]] || fail "raw session directory missing: $RAW_ROOT/ses-${age}mo"
done
ok "Baby Open Brains session directories"

mkdir -p "$WORK_ROOT" "$IBEAT_OUTPUT_ROOT" "$FS_SUBJECTS_DIR" "$SYNTHSEG_ROOT" "$CHARM_ROOT"
ok "output directories"
echo "Preflight complete. The pipeline is ready to submit."
