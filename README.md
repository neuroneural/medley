# Medley

An anatomy-aware, multi-tool curation pipeline for infant brain MRI
segmentation. Medley runs several complementary tools over Baby Open Brains
T1w data, aligns their outputs to a common native grid, and consolidates them
into a single label map using deterministic source-priority rules plus
boundary repair, leakage suppression, and topology cleanup.

The workflow is submitted as a chain of dependent SLURM job arrays, one array
task per subject/session.

## Pipeline overview

```
iBEAT ─▶ gather ─▶ enhance / enlarge ─▶ ┌ SynthSeg (robust + standard)
                                        ├ FreeSurfer (enhanced)
                                        ├ FreeSurfer (enhanced + enlarged)
                                        └ CHARM (whole-head, optional)
                                              │
                          gather tool outputs ─▶ inverse-scale / resample /
                          register to native ─▶ consolidation ─▶ label map
```

Each stage is a SLURM array; stages are chained with `afterok` dependencies,
and the four tool stages fan out in parallel after preprocessing.

## Requirements

**Cluster / system**

- SLURM (with `sbatch`, `srun`, `sinfo`)
- Singularity or Apptainer (for the iBEAT container)
- FreeSurfer 7.4.1 — provides `recon-all`, `mri_synthseg`, `mri_synthstrip`
- SimNIBS / CHARM (only if using the whole-head CHARM stage)
- An iBEAT v2 Singularity image (`.sif`)

**Python** (a conda environment is recommended)

```
numpy
scipy
nibabel
```

`cupy` is optional. The consolidation step uses a CuPy GPU path when available
and falls back to SciPy on CPU otherwise. Install a CUDA-matched CuPy wheel
(e.g. `cupy-cuda12x`) only if you intend to run consolidation on a GPU;
otherwise run that stage on CPU (see *Resource notes*).

## Installation

```bash
git clone https://github.com/neuroneural/medley.git
cd medley

# create and populate a Python environment
conda create -n labelling python=3.9 -y
conda activate labelling
pip install -r requirements.txt
```

## Configuration

Site-specific paths live in `config/site.env`, which is **not** tracked by Git.
Create it from the template and edit for your cluster:

```bash
cp config/site.env.example config/site.env
$EDITOR config/site.env
```

Key settings:

- `RAW_ROOT`, `WORK_ROOT`, `IBEAT_*`, `FS_SUBJECTS_DIR`, `SYNTHSEG_ROOT`,
  `CHARM_ROOT` — data and output locations
- `CONDA_BASE`, `CONDA_ENV` — Python environment
- `ACCOUNT`, `GPU_PARTITION`, `CPU_PARTITION`, `LONG_PARTITION` — SLURM targets
- `FREESURFER_HOME` — FreeSurfer 7.4.1 install
- `FREESURFER_MODULE` — environment-module name used to load FreeSurfer
  (default `freesurfer/7.4.1`). Confirm the exact name on your cluster with
  `module avail freesurfer`.
- `FREESURFER_EXPECT_VERSION` — setup aborts if the loaded build does not
  contain this string (guards against picking up a different local build).
  Set empty to disable.
- `SIMNIBS_ENV_SCRIPT` — SimNIBS activation script (required for CHARM)
- `*_PARALLEL` — per-stage array concurrency caps (see *Resource notes*)

Because `config/site.env` is git-ignored, your paths, account, and email are
never committed. Do not put credentials in it.

## Preflight

Validate the environment before submitting anything. The FreeSurfer/SynthSeg
check runs on a compute node (login nodes often lack `tcsh`):

```bash
scripts/preflight.sh --config config/site.env
```

Non-zero exit codes identify the failure: `11` SetUp script missing,
`12` license missing, `13` `SetUpFreeSurfer.sh` source error, `14` tool not on
`PATH`, `15` module load failed, `16` FreeSurfer version mismatch.

## Running

The example runner sources `config/site.env`, runs preflight, and submits:

```bash
examples/run_example.sh
```

Extra flags pass through, e.g. a safe rehearsal that submits nothing:

```bash
examples/run_example.sh --dry-run
```

Or call `run_medley.sh` directly:

```bash
./run_medley.sh \
  --raw-root       /path/to/raw/BabyOpen \
  --work-root      /path/to/medley_work \
  --ibeat-sif      /path/to/ibeat_v2.sif \
  --fs-subjects-dir /path/to/freesurfer_subjects \
  --scale-map      ./config/babyopenbrains_scales.tsv \
  --account        your_account \
  --gpu-partition  your_gpu_partition \
  --cpu-partition  your_cpu_partition \
  --long-partition your_long_partition \
  --freesurfer-home /path/to/freesurfer/7.4.1
```

### Useful options

| Option | Purpose |
|---|---|
| `--ages SPEC` | Sessions to process (default `1-8`; accepts `1-4,6,8`) |
| `--dry-run` | Build the manifest and print `sbatch` commands without submitting |
| `--skip-ibeat` / `--skip-synthseg` / `--skip-freesurfer` / `--skip-charm` | Reuse existing outputs for a stage |
| `--freesurfer-final MODE` | Which FreeSurfer aseg feeds consolidation: `enhanced` or `enhanced_enlarged` (default) |
| `--synthseg-final MODE` | Which SynthSeg map feeds consolidation: `raw_robust` (default), `enhanced_robust`, `raw_standard`, `enhanced_standard` |
| `--overwrite` | Recompute and replace existing outputs |
| `--run-dir DIR` | Where logs and the subject manifest are written |

Run `./run_medley.sh --help` for the complete list.

## Outputs and monitoring

The final label map is written per subject as:

```
<work-root>/<age>/<subject>/medley_segmentation.nii.gz
```

Check progress against the manifest at any time:

```bash
python scripts/check_status.py <run-dir>/subjects.tsv \
  --final-name medley_segmentation.nii.gz
```

## FreeSurfer loading

FreeSurfer is loaded through the cluster module system and then completed by
sourcing `SetUpFreeSurfer.sh`. The loader clears inherited FreeSurfer variables
first, so a build auto-loaded by a login profile cannot leak into the run, and
it verifies the loaded version against `FREESURFER_EXPECT_VERSION`. Two
FreeSurfer variants are produced per subject (`enhanced` and
`enhanced + enlarged`); `--freesurfer-final` selects which one is consolidated.

## CHARM

The whole-head CHARM stage is optional (`--skip-charm` to disable). When
enabled, `SIMNIBS_ENV_SCRIPT` must point at a valid SimNIBS activation script;
if it is empty the loader falls back to `module load simnibs`.

## Resource notes

Concurrency is controlled per stage by the `*_PARALLEL` caps in `site.env`,
not by the number of subjects: each stage is an array of all subjects with a
`%N` throttle. Guidance:

- Keep FreeSurfer and GPU stages at low concurrency; run the light CPU stages
  (gather, postprocess) wider.
- FreeSurfer runs twice per subject, so FreeSurfer job counts are `2 ×
  subjects`.
- Send CPU-only stages to a CPU partition, not a GPU partition.
- The consolidation stage requests a GPU only if you intend to use CuPy;
  without CuPy installed, run it on CPU and drop the GPU request.

Before scaling up, check your cluster's limits (`sinfo -o "%P %l %c %m %G"`,
`sacctmgr show qos`) and size `--mem` / `--cpus-per-task` from observed usage
(`seff <jobid>` or `sacct ... MaxRSS,TotalCPU,Elapsed`).

## Citation

If you use Medley, please cite the associated manuscript and this software.
Citation metadata is in [`CITATION.cff`](CITATION.cff).

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
