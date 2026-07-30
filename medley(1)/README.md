# Medley

Medley is an anatomy-aware, multi-tool curation pipeline for infant brain MRI
segmentation. It combines iBEAT tissue maps, intensity enhancement,
age-dependent center-preserving scaling, FreeSurfer, SynthSeg, SynthStrip,
SimNIBS CHARM, and deterministic label consolidation.

## Pipeline

1. Run iBEAT and gather `T1.nii.gz` plus the three-tissue map.
2. Enhance the T1 image and apply the age-specific scale factor.
3. Run SynthSeg/SynthStrip, two FreeSurfer variants, and CHARM.
4. Gather tool outputs into each working subject directory.
5. Convert, inverse-scale, and resample label maps to the original T1 grid.
6. Validate geometry and run final Medley consolidation.

FreeSurfer is evaluated on:

- enhanced T1: `T1_enhanced-18.nii.gz`
- enhanced and enlarged T1: `oT1e-18-big3.nii.gz`

CHARM uses the enhanced and enlarged T1. The final FreeSurfer source is selected with `--freesurfer-final enhanced` or
`--freesurfer-final enhanced_enlarged`. The final SynthSeg source is separately
selectable as raw/enhanced and robust/standard; the default is `raw_robust`,
matching the historical Medley consolidation input.

## Repository layout

```text
medley/
├── run_medley.sh
├── README.md
├── CITATION.cff
├── requirements.txt
├── config/
│   ├── babyopenbrains_scales.tsv
│   └── site.env.example
├── examples/
│   └── run_example.sh
├── scripts/
│   ├── medley_worker.sh
│   ├── preflight.sh
│   ├── build_manifest.py
│   ├── enhance_t1.py
│   ├── scale_nifti.py
│   ├── resample_labels.py
│   ├── validate_inputs.py
│   ├── consolidate_labels.py
│   ├── label_utils.py
│   └── check_status.py
└── docs/
    └── FILE_MIGRATION.md
```

## Requirements

Python requirements:

```bash
python -m pip install -r requirements.txt
```

External tools are not installed by this repository:

- Singularity/Apptainer and iBEAT v2
- FreeSurfer 7.4.1
- SynthSeg and SynthStrip distributed with FreeSurfer
- SimNIBS CHARM
- SLURM

The example wrapper runs `scripts/preflight.sh` before submitting anything. It
verifies Conda/Python packages, FreeSurfer 7.4.1, SynthSeg, SynthStrip, SimNIBS
CHARM, the iBEAT container, Slurm connectivity, input sessions, and output paths.
The workers then activate only the environment required by each stage.

## Configuration

Create a local configuration file:

```bash
cp config/site.env.example config/site.env
nano config/site.env
```

`config/site.env` contains cluster paths and is excluded from Git.

Age-dependent Baby Open Brains factors are stored in:

```text
config/babyopenbrains_scales.tsv
```

### SimNIBS CHARM

For the TReNDS installation used in this project:

```bash
export SIMNIBS_HOME=$HOME/SimNIBS-4.1
source "$HOME/SimNIBS-4.1/simnibs_env/bin/activate"
```

The pipeline invokes CHARM as:

```bash
charm <subject-id> oT1e-18-big3.nii.gz --forceqform --forcerun
```

The subject ID is generated uniquely from age and subject name. The input is the
enhanced and enlarged T1 image. Both CHARM flags can be disabled in
`config/site.env` by setting `CHARM_FORCE_QFORM=0` or `CHARM_FORCE_RUN=0`.

## Example configuration

`examples/run_example.sh` is a reusable site-configured wrapper. The included
configuration demonstrates the Baby Open Brains workflow, but the filename is
kept generic so additional dataset examples can be added later.

## Run

The example runner performs a full preflight before creating any jobs. A failed
tool activation therefore stops locally rather than producing a broken SLURM
dependency chain.


Check the submission without creating jobs:

```bash
examples/run_example.sh --skip-ibeat --dry-run
```

Submit while reusing existing iBEAT outputs:

```bash
examples/run_example.sh --skip-ibeat
```

The default final output is:

```text
<work-root>/<age>/<subject>/medley_segmentation.nii.gz
```

Check progress using the command printed after submission, or run:

```bash
python scripts/check_status.py /path/to/subjects.tsv \
  --final-name medley_segmentation.nii.gz
```

## Reproducibility note

The public script names and final output name are professionalized in this
release. Legacy intermediate image names are intentionally preserved because
they identify the exact processing branches used by the existing experiments.
See `docs/FILE_MIGRATION.md`.


## Run provenance

Each submission creates `provenance.txt` and a copy of the scale map in the run
directory. The record includes the command line, Git commit when available,
selected FreeSurfer and SynthSeg branches, CHARM configuration, and SHA-256
checksums of pipeline scripts.

### FreeSurfer preflight

Some login nodes do not provide `/bin/tcsh`, while qTRDGPU compute nodes do.
The preflight therefore validates FreeSurfer, SynthSeg, and SynthStrip through
a short one-task `srun` allocation on the configured long partition.
### FreeSurfer, SynthSeg, and SynthStrip initialization

All three commands are loaded using the same explicit FreeSurfer 7.4.1
sequence used by the verified cluster SLURM script: unload inherited modules,
unset inherited FreeSurfer variables, set `FREESURFER_HOME` and `FS_LICENSE`,
source `SetUpFreeSurfer.sh`, then assign the stage-specific `SUBJECTS_DIR`.
The worker verifies that `recon-all`, `mri_synthseg`, and `mri_synthstrip` all
resolve inside the configured FreeSurfer 7.4.1 installation before processing.
