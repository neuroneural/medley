# File migration map

The following source files were renamed for the public repository.

| Historical name | Public name |
|---|---|
| `run_medley_pipeline.sh` | `run_medley.sh` |
| `run_babyopen_example.sh` | `examples/run_example.sh` |
| `enhance_O_fixed.py` | `scripts/enhance_t1.py` |
| `center_scale_nifti.py` | `scripts/scale_nifti.py` |
| `resample_label_to_reference.py` | `scripts/resample_labels.py` |
| `validate_medley_inputs.py` | `scripts/validate_inputs.py` |
| `babyHead3more2op3_portable.py` | `scripts/consolidate_labels.py` |
| `editData4tfixed.py` | `scripts/label_utils.py` |
| `check_pipeline.py` | `scripts/check_status.py` |
| `medley_scales.tsv` | `config/babyopenbrains_scales.tsv` |
| `babyHead_medley.nii.gz` | `medley_segmentation.nii.gz` |

Legacy intermediate NIfTI names remain unchanged to preserve reproducibility
with existing experimental outputs and manuscript provenance.
