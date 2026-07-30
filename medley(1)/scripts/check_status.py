#!/usr/bin/env python3
"""Report per-subject completion status for a Medley run."""
from __future__ import annotations

import argparse
from pathlib import Path

STAGES = {
    "iBEAT gathered": ("T1.nii.gz", "T1-skullstripped-tissue.nii.gz"),
    "enhanced/enlarged": ("T1_enhanced-18.nii.gz", "T1-big3.nii.gz", "oT1e-18-big3.nii.gz"),
    "tool outputs gathered": ("T1_ss1rr.nii.gz", "fs_enhanced_aseg.mgz", "fs_enhanced_rawavg.mgz", "fs_enhanced_enlarged_aseg.mgz", "fs_enhanced_enlarged_rawavg.mgz", "mask.nii.gz", "mask-csf.nii.gz"),
    "native aligned": ("T1-ss1rrf.nii.gz", "aseg_enhanced1r.nii.gz", "aseg_enhanced_enlarged1r.nii.gz"),
    "final": ("medley_segmentation.nii.gz",),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--final-name", default="medley_segmentation.nii.gz")
    args = ap.parse_args()
    stages = dict(STAGES)
    stages["final"] = (args.final_name,)
    counts = {name: 0 for name in stages}
    total = 0
    incomplete: list[str] = []

    for raw in args.manifest.read_text().splitlines():
        if not raw.strip():
            continue
        fields = raw.split("\t")
        if len(fields) != 10:
            raise SystemExit(f"Bad manifest row: {raw}")
        age_name, subject, work_dir = fields[1], fields[2], Path(fields[5])
        total += 1
        missing_summary = []
        for stage, names in stages.items():
            missing = [name for name in names if not (work_dir / name).exists()]
            if not missing:
                counts[stage] += 1
            else:
                missing_summary.append(f"{stage}: {','.join(missing)}")
        if missing_summary:
            incomplete.append(f"{age_name}/{subject} | " + " | ".join(missing_summary))

    print(f"Subjects: {total}")
    for stage, count in counts.items():
        print(f"{stage:24s} {count}/{total}")
    if incomplete:
        print("\nIncomplete subjects:")
        for line in incomplete:
            print(line)


if __name__ == "__main__":
    main()
