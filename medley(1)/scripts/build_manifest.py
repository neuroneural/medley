#!/usr/bin/env python3
"""Build a deterministic subject manifest for a Medley SLURM run."""
from __future__ import annotations

import argparse
import re
from fractions import Fraction
from pathlib import Path


def natural_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def parse_ages(value: str) -> list[int]:
    ages: set[int] = set()
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            first, last = token.split("-", 1)
            start, stop = int(first), int(last)
            if start > stop:
                start, stop = stop, start
            ages.update(range(start, stop + 1))
        else:
            ages.add(int(token))
    if not ages or any(age < 0 for age in ages):
        raise argparse.ArgumentTypeError("--ages must contain nonnegative months, e.g. 1-8")
    return sorted(ages)


def read_scales(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for number, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            raise SystemExit(f"Bad scale-map line {number}: {raw}")
        age_name, value = parts[0], parts[1]
        try:
            parsed = Fraction(value)
        except Exception as exc:
            raise SystemExit(f"Bad scale on line {number}: {value}") from exc
        if parsed <= 0:
            raise SystemExit(f"Scale must be positive on line {number}: {value}")
        mapping[age_name] = value
    return mapping


def subject_from_filename(path: Path, age: int) -> str:
    match = re.search(r"(sub-[^_./]+)", path.name)
    if match:
        return match.group(1)
    marker = f"_ses-{age}mo"
    if marker in path.name:
        return path.name.split(marker, 1)[0]
    return path.name.split("_", 1)[0].replace(".nii.gz", "")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--ibeat-output-root", type=Path, required=True)
    parser.add_argument("--synthseg-root", type=Path, required=True)
    parser.add_argument("--charm-root", type=Path, required=True)
    parser.add_argument("--fs-subjects-dir", type=Path, required=True)
    parser.add_argument("--scale-map", type=Path, required=True)
    parser.add_argument("--ages", type=parse_ages, default=parse_ages("1-8"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scales = read_scales(args.scale_map)
    rows: list[tuple[str, ...]] = []
    seen: set[tuple[int, str]] = set()

    for age in args.ages:
        age_name = f"{age}m"
        if age_name not in scales:
            raise SystemExit(f"Scale map is missing {age_name}")
        session = args.raw_root / f"ses-{age}mo"
        if not session.is_dir():
            raise SystemExit(f"Raw session directory not found: {session}")
        files = sorted(session.rglob("*_T1w.nii.gz"), key=lambda p: natural_key(str(p)))
        if not files:
            raise SystemExit(f"No *_T1w.nii.gz files found under {session}")

        per_subject: dict[str, Path] = {}
        for raw_t1 in files:
            subject = subject_from_filename(raw_t1, age)
            if subject in per_subject:
                raise SystemExit(
                    f"Multiple T1 files found for {age_name}/{subject}: "
                    f"{per_subject[subject]} and {raw_t1}"
                )
            per_subject[subject] = raw_t1.resolve()

        for subject in sorted(per_subject, key=natural_key):
            key = (age, subject)
            if key in seen:
                raise SystemExit(f"Duplicate manifest key: {age_name}/{subject}")
            seen.add(key)
            raw_t1 = per_subject[subject]
            ibeat_subject = args.ibeat_output_root / f"T1-{age}m" / subject
            work_dir = args.work_root / age_name / subject
            synthseg_dir = args.synthseg_root / age_name / subject
            charm_case = args.charm_root / age_name / subject
            fs_id = subject
            rows.append(
                (
                    str(age), age_name, subject, str(raw_t1),
                    str(ibeat_subject), str(work_dir), str(synthseg_dir),
                    str(charm_case), fs_id, scales[age_name],
                )
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join("\t".join(row) + "\n" for row in rows))
    print(f"Wrote {len(rows)} subjects to {args.output}")


if __name__ == "__main__":
    main()
