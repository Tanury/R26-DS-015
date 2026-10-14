"""
generate_lemon_download_command.py
=====================================
R26-DS-015 -- MRI-MS branch (MindGlide-based)

Filters LEMON's participants.tsv to subjects whose age bin falls fully
within MS3SEG's real age range (18-55) -- avoiding the age-confound
problem an unfiltered LEMON sample would introduce, since LEMON's full
cohort spans 20-25 up through 75-80 in 5-year bins, not just a single
"young" group.

Generates ONE aws s3 sync command covering all selected subjects' anat/
folders (both sessions, since LEMON subjects aren't consistently in
ses-01 vs ses-02), rather than requiring one command per subject.

Usage:
    python3 generate_lemon_download_command.py participants.tsv --n 50 --out download_lemon.sh
    bash download_lemon.sh
"""

import argparse
import csv
import random
import re

AGE_COL = "age (5-year bins)"
DATASET = "ds000221"


def bin_bounds(age_str: str):
    match = re.match(r"(\d+)-(\d+)", age_str)
    return (int(match.group(1)), int(match.group(2))) if match else None


def filter_age_matched(participants_tsv: str, min_age: int, max_age: int) -> list:
    with open(participants_tsv, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    qualifying = []
    for row in rows:
        bounds = bin_bounds(row[AGE_COL])
        if bounds is None:
            continue
        low, high = bounds
        if low >= min_age and high <= max_age:
            qualifying.append(row["participant_id"])
    return qualifying


def main(participants_tsv: str, n: int, min_age: int, max_age: int, out_path: str, seed: int) -> None:
    qualifying = filter_age_matched(participants_tsv, min_age, max_age)
    print(f"{len(qualifying)} subjects have an age bin fully within {min_age}-{max_age}")

    if n < len(qualifying):
        random.seed(seed)  # reproducible selection, not cherry-picked
        selected = sorted(random.sample(qualifying, n))
    else:
        selected = sorted(qualifying)
        if n > len(qualifying):
            print(f"Requested {n} but only {len(qualifying)} qualify -- using all of them.")

    print(f"Selected {len(selected)} subjects for download.\n")

    include_flags = []
    for sub_id in selected:
        include_flags.append(f'--include "{sub_id}/ses-01/anat/*"')
        include_flags.append(f'--include "{sub_id}/ses-02/anat/*"')

    command = (
        f'aws s3 sync --no-sign-request s3://openneuro.org/{DATASET} ./LEMON '
        f'--exclude "*" ' + " ".join(include_flags)
    )

    with open(out_path, "w") as f:
        f.write("#!/bin/bash\n")
        f.write(f"# Downloads anat/ folders (both sessions) for {len(selected)} LEMON subjects\n")
        f.write(f"# age-matched to MS3SEG's {min_age}-{max_age} range.\n")
        f.write(command + "\n")

    print(f"Command saved to: {out_path}")
    print(f"Run it with: bash {out_path}")
    print(f"\nSelected subject IDs: {selected}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("participants_tsv", help="Path to LEMON's participants.tsv")
    parser.add_argument("--n", type=int, default=50, help="Number of subjects to select (default 50)")
    parser.add_argument("--min-age", type=int, default=18)
    parser.add_argument("--max-age", type=int, default=55)
    parser.add_argument("--out", default="download_lemon.sh")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for subject selection (reproducible)")
    args = parser.parse_args()
    main(args.participants_tsv, args.n, args.min_age, args.max_age, args.out, args.seed)