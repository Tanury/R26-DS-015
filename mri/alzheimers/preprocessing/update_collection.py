"""
update_collection.py - Utility for R26-DS-015 preprocessing pipeline
    The master collection lives at:
        mri/alzheimers/data/raw/collection.csv
"""

import os
import argparse
import pandas as pd
from pathlib import Path

DEFAULT_MASTER_CSV = "mri/alzheimers/data/raw/collection.csv"
IMAGE_ID_COL       = "Image Data ID"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge a new IDA collection CSV export into the master collection.csv."
    )
    parser.add_argument("--new_csv",    type=str, required=True,
                        help="Path to the newly downloaded IDA collection CSV.")
    parser.add_argument("--master_csv", type=str, default=DEFAULT_MASTER_CSV,
                        help="Path to the master collection CSV (default: mri/alzheimers/data/raw/collection.csv).")
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 55)
    print("  Update Master Collection CSV")
    print("=" * 55)

    # Load new CSV
    if not os.path.exists(args.new_csv):
        raise FileNotFoundError(f"New CSV not found: {args.new_csv}")
    new_df = pd.read_csv(args.new_csv)
    print(f"\n  New CSV    : {args.new_csv}")
    print(f"  New rows   : {len(new_df)}")

    # Load or initialise master CSV
    if os.path.exists(args.master_csv):
        master_df = pd.read_csv(args.master_csv)
        print(f"  Master CSV : {args.master_csv}")
        print(f"  Master rows (before): {len(master_df)}")
    else:
        print(f"  Master CSV not found — creating new one at: {args.master_csv}")
        Path(args.master_csv).parent.mkdir(parents=True, exist_ok=True)
        master_df = pd.DataFrame(columns=new_df.columns)

    # Merge and deduplicate
    combined = pd.concat([master_df, new_df], ignore_index=True)
    before   = len(combined)
    combined = combined.drop_duplicates(subset=IMAGE_ID_COL, keep="first")
    dupes    = before - len(combined)

    # Save
    combined.to_csv(args.master_csv, index=False)

    print(f"\n  Added      : {len(new_df) - dupes} new images")
    if dupes:
        print(f"  Duplicates : {dupes} skipped (already in master)")
    print(f"  Master rows (after): {len(combined)}")
    print(f"\n  Master collection updated: {os.path.abspath(args.master_csv)}")
    print(f"    Run compile.py to sort the new images.\n")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()