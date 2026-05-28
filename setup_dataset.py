#!/usr/bin/env python3
"""
Dataset Setup Helper - Locates and builds pose statistics from yoga dataset.
Use this script after placing the Kaggle yoga dataset in the project.

Usage:
    python setup_dataset.py                           # Auto-detects dataset
    python setup_dataset.py --dataset-dir "path/to/DATASET"  # Explicit path
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from build_pose_database import process_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Build pose statistics from yoga dataset for AI Yoga Instructor."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to the dataset root. Auto-detects if not provided.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Directory to save pose statistics (default: ./data).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.3,
        help="Minimum MediaPipe detection confidence (default: 0.3).",
    )
    
    args = parser.parse_args()

    # Resolve dataset directory
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
    else:
        # Auto-detect common locations
        candidates = [
            PROJECT_ROOT / "dataset",
            PROJECT_ROOT / "DATASET",
            Path.home() / "Downloads" / "DATASET",
            Path.home() / "Downloads" / "yoga_dataset",
        ]
        dataset_dir = None
        print("Searching for dataset in common locations...")
        for candidate in candidates:
            if candidate.is_dir():
                print(f"  ✓ Found at: {candidate}")
                dataset_dir = candidate
                break

        if dataset_dir is None:
            print("\n[ERROR] Could not locate dataset. Please provide --dataset-dir")
            print("\nExpected folder structure:")
            print("  dataset/")
            print("  ├── train/")
            print("  │   ├── plank/")
            print("  │   ├── tree_pose/")
            print("  │   ├── warrior_2/")
            print("  │   ├── goddess_pose/")
            print("  │   └── downward_dog/")
            print("  └── test/")
            print("      └── (similar structure)")
            sys.exit(1)

    if not dataset_dir.is_dir():
        print(f"[ERROR] Dataset directory not found: {dataset_dir}")
        sys.exit(1)

    print(f"\n✓ Using dataset: {dataset_dir}")
    print(f"✓ Output directory: {args.output_dir}")
    
    # Run the dataset processing pipeline
    try:
        process_dataset(dataset_dir, Path(args.output_dir), args.min_confidence)
        print("\n✓ Dataset processing complete!")
        print(f"✓ Pose statistics saved to: {args.output_dir}")
        print("\nThe AI Yoga Instructor app will now use these statistics for improved detection accuracy.")
    except Exception as e:
        print(f"\n[ERROR] Processing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
