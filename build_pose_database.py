#!/usr/bin/env python3
"""
build_pose_database.py -- Dataset-Driven Pose Reference Database Builder.

Scans a Kaggle-style yoga pose dataset, extracts MediaPipe landmarks from
every image using the multi-stage preprocessing pipeline, computes joint
angles and visibility features, and produces:

  1. pose_reference_db.json  -- per-image feature vectors
  2. pose_stats.json         -- per-class aggregated statistics

Usage
-----
    python build_pose_database.py
    python build_pose_database.py --dataset-dir "path/to/DATASET" --output-dir "data"
    python build_pose_database.py --min-confidence 0.3

The script is intentionally lightweight -- NO deep learning training occurs.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from pose_detector import PoseDetector
from utils.image_preprocessor import ImagePreprocessor
from utils.angle_utils import compute_all_angles, average_visibility, compute_feature_vector
from utils.pose_statistics import PoseStatsAccumulator, save_statistics

# ---------------------------------------------------------------------------
# Dataset folder name -> internal pose class key mapping
# ---------------------------------------------------------------------------
# Maps the actual folder names in the Kaggle dataset to the keys used in
# yoga_config.POSE_DATABASE. Add new entries here when expanding the dataset.

FOLDER_TO_CLASS = {
    "tree":       "tree",
    "warrior2":   "warrior_ii",
    "plank":      "plank",
    "goddess":    "goddess_pose",
    "downdog":    "downward_dog",
    # Aliases (in case folders are renamed)
    "tree_pose":      "tree",
    "warrior_2":      "warrior_ii",
    "goddess_pose":   "goddess_pose",
    "downward_dog":   "downward_dog",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def discover_images(dataset_dir: Path) -> dict[str, list[Path]]:
    """
    Walk the dataset directory and group image paths by class label.

    Expects structure:
        dataset_dir/
        ├── TRAIN/
        │   ├── <pose_folder>/
        │   │   ├── img1.jpg
        │   │   └── ...
        └── TEST/
            └── ...

    Returns
    -------
    dict[str, list[Path]]
        { class_key: [image_path, ...] }
    """
    class_images: dict[str, list[Path]] = {}

    for split in ["TRAIN", "TEST", "train", "test"]:
        split_dir = dataset_dir / split
        if not split_dir.is_dir():
            continue

        for folder in sorted(split_dir.iterdir()):
            if not folder.is_dir():
                continue

            folder_name = folder.name.lower().strip()
            class_key = FOLDER_TO_CLASS.get(folder_name, folder_name)

            images = [
                p for p in sorted(folder.iterdir())
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            ]

            if images:
                existing = class_images.get(class_key, [])
                existing.extend(images)
                class_images[class_key] = existing

    return class_images


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    min_confidence: float = 0.3,
) -> None:
    """
    Main processing loop. Iterates over all discovered images, extracts
    features, accumulates statistics, and writes output files.
    """
    print("=" * 65)
    print("  YOGA POSE DATABASE BUILDER")
    print("=" * 65)
    print(f"  Dataset dir  : {dataset_dir}")
    print(f"  Output dir   : {output_dir}")
    print(f"  Min confidence: {min_confidence}")
    print("=" * 65)

    # Discover images
    class_images = discover_images(dataset_dir)
    if not class_images:
        print("[ERROR] No images found. Verify --dataset-dir path and folder structure.")
        sys.exit(1)

    total_images = sum(len(v) for v in class_images.values())
    print(f"\nDiscovered {total_images} images across {len(class_images)} classes:")
    for cls, imgs in sorted(class_images.items()):
        print(f"  - {cls:20s} : {len(imgs)} images")

    # Initialize detector (low confidence for maximum recall on diverse images)
    print("\nInitializing MediaPipe PoseDetector ...")
    detector = PoseDetector(
        min_detection_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )

    # Accumulators
    accumulator = PoseStatsAccumulator()
    reference_records: list[dict] = []
    skipped_log: list[str] = []

    start_time = time.time()

    for cls_key, image_paths in sorted(class_images.items()):
        print(f"\n[CLASS] Processing '{cls_key}' ({len(image_paths)} images) ...")
        cls_processed = 0
        cls_skipped = 0

        for idx, img_path in enumerate(image_paths):
            # Progress indicator every 20 images
            if (idx + 1) % 20 == 0 or idx == 0:
                print(f"  [{idx + 1}/{len(image_paths)}] {img_path.name}")

            # Extract landmarks via the multi-stage retry pipeline
            landmark_list = ImagePreprocessor.process_image_for_pose(
                str(img_path), detector
            )

            if landmark_list is None:
                cls_skipped += 1
                accumulator.record_skip(cls_key)
                skipped_log.append(str(img_path))
                continue

            # Compute feature vector
            features = compute_feature_vector(detector, landmark_list)
            if features is None:
                cls_skipped += 1
                accumulator.record_skip(cls_key)
                skipped_log.append(str(img_path))
                continue

            # Record
            accumulator.add_observation(cls_key, features)
            reference_records.append({
                "class": cls_key,
                "image": str(img_path.relative_to(dataset_dir)),
                "angles": features["angles"],
                "visibility": features["visibility"],
            })
            cls_processed += 1

        print(f"  [OK] {cls_processed} processed, [SKIP] {cls_skipped} skipped")

    elapsed = time.time() - start_time

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reference database (per-image records)
    db_path = output_dir / "pose_reference_db.json"
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(reference_records, f, indent=2, ensure_ascii=False)
    print(f"\n[OUTPUT] Pose reference database -> {db_path}  ({len(reference_records)} records)")

    # 2. Aggregated statistics
    stats = accumulator.compute_statistics()
    stats_path = output_dir / "pose_stats.json"
    save_statistics(stats, stats_path)

    # 3. Skipped images log
    if skipped_log:
        skip_path = output_dir / "skipped_images.txt"
        with open(skip_path, "w", encoding="utf-8") as f:
            f.write("\n".join(skipped_log))
        print(f"[OUTPUT] Skipped images log -> {skip_path}  ({len(skipped_log)} entries)")

    # Summary
    print(f"\n{'=' * 65}")
    print(f"  COMPLETE -- {elapsed:.1f}s elapsed")
    print(f"  Total processed : {stats['global']['total_processed']}")
    print(f"  Total skipped   : {stats['global']['total_skipped']}")
    print(f"  Success rate    : {stats['global']['success_rate']:.1f}%")
    print(f"{'=' * 65}")

    # Print per-class confidence thresholds
    print("\n  Recommended confidence thresholds:")
    for cls_name, cls_stats in stats["classes"].items():
        print(f"    {cls_name:20s} -> {cls_stats['confidence_threshold']:.1f}%")

    detector.close()
    print("\nDone. Generated files are ready in:", output_dir.resolve())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build a lightweight pose reference database from a Kaggle yoga dataset."
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Path to the dataset root containing TRAIN/ and TEST/ folders. "
             "Defaults to searching common locations.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data"),
        help="Directory to write output files (default: ./data).",
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
            Path(r"c:\Users\Keerthana\Downloads\archive (1)\DATASET"),
        ]
        dataset_dir = None
        for c in candidates:
            if c.is_dir():
                dataset_dir = c
                break

        if dataset_dir is None:
            print("[ERROR] Could not auto-detect dataset directory.")
            print("Please pass --dataset-dir explicitly.")
            sys.exit(1)

    if not dataset_dir.is_dir():
        print(f"[ERROR] Dataset directory does not exist: {dataset_dir}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    process_dataset(dataset_dir, output_dir, args.min_confidence)


if __name__ == "__main__":
    main()
