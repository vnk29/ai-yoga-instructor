"""
Pose Statistics Aggregation Module.

Accumulates per-class joint angle observations from the dataset and computes
summary statistics: mean, std, min, max, and recommended confidence thresholds.

All data is serialized as portable JSON.
"""

import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np


class PoseStatsAccumulator:
    """Collects angle observations per pose class and computes aggregate statistics."""

    def __init__(self):
        # Structure: { class_name: { joint_name: [angle_values...] } }
        self._data = defaultdict(lambda: defaultdict(list))
        # Visibility averages per class
        self._visibility = defaultdict(list)
        # Total image counts (processed / skipped)
        self._processed = defaultdict(int)
        self._skipped = defaultdict(int)

    # ------------------------------------------------------------------
    # Data collection
    # ------------------------------------------------------------------

    def add_observation(self, class_name: str, feature_vector: dict) -> None:
        """
        Record one image's feature vector.

        Parameters
        ----------
        class_name : str
            Pose class label (e.g. "tree", "plank").
        feature_vector : dict
            {"angles": {joint: angle, ...}, "visibility": float}
        """
        angles = feature_vector.get("angles", {})
        vis = feature_vector.get("visibility", 0.0)

        for joint, value in angles.items():
            self._data[class_name][joint].append(value)

        self._visibility[class_name].append(vis)
        self._processed[class_name] += 1

    def record_skip(self, class_name: str) -> None:
        """Record that an image was skipped (landmark extraction failed)."""
        self._skipped[class_name] += 1

    # ------------------------------------------------------------------
    # Statistics computation
    # ------------------------------------------------------------------

    def compute_statistics(self) -> dict:
        """
        Compute per-class, per-joint summary statistics.

        Returns
        -------
        dict
            {
                "classes": {
                    "<class_name>": {
                        "joints": {
                            "<joint>": {
                                "mean": float,
                                "std": float,
                                "min": float,
                                "max": float,
                                "count": int
                            }
                        },
                        "visibility_avg": float,
                        "confidence_threshold": float,
                        "images_processed": int,
                        "images_skipped": int
                    }
                },
                "global": {
                    "total_processed": int,
                    "total_skipped": int,
                    "success_rate": float
                }
            }
        """
        result = {"classes": {}, "global": {}}

        total_proc = 0
        total_skip = 0

        for cls_name in sorted(self._data.keys()):
            joints_stats = {}
            joint_data = self._data[cls_name]

            for joint_name in sorted(joint_data.keys()):
                values = np.array(joint_data[joint_name])
                joints_stats[joint_name] = {
                    "mean": round(float(np.mean(values)), 2),
                    "std": round(float(np.std(values)), 2),
                    "min": round(float(np.min(values)), 2),
                    "max": round(float(np.max(values)), 2),
                    "count": len(values),
                }

            vis_vals = self._visibility.get(cls_name, [])
            vis_avg = round(float(np.mean(vis_vals)), 4) if vis_vals else 0.0

            # Confidence threshold: use (100 - 2*avg_std) clamped to [30, 70]
            # Lower std → higher confidence threshold (tighter matching required)
            all_stds = [v["std"] for v in joints_stats.values()]
            avg_std = float(np.mean(all_stds)) if all_stds else 20.0
            conf_threshold = float(np.clip(100.0 - 2.0 * avg_std, 30.0, 70.0))

            proc = self._processed.get(cls_name, 0)
            skip = self._skipped.get(cls_name, 0)
            total_proc += proc
            total_skip += skip

            result["classes"][cls_name] = {
                "joints": joints_stats,
                "visibility_avg": vis_avg,
                "confidence_threshold": round(conf_threshold, 2),
                "images_processed": proc,
                "images_skipped": skip,
            }

        success_rate = (
            round(total_proc / max(total_proc + total_skip, 1) * 100.0, 2)
        )
        result["global"] = {
            "total_processed": total_proc,
            "total_skipped": total_skip,
            "success_rate": success_rate,
        }
        return result


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_statistics(stats: dict, out_path: str | Path) -> None:
    """Serialize statistics dict to a JSON file."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"[STATS] Saved pose statistics → {out_path}")


def load_statistics(stats_path: str | Path) -> dict | None:
    """Load previously saved statistics JSON. Returns None if file missing."""
    stats_path = Path(stats_path)
    if not stats_path.exists():
        return None
    with open(stats_path, "r", encoding="utf-8") as f:
        return json.load(f)
