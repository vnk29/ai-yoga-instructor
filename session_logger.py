"""
Session statistics tracking and persistence for the AI Yoga Instructor.
Saves session summaries in a JSON format.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime


class SessionLogger:
    """Tracks pose detection metrics, corrections, and overall stability for a yoga session."""

    def __init__(self, active_pose: str = "Unknown") -> None:
        self.active_pose = active_pose
        self._start = time.monotonic()
        self.corrections: dict[str, int] = defaultdict(int)
        self.accuracies: list[float] = []
        self.best_accuracy: float = 0.0

    @property
    def duration(self) -> float:
        """Elapsed session time in seconds."""
        return time.monotonic() - self._start

    @property
    def total_corrections(self) -> int:
        """Sum of all postural corrections logged."""
        return sum(self.corrections.values())

    @property
    def average_accuracy(self) -> float:
        """Compute the average pose accuracy percentage for the session."""
        if not self.accuracies:
            return 0.0
        return sum(self.accuracies) / len(self.accuracies)

    def record_accuracy(self, accuracy: float) -> None:
        """Record the current frame's accuracy percentage and check for a new personal best."""
        self.accuracies.append(accuracy)
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy

    def record_correction(self, correction_key: str) -> None:
        """Increment count for a specific posture adjustment suggestion."""
        self.corrections[correction_key] += 1

    def save(self, output_file: str = "session_log.json") -> dict:
        """Save/Append the session summary into session_log.json."""
        summary = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "pose_name": self.active_pose,
            "duration_seconds": round(self.duration, 1),
            "total_corrections": self.total_corrections,
            "corrections_breakdown": dict(self.corrections),
            "average_accuracy_pct": round(self.average_accuracy, 1),
            "best_accuracy_pct": round(self.best_accuracy, 1)
        }

        log = []
        if os.path.exists(output_file):
            try:
                with open(output_file, "r") as fh:
                    log = json.load(fh)
                    if not isinstance(log, list):
                        log = []
            except (json.JSONDecodeError, ValueError):
                log = []

        log.append(summary)

        try:
            with open(output_file, "w") as fh:
                json.dump(log, fh, indent=2)
        except IOError as e:
            print(f"Error saving session log: {e}")

        return summary
