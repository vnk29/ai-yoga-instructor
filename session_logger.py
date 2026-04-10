"""
Session statistics tracking and persistence.
On exit, appends a JSON entry to session_log.json.
"""

import json
import os
import time
from collections import defaultdict
from datetime import datetime


class SessionLogger:
    """Tracks corrections and compliments during a plank session."""

    def __init__(self) -> None:
        self._start = time.monotonic()
        self.corrections: dict[str, int] = defaultdict(int)
        self.compliments: int = 0

    # ------------------------------------------------------------------

    @property
    def duration(self) -> float:
        """Elapsed session time in seconds."""
        return time.monotonic() - self._start

    @property
    def total_corrections(self) -> int:
        return sum(self.corrections.values())

    # ------------------------------------------------------------------

    def record_correction(self, issue_key: str) -> None:
        self.corrections[issue_key] += 1

    def record_compliment(self) -> None:
        self.compliments += 1

    # ------------------------------------------------------------------

    def save(self, sensitivity: str, output_file: str = "session_log.json") -> None:
        """Append a session summary to *output_file* as a JSON array entry."""
        entry = {
            "date": datetime.now().isoformat(timespec="seconds"),
            "duration_seconds": round(self.duration, 1),
            "sensitivity": sensitivity,
            "total_corrections": self.total_corrections,
            "corrections_by_type": dict(self.corrections),
            "compliments": self.compliments,
        }

        log: list = []
        if os.path.exists(output_file):
            with open(output_file) as fh:
                try:
                    log = json.load(fh)
                except (json.JSONDecodeError, ValueError):
                    log = []

        log.append(entry)

        with open(output_file, "w") as fh:
            json.dump(log, fh, indent=2)

        print(f"\nSession saved → {output_file}")
        print(
            f"  Duration : {self.duration / 60:.1f} min  |  "
            f"Corrections: {self.total_corrections}  |  "
            f"Compliments: {self.compliments}"
        )
