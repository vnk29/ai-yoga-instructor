"""
Plank posture analysis.

All landmark coordinates are normalised (x, y in [0,1], y increases downward).

Key geometry
------------
Side-view plank: person lies horizontally, camera placed to their side.

  HEAD ──── SHOULDER ──────── HIP ──────── ANKLE
              S                H              A

The ideal spine line runs from S to A.
- Hip deviation > 0  →  hip is BELOW the S-A line  →  hips sagging
- Hip deviation < 0  →  hip is ABOVE the S-A line  →  hips piked

Head: nose should be roughly level with the shoulder (eyes toward floor).
- head_rise > 0  →  nose is ABOVE shoulder  →  head looking up
- head_rise < 0  →  nose is BELOW shoulder  →  head drooping
"""

import numpy as np

from config import (
    ISSUE_MESSAGES,
    MIN_BODY_LENGTH,
    PLANK_BODY_ANGLE_THRESHOLD,
    SENSITIVITY_PRESETS,
)
from pose_detector import PoseDetector


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _body_angle_deg(shoulder: tuple, ankle: tuple) -> float:
    """
    Angle of the shoulder→ankle line with the horizontal axis, in degrees.
    0° = perfectly horizontal; 90° = vertical.
    """
    dx = ankle[0] - shoulder[0]
    dy = ankle[1] - shoulder[1]
    return float(np.degrees(np.arctan2(abs(dy), abs(dx) + 1e-8)))


def _signed_hip_deviation(shoulder: tuple, hip: tuple, ankle: tuple) -> float:
    """
    Signed perpendicular distance of the hip from the shoulder-ankle line,
    normalised by body length (shoulder-ankle distance).

    Convention (image coords, y increases downward):
      +  hip is BELOW the ideal line  →  hips sagging
      -  hip is ABOVE the ideal line  →  hips piked

    Works for both left-facing and right-facing orientations.
    """
    sx, sy = shoulder[0], shoulder[1]
    hx, hy = hip[0], hip[1]
    ax, ay = ankle[0], ankle[1]

    body_len = np.hypot(ax - sx, ay - sy)
    if body_len < 1e-4:
        return 0.0

    # Signed area of triangle S-A-H via cross product
    # cross > 0 when H is to the left of the S→A ray (above in image for rightward ray)
    cross = (ax - sx) * (hy - sy) - (ay - sy) * (hx - sx)
    signed_dist = cross / body_len

    # When shoulder is to the RIGHT of ankle (person faces left in frame), the
    # cross-product sign is flipped relative to our convention — normalise it.
    if shoulder[0] > ankle[0]:
        signed_dist = -signed_dist

    return float(signed_dist)


def _head_rise(nose: tuple, shoulder: tuple, body_len: float) -> float:
    """
    How high the nose is relative to the shoulder, normalised by body length.

    In image coordinates (y↓):
      shoulder[1] - nose[1] > 0  →  shoulder is lower than nose
                                  →  nose is higher in the image  →  head up
      shoulder[1] - nose[1] < 0  →  nose is below shoulder       →  head drooping
    """
    if body_len < 1e-4:
        return 0.0
    return float((shoulder[1] - nose[1]) / body_len)


# ---------------------------------------------------------------------------
# Analyser
# ---------------------------------------------------------------------------


class PlankAnalyzer:
    """
    Analyses MediaPipe pose landmarks each frame and produces a list of
    posture issues with descriptive messages.
    """

    def __init__(self, sensitivity: str = "medium") -> None:
        self.sensitivity = sensitivity
        # Cycle through messages so the same phrase is not repeated consecutively
        self._msg_idx: dict[str, int] = {key: 0 for key in ISSUE_MESSAGES}

    # ------------------------------------------------------------------

    def _next_message(self, key: str) -> str:
        msgs = ISSUE_MESSAGES[key]
        idx = self._msg_idx[key] % len(msgs)
        self._msg_idx[key] += 1
        return msgs[idx]

    # ------------------------------------------------------------------

    def analyze(self, pose_detector: PoseDetector, landmark_list: list) -> dict:
        """
        Analyse one frame of landmarks.

        Parameters
        ----------
        landmark_list : list
            Single-person landmark list from result.pose_landmarks[0].

        Returns
        -------
        {
          "in_plank": bool,
          "issues": [{"key": str, "message": str, "severity": float}, ...],
          "metrics": {
              "shoulder": tuple, "hip": tuple, "ankle": tuple, "nose": tuple,
              "body_angle": float, "body_len": float,
              "hip_deviation": float | None,
              "head_rise": float | None,
          }
        }

        ``severity`` is a multiple of the threshold (≥1 when issue is active).
        ``good_form`` has severity 0.
        """
        pd = pose_detector
        preset = SENSITIVITY_PRESETS[self.sensitivity]

        # Midpoints of bilateral landmarks give a stable centre-line estimate
        shoulder = pd.midpoint(
            pd.get_coords(landmark_list, "LEFT_SHOULDER"),
            pd.get_coords(landmark_list, "RIGHT_SHOULDER"),
        )
        hip = pd.midpoint(
            pd.get_coords(landmark_list, "LEFT_HIP"),
            pd.get_coords(landmark_list, "RIGHT_HIP"),
        )
        ankle = pd.midpoint(
            pd.get_coords(landmark_list, "LEFT_ANKLE"),
            pd.get_coords(landmark_list, "RIGHT_ANKLE"),
        )
        nose = pd.get_coords(landmark_list, "NOSE")

        body_len = float(np.hypot(ankle[0] - shoulder[0], ankle[1] - shoulder[1]))
        body_angle = _body_angle_deg(shoulder, ankle)

        # Is the body roughly horizontal and large enough in frame?
        in_plank = (
            body_angle < PLANK_BODY_ANGLE_THRESHOLD
            and body_len > MIN_BODY_LENGTH
        )

        metrics = {
            "shoulder": shoulder,
            "hip": hip,
            "ankle": ankle,
            "nose": nose,
            "body_angle": body_angle,
            "body_len": body_len,
            "hip_deviation": None,
            "head_rise": None,
        }

        issues: list[dict] = []

        if not in_plank:
            issues.append(
                {
                    "key": "not_in_plank",
                    "message": self._next_message("not_in_plank"),
                    "severity": 1.0,
                }
            )
            return {"in_plank": False, "issues": issues, "metrics": metrics}

        # ---- Hip deviation -------------------------------------------------
        hip_dev = _signed_hip_deviation(shoulder, hip, ankle)
        metrics["hip_deviation"] = hip_dev

        thr = preset["hip_threshold"]
        if hip_dev < -thr:
            issues.append(
                {
                    "key": "hip_high",
                    "message": self._next_message("hip_high"),
                    "severity": abs(hip_dev) / thr,
                }
            )
        elif hip_dev > thr:
            issues.append(
                {
                    "key": "hip_low",
                    "message": self._next_message("hip_low"),
                    "severity": hip_dev / thr,
                }
            )

        # ---- Head position -------------------------------------------------
        head_r = _head_rise(nose, shoulder, body_len)
        metrics["head_rise"] = head_r

        # Use separate thresholds for each direction:
        # • head_up  – nose above shoulder level. Tight threshold because
        #              craning the neck upward is always wrong in a plank.
        # • head_down – nose below shoulder level. Normal in a plank because
        #              you look at the floor; only flag a severe droop.
        up_thr   = preset["head_up_threshold"]
        down_thr = preset["head_down_threshold"]

        if head_r > up_thr:
            issues.append(
                {
                    "key": "head_up",
                    "message": self._next_message("head_up"),
                    "severity": head_r / up_thr,
                }
            )
        elif head_r < -down_thr:
            issues.append(
                {
                    "key": "head_down",
                    "message": self._next_message("head_down"),
                    "severity": abs(head_r) / down_thr,
                }
            )

        # ---- Good form (no issues) -----------------------------------------
        if not issues:
            issues.append(
                {
                    "key": "good_form",
                    "message": self._next_message("good_form"),
                    "severity": 0.0,
                }
            )

        return {"in_plank": True, "issues": issues, "metrics": metrics}
