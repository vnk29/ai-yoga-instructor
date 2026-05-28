"""
Joint Angle Computation Utilities for Dataset-Driven Pose Analysis.

Provides lightweight functions to calculate angles between three landmarks using
vector geometry (consistent with YogaAnalyzer.calculate_angle), and a predefined
mapping of human joint names to MediaPipe landmark triples.

All functions operate on normalized (x, y, z, visibility) tuples returned by
PoseDetector.get_coords().
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Core angle computation
# ---------------------------------------------------------------------------

def angle_between(p1: tuple, p2: tuple, p3: tuple) -> float:
    """
    Calculate the angle (in degrees) at vertex p2 formed by rays p2→p1 and p2→p3.

    Each point is a tuple of at least (x, y, ...).
    Uses 2D projection (x, y) matching the existing YogaAnalyzer approach.

    Returns 0.0 if the vectors are degenerate (zero-length).
    """
    a = np.array([p1[0], p1[1]])
    b = np.array([p2[0], p2[1]])
    c = np.array([p3[0], p3[1]])

    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-8 or norm_bc < 1e-8:
        return 0.0

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc + 1e-9)
    angle_rad = np.arccos(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(angle_rad))


# ---------------------------------------------------------------------------
# Landmark-triple mapping for standard joint angles
# ---------------------------------------------------------------------------
# Each entry: joint_name → (landmark_A, landmark_VERTEX, landmark_C)
# The angle is measured at the VERTEX landmark.

JOINT_LANDMARK_MAP = {
    # Arms
    "left_elbow":     ("LEFT_SHOULDER",  "LEFT_ELBOW",    "LEFT_WRIST"),
    "right_elbow":    ("RIGHT_SHOULDER", "RIGHT_ELBOW",   "RIGHT_WRIST"),
    "left_shoulder":  ("LEFT_ELBOW",     "LEFT_SHOULDER", "LEFT_HIP"),
    "right_shoulder": ("RIGHT_ELBOW",    "RIGHT_SHOULDER","RIGHT_HIP"),

    # Hips
    "left_hip":       ("LEFT_SHOULDER",  "LEFT_HIP",      "LEFT_KNEE"),
    "right_hip":      ("RIGHT_SHOULDER", "RIGHT_HIP",     "RIGHT_KNEE"),

    # Knees
    "left_knee":      ("LEFT_HIP",       "LEFT_KNEE",     "LEFT_ANKLE"),
    "right_knee":     ("RIGHT_HIP",      "RIGHT_KNEE",    "RIGHT_ANKLE"),

    # Ankles
    "left_ankle":     ("LEFT_KNEE",      "LEFT_ANKLE",    "LEFT_FOOT_INDEX"),
    "right_ankle":    ("RIGHT_KNEE",     "RIGHT_ANKLE",   "RIGHT_FOOT_INDEX"),
}


def compute_all_angles(detector, landmark_list) -> dict:
    """
    Compute all joint angles defined in JOINT_LANDMARK_MAP.

    Parameters
    ----------
    detector : PoseDetector
        Instance used to resolve landmark names to (x, y, z, vis) tuples.
    landmark_list : list
        Single-person landmark list (e.g. result.pose_landmarks[0]).

    Returns
    -------
    dict[str, float]
        Mapping of joint_name → angle in degrees.
        Joints whose landmarks cannot be resolved are silently skipped.
    """
    angles = {}
    for joint_name, (lm_a, lm_b, lm_c) in JOINT_LANDMARK_MAP.items():
        try:
            pa = detector.get_coords(landmark_list, lm_a)
            pb = detector.get_coords(landmark_list, lm_b)
            pc = detector.get_coords(landmark_list, lm_c)
            angles[joint_name] = round(angle_between(pa, pb, pc), 2)
        except Exception:
            # Landmark not available — skip
            continue
    return angles


def average_visibility(detector, landmark_list) -> float:
    """
    Return the mean visibility score across all 33 MediaPipe pose landmarks.

    Parameters
    ----------
    detector : PoseDetector
        Instance used to look up landmarks.
    landmark_list : list
        Single-person landmark list.

    Returns
    -------
    float
        Mean visibility in [0.0, 1.0].
    """
    vis_values = []
    for lm in landmark_list:
        vis_values.append(getattr(lm, "visibility", 0.0))
    if not vis_values:
        return 0.0
    return float(np.mean(vis_values))


def compute_feature_vector(detector, landmark_list) -> dict | None:
    """
    Build a complete feature vector for a single pose observation.

    Returns
    -------
    dict
        {"angles": {…}, "visibility": float}  or  None if no angles extracted.
    """
    angles = compute_all_angles(detector, landmark_list)
    if not angles:
        return None
    vis = average_visibility(detector, landmark_list)
    return {"angles": angles, "visibility": round(vis, 4)}
