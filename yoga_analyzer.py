"""
Industrial Yoga Pose Classification Engine — v2.0
===================================================

4-Stage Pipeline:
  STAGE 1: Body Category Detection (Standing/Sitting/Prone/Inverted/Balance/Backbend)
  STAGE 2: Pose Gating (only evaluate poses within detected category)
  STAGE 3: Multi-Factor Weighted Confidence Scoring
  STAGE 4: Hard Rejection Rules + Boost Rules + 65% Uncertainty Threshold

Design Principles:
  - NEVER classify a pose based on angles alone
  - ALWAYS determine body orientation FIRST
  - REJECT impossible combinations (e.g. vertical torso → Cobra)
  - Show "Pose Uncertain" if confidence < 65%
  - Provide full debug output for every analysis
"""

import numpy as np
from collections import deque
from yoga_config import POSE_DATABASE, GLOBAL_UNCERTAINTY_THRESHOLD


class YogaAnalyzer:
    """Industrial-grade hierarchical yoga pose classifier with strict orientation gating."""

    def __init__(self) -> None:
        # Temporal smoothing buffers (webcam mode only)
        self.pose_buffer = deque(maxlen=15)
        self.landmark_history = deque(maxlen=15)
        self.current_stable_pose = "unknown"
        self.locked_pose = "unknown"
        self.low_conf_frames = 0

    # ==================================================================
    # GEOMETRY PRIMITIVES
    # ==================================================================

    @staticmethod
    def _angle(p1, p2, p3) -> float:
        """Angle (degrees) at vertex p2 formed by p1→p2→p3."""
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])
        ba, bc = a - b, c - b
        cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
        return float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))

    @staticmethod
    def _dist(p1, p2) -> float:
        """Euclidean distance between two (x, y, ...) points."""
        return float(np.linalg.norm(np.array(p1[:2]) - np.array(p2[:2])))

    # ==================================================================
    # LANDMARK EXTRACTION
    # ==================================================================

    @staticmethod
    def _extract_coords(detector, lm_list) -> dict:
        """Pull named landmark coordinates into a flat dict."""
        names = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
            "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
            "LEFT_EAR", "RIGHT_EAR", "LEFT_HEEL", "RIGHT_HEEL",
            "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX"
        ]
        coords = {}
        for n in names:
            try:
                coords[n] = detector.get_coords(lm_list, n)
            except Exception:
                coords[n] = None
        return coords

    # ==================================================================
    # PRECOMPUTED ANGLE CACHE
    # ==================================================================

    def _build_angle_cache(self, c: dict) -> dict:
        """Compute all reusable angles once."""
        a = {}

        # Knee angles
        a["left_knee"] = self._angle(c["LEFT_HIP"], c["LEFT_KNEE"], c["LEFT_ANKLE"]) \
            if all([c.get("LEFT_HIP"), c.get("LEFT_KNEE"), c.get("LEFT_ANKLE")]) else 180.0
        a["right_knee"] = self._angle(c["RIGHT_HIP"], c["RIGHT_KNEE"], c["RIGHT_ANKLE"]) \
            if all([c.get("RIGHT_HIP"), c.get("RIGHT_KNEE"), c.get("RIGHT_ANKLE")]) else 180.0

        # Hip angles
        a["left_hip"] = self._angle(c["LEFT_SHOULDER"], c["LEFT_HIP"], c["LEFT_KNEE"]) \
            if all([c.get("LEFT_SHOULDER"), c.get("LEFT_HIP"), c.get("LEFT_KNEE")]) else 180.0
        a["right_hip"] = self._angle(c["RIGHT_SHOULDER"], c["RIGHT_HIP"], c["RIGHT_KNEE"]) \
            if all([c.get("RIGHT_SHOULDER"), c.get("RIGHT_HIP"), c.get("RIGHT_KNEE")]) else 180.0

        # Arm (elbow) angles
        a["left_arm"] = self._angle(c["LEFT_SHOULDER"], c["LEFT_ELBOW"], c["LEFT_WRIST"]) \
            if all([c.get("LEFT_SHOULDER"), c.get("LEFT_ELBOW"), c.get("LEFT_WRIST")]) else 180.0
        a["right_arm"] = self._angle(c["RIGHT_SHOULDER"], c["RIGHT_ELBOW"], c["RIGHT_WRIST"]) \
            if all([c.get("RIGHT_SHOULDER"), c.get("RIGHT_ELBOW"), c.get("RIGHT_WRIST")]) else 180.0

        # Shoulder angles (arm-to-torso)
        a["left_shoulder_angle"] = self._angle(c["LEFT_ELBOW"], c["LEFT_SHOULDER"], c["LEFT_HIP"]) \
            if all([c.get("LEFT_ELBOW"), c.get("LEFT_SHOULDER"), c.get("LEFT_HIP")]) else 30.0
        a["right_shoulder_angle"] = self._angle(c["RIGHT_ELBOW"], c["RIGHT_SHOULDER"], c["RIGHT_HIP"]) \
            if all([c.get("RIGHT_ELBOW"), c.get("RIGHT_SHOULDER"), c.get("RIGHT_HIP")]) else 30.0

        # Midpoints
        ls, rs = c.get("LEFT_SHOULDER"), c.get("RIGHT_SHOULDER")
        lh, rh = c.get("LEFT_HIP"), c.get("RIGHT_HIP")
        if all([ls, rs, lh, rh]):
            a["shoulder_mid"] = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
            a["hip_mid"] = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
        else:
            a["shoulder_mid"] = None
            a["hip_mid"] = None

        return a

    # ==================================================================
    # BODY VISIBILITY CHECK
    # ==================================================================

    def _check_visibility(self, detector, lm_list) -> tuple:
        """Core joints (shoulders + hips) must have visibility > 0.45."""
        core = ["LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP"]
        missing = []
        for name in core:
            try:
                _, _, _, vis = detector.get_coords(lm_list, name)
                if vis < 0.45:
                    missing.append(name.replace("_", " ").title())
            except Exception:
                missing.append(name.replace("_", " ").title())
        if missing:
            return False, "Move fully into camera frame. Missing: " + ", ".join(missing[:2])
        return True, ""

    # ==================================================================
    # STABILITY SCORING
    # ==================================================================

    def _compute_stability(self, lm_list) -> float:
        """Rolling variance of key joints across frames → 0-100% stability."""
        if len(lm_list) < 25:
            return 100.0
        frame = []
        for idx in [0, 11, 12, 23, 24]:
            lm = lm_list[idx]
            frame.append((lm.x, lm.y))
        self.landmark_history.append(frame)
        if len(self.landmark_history) < 5:
            return 100.0
        arr = np.array(self.landmark_history)
        var = np.mean(np.var(arr, axis=0))
        return float(np.clip(100.0 - var * 80000.0, 0.0, 100.0))

    # ==================================================================
    # BODY ORIENTATION METRICS
    # ==================================================================

    def _compute_orientation(self, c: dict, ac: dict) -> dict:
        """
        Comprehensive body orientation metrics.
        These are the foundation for category classification and condition checks.
        """
        ls, rs = c.get("LEFT_SHOULDER"), c.get("RIGHT_SHOULDER")
        lh, rh = c.get("LEFT_HIP"), c.get("RIGHT_HIP")
        la, ra = c.get("LEFT_ANKLE"), c.get("RIGHT_ANKLE")
        lw, rw = c.get("LEFT_WRIST"), c.get("RIGHT_WRIST")
        lk, rk = c.get("LEFT_KNEE"), c.get("RIGHT_KNEE")

        o = {}

        # === Y-positions (image coords: 0=top, 1=bottom) ===
        o["hip_y"] = (lh[1] + rh[1]) / 2.0 if (lh and rh) else 0.5
        o["shoulder_y"] = (ls[1] + rs[1]) / 2.0 if (ls and rs) else 0.5
        o["ankle_y"] = (la[1] + ra[1]) / 2.0 if (la and ra) else o["hip_y"] + 0.35
        o["wrist_y"] = (lw[1] + rw[1]) / 2.0 if (lw and rw) else o["shoulder_y"]
        o["knee_y"] = (lk[1] + rk[1]) / 2.0 if (lk and rk) else o["hip_y"] + 0.15

        # === Vertical gaps ===
        o["hip_ankle_gap"] = abs(o["hip_y"] - o["ankle_y"])
        o["hip_shoulder_gap"] = o["hip_y"] - o["shoulder_y"]  # positive = hips below shoulders (normal)
        o["hip_knee_gap"] = abs(o["hip_y"] - o["knee_y"])
        o["shoulder_ankle_gap"] = abs(o["shoulder_y"] - o["ankle_y"])

        # === Torso slope: angle of shoulder→hip from horizontal ===
        # High (near 90°) = upright. Low (near 0°) = horizontal/prone
        if ac["shoulder_mid"] and ac["hip_mid"]:
            dx = abs(ac["shoulder_mid"][0] - ac["hip_mid"][0])
            dy = abs(ac["shoulder_mid"][1] - ac["hip_mid"][1])
            o["torso_slope"] = float(np.degrees(np.arctan2(dy, dx + 1e-6)))
        else:
            o["torso_slope"] = 90.0

        # === Knee angles ===
        o["avg_knee_angle"] = (ac["left_knee"] + ac["right_knee"]) / 2.0
        o["min_knee_angle"] = min(ac["left_knee"], ac["right_knee"])
        o["max_knee_angle"] = max(ac["left_knee"], ac["right_knee"])

        # === Spreads ===
        o["knee_x_spread"] = abs(lk[0] - rk[0]) if (lk and rk) else 0.0
        o["ankle_x_spread"] = abs(la[0] - ra[0]) if (la and ra) else 0.0
        o["shoulder_x_spread"] = abs(ls[0] - rs[0]) if (ls and rs) else 0.15
        o["hip_x_spread"] = abs(lh[0] - rh[0]) if (lh and rh) else 0.10

        # === Floor reference: lowest visible point (highest Y in image coords) ===
        all_y = []
        for pt in [la, ra, lk, rk, lh, rh, ls, rs, lw, rw]:
            if pt:
                all_y.append(pt[1])
        o["floor_y"] = max(all_y) if all_y else 1.0

        # === How close are hips to the floor? ===
        o["hip_floor_ratio"] = abs(o["hip_y"] - o["floor_y"])

        # === Body bounding box aspect ratio ===
        all_x = []
        for pt in [la, ra, lk, rk, lh, rh, ls, rs, lw, rw]:
            if pt:
                all_x.append(pt[0])
                all_y.append(pt[1])
        if all_x and all_y:
            bbox_w = max(all_x) - min(all_x) + 1e-6
            bbox_h = max(all_y) - min(all_y) + 1e-6
            o["aspect_ratio"] = bbox_h / bbox_w  # tall = standing, wide = sitting
        else:
            o["aspect_ratio"] = 2.0

        # === Hip symmetry (Y-axis difference between left and right hip) ===
        o["hip_symmetry"] = abs(lh[1] - rh[1]) if (lh and rh) else 0.0

        # === Wrist proximity to knees ===
        if all([lw, rw, lk, rk]):
            w_mid = ((lw[0]+rw[0])/2, (lw[1]+rw[1])/2)
            k_mid = ((lk[0]+rk[0])/2, (lk[1]+rk[1])/2)
            o["wrist_knee_dist"] = self._dist(w_mid, k_mid)
        else:
            o["wrist_knee_dist"] = 0.3

        # === Wrist proximity to ankles ===
        if all([lw, rw, la, ra]):
            w_mid = ((lw[0]+rw[0])/2, (lw[1]+rw[1])/2)
            a_mid = ((la[0]+ra[0])/2, (la[1]+ra[1])/2)
            o["wrist_ankle_dist"] = self._dist(w_mid, a_mid)
        else:
            o["wrist_ankle_dist"] = 0.3

        return o

    # ==================================================================
    # STAGE 1: BODY CATEGORY DETECTION
    # ==================================================================

    def classify_category(self, c: dict, ac: dict, o: dict) -> tuple:
        """
        Classify body orientation using composite metrics with explicit ordering.

        Priority order prevents cross-contamination:
          1. Inverted (hips clearly above shoulders)
          2. Prone (torso horizontal + specific geometry)
          3. Backbend (special body shapes)
          4. Balance/Boat (legs elevated + torso leaning)
          5. Sitting (hips near floor + torso upright + knees folded)
          6. Standing (default)

        Returns (category, confidence, reason_str)
        """
        reasons = []

        torso_upright = o["torso_slope"] > 55.0
        torso_horizontal = o["torso_slope"] < 40.0
        hips_near_floor = o["hip_floor_ratio"] < 0.12
        hips_well_above_ankles = o["hip_ankle_gap"] > 0.22
        knees_folded = o["avg_knee_angle"] < 130.0
        knees_deeply_folded = o["avg_knee_angle"] < 100.0

        # --- 1. INVERTED: hips clearly ABOVE shoulders ---
        if o["hip_shoulder_gap"] < -0.06:
            reasons.append(f"hips above shoulders (gap={o['hip_shoulder_gap']:.3f})")
            # Distinguish Downward Dog from other inversions
            if o["wrist_y"] > o["shoulder_y"] + 0.03:  # wrists below shoulders (near floor)
                reasons.append("wrists near floor → Downward Dog geometry")
            return "Inverted", 90.0, "; ".join(reasons)

        # --- 2. PRONE: torso clearly horizontal + hips near lowest point ---
        # Critical: shoulders must be ABOVE hips (in image: shoulder_y < hip_y)
        if torso_horizontal and o["shoulder_y"] < o["hip_y"]:
            if o["hip_floor_ratio"] < 0.18:
                reasons.append(f"torso horizontal (slope={o['torso_slope']:.0f}°)")
                reasons.append(f"hips near floor (ratio={o['hip_floor_ratio']:.3f})")
                reasons.append(f"shoulders above hips in frame")
                return "Prone", 85.0, "; ".join(reasons)

        # --- 3. BACKBEND: special body shapes ---
        # Bridge: shoulders near floor, hips elevated above shoulders, knees bent
        if (o["hip_shoulder_gap"] < -0.02 and
            knees_folded and
            o["shoulder_y"] > o["hip_y"]):
            reasons.append("shoulders below hips + knees bent → backbend geometry")
            return "Backbend", 80.0, "; ".join(reasons)

        # Camel: kneeling with torso arching back
        la, ra = c.get("LEFT_ANKLE"), c.get("RIGHT_ANKLE")
        lk, rk = c.get("LEFT_KNEE"), c.get("RIGHT_KNEE")
        if (lk and rk and la and ra):
            knee_ankle_y_gap = abs(o["knee_y"] - o["ankle_y"])
            if (knees_folded and
                knee_ankle_y_gap < 0.15 and
                not torso_upright and
                o["torso_slope"] < 55.0 and
                o["shoulder_y"] < o["hip_y"]):
                reasons.append(f"kneeling (knee-ankle gap={knee_ankle_y_gap:.3f}) + torso arching back")
                return "Backbend", 75.0, "; ".join(reasons)

        # --- 4. BALANCE / BOAT: legs elevated + torso leaning ---
        if la and ra:
            avg_ankle_y = (la[1] + ra[1]) / 2.0
            if (avg_ankle_y < o["hip_y"] - 0.03 and
                o["shoulder_y"] < o["hip_y"] and
                not torso_upright):
                reasons.append("legs elevated above hips + torso leaning back")
                return "Sitting", 80.0, "; ".join(reasons)  # Boat is classified under Sitting

        # --- 5. SITTING: hips near floor + torso upright + knees folded ---
        if hips_near_floor and torso_upright:
            reasons.append(f"hips near floor (ratio={o['hip_floor_ratio']:.3f})")
            reasons.append(f"torso upright (slope={o['torso_slope']:.0f}°)")
            return "Sitting", 90.0, "; ".join(reasons)

        # Sitting fallback: hips near floor + knees folded (even if slouching)
        if hips_near_floor and knees_folded:
            reasons.append(f"hips near floor + knees folded ({o['avg_knee_angle']:.0f}°)")
            return "Sitting", 75.0, "; ".join(reasons)

        # Extra sitting check: very compact body (low aspect ratio), knees deeply bent
        if (o["aspect_ratio"] < 1.2 and knees_deeply_folded and
            o["hip_ankle_gap"] < 0.30):
            reasons.append(f"compact body (AR={o['aspect_ratio']:.2f}) + deep knee fold")
            return "Sitting", 70.0, "; ".join(reasons)

        # --- 6. STANDING: default ---
        reasons.append(f"upright body, hips above ankles (gap={o['hip_ankle_gap']:.3f})")
        return "Standing", 80.0, "; ".join(reasons)

    # ==================================================================
    # CONDITION EVALUATORS (for rejection_rules and boost_rules)
    # ==================================================================

    def _evaluate_condition(self, condition: str, c: dict, ac: dict, o: dict) -> bool:
        """Evaluate a named body-state condition. Returns True if condition is met."""

        # --- Torso conditions ---
        if condition == "torso_vertical":
            return o["torso_slope"] > 75.0
        if condition == "torso_upright":
            return o["torso_slope"] > 55.0
        if condition == "torso_horizontal":
            return o["torso_slope"] < 40.0
        if condition == "torso_leaning_back":
            # Shoulder behind hip (in x)
            if ac["shoulder_mid"] and ac["hip_mid"]:
                return ac["shoulder_mid"][1] < ac["hip_mid"][1] and o["torso_slope"] < 65.0
            return False
        if condition == "torso_lateral_tilt":
            if ac["shoulder_mid"] and ac["hip_mid"]:
                dx = abs(ac["shoulder_mid"][0] - ac["hip_mid"][0])
                return dx > 0.08
            return False

        # --- Hip conditions ---
        if condition == "hips_on_floor":
            return o["hip_floor_ratio"] < 0.12
        if condition == "hips_elevated":
            return o["hip_ankle_gap"] > 0.22
        if condition == "hips_above_ankles":
            return o["hip_ankle_gap"] > 0.20
        if condition == "hips_elevated_above_shoulders":
            return o["hip_shoulder_gap"] < -0.02
        if condition == "hips_highest_point":
            return o["hip_shoulder_gap"] < -0.04
        if condition == "hips_pushed_forward":
            if ac["shoulder_mid"] and ac["hip_mid"]:
                return ac["hip_mid"][0] != ac["shoulder_mid"][0]  # basic check
            return False

        # --- Knee conditions ---
        if condition == "legs_crossed":
            return o["avg_knee_angle"] < 120.0 and o["knee_x_spread"] < 0.20
        if condition == "legs_straight":
            return o["avg_knee_angle"] > 155.0
        if condition == "both_legs_straight":
            return ac["left_knee"] > 155.0 and ac["right_knee"] > 155.0
        if condition == "both_knees_bent":
            return ac["left_knee"] < 140.0 and ac["right_knee"] < 140.0
        if condition == "both_knees_straight":
            return ac["left_knee"] > 155.0 and ac["right_knee"] > 155.0
        if condition == "both_knees_deeply_bent":
            return ac["left_knee"] < 110.0 and ac["right_knee"] < 110.0
        if condition == "one_knee_bent":
            lk, rk = ac["left_knee"], ac["right_knee"]
            return (lk < 135 and rk > 150) or (rk < 135 and lk > 150)
        if condition == "one_knee_bent_warrior":
            lk, rk = ac["left_knee"], ac["right_knee"]
            return (lk < 140 and rk > 145) or (rk < 140 and lk > 145)
        if condition == "knees_bent_90":
            return 70 < o["avg_knee_angle"] < 110
        if condition == "knees_wide":
            return o["knee_x_spread"] > o["ankle_x_spread"] * 0.9 and o["knee_x_spread"] > 0.10
        if condition == "legs_elevated":
            la, ra = c.get("LEFT_ANKLE"), c.get("RIGHT_ANKLE")
            if la and ra:
                avg_ankle_y = (la[1] + ra[1]) / 2.0
                return avg_ankle_y < o["hip_y"] - 0.02
            return False
        if condition == "legs_on_floor":
            la, ra = c.get("LEFT_ANKLE"), c.get("RIGHT_ANKLE")
            if la and ra:
                avg_ankle_y = (la[1] + ra[1]) / 2.0
                return avg_ankle_y > o["hip_y"] + 0.05
            return True
        if condition == "legs_straight_on_floor":
            return o["avg_knee_angle"] > 155.0 and o["hip_floor_ratio"] < 0.15
        if condition == "kneeling":
            lk, rk = c.get("LEFT_KNEE"), c.get("RIGHT_KNEE")
            la, ra = c.get("LEFT_ANKLE"), c.get("RIGHT_ANKLE")
            if lk and rk and la and ra:
                knee_y = (lk[1] + rk[1]) / 2.0
                ankle_y = (la[1] + ra[1]) / 2.0
                return abs(knee_y - ankle_y) < 0.15 and o["avg_knee_angle"] < 120.0
            return False
        if condition == "torso_arching_back":
            return o["torso_slope"] < 55.0 and o["shoulder_y"] < o["hip_y"]

        # --- Stance conditions ---
        if condition == "wide_stance":
            return o["ankle_x_spread"] > 0.20
        if condition == "feet_together":
            return o["ankle_x_spread"] < 0.12

        # --- Arm conditions ---
        if condition == "arms_at_sides":
            return (ac["left_shoulder_angle"] < 30 and ac["right_shoulder_angle"] < 30)
        if condition == "arms_overhead":
            return (ac["left_shoulder_angle"] > 140 and ac["right_shoulder_angle"] > 140)
        if condition == "arms_horizontal":
            ls, rs = c.get("LEFT_SHOULDER"), c.get("RIGHT_SHOULDER")
            lw, rw = c.get("LEFT_WRIST"), c.get("RIGHT_WRIST")
            if all([ls, rs, lw, rw]):
                left_diff = abs(ls[1] - lw[1])
                right_diff = abs(rs[1] - rw[1])
                return left_diff < 0.08 and right_diff < 0.08
            return False
        if condition == "arms_forward":
            lw, rw = c.get("LEFT_WRIST"), c.get("RIGHT_WRIST")
            ls, rs = c.get("LEFT_SHOULDER"), c.get("RIGHT_SHOULDER")
            if all([lw, rw, ls, rs]):
                avg_w_y = (lw[1] + rw[1]) / 2
                avg_s_y = (ls[1] + rs[1]) / 2
                return abs(avg_w_y - avg_s_y) < 0.12
            return False
        if condition == "standing_leg_straight":
            return max(ac["left_knee"], ac["right_knee"]) > 165.0

        # --- Hand/Floor conditions ---
        if condition == "hands_on_knees":
            return o["wrist_knee_dist"] < 0.12
        if condition == "hands_on_floor":
            lw, rw = c.get("LEFT_WRIST"), c.get("RIGHT_WRIST")
            if lw and rw:
                avg_wrist_y = (lw[1] + rw[1]) / 2.0
                return avg_wrist_y > o["floor_y"] - 0.15
            return False
        if condition == "hands_on_heels":
            lw, rw = c.get("LEFT_WRIST"), c.get("RIGHT_WRIST")
            lh_pt, rh_pt = c.get("LEFT_HEEL"), c.get("RIGHT_HEEL")
            if all([lw, rw, lh_pt, rh_pt]):
                return self._dist(lw, lh_pt) < 0.15 or self._dist(rw, rh_pt) < 0.15
            return False
        if condition == "hands_touching_floor_and_hips_elevated":
            hands_down = self._evaluate_condition("hands_on_floor", c, ac, o)
            hips_up = o["hip_shoulder_gap"] < -0.04
            return hands_down and hips_up
        if condition == "chest_lifted":
            return o["shoulder_y"] < o["hip_y"] and o["torso_slope"] > 15.0
        if condition == "shoulders_on_floor":
            ls, rs = c.get("LEFT_SHOULDER"), c.get("RIGHT_SHOULDER")
            if ls and rs:
                avg_s_y = (ls[1] + rs[1]) / 2.0
                return avg_s_y > o["floor_y"] - 0.10
            return False
        if condition == "v_shape":
            la, ra = c.get("LEFT_ANKLE"), c.get("RIGHT_ANKLE")
            if la and ra:
                avg_ankle_y = (la[1] + ra[1]) / 2.0
                return avg_ankle_y < o["hip_y"] - 0.02
            return False

        # --- Inversion ---
        if condition == "not_inverted":
            return o["hip_shoulder_gap"] > -0.05
        if condition == "not_prone":
            return o["torso_slope"] > 40.0

        # Unknown condition — don't block
        return False

    # ==================================================================
    # STAGE 2: POSE GATING
    # ==================================================================

    def _gate_candidates(self, category: str) -> tuple:
        """
        Return (allowed_keys, rejected_dict).
        Only poses matching the detected category pass through.
        """
        allowed = []
        rejected = {}

        for pose_key, data in POSE_DATABASE.items():
            if data["category"] == category:
                allowed.append(pose_key)
            else:
                rejected[data["display_name"]] = [
                    f"Category mismatch: detected '{category}' but pose requires '{data['category']}'"
                ]

        return allowed, rejected

    # ==================================================================
    # MIRRORED SIDE DETECTION
    # ==================================================================

    def _detect_sides(self, pose_key: str, ac: dict) -> dict:
        """Determine which side is standing/bent or front/back."""
        sides = {
            "standing_side": "left", "bent_side": "right",
            "front_side": "left", "back_side": "right",
        }
        lk, rk = ac["left_knee"], ac["right_knee"]

        if pose_key == "tree":
            if lk > rk:
                sides["standing_side"], sides["bent_side"] = "left", "right"
            else:
                sides["standing_side"], sides["bent_side"] = "right", "left"

        elif pose_key in ("warrior_i", "warrior_ii", "triangle"):
            if lk < rk:
                sides["front_side"], sides["back_side"] = "left", "right"
            else:
                sides["front_side"], sides["back_side"] = "right", "left"

        return sides

    # ==================================================================
    # JOINT VALUE COMPUTATION
    # ==================================================================

    def _compute_joint_value(self, joint_type: str, c: dict, ac: dict, o: dict, sides: dict) -> float:
        """Map any joint_type string → computed float."""
        ls, rs = c.get("LEFT_SHOULDER"), c.get("RIGHT_SHOULDER")
        lh, rh = c.get("LEFT_HIP"), c.get("RIGHT_HIP")
        lw, rw = c.get("LEFT_WRIST"), c.get("RIGHT_WRIST")
        la, ra = c.get("LEFT_ANKLE"), c.get("RIGHT_ANKLE")
        lk, rk = c.get("LEFT_KNEE"), c.get("RIGHT_KNEE")

        if joint_type == "torso_vertical":
            if ac["shoulder_mid"] and ac["hip_mid"]:
                return self._angle(
                    ac["shoulder_mid"], ac["hip_mid"],
                    (ac["hip_mid"][0] + 0.5, ac["hip_mid"][1], 0.0, 1.0)
                )
            return 90.0

        elif joint_type == "avg_knee":
            return (ac["left_knee"] + ac["right_knee"]) / 2.0

        elif joint_type == "avg_arm":
            return (ac["left_arm"] + ac["right_arm"]) / 2.0

        elif joint_type == "avg_arm_body_angle":
            avg = (ac["left_shoulder_angle"] + ac["right_shoulder_angle"]) / 2.0
            return avg

        elif joint_type == "avg_arm_elevation":
            return (ac["left_shoulder_angle"] + ac["right_shoulder_angle"]) / 2.0

        elif joint_type == "standing_knee":
            return ac["left_knee"] if sides.get("standing_side") == "left" else ac["right_knee"]

        elif joint_type == "bent_knee":
            return ac["left_knee"] if sides.get("bent_side") == "left" else ac["right_knee"]

        elif joint_type == "front_knee":
            return ac["left_knee"] if sides.get("front_side") == "left" else ac["right_knee"]

        elif joint_type == "back_knee":
            return ac["left_knee"] if sides.get("back_side") == "left" else ac["right_knee"]

        elif joint_type == "arms_horizontal_score":
            if all([ls, lw, rs, rw]):
                left_v = abs(ls[1] - lw[1])
                right_v = abs(rs[1] - rw[1])
                avg_v = (left_v + right_v) / 2.0
                return max(0.0, 100.0 - avg_v * 120.0)
            return 100.0

        elif joint_type == "hip_flexion":
            return (ac["left_hip"] + ac["right_hip"]) / 2.0

        elif joint_type == "shoulder_shrug":
            return 90.0

        elif joint_type == "knees_drop_ratio":
            if all([lk, rk, lh, rh]):
                hip_y = (lh[1] + rh[1]) / 2.0
                knee_y = (lk[1] + rk[1]) / 2.0
                return abs(knee_y - hip_y) * 100.0
            return 30.0

        elif joint_type == "hands_feet_dist":
            if all([lw, rw, la, ra]):
                w_mid = ((lw[0]+rw[0])/2, (lw[1]+rw[1])/2)
                a_mid = ((la[0]+ra[0])/2, (la[1]+ra[1])/2)
                return self._dist(w_mid, a_mid)
            return 0.05

        elif joint_type == "hands_knee_proximity":
            if all([lw, rw, lk, rk]):
                w_mid = ((lw[0]+rw[0])/2, (lw[1]+rw[1])/2)
                k_mid = ((lk[0]+rk[0])/2, (lk[1]+rk[1])/2)
                return self._dist(w_mid, k_mid)
            return 0.05

        elif joint_type == "hip_symmetry":
            return o.get("hip_symmetry", 0.0)

        elif joint_type == "ankle_x_spread":
            return o.get("ankle_x_spread", 0.05)

        elif joint_type == "hip_floor_ratio":
            return o.get("hip_floor_ratio", 0.05)

        elif joint_type == "torso_lateral_angle":
            # Angle of torso tilt from vertical in the frontal plane
            if ac["shoulder_mid"] and ac["hip_mid"]:
                dx = abs(ac["shoulder_mid"][0] - ac["hip_mid"][0])
                dy = abs(ac["shoulder_mid"][1] - ac["hip_mid"][1])
                return float(np.degrees(np.arctan2(dx, dy + 1e-6)))
            return 0.0

        elif joint_type == "arms_vertical_spread":
            # Angle between left and right wrist relative to shoulders
            if all([lw, rw, ls, rs]):
                left_arm_angle = ac["left_shoulder_angle"]
                right_arm_angle = ac["right_shoulder_angle"]
                return left_arm_angle + right_arm_angle
            return 60.0

        elif joint_type == "body_v_angle":
            # Angle at hips between shoulder-hip and hip-ankle lines
            if all([ac["shoulder_mid"], ac["hip_mid"], la, ra]):
                ankle_mid = ((la[0]+ra[0])/2, (la[1]+ra[1])/2)
                return self._angle(ac["shoulder_mid"], ac["hip_mid"], ankle_mid)
            return 90.0

        elif joint_type == "hip_elevation_ratio":
            # Ratio of hip height relative to shoulder-floor range
            if o["shoulder_ankle_gap"] > 0.05:
                return (o["hip_y"] - o["shoulder_y"]) / o["shoulder_ankle_gap"]
            return 0.5

        elif joint_type == "shoulder_floor_proximity":
            # How close shoulders are to the lowest point
            ls, rs = c.get("LEFT_SHOULDER"), c.get("RIGHT_SHOULDER")
            if ls and rs:
                avg_s_y = (ls[1] + rs[1]) / 2.0
                return abs(avg_s_y - o["floor_y"])
            return 0.3

        return 0.0

    # ==================================================================
    # STAGE 3: CONFIDENCE SCORING
    # ==================================================================

    def _score_pose(self, pose_key: str, c: dict, ac: dict, o: dict, sides: dict) -> tuple:
        """
        Weighted multi-factor confidence scoring.
        Returns (confidence_0_to_100, detail_list).
        """
        rules = POSE_DATABASE[pose_key]["rules"]
        total = 0.0
        details = []

        for rule_name, rule in rules.items():
            target = rule["target"]
            tol = rule["tolerance"]
            weight = rule["weight"]

            val = self._compute_joint_value(rule["joint"], c, ac, o, sides)
            diff = abs(val - target)

            if diff <= tol:
                score = 1.0
            else:
                excess = diff - tol
                score = max(0.0, 1.0 - excess / max(tol * 1.5, 1.0))

            weighted = score * weight
            total += weighted

            details.append({
                "rule": rule_name,
                "joint": rule["joint"],
                "target": target,
                "computed": round(val, 1),
                "diff": round(diff, 2),
                "score": round(score, 3),
                "weight": round(weight, 2),
                "weighted": round(weighted, 3),
                "status": "PASS" if score >= 0.75 else "PARTIAL" if score >= 0.35 else "FAIL"
            })

        confidence = float(np.clip(total * 100.0, 0.0, 100.0))
        return confidence, details

    # ==================================================================
    # STAGE 4: HARD REJECTION + BOOST RULES
    # ==================================================================

    def _apply_rejection_rules(self, pose_key: str, c: dict, ac: dict, o: dict) -> tuple:
        """
        Check all rejection_rules for a pose.
        Returns (rejected: bool, reasons: list[str])
        """
        data = POSE_DATABASE[pose_key]
        reasons = []

        for rule in data.get("rejection_rules", []):
            if self._evaluate_condition(rule["condition"], c, ac, o):
                reasons.append(rule["reason"])

        return len(reasons) > 0, reasons

    def _apply_required_body_state(self, pose_key: str, c: dict, ac: dict, o: dict) -> tuple:
        """
        Check all required_body_state for a pose.
        Returns (passed: bool, failing_reasons: list[str])
        """
        data = POSE_DATABASE[pose_key]
        reasons = []

        for state_name, required in data.get("required_body_state", {}).items():
            if not required:
                continue
            if not self._evaluate_condition(state_name, c, ac, o):
                reasons.append(f"Required state '{state_name}' not met")

        return len(reasons) == 0, reasons

    def _apply_boost_rules(self, pose_key: str, confidence: float,
                           c: dict, ac: dict, o: dict) -> tuple:
        """
        Apply boost rules to increase confidence for strong pose indicators.
        Returns (new_confidence, boost_reasons)
        """
        data = POSE_DATABASE[pose_key]
        boost_total = 0
        boost_reasons = []

        for rule in data.get("boost_rules", []):
            if self._evaluate_condition(rule["condition"], c, ac, o):
                boost_total += rule["bonus"]
                boost_reasons.append(f"+{rule['bonus']} ({rule['reason']})")

        new_conf = min(100.0, confidence + boost_total)
        return new_conf, boost_reasons

    # ==================================================================
    # FULL CANDIDATE EVALUATION
    # ==================================================================

    def _evaluate_candidates(self, allowed_keys: list, c: dict, ac: dict, o: dict) -> tuple:
        """
        Score all allowed poses through the full pipeline:
          1. Check required_body_state → reject if not met
          2. Check rejection_rules → reject if triggered
          3. Score with weighted rules
          4. Apply boost rules
        Returns (sorted_candidates, rejection_log)
        """
        candidates = []
        rejection_log = {}

        for pose_key in allowed_keys:
            data = POSE_DATABASE[pose_key]
            display = data["display_name"]

            # Step 1: Required body state
            state_ok, state_reasons = self._apply_required_body_state(pose_key, c, ac, o)
            if not state_ok:
                rejection_log[display] = [f"Body state: {r}" for r in state_reasons]
                continue

            # Step 2: Hard rejection rules
            rejected, rej_reasons = self._apply_rejection_rules(pose_key, c, ac, o)
            if rejected:
                rejection_log[display] = rej_reasons
                continue

            # Step 3: Score
            sides = self._detect_sides(pose_key, ac)
            conf, details = self._score_pose(pose_key, c, ac, o, sides)

            # Step 4: Boost
            boosted_conf, boost_reasons = self._apply_boost_rules(pose_key, conf, c, ac, o)

            candidates.append({
                "key": pose_key,
                "confidence": boosted_conf,
                "raw_confidence": conf,
                "sides": sides,
                "details": details,
                "boost_reasons": boost_reasons,
            })

        sorted_cands = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
        return sorted_cands, rejection_log

    # ==================================================================
    # TEMPORAL SMOOTHING (webcam mode only)
    # ==================================================================

    def _temporal_smooth(self, best_key: str, confidence: float) -> str:
        """
        15-frame buffer with majority vote and pose locking.
        Lock at >75%, unlock when <40% for 5+ frames.
        """
        self.pose_buffer.append(best_key)

        if confidence < 40.0:
            self.low_conf_frames += 1
        else:
            self.low_conf_frames = 0

        if confidence > 75.0 and best_key != "unknown":
            self.locked_pose = best_key
            self.current_stable_pose = best_key

        if self.low_conf_frames >= 5:
            self.locked_pose = "unknown"

        freq = {}
        for p in self.pose_buffer:
            freq[p] = freq.get(p, 0) + 1
        majority = max(freq, key=freq.get)
        majority_count = freq[majority]

        if self.locked_pose != "unknown":
            if majority_count >= 10 and majority != "unknown" and majority != self.locked_pose:
                self.locked_pose = majority
                self.current_stable_pose = majority
            else:
                self.current_stable_pose = self.locked_pose
        else:
            if majority_count >= 10:
                self.current_stable_pose = majority

        return self.current_stable_pose

    # ==================================================================
    # MAIN ENTRY POINT
    # ==================================================================

    def analyze(self, detector, lm_list, selected_pose: str = None, is_static: bool = False) -> dict:
        """
        Central classification entrypoint — 4-stage pipeline.

        Parameters
        ----------
        detector     : PoseDetector instance
        lm_list      : MediaPipe landmark list for one person
        selected_pose: Optional manual pose selection string
        is_static    : If True (image upload), bypass temporal smoothing
        """
        result = {
            "category": "Detecting...",
            "category_confidence": 0.0,
            "category_reason": "",
            "detected_pose": "unknown",
            "confidence": 0.0,
            "accuracy": 0.0,
            "corrections": [],
            "stability": 100.0,
            "entry_guidance": "",
            "candidate_scores": {},
            "rejected_poses": {},
            "orientation": {},
            "angles": {},
            "is_low_confidence": False,
            "boost_applied": [],
            "rejection_applied": [],
        }

        # ── Step 1: Visibility check ──
        visible, warning = self._check_visibility(detector, lm_list)
        if not visible:
            self.locked_pose = "unknown"
            self.current_stable_pose = "unknown"
            self.pose_buffer.clear()
            result["corrections"].append({"key": "no_pose", "message": warning, "priority": 1})
            result["entry_guidance"] = warning
            return result

        # ── Step 2: Stability ──
        result["stability"] = round(self._compute_stability(lm_list), 1)

        # ── Step 3: Extract coords + angles ──
        c = self._extract_coords(detector, lm_list)
        ac = self._build_angle_cache(c)

        result["angles"] = {
            "Left Knee": round(ac["left_knee"], 1),
            "Right Knee": round(ac["right_knee"], 1),
            "Left Hip": round(ac["left_hip"], 1),
            "Right Hip": round(ac["right_hip"], 1),
            "Left Arm": round(ac["left_arm"], 1),
            "Right Arm": round(ac["right_arm"], 1),
            "Left Shoulder": round(ac["left_shoulder_angle"], 1),
            "Right Shoulder": round(ac["right_shoulder_angle"], 1),
        }

        # ── Step 4: Body orientation metrics ──
        o = self._compute_orientation(c, ac)
        result["orientation"] = {
            "torso_slope": round(o["torso_slope"], 1),
            "hip_floor_ratio": round(o["hip_floor_ratio"], 3),
            "hip_ankle_gap": round(o["hip_ankle_gap"], 3),
            "hip_shoulder_gap": round(o["hip_shoulder_gap"], 3),
            "avg_knee_angle": round(o["avg_knee_angle"], 1),
            "aspect_ratio": round(o["aspect_ratio"], 2),
            "knee_x_spread": round(o["knee_x_spread"], 3),
            "ankle_x_spread": round(o["ankle_x_spread"], 3),
        }

        # ── STAGE 1: Body category detection ──
        category, cat_conf, cat_reason = self.classify_category(c, ac, o)
        result["category"] = category
        result["category_confidence"] = cat_conf
        result["category_reason"] = cat_reason

        # ── STAGE 2: Pose gating ──
        allowed_keys, gating_rejections = self._gate_candidates(category)
        result["rejected_poses"].update(gating_rejections)

        # ── STAGE 3 + 4: Evaluate candidates (score + reject + boost) ──
        candidates, eval_rejections = self._evaluate_candidates(allowed_keys, c, ac, o)
        result["rejected_poses"].update(eval_rejections)

        # Store all candidate scores for debug
        for cand in candidates:
            display = POSE_DATABASE[cand["key"]]["display_name"]
            result["candidate_scores"][display] = round(cand["confidence"], 1)

        # ── Pick best candidate ──
        best_key = "unknown"
        best_conf = 0.0
        best_sides = {}
        best_boosts = []

        if candidates:
            top = candidates[0]
            best_key = top["key"]
            best_conf = top["confidence"]
            best_sides = top["sides"]
            best_boosts = top["boost_reasons"]

        # ── Apply global uncertainty threshold ──
        threshold = GLOBAL_UNCERTAINTY_THRESHOLD
        if best_key != "unknown":
            pose_threshold = POSE_DATABASE[best_key].get("confidence_threshold", threshold)
            if best_conf < pose_threshold:
                result["is_low_confidence"] = True
                best_key = "unknown"

        # ── Manual pose override ──
        if selected_pose and selected_pose != "Auto-Detect":
            for k, data in POSE_DATABASE.items():
                if data["display_name"] == selected_pose:
                    sides = self._detect_sides(k, ac)
                    conf, _ = self._score_pose(k, c, ac, o, sides)
                    boosted, boost_r = self._apply_boost_rules(k, conf, c, ac, o)
                    if boosted >= data.get("confidence_threshold", threshold):
                        best_key = k
                        best_conf = boosted
                        best_sides = sides
                        best_boosts = boost_r
                    else:
                        best_key = "unknown"
                    break

        # ── Temporal smoothing (skip for static images) ──
        if is_static:
            final_key = best_key
        else:
            final_key = self._temporal_smooth(best_key, best_conf)

        # ── Build result ──
        if final_key == "unknown":
            result["detected_pose"] = "unknown"
            result["confidence"] = round(best_conf, 1)
            result["accuracy"] = 0.0
            result["boost_applied"] = best_boosts

            if selected_pose and selected_pose != "Auto-Detect":
                for k, data in POSE_DATABASE.items():
                    if data["display_name"] == selected_pose:
                        result["entry_guidance"] = data["entry_guidance"]
                        break
            else:
                result["entry_guidance"] = f"Pose Uncertain — checking {category} candidates..."

            result["corrections"].append({
                "key": "uncertain",
                "message": result["entry_guidance"],
                "priority": 1
            })
            return result

        pose_data = POSE_DATABASE[final_key]
        result["detected_pose"] = final_key
        result["confidence"] = round(best_conf, 1)
        result["accuracy"] = round(best_conf, 1)
        result["entry_guidance"] = pose_data["entry_guidance"]
        result["boost_applied"] = best_boosts

        # ── Generate corrections ──
        errors = []
        for rule_name, rule in pose_data["rules"].items():
            val = self._compute_joint_value(rule["joint"], c, ac, o, best_sides)
            if abs(val - rule["target"]) > rule["tolerance"]:
                errors.append({
                    "key": f"{final_key}_{rule_name}",
                    "message": rule["error_msg"],
                    "priority": rule["priority"]
                })

        if errors:
            errors.sort(key=lambda x: x["priority"])
            result["corrections"] = [errors[0]]
        else:
            result["corrections"] = [{
                "key": f"{final_key}_perfect",
                "message": "Excellent posture! Maintain this position.",
                "priority": 9
            }]

        return result
