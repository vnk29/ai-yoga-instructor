"""
Industrial Yoga Pose Classification Engine — Strict Multi-Stage Gating.

Architecture:
  1. Visibility check (shoulders + hips mandatory)
  2. Extract landmarks + precompute angles
  3. Compute BODY ORIENTATION metrics (torso slope, hip height, floor relationship)
  4. Classify body CATEGORY using strict gating rules
  5. Apply hard geometry filters to eliminate impossible candidates
  6. Score remaining candidates with weighted angle rules
  7. Apply temporal smoothing (webcam only) or return raw result (image upload)
  8. Generate prioritized posture corrections
  9. Build debug/rejection panel
"""

import numpy as np
from collections import deque
from yoga_config import POSE_DATABASE


class YogaAnalyzer:
    """Industrial-grade hierarchical yoga pose classifier with strict orientation gating."""

    def __init__(self) -> None:
        # Temporal smoothing buffers (webcam mode only)
        self.pose_buffer = deque(maxlen=15)
        self.landmark_history = deque(maxlen=15)
        self.current_stable_pose = "unknown"
        self.locked_pose = "unknown"
        self.low_conf_frames = 0

    # ------------------------------------------------------------------
    # GEOMETRY PRIMITIVES
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # LANDMARK EXTRACTION
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_coords(detector, lm_list) -> dict:
        """Pull named landmark coordinates into a flat dict. Missing joints → None."""
        names = [
            "NOSE", "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_HIP", "RIGHT_HIP",
            "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE",
            "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
            "LEFT_EAR", "RIGHT_EAR"
        ]
        coords = {}
        for n in names:
            try:
                coords[n] = detector.get_coords(lm_list, n)
            except Exception:
                coords[n] = None
        return coords

    # ------------------------------------------------------------------
    # PRECOMPUTED ANGLE CACHE
    # ------------------------------------------------------------------

    def _build_angle_cache(self, c: dict) -> dict:
        """Compute all reusable angles once and return as a flat dict."""
        a = {}
        a["left_knee"]  = self._angle(c["LEFT_HIP"],  c["LEFT_KNEE"],  c["LEFT_ANKLE"])  if all([c.get("LEFT_HIP"),  c.get("LEFT_KNEE"),  c.get("LEFT_ANKLE")])  else 180.0
        a["right_knee"] = self._angle(c["RIGHT_HIP"], c["RIGHT_KNEE"], c["RIGHT_ANKLE"]) if all([c.get("RIGHT_HIP"), c.get("RIGHT_KNEE"), c.get("RIGHT_ANKLE")]) else 180.0
        a["left_hip"]   = self._angle(c["LEFT_SHOULDER"],  c["LEFT_HIP"],  c["LEFT_KNEE"])  if all([c.get("LEFT_SHOULDER"),  c.get("LEFT_HIP"),  c.get("LEFT_KNEE")])  else 180.0
        a["right_hip"]  = self._angle(c["RIGHT_SHOULDER"], c["RIGHT_HIP"], c["RIGHT_KNEE"]) if all([c.get("RIGHT_SHOULDER"), c.get("RIGHT_HIP"), c.get("RIGHT_KNEE")]) else 180.0
        a["left_arm"]   = self._angle(c["LEFT_SHOULDER"],  c["LEFT_ELBOW"],  c["LEFT_WRIST"])  if all([c.get("LEFT_SHOULDER"),  c.get("LEFT_ELBOW"),  c.get("LEFT_WRIST")])  else 180.0
        a["right_arm"]  = self._angle(c["RIGHT_SHOULDER"], c["RIGHT_ELBOW"], c["RIGHT_WRIST"]) if all([c.get("RIGHT_SHOULDER"), c.get("RIGHT_ELBOW"), c.get("RIGHT_WRIST")]) else 180.0

        # Midpoints
        ls, rs = c.get("LEFT_SHOULDER"), c.get("RIGHT_SHOULDER")
        lh, rh = c.get("LEFT_HIP"), c.get("RIGHT_HIP")
        if all([ls, rs, lh, rh]):
            a["shoulder_mid"] = ((ls[0]+rs[0])/2, (ls[1]+rs[1])/2)
            a["hip_mid"]      = ((lh[0]+rh[0])/2, (lh[1]+rh[1])/2)
        else:
            a["shoulder_mid"] = None
            a["hip_mid"] = None

        return a

    # ------------------------------------------------------------------
    # BODY VISIBILITY CHECK
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # STABILITY SCORING
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # BODY ORIENTATION METRICS
    # ------------------------------------------------------------------

    def _compute_orientation(self, c: dict, ac: dict) -> dict:
        """
        Compute body orientation metrics used by both category classification
        and hard filters. Returns a dict of floats describing body geometry.
        """
        ls, rs = c.get("LEFT_SHOULDER"), c.get("RIGHT_SHOULDER")
        lh, rh = c.get("LEFT_HIP"), c.get("RIGHT_HIP")
        la, ra = c.get("LEFT_ANKLE"), c.get("RIGHT_ANKLE")
        lw, rw = c.get("LEFT_WRIST"), c.get("RIGHT_WRIST")
        lk, rk = c.get("LEFT_KNEE"), c.get("RIGHT_KNEE")

        o = {}

        # Y-positions (in image coords: 0=top, 1=bottom)
        o["hip_y"]      = (lh[1] + rh[1]) / 2.0 if (lh and rh) else 0.5
        o["shoulder_y"] = (ls[1] + rs[1]) / 2.0 if (ls and rs) else 0.5
        o["ankle_y"]    = (la[1] + ra[1]) / 2.0 if (la and ra) else o["hip_y"] + 0.35
        o["wrist_y"]    = (lw[1] + rw[1]) / 2.0 if (lw and rw) else o["shoulder_y"]
        o["knee_y"]     = (lk[1] + rk[1]) / 2.0 if (lk and rk) else o["hip_y"] + 0.15

        # Gaps
        o["hip_ankle_gap"] = abs(o["hip_y"] - o["ankle_y"])
        o["hip_shoulder_gap"] = o["hip_y"] - o["shoulder_y"]  # positive = hips below shoulders (normal)
        o["hip_knee_gap"] = abs(o["hip_y"] - o["knee_y"])

        # Torso slope: angle of shoulder-hip line from horizontal
        # High value (near 90) = upright torso. Low value = horizontal/leaning
        if ac["shoulder_mid"] and ac["hip_mid"]:
            dx = abs(ac["shoulder_mid"][0] - ac["hip_mid"][0])
            dy = abs(ac["shoulder_mid"][1] - ac["hip_mid"][1])
            o["torso_slope"] = float(np.degrees(np.arctan2(dy, dx + 1e-6)))
        else:
            o["torso_slope"] = 90.0  # default upright

        # Knee angles
        o["avg_knee_angle"] = (ac["left_knee"] + ac["right_knee"]) / 2.0
        o["min_knee_angle"] = min(ac["left_knee"], ac["right_knee"])
        o["max_knee_angle"] = max(ac["left_knee"], ac["right_knee"])

        # Knee spread (x-axis)
        if lk and rk:
            o["knee_x_spread"] = abs(lk[0] - rk[0])
        else:
            o["knee_x_spread"] = 0.0

        # Ankle spread (x-axis)
        if la and ra:
            o["ankle_x_spread"] = abs(la[0] - ra[0])
        else:
            o["ankle_x_spread"] = 0.0

        return o

    # ------------------------------------------------------------------
    # STEP 1: STRICT CATEGORY CLASSIFICATION
    # ------------------------------------------------------------------

    def classify_category(self, c: dict, ac: dict, o: dict) -> tuple:
        """
        Classify body orientation into one of: Standing, Sitting, Prone, Inverted, Balance.
        Uses strict body orientation metrics with explicit gating.

        Returns (category_str, reason_str)
        """
        reasons = []

        # --- INVERTED: hips clearly ABOVE shoulders ---
        # In image coords: hip_y < shoulder_y
        if o["hip_shoulder_gap"] < -0.05:
            reasons.append("hips above shoulders")
            return "Inverted", "; ".join(reasons)

        # --- Key decision: UPRIGHT TORSO vs HORIZONTAL TORSO ---
        torso_is_upright = o["torso_slope"] > 50.0  # >50° from horizontal = upright-ish
        torso_is_horizontal = o["torso_slope"] < 35.0  # <35° = lying down / prone
        hips_near_floor = o["hip_ankle_gap"] < 0.25  # hips near ankle Y = on floor
        knees_folded = o["avg_knee_angle"] < 130.0

        # --- PRONE: torso horizontal AND hips near floor level ---
        # This MUST check torso orientation, not just hip height
        if torso_is_horizontal and hips_near_floor and o["shoulder_y"] < o["hip_y"]:
            reasons.append(f"torso horizontal (slope={o['torso_slope']:.0f}°)")
            reasons.append(f"hips near floor (gap={o['hip_ankle_gap']:.2f})")
            return "Prone", "; ".join(reasons)

        # --- SITTING: hips near floor AND torso upright ---
        if hips_near_floor and torso_is_upright:
            reasons.append(f"hips near floor (gap={o['hip_ankle_gap']:.2f})")
            reasons.append(f"torso upright (slope={o['torso_slope']:.0f}°)")
            return "Sitting", "; ".join(reasons)

        # --- BALANCE (Boat): ankles elevated near/above hip level, torso leaning back ---
        la, ra = c.get("LEFT_ANKLE"), c.get("RIGHT_ANKLE")
        if la and ra:
            avg_ankle_y = (la[1] + ra[1]) / 2.0
            if avg_ankle_y < o["hip_y"] - 0.02 and o["shoulder_y"] < o["hip_y"]:
                reasons.append("legs elevated above hips")
                return "Balance", "; ".join(reasons)

        # --- SITTING fallback: hips near floor but torso is in-between ---
        # Catches slouching / uncertain sitting postures
        if hips_near_floor and knees_folded:
            reasons.append(f"hips near floor + knees folded ({o['avg_knee_angle']:.0f}°)")
            return "Sitting", "; ".join(reasons)

        # --- STANDING: default ---
        reasons.append("default (upright, hips above ankles)")
        return "Standing", "; ".join(reasons)

    # ------------------------------------------------------------------
    # STEP 2: HARD GEOMETRY FILTERS
    # ------------------------------------------------------------------

    def _passes_hard_filters(self, pose_key: str, c: dict, ac: dict, o: dict) -> tuple:
        """
        Check boolean geometry constraints for a pose.
        Returns (passed: bool, rejection_reasons: list[str])
        """
        filters = POSE_DATABASE[pose_key].get("hard_filters", {})
        rejections = []

        ls, rs = c.get("LEFT_SHOULDER"), c.get("RIGHT_SHOULDER")
        lh, rh = c.get("LEFT_HIP"), c.get("RIGHT_HIP")
        la, ra = c.get("LEFT_ANKLE"), c.get("RIGHT_ANKLE")
        lk, rk = c.get("LEFT_KNEE"), c.get("RIGHT_KNEE")
        lw, rw = c.get("LEFT_WRIST"), c.get("RIGHT_WRIST")

        for f_name, f_val in filters.items():
            if not f_val:
                continue

            if f_name == "not_inverted":
                if o["hip_shoulder_gap"] < -0.05:
                    rejections.append("hips above shoulders (inverted)")

            elif f_name == "not_seated":
                # For standing poses: hips must be well above ankles
                if o["hip_ankle_gap"] < 0.22:
                    rejections.append(f"hips too close to floor (gap={o['hip_ankle_gap']:.2f})")

            elif f_name == "not_prone":
                # For sitting poses: torso must NOT be horizontal
                if o["torso_slope"] < 35.0:
                    rejections.append(f"torso is horizontal (slope={o['torso_slope']:.0f}°), not seated")

            elif f_name == "body_upright":
                if ac["shoulder_mid"] and ac["hip_mid"]:
                    dx = abs(ac["shoulder_mid"][0] - ac["hip_mid"][0])
                    if dx > 0.25:
                        rejections.append(f"torso too tilted (dx={dx:.2f})")

            elif f_name == "torso_upright":
                if o["torso_slope"] < 45.0:
                    rejections.append(f"torso not upright (slope={o['torso_slope']:.0f}°)")

            elif f_name == "torso_horizontal":
                # Cobra: torso MUST be near-horizontal
                if o["torso_slope"] > 55.0:
                    rejections.append(f"torso too vertical for prone pose (slope={o['torso_slope']:.0f}°)")

            elif f_name == "legs_straight":
                if ac["left_knee"] < 145 or ac["right_knee"] < 145:
                    rejections.append(f"knees bent (L={ac['left_knee']:.0f}°, R={ac['right_knee']:.0f}°)")

            elif f_name == "one_knee_bent":
                lk_a, rk_a = ac["left_knee"], ac["right_knee"]
                if not ((lk_a < 135 and rk_a > 145) or (rk_a < 135 and lk_a > 145)):
                    rejections.append(f"no single bent knee (L={lk_a:.0f}°, R={rk_a:.0f}°)")

            elif f_name == "standing_height":
                if o["hip_ankle_gap"] < 0.20:
                    rejections.append(f"not standing height (hip-ankle gap={o['hip_ankle_gap']:.2f})")

            elif f_name == "wide_stance":
                if la and ra:
                    if abs(la[0] - ra[0]) < 0.20:
                        rejections.append("stance too narrow")

            elif f_name == "one_knee_bent_warrior":
                lk_a, rk_a = ac["left_knee"], ac["right_knee"]
                if not ((lk_a < 140 and rk_a > 145) or (rk_a < 140 and lk_a > 145)):
                    rejections.append(f"no warrior knee bend (L={lk_a:.0f}°, R={rk_a:.0f}°)")

            elif f_name == "seated":
                if o["hip_ankle_gap"] > 0.30:
                    rejections.append(f"not seated (hip-ankle gap={o['hip_ankle_gap']:.2f})")

            elif f_name == "knees_wide":
                if lk and rk and la and ra:
                    knees_x = abs(lk[0] - rk[0])
                    ankles_x = abs(la[0] - ra[0])
                    if knees_x < ankles_x * 0.9:
                        rejections.append("knees not wider than ankles")

            elif f_name == "knees_moderate_spread":
                pass  # Soft check, don't hard-reject

            elif f_name == "prone_body":
                if o["hip_ankle_gap"] > 0.25:
                    rejections.append(f"hips too high for prone (gap={o['hip_ankle_gap']:.2f})")
                if o["shoulder_y"] > o["hip_y"]:
                    rejections.append("shoulders below hips (not prone)")

            elif f_name == "hips_above_shoulders":
                if o["hip_y"] > o["shoulder_y"] - 0.03:
                    rejections.append("hips not above shoulders")

            elif f_name == "hands_on_floor":
                if lw and rw:
                    wrist_y = (lw[1] + rw[1]) / 2.0
                    if wrist_y < o["shoulder_y"] - 0.15:
                        rejections.append("wrists not near floor")

            elif f_name == "v_shape":
                if la and ra:
                    avg_ankle_y = (la[1] + ra[1]) / 2.0
                    if avg_ankle_y > o["hip_y"] + 0.05:
                        rejections.append("legs not elevated for V-shape")

            elif f_name == "leaning_back":
                if ac["shoulder_mid"] and ac["hip_mid"]:
                    torso_vert = self._angle(
                        ac["shoulder_mid"], ac["hip_mid"],
                        (ac["hip_mid"][0] + 0.5, ac["hip_mid"][1], 0.0, 1.0)
                    )
                    if torso_vert > 80.0:
                        rejections.append(f"torso too upright for boat (angle={torso_vert:.0f}°)")

        return (len(rejections) == 0), rejections

    # ------------------------------------------------------------------
    # STEP 3: MIRRORED SIDE DETECTION
    # ------------------------------------------------------------------

    def _detect_sides(self, pose_key: str, ac: dict) -> dict:
        """Determine which side is standing/bent or front/back for asymmetric poses."""
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

        elif pose_key == "warrior_ii":
            if lk < rk:
                sides["front_side"], sides["back_side"] = "left", "right"
            else:
                sides["front_side"], sides["back_side"] = "right", "left"

        return sides

    # ------------------------------------------------------------------
    # STEP 4: UNIFIED JOINT VALUE COMPUTATION
    # ------------------------------------------------------------------

    def _compute_joint_value(self, joint_type: str, c: dict, ac: dict, sides: dict) -> float:
        """Single function that maps any joint_type string → computed float."""
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
            if all([ls, lw, lh]):
                return self._angle(lw, ls, lh)
            return 15.0

        elif joint_type == "standing_knee":
            return ac["left_knee"] if sides["standing_side"] == "left" else ac["right_knee"]

        elif joint_type == "bent_knee":
            return ac["left_knee"] if sides["bent_side"] == "left" else ac["right_knee"]

        elif joint_type == "front_knee":
            return ac["left_knee"] if sides["front_side"] == "left" else ac["right_knee"]

        elif joint_type == "back_knee":
            return ac["left_knee"] if sides["back_side"] == "left" else ac["right_knee"]

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
            return 90.0  # Approximation

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

        return 0.0

    # ------------------------------------------------------------------
    # STEP 5: CONFIDENCE SCORING
    # ------------------------------------------------------------------

    def _score_pose(self, pose_key: str, c: dict, ac: dict, sides: dict) -> tuple:
        """
        Weighted angle scoring with soft thresholds.
        Returns (confidence_0_to_100, debug_details_list).
        """
        rules = POSE_DATABASE[pose_key]["rules"]
        total = 0.0
        details = []

        for rule_name, rule in rules.items():
            target = rule["target"]
            tol = rule["tolerance"]
            weight = rule["weight"]

            val = self._compute_joint_value(rule["joint"], c, ac, sides)
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

    # ------------------------------------------------------------------
    # STEP 6: EVALUATE ALL CANDIDATES IN CATEGORY
    # ------------------------------------------------------------------

    def _evaluate_candidates(self, category: str, c: dict, ac: dict, o: dict) -> tuple:
        """
        Score all poses in the active category that pass hard filters.
        Returns (sorted_candidates, rejection_log).
        """
        candidates = []
        rejection_log = {}

        for pose_key, data in POSE_DATABASE.items():
            if data["category"] != category:
                continue

            passed, rejections = self._passes_hard_filters(pose_key, c, ac, o)
            if not passed:
                rejection_log[data["display_name"]] = rejections
                continue

            sides = self._detect_sides(pose_key, ac)
            conf, details = self._score_pose(pose_key, c, ac, sides)
            candidates.append((pose_key, conf, sides, details))

        sorted_cands = sorted(candidates, key=lambda x: x[1], reverse=True)
        return sorted_cands, rejection_log

    # ------------------------------------------------------------------
    # STEP 7: TEMPORAL SMOOTHING (webcam only)
    # ------------------------------------------------------------------

    def _temporal_smooth(self, best_key: str, confidence: float) -> str:
        """
        15-frame buffer with majority vote and pose locking.
        Lock at >75% confidence, unlock when <40% for 5+ frames.
        """
        self.pose_buffer.append(best_key)

        if confidence < 40.0:
            self.low_conf_frames += 1
        else:
            self.low_conf_frames = 0

        # Lock logic
        if confidence > 75.0 and best_key != "unknown":
            self.locked_pose = best_key
            self.current_stable_pose = best_key

        # Unlock logic
        if self.low_conf_frames >= 5:
            self.locked_pose = "unknown"

        # Majority vote
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

    # ------------------------------------------------------------------
    # MAIN ENTRY POINT
    # ------------------------------------------------------------------

    def analyze(self, detector, lm_list, selected_pose: str = None, is_static: bool = False) -> dict:
        """
        Central classification entrypoint.

        Parameters
        ----------
        detector   : PoseDetector instance
        lm_list    : MediaPipe landmark list for one person
        selected_pose : Optional manual pose selection string
        is_static  : If True (image upload), bypass temporal smoothing entirely
        """
        result = {
            "category": "Auto Detecting",
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
        }

        # 1. Visibility check
        visible, warning = self._check_visibility(detector, lm_list)
        if not visible:
            self.locked_pose = "unknown"
            self.current_stable_pose = "unknown"
            self.pose_buffer.clear()
            result["corrections"].append({"key": "no_pose", "message": warning, "priority": 1})
            result["entry_guidance"] = warning
            return result

        # 2. Stability (rolling, only meaningful for webcam)
        result["stability"] = round(self._compute_stability(lm_list), 1)

        # 3. Extract coordinates and precompute angles
        c = self._extract_coords(detector, lm_list)
        ac = self._build_angle_cache(c)

        # Populate debug angles
        result["angles"] = {
            "Left Knee": round(ac["left_knee"], 1),
            "Right Knee": round(ac["right_knee"], 1),
            "Left Hip": round(ac["left_hip"], 1),
            "Right Hip": round(ac["right_hip"], 1),
            "Left Arm": round(ac["left_arm"], 1),
            "Right Arm": round(ac["right_arm"], 1),
        }

        # 4. Compute body orientation metrics
        o = self._compute_orientation(c, ac)
        result["orientation"] = {
            "torso_slope": round(o["torso_slope"], 1),
            "hip_ankle_gap": round(o["hip_ankle_gap"], 3),
            "hip_shoulder_gap": round(o["hip_shoulder_gap"], 3),
            "avg_knee_angle": round(o["avg_knee_angle"], 1),
        }

        # 5. Classify body category (STRICT GATING)
        category, cat_reason = self.classify_category(c, ac, o)
        result["category"] = category
        result["category_reason"] = cat_reason

        # 6. Evaluate candidates within category (with rejection logging)
        candidates, rejection_log = self._evaluate_candidates(category, c, ac, o)

        # Store all scores for debug panel
        for p_key, conf, _, _ in candidates:
            display = POSE_DATABASE[p_key]["display_name"]
            result["candidate_scores"][display] = round(conf, 1)

        # Store rejection log for debug panel
        result["rejected_poses"] = rejection_log

        # 7. Pick best candidate
        best_key = "unknown"
        best_conf = 0.0
        best_sides = {}

        if candidates:
            best_key, best_conf, best_sides, _ = candidates[0]

        # Apply confidence threshold
        if best_key != "unknown":
            threshold = POSE_DATABASE[best_key].get("threshold", 38.0)
            if best_conf < threshold:
                result["is_low_confidence"] = True
                if best_conf < 25.0:
                    best_key = "unknown"

        # Manual pose override
        if selected_pose and selected_pose != "Auto-Detect":
            for k, data in POSE_DATABASE.items():
                if data["display_name"] == selected_pose:
                    sides = self._detect_sides(k, ac)
                    conf, _ = self._score_pose(k, c, ac, sides)
                    if conf >= data.get("threshold", 38.0):
                        best_key = k
                        best_conf = conf
                        best_sides = sides
                    else:
                        best_key = "unknown"
                    break

        # 8. Temporal smoothing (SKIP for static image uploads)
        if is_static:
            final_key = best_key
        else:
            final_key = self._temporal_smooth(best_key, best_conf)

        # 9. Build result
        if final_key == "unknown":
            result["detected_pose"] = "unknown"
            result["confidence"] = 0.0
            result["accuracy"] = 0.0
            if selected_pose and selected_pose != "Auto-Detect":
                for k, data in POSE_DATABASE.items():
                    if data["display_name"] == selected_pose:
                        result["entry_guidance"] = data["entry_guidance"]
                        break
            else:
                result["entry_guidance"] = f"Pose Uncertain — checking {category} candidates..."
            result["corrections"].append({"key": "no_pose", "message": result["entry_guidance"], "priority": 1})
            return result

        pose_data = POSE_DATABASE[final_key]
        result["detected_pose"] = final_key
        result["confidence"] = round(best_conf, 1)
        result["accuracy"] = round(best_conf, 1)
        result["entry_guidance"] = pose_data["entry_guidance"]

        # 10. Generate corrections
        errors = []
        for rule_name, rule in pose_data["rules"].items():
            val = self._compute_joint_value(rule["joint"], c, ac, best_sides)
            if abs(val - rule["target"]) > rule["tolerance"]:
                errors.append({
                    "key": f"{final_key}_{rule_name}",
                    "message": rule["error_msg"],
                    "priority": rule["priority"]
                })

        if errors:
            errors.sort(key=lambda x: x["priority"])
            result["corrections"] = [errors[0]]  # Single highest-priority correction
        else:
            result["corrections"] = [{
                "key": f"{final_key}_perfect",
                "message": "Excellent posture! Maintain this position.",
                "priority": 9
            }]

        return result
