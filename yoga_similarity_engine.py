"""
Pose Similarity Engine for Static Image Analysis.

This module provides a robust continuous-similarity engine for comparing
extracted landmarks against dataset-derived "Pose Signatures".

Unlike the live camera which uses rigid bounding rules, this engine calculates
Euclidean and RBF (Radial Basis Function) distances for:
- Joint angles
- Limb proportions
- Torso orientation
- Body geometry (aspect ratios)
"""

import numpy as np

# ---------------------------------------------------------------------------
# DATASET SIGNATURES FOR THE 11 SUPPORTED POSES
# ---------------------------------------------------------------------------
# These signatures act as the "average dataset statistics" for similarity matching.
# They define the ideal geometric and angular state of a perfect pose.

POSE_SIGNATURES = {
    # ================= STANDING =================
    "mountain": {
        "category": "Standing",
        "angles": {
            "left_knee": 180, "right_knee": 180,
            "left_hip": 180, "right_hip": 180,
            "left_arm": 180, "right_arm": 180,
            "left_shoulder_angle": 15, "right_shoulder_angle": 15
        },
        "torso_slope": 90,
        "aspect_ratio_min": 2.5, # Very tall
        "limb_ratios": {
            "knee_to_shoulder_spread": 0.5  # Knees together, shoulders wider
        }
    },
    "tree": {
        "category": "Standing",
        "angles": {
            # One leg straight (180), one bent (~45 for hip, ~45 for knee)
            # We use a symmetric matching approach: we match the best leg to straight and the other to bent
            "straight_knee": 180,
            "bent_knee": 45,
            "straight_hip": 180,
            "bent_hip": 135
        },
        "torso_slope": 90,
        "aspect_ratio_min": 1.8,
        "limb_ratios": {}
    },
    "warrior_i": {
        "category": "Standing",
        "angles": {
            "front_knee": 90,
            "back_knee": 180,
            "left_arm": 180, "right_arm": 180,
            "left_shoulder_angle": 160, "right_shoulder_angle": 160 # Arms up
        },
        "torso_slope": 90,
        "aspect_ratio_min": 1.0,
        "limb_ratios": {}
    },
    "warrior_ii": {
        "category": "Standing",
        "angles": {
            "front_knee": 90,
            "back_knee": 180,
            "left_arm": 180, "right_arm": 180,
            "left_shoulder_angle": 90, "right_shoulder_angle": 90 # Arms out
        },
        "torso_slope": 90,
        "aspect_ratio_min": 0.8,
        "limb_ratios": {}
    },
    "triangle": {
        "category": "Standing",
        "angles": {
            "left_knee": 180, "right_knee": 180,
            "front_hip": 90,
            "back_hip": 135,
            "left_arm": 180, "right_arm": 180,
            "arms_spread": 180 # Arms form a straight line
        },
        "torso_slope": 45, # Torso is leaning
        "aspect_ratio_min": 0.8,
        "limb_ratios": {}
    },

    # ================= SITTING =================
    "sukhasana": {
        "category": "Sitting",
        "angles": {
            "left_knee": 45, "right_knee": 45,
            "left_hip": 90, "right_hip": 90
        },
        "torso_slope": 90,
        "aspect_ratio_min": 0.8,
        "limb_ratios": {
            "knee_to_shoulder_spread": 1.5 # Knees wider than shoulders
        }
    },
    "butterfly": {
        "category": "Sitting",
        "angles": {
            "left_knee": 30, "right_knee": 30,  # Tighter bend than sukhasana
            "left_hip": 130, "right_hip": 130,
            "left_arm": 160, "right_arm": 160
        },
        "torso_slope": 90,
        "aspect_ratio_min": 0.8,
        "limb_ratios": {
            "knee_to_shoulder_spread": 1.8 # Knees very wide
        }
    },
    "lotus": {
        "category": "Sitting",
        "angles": {
            "left_knee": 20, "right_knee": 20, # Extreme bend
            "left_hip": 110, "right_hip": 110
        },
        "torso_slope": 90,
        "aspect_ratio_min": 0.8,
        "limb_ratios": {}
    },

    # ================= PRONE =================
    "cobra": {
        "category": "Prone",
        "angles": {
            "left_knee": 180, "right_knee": 180,
            "left_hip": 160, "right_hip": 160,
            "left_arm": 160, "right_arm": 160
        },
        "torso_slope": 30, # Torso raised slightly
        "aspect_ratio_min": 0.3, # Very wide, low height
        "limb_ratios": {}
    },

    # ================= INVERTED =================
    "downward_dog": {
        "category": "Inverted",
        "angles": {
            "left_knee": 180, "right_knee": 180,
            "left_hip": 70, "right_hip": 70,
            "left_arm": 180, "right_arm": 180,
            "left_shoulder_angle": 160, "right_shoulder_angle": 160
        },
        "torso_slope": 45, # Hips high
        "aspect_ratio_min": 0.5,
        "limb_ratios": {}
    },

    # ================= BALANCE =================
    "boat": {
        "category": "Sitting", # Classified under sitting initially but acts as balance
        "angles": {
            "left_knee": 180, "right_knee": 180,
            "left_hip": 60, "right_hip": 60, # V-shape
            "left_arm": 180, "right_arm": 180,
            "left_shoulder_angle": 90, "right_shoulder_angle": 90
        },
        "torso_slope": 60, # Leaning back
        "aspect_ratio_min": 0.8,
        "limb_ratios": {}
    },

    # ================= BACKBEND =================
    "bridge": {
        "category": "Backbend",
        "angles": {
            "left_knee": 90, "right_knee": 90,
            "left_hip": 150, "right_hip": 150,
            "left_arm": 180, "right_arm": 180,
            "left_shoulder_angle": 10, "right_shoulder_angle": 10
        },
        "torso_slope": 35,
        "aspect_ratio_min": 0.5,
        "limb_ratios": {}
    },
    "camel": {
        "category": "Backbend",
        "angles": {
            "left_knee": 90, "right_knee": 90,
            "left_hip": 160, "right_hip": 160,
            "left_arm": 160, "right_arm": 160,
            "left_shoulder_angle": 70, "right_shoulder_angle": 70
        },
        "torso_slope": 40,
        "aspect_ratio_min": 0.8,
        "limb_ratios": {}
    }
}


class PoseSimilarityEngine:
    """Calculates continuous similarity scores between extracted features and dataset signatures."""
    
    @staticmethod
    def gaussian_similarity(val, target, sigma):
        """Returns 1.0 when val == target, approaches 0.0 as distance increases."""
        return float(np.exp(-0.5 * ((val - target) / sigma) ** 2))

    def evaluate_image(self, category: str, c: dict, ac: dict, o: dict) -> dict:
        """
        Evaluate the image against all signatures in the matching category.
        
        Returns:
            scores: dict mapping pose_name to confidence_percentage
            details: dict mapping pose_name to match/fail features
        """
        candidate_scores = {}
        candidate_details = {}

        # 1. Extract dynamic features safely with explicit fallback defaults to avoid KeyError or TypeError
        lk = ac.get("left_knee") if ac.get("left_knee") is not None else 180.0
        rk = ac.get("right_knee") if ac.get("right_knee") is not None else 180.0
        lh = ac.get("left_hip") if ac.get("left_hip") is not None else 180.0
        rh = ac.get("right_hip") if ac.get("right_hip") is not None else 180.0
        ls = ac.get("left_shoulder_angle") if ac.get("left_shoulder_angle") is not None else 30.0
        rs = ac.get("right_shoulder_angle") if ac.get("right_shoulder_angle") is not None else 30.0

        img_features = {
            "max_knee": max(lk, rk),
            "min_knee": min(lk, rk),
            "max_hip": max(lh, rh),
            "min_hip": min(lh, rh),
            "arms_spread": ls + rs,
            "knee_to_shoulder_spread": (o.get("knee_x_spread", 0.1) or 0.1) / ((o.get("shoulder_x_spread", 0.15) or 0.15) + 1e-6)
        }

        # 2. Iterate through all supported poses
        for pose_name, signature in POSE_SIGNATURES.items():
            # Strict Category Gating: Only evaluate poses in the detected category
            # (Exception: Boat can be Sitting or Balance)
            if signature["category"] != category and not (pose_name == "boat" and category in ["Sitting", "Balance"]):
                continue

            score, details = self._calculate_similarity(pose_name, signature, ac, o, img_features)
            candidate_scores[pose_name] = score
            candidate_details[pose_name] = details

        return candidate_scores, candidate_details

    def _calculate_similarity(self, pose_name: str, signature: dict, ac: dict, o: dict, img: dict) -> tuple:
        """Calculate multi-modal similarity for a specific pose with vigorous tuning."""
        total_weight = 0.0
        weighted_sum = 0.0
        details = {"matched": [], "failed": []}

        # 1. Angle Similarity (Weight: 60%)
        angle_weight = 60.0 / max(len(signature["angles"]), 1)
        for angle_name, target_val in signature["angles"].items():
            # Dynamic sigma (tolerance) based on the joint type for maximum robustness
            # Hips and shoulders vary more widely than knees/elbows across different body types
            if "hip" in angle_name or "shoulder" in angle_name:
                sigma = 35.0
            else:
                sigma = 25.0
                
            val = None
            if angle_name in ac: val = ac[angle_name]
            elif angle_name == "straight_knee": val = img["max_knee"]
            elif angle_name == "bent_knee": val = img["min_knee"]
            elif angle_name == "straight_hip": val = img["max_hip"]
            elif angle_name == "bent_hip": val = img["min_hip"]
            elif angle_name == "front_knee": val = img["min_knee"]
            elif angle_name == "back_knee": val = img["max_knee"]
            elif angle_name == "front_hip": val = img["min_hip"]
            elif angle_name == "back_hip": val = img["max_hip"]
            elif angle_name == "arms_spread": val = img["arms_spread"]

            if val is not None:
                sim = self.gaussian_similarity(val, target_val, sigma=sigma)
                weighted_sum += sim * angle_weight
                total_weight += angle_weight
                
                label = f"{angle_name}: {val:.1f}° (target: {target_val}°)"
                if sim > 0.6: details["matched"].append(label)
                else: details["failed"].append(label)

        # 2. Torso Orientation Similarity (Weight: 20%)
        # Slope can vary heavily depending on camera angle, so we use a loose sigma of 30.0
        target_slope = signature["torso_slope"]
        slope_sim = self.gaussian_similarity(o.get("torso_slope", 90.0), target_slope, sigma=30.0)
        weighted_sum += slope_sim * 20.0
        total_weight += 20.0
        label = f"torso_slope: {o.get('torso_slope', 90.0):.1f}° (target: {target_slope}°)"
        if slope_sim > 0.6: details["matched"].append(label)
        else: details["failed"].append(label)

        # 3. Body Geometry & Aspect Ratio (Weight: 10%)
        # Aspect ratios vary significantly between thin/wide bodies
        target_ar = signature["aspect_ratio_min"]
        ar_sim = self.gaussian_similarity(o.get("aspect_ratio", 1.5), target_ar, sigma=1.5)
        weighted_sum += ar_sim * 10.0
        total_weight += 10.0
        label = f"aspect_ratio: {o.get('aspect_ratio', 1.5):.2f} (target: {target_ar:.1f})"
        if ar_sim > 0.6: details["matched"].append(label)
        else: details["failed"].append(label)

        # 4. Limb Ratios (Weight: 10%)
        for ratio_name, target_ratio in signature["limb_ratios"].items():
            if ratio_name == "knee_to_shoulder_spread":
                val = img.get("knee_to_shoulder_spread", 1.0)
                sim = self.gaussian_similarity(val, target_ratio, sigma=0.8)
                weighted_sum += sim * 10.0
                total_weight += 10.0
                label = f"knee_spread_ratio: {val:.1f} (target: {target_ratio:.1f})"
                if sim > 0.6: details["matched"].append(label)
                else: details["failed"].append(label)

        # Calculate final percentage
        if total_weight == 0:
            return 0.0, details
            
        final_score = (weighted_sum / total_weight) * 100.0
        return round(final_score, 1), details
