"""
Industrial Yoga Pose Configuration Database — v2.0
===================================================

14 yoga poses with hierarchical body-orientation gating.

Each pose defines:
  - category: strict body orientation bucket
  - required_body_state: biomechanical prerequisites checked BEFORE scoring
  - rejection_rules: explicit hard-rejection conditions (overrides scoring)
  - boost_rules: conditions that add bonus confidence points
  - rules: weighted angle-based scoring factors
  - confidence_threshold: minimum confidence to accept (65% for all)
"""

CAMERA_INDEX = 0

# Cooldown parameters for pyttsx3 offline audio guides (seconds)
VOICE_COOLDOWN_DEFAULT = 6.0

# General wellness categories
WELLNESS_GOALS = ["stress", "back pain", "flexibility", "weight loss"]

# All supported pose categories for hierarchical classification
POSE_CATEGORIES = ["Standing", "Sitting", "Prone", "Inverted", "Balance", "Backbend"]

# Confidence below this → "Pose Uncertain"
GLOBAL_UNCERTAINTY_THRESHOLD = 65.0

# ---------------------------------------------------------------------------
# 14-Pose Industrial Database with Multi-Stage Gating
# ---------------------------------------------------------------------------
POSE_DATABASE = {

    # ===================================================================
    # STANDING POSES
    # ===================================================================

    "mountain": {
        "category": "Standing",
        "display_name": "Mountain Pose (Tadasana)",
        "mirrored": False,
        "confidence_threshold": 65.0,
        "entry_guidance": "Stand tall with feet together, relax your shoulders, and let arms hang active at your sides.",

        "required_body_state": {
            "torso_upright": True,       # torso slope > 55°
            "hips_above_ankles": True,   # hip-ankle gap > 0.20
            "not_inverted": True,        # hips below shoulders
        },
        "rejection_rules": [
            {"condition": "legs_crossed", "reason": "Legs are crossed — not Mountain Pose"},
            {"condition": "torso_horizontal", "reason": "Torso is horizontal — not standing"},
            {"condition": "hips_on_floor", "reason": "Hips on floor — not standing"},
        ],
        "boost_rules": [
            {"condition": "legs_straight", "bonus": 8, "reason": "Legs fully straight"},
            {"condition": "arms_at_sides", "bonus": 5, "reason": "Arms at sides"},
        ],
        "rules": {
            "spine_alignment": {
                "joint": "torso_vertical",
                "target": 88.0,
                "tolerance": 15.0,
                "weight": 0.25,
                "error_msg": "Straighten your spine and stand taller.",
                "priority": 1
            },
            "knee_straightness": {
                "joint": "avg_knee",
                "target": 175.0,
                "tolerance": 18.0,
                "weight": 0.25,
                "error_msg": "Keep your legs straight without bending your knees.",
                "priority": 2
            },
            "arm_position": {
                "joint": "avg_arm_body_angle",
                "target": 12.0,
                "tolerance": 18.0,
                "weight": 0.20,
                "error_msg": "Keep your arms straight and active down beside your body.",
                "priority": 3
            },
            "hip_alignment": {
                "joint": "hip_symmetry",
                "target": 0.0,
                "tolerance": 0.06,
                "weight": 0.15,
                "error_msg": "Level your hips — distribute weight evenly.",
                "priority": 4
            },
            "feet_together": {
                "joint": "ankle_x_spread",
                "target": 0.04,
                "tolerance": 0.10,
                "weight": 0.15,
                "error_msg": "Bring your feet closer together.",
                "priority": 5
            },
        }
    },

    "tree": {
        "category": "Standing",
        "display_name": "Tree Pose (Vrksasana)",
        "mirrored": True,
        "confidence_threshold": 65.0,
        "entry_guidance": "Find balance on one leg, lift the opposite foot against your inner thigh, and raise hands overhead.",

        "required_body_state": {
            "torso_upright": True,
            "hips_above_ankles": True,
            "not_inverted": True,
            "one_knee_bent": True,
        },
        "rejection_rules": [
            {"condition": "both_knees_bent", "reason": "Both knees bent — not Tree Pose"},
            {"condition": "torso_horizontal", "reason": "Torso horizontal — not standing"},
            {"condition": "hips_on_floor", "reason": "Hips on floor — not standing"},
        ],
        "boost_rules": [
            {"condition": "arms_overhead", "bonus": 10, "reason": "Arms raised overhead"},
            {"condition": "standing_leg_straight", "bonus": 8, "reason": "Standing leg fully straight"},
        ],
        "rules": {
            "standing_leg": {
                "joint": "standing_knee",
                "target": 175.0,
                "tolerance": 22.0,
                "weight": 0.30,
                "error_msg": "Straighten your standing leg to stabilize.",
                "priority": 1
            },
            "bent_knee_outward": {
                "joint": "bent_knee",
                "target": 50.0,
                "tolerance": 40.0,
                "weight": 0.25,
                "error_msg": "Open your bent knee outward away from center.",
                "priority": 2
            },
            "arms_overhead": {
                "joint": "avg_arm_elevation",
                "target": 160.0,
                "tolerance": 30.0,
                "weight": 0.25,
                "error_msg": "Raise your arms fully overhead and join your palms.",
                "priority": 3
            },
            "spine_alignment": {
                "joint": "torso_vertical",
                "target": 88.0,
                "tolerance": 15.0,
                "weight": 0.20,
                "error_msg": "Keep your torso upright and centered.",
                "priority": 4
            },
        }
    },

    "warrior_i": {
        "category": "Standing",
        "display_name": "Warrior I (Virabhadrasana I)",
        "mirrored": True,
        "confidence_threshold": 65.0,
        "entry_guidance": "Step one foot back into a lunge, bend the front knee to 90°, square your hips forward, and raise both arms overhead.",

        "required_body_state": {
            "torso_upright": True,
            "hips_above_ankles": True,
            "not_inverted": True,
            "wide_stance": True,
            "one_knee_bent_warrior": True,
        },
        "rejection_rules": [
            {"condition": "hips_on_floor", "reason": "Hips on floor — not standing"},
            {"condition": "torso_horizontal", "reason": "Torso horizontal — not Warrior I"},
            {"condition": "both_knees_straight", "reason": "Both knees straight — need front knee bend"},
        ],
        "boost_rules": [
            {"condition": "arms_overhead", "bonus": 10, "reason": "Arms raised overhead"},
        ],
        "rules": {
            "front_knee_bend": {
                "joint": "front_knee",
                "target": 100.0,
                "tolerance": 25.0,
                "weight": 0.30,
                "error_msg": "Bend your front knee closer to 90 degrees.",
                "priority": 1
            },
            "back_leg_straight": {
                "joint": "back_knee",
                "target": 170.0,
                "tolerance": 22.0,
                "weight": 0.25,
                "error_msg": "Straighten your back leg fully.",
                "priority": 2
            },
            "arms_up": {
                "joint": "avg_arm_elevation",
                "target": 160.0,
                "tolerance": 30.0,
                "weight": 0.25,
                "error_msg": "Raise both arms overhead.",
                "priority": 3
            },
            "spine_alignment": {
                "joint": "torso_vertical",
                "target": 85.0,
                "tolerance": 18.0,
                "weight": 0.20,
                "error_msg": "Keep your torso upright, don't lean forward.",
                "priority": 4
            },
        }
    },

    "warrior_ii": {
        "category": "Standing",
        "display_name": "Warrior II (Virabhadrasana II)",
        "mirrored": True,
        "confidence_threshold": 65.0,
        "entry_guidance": "Adopt a wide stance, bend your front knee to 90 degrees, and stretch both arms out horizontally.",

        "required_body_state": {
            "torso_upright": True,
            "hips_above_ankles": True,
            "not_inverted": True,
            "wide_stance": True,
            "one_knee_bent_warrior": True,
        },
        "rejection_rules": [
            {"condition": "hips_on_floor", "reason": "Hips on floor — not standing"},
            {"condition": "torso_horizontal", "reason": "Torso horizontal — not Warrior"},
            {"condition": "both_knees_straight", "reason": "Both knees straight — need front knee bend"},
        ],
        "boost_rules": [
            {"condition": "arms_horizontal", "bonus": 10, "reason": "Arms stretched horizontally"},
        ],
        "rules": {
            "front_knee_bend": {
                "joint": "front_knee",
                "target": 100.0,
                "tolerance": 25.0,
                "weight": 0.30,
                "error_msg": "Lower your hips and bend your front knee closer to 90 degrees.",
                "priority": 1
            },
            "back_leg_straight": {
                "joint": "back_knee",
                "target": 175.0,
                "tolerance": 22.0,
                "weight": 0.25,
                "error_msg": "Engage your thigh and straighten your back leg.",
                "priority": 2
            },
            "arms_horizontal": {
                "joint": "arms_horizontal_score",
                "target": 95.0,
                "tolerance": 25.0,
                "weight": 0.25,
                "error_msg": "Raise your arms horizontally at shoulder level.",
                "priority": 3
            },
            "spine_alignment": {
                "joint": "torso_vertical",
                "target": 88.0,
                "tolerance": 15.0,
                "weight": 0.20,
                "error_msg": "Keep your torso upright and centered over your hips.",
                "priority": 4
            },
        }
    },

    "triangle": {
        "category": "Standing",
        "display_name": "Triangle Pose (Trikonasana)",
        "mirrored": True,
        "confidence_threshold": 65.0,
        "entry_guidance": "Adopt a wide stance, keep both legs straight, extend torso sideways over the front leg, and stretch arms vertically.",

        "required_body_state": {
            "hips_above_ankles": True,
            "not_inverted": True,
            "wide_stance": True,
        },
        "rejection_rules": [
            {"condition": "hips_on_floor", "reason": "Hips on floor — not standing"},
            {"condition": "both_knees_deeply_bent", "reason": "Both knees deeply bent — not Triangle"},
        ],
        "boost_rules": [
            {"condition": "torso_lateral_tilt", "bonus": 10, "reason": "Torso tilted sideways"},
            {"condition": "both_legs_straight", "bonus": 8, "reason": "Both legs straight"},
        ],
        "rules": {
            "leg_straightness": {
                "joint": "avg_knee",
                "target": 170.0,
                "tolerance": 20.0,
                "weight": 0.25,
                "error_msg": "Keep both legs straight.",
                "priority": 1
            },
            "torso_lateral": {
                "joint": "torso_lateral_angle",
                "target": 45.0,
                "tolerance": 25.0,
                "weight": 0.30,
                "error_msg": "Tilt your torso more sideways over the front leg.",
                "priority": 2
            },
            "arm_extension": {
                "joint": "arms_vertical_spread",
                "target": 160.0,
                "tolerance": 30.0,
                "weight": 0.25,
                "error_msg": "Extend your arms into a vertical line — one up, one down.",
                "priority": 3
            },
            "wide_stance_check": {
                "joint": "ankle_x_spread",
                "target": 0.35,
                "tolerance": 0.15,
                "weight": 0.20,
                "error_msg": "Widen your stance more.",
                "priority": 4
            },
        }
    },

    # ===================================================================
    # SITTING POSES
    # ===================================================================

    "sukhasana": {
        "category": "Sitting",
        "display_name": "Sukhasana (Easy Pose)",
        "mirrored": False,
        "confidence_threshold": 65.0,
        "entry_guidance": "Sit cross-legged on the floor with a tall spine, hands resting on your knees, shoulders relaxed.",

        "required_body_state": {
            "torso_upright": True,
            "hips_on_floor": True,
            "not_prone": True,
        },
        "rejection_rules": [
            {"condition": "torso_horizontal", "reason": "Torso horizontal — not seated upright"},
            {"condition": "legs_straight", "reason": "Legs straight — not cross-legged"},
            {"condition": "hips_elevated", "reason": "Hips elevated — not on floor"},
        ],
        "boost_rules": [
            {"condition": "legs_crossed", "bonus": 15, "reason": "Legs crossed"},
            {"condition": "hands_on_knees", "bonus": 8, "reason": "Hands on knees"},
        ],
        "rules": {
            "spine_upright": {
                "joint": "torso_vertical",
                "target": 88.0,
                "tolerance": 22.0,
                "weight": 0.30,
                "error_msg": "Sit tall and lengthen your spine upward.",
                "priority": 1
            },
            "knee_fold": {
                "joint": "avg_knee",
                "target": 75.0,
                "tolerance": 40.0,
                "weight": 0.25,
                "error_msg": "Cross your legs comfortably in front of you.",
                "priority": 2
            },
            "hands_on_knees": {
                "joint": "hands_knee_proximity",
                "target": 0.06,
                "tolerance": 0.18,
                "weight": 0.20,
                "error_msg": "Rest your hands gently on your knees.",
                "priority": 3
            },
            "hip_grounding": {
                "joint": "hip_floor_ratio",
                "target": 0.05,
                "tolerance": 0.15,
                "weight": 0.25,
                "error_msg": "Lower your hips closer to the ground.",
                "priority": 4
            },
        }
    },

    "butterfly": {
        "category": "Sitting",
        "display_name": "Butterfly Pose (Baddha Konasana)",
        "mirrored": False,
        "confidence_threshold": 65.0,
        "entry_guidance": "Sit upright, bend knees pulling heels close, bring soles of feet together, and let knees drop outward.",

        "required_body_state": {
            "torso_upright": True,
            "hips_on_floor": True,
            "not_prone": True,
            "knees_wide": True,
        },
        "rejection_rules": [
            {"condition": "torso_horizontal", "reason": "Torso horizontal — not seated"},
            {"condition": "legs_straight", "reason": "Legs straight — not Butterfly"},
            {"condition": "hips_elevated", "reason": "Hips elevated — not on floor"},
        ],
        "boost_rules": [
            {"condition": "knees_wide", "bonus": 10, "reason": "Knees wide apart"},
            {"condition": "feet_together", "bonus": 8, "reason": "Feet close together"},
        ],
        "rules": {
            "spine_upright": {
                "joint": "torso_vertical",
                "target": 88.0,
                "tolerance": 20.0,
                "weight": 0.30,
                "error_msg": "Sit tall and straighten your spine, do not round your back.",
                "priority": 1
            },
            "knees_low": {
                "joint": "knees_drop_ratio",
                "target": 40.0,
                "tolerance": 30.0,
                "weight": 0.25,
                "error_msg": "Relax your thighs and let your knees drop lower.",
                "priority": 2
            },
            "hands_feet": {
                "joint": "hands_feet_dist",
                "target": 0.06,
                "tolerance": 0.15,
                "weight": 0.20,
                "error_msg": "Hold your feet securely with both hands.",
                "priority": 3
            },
            "hip_grounding": {
                "joint": "hip_floor_ratio",
                "target": 0.05,
                "tolerance": 0.15,
                "weight": 0.25,
                "error_msg": "Ground your sitting bones into the mat.",
                "priority": 4
            },
        }
    },

    "lotus": {
        "category": "Sitting",
        "display_name": "Lotus Pose (Padmasana)",
        "mirrored": False,
        "confidence_threshold": 65.0,
        "entry_guidance": "Sit upright with legs crossed, each foot resting on the opposite thigh. Keep spine tall and shoulders relaxed.",

        "required_body_state": {
            "torso_upright": True,
            "hips_on_floor": True,
            "not_prone": True,
        },
        "rejection_rules": [
            {"condition": "torso_horizontal", "reason": "Torso horizontal — not seated"},
            {"condition": "legs_straight", "reason": "Legs straight — not Lotus"},
            {"condition": "hips_elevated", "reason": "Hips elevated — not on floor"},
        ],
        "boost_rules": [
            {"condition": "legs_crossed", "bonus": 15, "reason": "Legs tightly crossed"},
            {"condition": "hands_on_knees", "bonus": 5, "reason": "Hands resting on knees"},
        ],
        "rules": {
            "spine_upright": {
                "joint": "torso_vertical",
                "target": 88.0,
                "tolerance": 18.0,
                "weight": 0.30,
                "error_msg": "Sit tall, lengthen your spine upward.",
                "priority": 1
            },
            "knee_angle_low": {
                "joint": "avg_knee",
                "target": 65.0,
                "tolerance": 35.0,
                "weight": 0.30,
                "error_msg": "Cross your legs more tightly, draw heels closer.",
                "priority": 2
            },
            "hands_on_knees": {
                "joint": "hands_knee_proximity",
                "target": 0.06,
                "tolerance": 0.15,
                "weight": 0.20,
                "error_msg": "Rest your hands gently on your knees.",
                "priority": 3
            },
            "hip_grounding": {
                "joint": "hip_floor_ratio",
                "target": 0.05,
                "tolerance": 0.15,
                "weight": 0.20,
                "error_msg": "Sit firmly on the mat.",
                "priority": 4
            },
        }
    },

    "boat": {
        "category": "Sitting",
        "display_name": "Boat Pose (Navasana)",
        "mirrored": False,
        "confidence_threshold": 65.0,
        "entry_guidance": "Sit and lean back, lift your legs off the floor forming a V-shape with your body. Extend arms forward parallel to the ground.",

        "required_body_state": {
            "not_inverted": True,
            "legs_elevated": True,
            "torso_leaning_back": True,
        },
        "rejection_rules": [
            {"condition": "torso_horizontal", "reason": "Lying flat — not Boat Pose"},
            {"condition": "legs_on_floor", "reason": "Legs on floor — need legs elevated"},
        ],
        "boost_rules": [
            {"condition": "v_shape", "bonus": 12, "reason": "V-shape body position"},
            {"condition": "arms_forward", "bonus": 8, "reason": "Arms extended forward"},
        ],
        "rules": {
            "torso_lean": {
                "joint": "torso_vertical",
                "target": 55.0,
                "tolerance": 25.0,
                "weight": 0.30,
                "error_msg": "Lean your torso back more to form the V-shape.",
                "priority": 1
            },
            "leg_elevation": {
                "joint": "avg_knee",
                "target": 150.0,
                "tolerance": 35.0,
                "weight": 0.30,
                "error_msg": "Lift and straighten your legs higher off the floor.",
                "priority": 2
            },
            "arms_forward": {
                "joint": "arms_horizontal_score",
                "target": 90.0,
                "tolerance": 30.0,
                "weight": 0.20,
                "error_msg": "Extend your arms straight forward, parallel to the floor.",
                "priority": 3
            },
            "hip_balance": {
                "joint": "hip_flexion",
                "target": 70.0,
                "tolerance": 25.0,
                "weight": 0.20,
                "error_msg": "Find the balance point on your sit bones.",
                "priority": 4
            },
        }
    },

    # ===================================================================
    # PRONE POSES
    # ===================================================================

    "cobra": {
        "category": "Prone",
        "display_name": "Cobra Pose (Bhujangasana)",
        "mirrored": False,
        "confidence_threshold": 65.0,
        "entry_guidance": "Lie prone on your stomach, place hands near chest, and gently lift your torso arching the back.",

        "required_body_state": {
            "torso_horizontal": True,    # torso slope < 40°
            "hips_on_floor": True,       # hips near lowest point
            "not_inverted": True,
        },
        "rejection_rules": [
            {"condition": "torso_vertical", "reason": "Torso is vertical — REJECT Cobra (must be prone)"},
            {"condition": "torso_upright", "reason": "Torso is upright — REJECT Cobra (must lie on floor)"},
            {"condition": "legs_crossed", "reason": "Legs crossed — not prone on floor"},
            {"condition": "hips_elevated", "reason": "Hips elevated — not lying on floor"},
        ],
        "boost_rules": [
            {"condition": "chest_lifted", "bonus": 10, "reason": "Chest lifted off floor"},
            {"condition": "legs_straight_on_floor", "bonus": 5, "reason": "Legs straight on floor"},
        ],
        "rules": {
            "chest_lift": {
                "joint": "hip_flexion",
                "target": 140.0,
                "tolerance": 30.0,
                "weight": 0.30,
                "error_msg": "Lift your chest higher off the mat.",
                "priority": 1
            },
            "elbows_bent": {
                "joint": "avg_arm",
                "target": 130.0,
                "tolerance": 35.0,
                "weight": 0.25,
                "error_msg": "Keep your elbows slightly bent close to your ribs.",
                "priority": 2
            },
            "leg_extension": {
                "joint": "avg_knee",
                "target": 170.0,
                "tolerance": 20.0,
                "weight": 0.25,
                "error_msg": "Keep your legs straight and pressed into the mat.",
                "priority": 3
            },
            "torso_angle": {
                "joint": "torso_vertical",
                "target": 30.0,
                "tolerance": 25.0,
                "weight": 0.20,
                "error_msg": "Maintain a gentle backbend — don't sit up.",
                "priority": 4
            },
        }
    },

    # ===================================================================
    # INVERTED POSES
    # ===================================================================

    "downward_dog": {
        "category": "Inverted",
        "display_name": "Downward Dog (Adho Mukha Svanasana)",
        "mirrored": False,
        "confidence_threshold": 65.0,
        "entry_guidance": "Press hands into the mat, lift hips high toward the ceiling, forming an inverted V-shape with your body.",

        "required_body_state": {
            "hips_highest_point": True,
            "hands_on_floor": True,
        },
        "rejection_rules": [
            {"condition": "torso_vertical", "reason": "Torso vertical — not inverted V"},
            {"condition": "hips_on_floor", "reason": "Hips on floor — need hips elevated"},
        ],
        "boost_rules": [
            {"condition": "hands_touching_floor_and_hips_elevated", "bonus": 15, "reason": "Hands on floor + hips high"},
            {"condition": "both_legs_straight", "bonus": 8, "reason": "Legs straight"},
        ],
        "rules": {
            "hip_pike_angle": {
                "joint": "hip_flexion",
                "target": 65.0,
                "tolerance": 30.0,
                "weight": 0.30,
                "error_msg": "Push your hips higher toward the ceiling.",
                "priority": 1
            },
            "leg_straightness": {
                "joint": "avg_knee",
                "target": 170.0,
                "tolerance": 25.0,
                "weight": 0.25,
                "error_msg": "Try to straighten your legs more, pressing heels down.",
                "priority": 2
            },
            "arm_straightness": {
                "joint": "avg_arm",
                "target": 170.0,
                "tolerance": 25.0,
                "weight": 0.25,
                "error_msg": "Extend your arms fully, pressing palms into the mat.",
                "priority": 3
            },
            "v_shape_angle": {
                "joint": "body_v_angle",
                "target": 75.0,
                "tolerance": 25.0,
                "weight": 0.20,
                "error_msg": "Form a sharper inverted V with your body.",
                "priority": 4
            },
        }
    },

    # ===================================================================
    # BACKBEND POSES
    # ===================================================================

    "bridge": {
        "category": "Backbend",
        "display_name": "Bridge Pose (Setu Bandhasana)",
        "mirrored": False,
        "confidence_threshold": 65.0,
        "entry_guidance": "Lie on your back, bend knees, place feet flat on floor hip-width apart, and lift hips toward the ceiling.",

        "required_body_state": {
            "hips_elevated_above_shoulders": True,
            "shoulders_on_floor": True,
        },
        "rejection_rules": [
            {"condition": "torso_vertical", "reason": "Torso vertical — not Bridge"},
            {"condition": "legs_straight", "reason": "Legs straight — knees must be bent"},
        ],
        "boost_rules": [
            {"condition": "hips_elevated", "bonus": 12, "reason": "Hips lifted high"},
            {"condition": "knees_bent_90", "bonus": 8, "reason": "Knees bent at 90°"},
        ],
        "rules": {
            "hip_elevation": {
                "joint": "hip_elevation_ratio",
                "target": 0.80,
                "tolerance": 0.25,
                "weight": 0.30,
                "error_msg": "Lift your hips higher toward the ceiling.",
                "priority": 1
            },
            "knee_bend": {
                "joint": "avg_knee",
                "target": 90.0,
                "tolerance": 25.0,
                "weight": 0.25,
                "error_msg": "Bend your knees to approximately 90 degrees.",
                "priority": 2
            },
            "shoulder_grounding": {
                "joint": "shoulder_floor_proximity",
                "target": 0.05,
                "tolerance": 0.15,
                "weight": 0.25,
                "error_msg": "Press your shoulders firmly into the mat.",
                "priority": 3
            },
            "torso_arc": {
                "joint": "torso_vertical",
                "target": 35.0,
                "tolerance": 25.0,
                "weight": 0.20,
                "error_msg": "Maintain a smooth arc through your torso.",
                "priority": 4
            },
        }
    },

    "camel": {
        "category": "Backbend",
        "display_name": "Camel Pose (Ustrasana)",
        "mirrored": False,
        "confidence_threshold": 65.0,
        "entry_guidance": "Kneel with knees hip-width apart, place hands on heels, and push hips forward while arching your back.",

        "required_body_state": {
            "kneeling": True,
            "torso_arching_back": True,
        },
        "rejection_rules": [
            {"condition": "hips_on_floor", "reason": "Hips on floor — must be kneeling"},
            {"condition": "legs_straight", "reason": "Legs straight — must be kneeling"},
        ],
        "boost_rules": [
            {"condition": "hands_on_heels", "bonus": 12, "reason": "Hands reaching back to heels"},
            {"condition": "hips_pushed_forward", "bonus": 8, "reason": "Hips pushed forward"},
        ],
        "rules": {
            "knee_bend": {
                "joint": "avg_knee",
                "target": 90.0,
                "tolerance": 25.0,
                "weight": 0.25,
                "error_msg": "Kneel with knees at approximately 90 degrees.",
                "priority": 1
            },
            "torso_arch": {
                "joint": "torso_vertical",
                "target": 40.0,
                "tolerance": 30.0,
                "weight": 0.30,
                "error_msg": "Arch your back more — push your chest toward the ceiling.",
                "priority": 2
            },
            "hip_extension": {
                "joint": "hip_flexion",
                "target": 160.0,
                "tolerance": 30.0,
                "weight": 0.25,
                "error_msg": "Push your hips forward over your knees.",
                "priority": 3
            },
            "arm_reach": {
                "joint": "avg_arm",
                "target": 160.0,
                "tolerance": 30.0,
                "weight": 0.20,
                "error_msg": "Reach your hands back toward your heels.",
                "priority": 4
            },
        }
    },
}
