"""
Industrial Yoga Pose Configuration Database.
Defines 9 stabilized yoga poses with hierarchical categories, hard geometry filters,
weighted angle rules, tolerances, and prioritized correction messages.
"""

CAMERA_INDEX = 0

# Cooldown parameters for pyttsx3 offline audio guides (seconds)
VOICE_COOLDOWN_DEFAULT = 6.0

# General wellness categories
WELLNESS_GOALS = ["stress", "back pain", "flexibility", "weight loss"]

# All supported pose categories for hierarchical classification
POSE_CATEGORIES = ["Standing", "Sitting", "Prone", "Inverted", "Balance"]

# ---------------------------------------------------------------------------
# 9-Pose Industrial Database with Hard Filters
# ---------------------------------------------------------------------------
POSE_DATABASE = {
    # ===================================================================
    # STANDING POSES
    # ===================================================================
    "mountain": {
        "category": "Standing",
        "display_name": "Mountain Pose (Tadasana)",
        "mirrored": False,
        "threshold": 40.0,
        "entry_guidance": "Stand tall with feet together, relax your shoulders, and let arms hang active at your sides.",
        "hard_filters": {
            "body_upright": True,       # torso must be near-vertical
            "legs_straight": True,      # both knees > 150
            "not_inverted": True,       # hips must be BELOW shoulders (in image coords: hip_y > shoulder_y)
            "not_seated": True,         # hips must be well above ankles
        },
        "rules": {
            "spine_alignment": {
                "joint": "torso_vertical",
                "target": 90.0,
                "tolerance": 15.0,
                "weight": 0.35,
                "error_msg": "Straighten your spine and stand taller.",
                "priority": 1
            },
            "knee_straightness": {
                "joint": "avg_knee",
                "target": 175.0,
                "tolerance": 18.0,
                "weight": 0.35,
                "error_msg": "Keep your legs straight without bending your knees.",
                "priority": 2
            },
            "arm_position": {
                "joint": "avg_arm_body_angle",
                "target": 12.0,
                "tolerance": 18.0,
                "weight": 0.30,
                "error_msg": "Keep your arms straight and active down beside your body.",
                "priority": 3
            }
        }
    },

    "tree": {
        "category": "Standing",
        "display_name": "Tree Pose (Vrksasana)",
        "mirrored": True,
        "threshold": 38.0,
        "entry_guidance": "Find balance on one leg, lift the opposite foot against your inner thigh, and raise hands overhead.",
        "hard_filters": {
            "one_knee_bent": True,      # one knee < 130, other > 150
            "standing_height": True,    # ankle-to-hip distance > 0.25
            "not_inverted": True,
            "not_seated": True,
        },
        "rules": {
            "standing_leg": {
                "joint": "standing_knee",
                "target": 175.0,
                "tolerance": 22.0,
                "weight": 0.35,
                "error_msg": "Straighten your standing leg to stabilize.",
                "priority": 1
            },
            "bent_knee_outward": {
                "joint": "bent_knee",
                "target": 50.0,
                "tolerance": 40.0,
                "weight": 0.30,
                "error_msg": "Open your bent knee outward away from center.",
                "priority": 2
            },
            "arms_overhead": {
                "joint": "avg_arm",
                "target": 160.0,
                "tolerance": 30.0,
                "weight": 0.35,
                "error_msg": "Raise your arms fully overhead and join your palms.",
                "priority": 3
            }
        }
    },

    "warrior_ii": {
        "category": "Standing",
        "display_name": "Warrior II (Virabhadrasana II)",
        "mirrored": True,
        "threshold": 38.0,
        "entry_guidance": "Adopt a wide stance, bend your front knee to 90 degrees, and stretch both arms out horizontally.",
        "hard_filters": {
            "wide_stance": True,        # ankle x-distance > 0.25
            "one_knee_bent_warrior": True,  # one knee ~90, other > 150
            "not_inverted": True,
            "not_seated": True,
        },
        "rules": {
            "front_knee_bend": {
                "joint": "front_knee",
                "target": 100.0,
                "tolerance": 25.0,
                "weight": 0.35,
                "error_msg": "Lower your hips and bend your front knee closer to 90 degrees.",
                "priority": 1
            },
            "back_leg_straight": {
                "joint": "back_knee",
                "target": 175.0,
                "tolerance": 22.0,
                "weight": 0.30,
                "error_msg": "Engage your thigh and straighten your back leg.",
                "priority": 2
            },
            "arms_horizontal": {
                "joint": "arms_horizontal_score",
                "target": 95.0,
                "tolerance": 25.0,
                "weight": 0.35,
                "error_msg": "Raise your arms horizontally at shoulder level.",
                "priority": 3
            }
        }
    },

    # ===================================================================
    # SITTING POSES
    # ===================================================================
    "sukhasana": {
        "category": "Sitting",
        "display_name": "Sukhasana (Easy Pose)",
        "mirrored": False,
        "threshold": 35.0,
        "entry_guidance": "Sit cross-legged on the floor with a tall spine, hands resting on your knees, shoulders relaxed.",
        "hard_filters": {
            "seated": True,             # hips near ankle level
            "torso_upright": True,      # torso must be near-vertical
            "not_prone": True,          # torso must NOT be horizontal
        },
        "rules": {
            "spine_upright": {
                "joint": "torso_vertical",
                "target": 90.0,
                "tolerance": 22.0,
                "weight": 0.40,
                "error_msg": "Sit tall and lengthen your spine upward.",
                "priority": 1
            },
            "knee_fold": {
                "joint": "avg_knee",
                "target": 80.0,
                "tolerance": 45.0,
                "weight": 0.30,
                "error_msg": "Cross your legs comfortably in front of you.",
                "priority": 2
            },
            "hands_on_knees": {
                "joint": "hands_knee_proximity",
                "target": 0.08,
                "tolerance": 0.18,
                "weight": 0.30,
                "error_msg": "Rest your hands gently on your knees.",
                "priority": 3
            }
        }
    },

    "butterfly": {
        "category": "Sitting",
        "display_name": "Butterfly Pose (Baddha Konasana)",
        "mirrored": False,
        "threshold": 38.0,
        "entry_guidance": "Sit upright, bend knees pulling heels close, bring soles of feet together, and let knees drop outward.",
        "hard_filters": {
            "seated": True,             # hips near ankle level
            "knees_wide": True,         # knee x-spread > ankle x-spread
            "torso_upright": True,
            "not_prone": True,
        },
        "rules": {
            "spine_upright": {
                "joint": "torso_vertical",
                "target": 90.0,
                "tolerance": 20.0,
                "weight": 0.40,
                "error_msg": "Sit tall and straighten your spine, do not round your back.",
                "priority": 1
            },
            "knees_low": {
                "joint": "knees_drop_ratio",
                "target": 40.0,
                "tolerance": 30.0,
                "weight": 0.35,
                "error_msg": "Relax your thighs and let your knees drop lower.",
                "priority": 2
            },
            "hands_feet": {
                "joint": "hands_feet_dist",
                "target": 0.08,
                "tolerance": 0.15,
                "weight": 0.25,
                "error_msg": "Hold your feet securely with both hands.",
                "priority": 3
            }
        }
    },

    "lotus": {
        "category": "Sitting",
        "display_name": "Lotus Pose (Padmasana)",
        "mirrored": False,
        "threshold": 38.0,
        "entry_guidance": "Sit upright with legs crossed, each foot resting on the opposite thigh. Keep spine tall and shoulders relaxed.",
        "hard_filters": {
            "seated": True,
            "torso_upright": True,
            "not_prone": True,
            "knees_moderate_spread": True,  # knees spread but not as wide as butterfly
        },
        "rules": {
            "spine_upright": {
                "joint": "torso_vertical",
                "target": 90.0,
                "tolerance": 18.0,
                "weight": 0.45,
                "error_msg": "Sit tall, lengthen your spine upward.",
                "priority": 1
            },
            "knee_angle_low": {
                "joint": "avg_knee",
                "target": 70.0,
                "tolerance": 40.0,
                "weight": 0.30,
                "error_msg": "Cross your legs more tightly, draw heels closer.",
                "priority": 2
            },
            "hands_on_knees": {
                "joint": "hands_knee_proximity",
                "target": 0.08,
                "tolerance": 0.15,
                "weight": 0.25,
                "error_msg": "Rest your hands gently on your knees.",
                "priority": 3
            }
        }
    },

    # ===================================================================
    # PRONE / BACKBEND POSES
    # ===================================================================
    "cobra": {
        "category": "Prone",
        "display_name": "Cobra Pose (Bhujangasana)",
        "mirrored": False,
        "threshold": 40.0,
        "entry_guidance": "Lie prone on your stomach, place hands near chest, and gently lift your torso arching the back.",
        "hard_filters": {
            "prone_body": True,         # torso horizontal, hips near ankle level
            "torso_horizontal": True,   # torso must NOT be vertical (reject sitting)
            "not_inverted": True,       # hips below shoulders in Y
        },
        "rules": {
            "chest_lift": {
                "joint": "hip_flexion",
                "target": 140.0,
                "tolerance": 30.0,
                "weight": 0.40,
                "error_msg": "Lift your chest higher off the mat.",
                "priority": 1
            },
            "elbows_bent": {
                "joint": "avg_arm",
                "target": 130.0,
                "tolerance": 35.0,
                "weight": 0.30,
                "error_msg": "Keep your elbows slightly bent close to your ribs.",
                "priority": 2
            },
            "relax_shoulders": {
                "joint": "shoulder_shrug",
                "target": 90.0,
                "tolerance": 20.0,
                "weight": 0.30,
                "error_msg": "Relax your shoulders down away from your ears.",
                "priority": 3
            }
        }
    },

    # ===================================================================
    # INVERTED POSES
    # ===================================================================
    "downward_dog": {
        "category": "Inverted",
        "display_name": "Downward Dog (Adho Mukha Svanasana)",
        "mirrored": False,
        "threshold": 38.0,
        "entry_guidance": "Press hands into the mat, lift hips high toward the ceiling, forming an inverted V-shape with your body.",
        "hard_filters": {
            "hips_above_shoulders": True,  # hip_y < shoulder_y in image coords
            "hands_on_floor": True,        # wrists near ankle Y level
        },
        "rules": {
            "hip_pike_angle": {
                "joint": "hip_flexion",
                "target": 65.0,
                "tolerance": 30.0,
                "weight": 0.40,
                "error_msg": "Push your hips higher toward the ceiling.",
                "priority": 1
            },
            "leg_straightness": {
                "joint": "avg_knee",
                "target": 170.0,
                "tolerance": 25.0,
                "weight": 0.30,
                "error_msg": "Try to straighten your legs more, pressing heels down.",
                "priority": 2
            },
            "arm_straightness": {
                "joint": "avg_arm",
                "target": 170.0,
                "tolerance": 25.0,
                "weight": 0.30,
                "error_msg": "Extend your arms fully, pressing palms into the mat.",
                "priority": 3
            }
        }
    },

    # ===================================================================
    # BALANCE POSES
    # ===================================================================
    "boat": {
        "category": "Balance",
        "display_name": "Boat Pose (Navasana)",
        "mirrored": False,
        "threshold": 38.0,
        "entry_guidance": "Sit and lean back, lift your legs off the floor forming a V-shape with your body. Extend arms forward parallel to the ground.",
        "hard_filters": {
            "v_shape": True,            # shoulders and ankles both above hips
            "leaning_back": True,       # torso angled back (not vertical)
        },
        "rules": {
            "torso_lean": {
                "joint": "torso_vertical",
                "target": 55.0,
                "tolerance": 25.0,
                "weight": 0.35,
                "error_msg": "Lean your torso back more to form the V-shape.",
                "priority": 1
            },
            "leg_elevation": {
                "joint": "avg_knee",
                "target": 150.0,
                "tolerance": 35.0,
                "weight": 0.35,
                "error_msg": "Lift and straighten your legs higher off the floor.",
                "priority": 2
            },
            "arms_forward": {
                "joint": "arms_horizontal_score",
                "target": 90.0,
                "tolerance": 30.0,
                "weight": 0.30,
                "error_msg": "Extend your arms straight forward, parallel to the floor.",
                "priority": 3
            }
        }
    },
}
