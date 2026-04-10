"""
Central configuration: sensitivity presets, thresholds, and voice messages.
"""

# ---------------------------------------------------------------------------
# TTS settings
# ---------------------------------------------------------------------------

# Gradium TTS (primary).  Set GRADIUM_API_KEY env var before running.
# Leave GRADIUM_VOICE_ID empty ("") to use the Gradium default voice.
GRADIUM_VOICE_ID: str = "YTpq7expH9539ERJ"

# Camera index (0 = default/built-in)
CAMERA_INDEX = 0

# If body angle with horizontal exceeds this, person is not considered in plank
PLANK_BODY_ANGLE_THRESHOLD = 30  # degrees

# Minimum normalised shoulder-to-ankle distance for person to count as visible
MIN_BODY_LENGTH = 0.25

# Seconds between positive-reinforcement messages
GOOD_FORM_INTERVAL = 15.0

# Per-issue sensitivity presets
# hip_threshold      : max allowed normalised hip deviation from ideal line
# head_up_threshold  : max nose-above-shoulder rise (craning neck — tight)
# head_down_threshold: max nose-below-shoulder droop (looking down is normal — lenient)
# consecutive_frames : how many consecutive bad frames before triggering alert
# cooldown           : seconds before the same issue can trigger again
SENSITIVITY_PRESETS = {
    "low": {
        "hip_threshold": 0.04,
        # head_up: nose above shoulder (serious error, tight threshold)
        "head_up_threshold": 0.05,
        # head_down: nose below shoulder – normal in plank, only flag severe droop
        "head_down_threshold": 0.18,
        "consecutive_frames": 12,
        "cooldown": 6.0,
    },
    "medium": {
        "hip_threshold": 0.025,
        "head_up_threshold": 0.035,
        "head_down_threshold": 0.14,
        "consecutive_frames": 8,
        "cooldown": 4.0,
    },
    "high": {
        "hip_threshold": 0.015,
        "head_up_threshold": 0.02,
        "head_down_threshold": 0.10,
        "consecutive_frames": 4,
        "cooldown": 2.5,
    },
}

# Voice messages per issue key (cycled in order, never repeat consecutive)
ISSUE_MESSAGES: dict[str, list[str]] = {
    "hip_high": [
        "Your hips are too high. Lower them to form a straight line.",
        "Bring those hips down. You are in a pike position.",
        "Lower your hips to align with your shoulders and feet.",
        "Pike detected. Flatten that back out.",
    ],
    "hip_low": [
        "Your hips are sagging. Engage your core and lift them up.",
        "Tighten your core! Your hips are dropping.",
        "Squeeze your glutes and raise those hips.",
        "Do not let gravity win. Lift your hips!",
        "Core engagement needed. Pull those hips up.",
    ],
    "head_up": [
        "Lower your head. Your neck should be aligned with your spine.",
        "You are craning your neck. Look straight down.",
        "Neutral neck! Do not look up during your plank.",
        "Bring your chin in. Keep your gaze at the floor.",
    ],
    "head_down": [
        "Lift your head slightly. Look at the floor a foot ahead of you.",
        "Your head is dropping. Keep your neck in neutral position.",
        "Do not tuck your chin. Look slightly forward.",
    ],
    "not_in_plank": [
        "Get into plank position to start. Camera should be to your side.",
        "Position your camera at torso height to your side, then get into plank.",
        "I am ready when you are. Get into your plank!",
    ],
    "good_form": [
        "Great form! Keep it up.",
        "Perfect alignment. Stay strong!",
        "Looking good. Maintain this position.",
        "Beautiful plank! You have got this.",
        "Excellent! Your body is perfectly straight.",
        "Solid plank! Hold that position.",
    ],
}
