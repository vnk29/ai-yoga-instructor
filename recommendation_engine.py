"""
Yoga Recommendation Engine based on wellness targets.
Directly maps wellness search queries to curated YouTube yoga tutorial videos with rich card metadata.
"""

import re


class YogaRecommendationEngine:
    """Curated video discovery database matching wellness goals directly."""

    # Richly annotated video catalog with difficulty, descriptions, benefit tags, duration, and type tags
    VIDEO_CATALOG = {
        "stress": [
            {
                "id": "sTANio_2E0Q",
                "url": "https://www.youtube.com/watch?v=sTANio_2E0Q",
                "title": "Yoga For Stress Relief — Gentle Practice",
                "description": "Unwind your body, release stress hormones, and calm the nervous system with simple calming folds.",
                "difficulty": "Beginner",
                "benefits": ["Stress reduction", "Deep relaxation"],
                "duration": "20 min",
                "tags": ["Gentle", "Relaxation"]
            },
            {
                "id": "Nw2oBIrQk84",
                "url": "https://www.youtube.com/watch?v=Nw2oBIrQk84",
                "title": "15 Min Yoga for Anxiety & Deep Calm",
                "description": "Slow, grounding floor stretches to relieve daily mental pressure and physical chest tightness.",
                "difficulty": "Beginner",
                "benefits": ["Anxiety relief", "Improves breathing"],
                "duration": "15 min",
                "tags": ["Calming", "Floor"]
            },
            {
                "id": "9J7GPje3y14",
                "url": "https://www.youtube.com/watch?v=9J7GPje3y14",
                "title": "Deep Restorative Yin Yoga Flow",
                "description": "Passive floor postures held for longer durations to release tension in deep connective tissues.",
                "difficulty": "Intermediate",
                "benefits": ["Joint mobility", "Stress reduction"],
                "duration": "30 min",
                "tags": ["Yin", "Restorative"]
            }
        ],
        "back pain": [
            {
                "id": "2eA2Koq6pTI",
                "url": "https://www.youtube.com/watch?v=2eA2Koq6pTI",
                "title": "Yoga for Lower Back Pain Relief",
                "description": "Specifically designed to lengthen back muscles, release spine stiffness, and build support.",
                "difficulty": "Beginner",
                "benefits": ["Relieves stiffness", "Spinal decompression"],
                "duration": "20 min",
                "tags": ["Tutorial", "Therapeutic"]
            },
            {
                "id": "XeXz8fIZcoY",
                "url": "https://www.youtube.com/watch?v=XeXz8fIZcoY",
                "title": "10 Min Daily Stretch for Back Tension",
                "description": "Quick and highly effective floor stretches to open up compressed lumbar vertebrae.",
                "difficulty": "Beginner",
                "benefits": ["Relieves backache", "Improves mobility"],
                "duration": "10 min",
                "tags": ["Quick", "Daily"]
            },
            {
                "id": "N643bZ2QvG0",
                "url": "https://www.youtube.com/watch?v=N643bZ2QvG0",
                "title": "Bridge Pose Core & Spine Support",
                "description": "Build glute and core stability to permanently relieve lower back pressure and strain.",
                "difficulty": "Intermediate",
                "benefits": ["Builds strength", "Glute activation"],
                "duration": "15 min",
                "tags": ["Strength", "Core"]
            }
        ],
        "flexibility": [
            {
                "id": "o0G10Z5bA54",
                "url": "https://www.youtube.com/watch?v=o0G10Z5bA54",
                "title": "yoga for flexibility — Complete 20 Min Guide",
                "description": "Comprehensive yoga for flexibility with expert guidance.",
                "difficulty": "Beginner",
                "benefits": ["Builds strength", "Improves flexibility"],
                "duration": "20 min",
                "tags": ["Tutorial", "Complete"]
            },
            {
                "id": "S6gB0QHvRD4",
                "url": "https://www.youtube.com/watch?v=S6gB0QHvRD4",
                "title": "yoga for flexibility For Beginners",
                "description": "Gentle accessible yoga for flexibility for beginners.",
                "difficulty": "Beginner",
                "benefits": ["Accessible", "No equipment"],
                "duration": "15 min",
                "tags": ["Beginner", "Gentle"]
            },
            {
                "id": "YXMskVfP3v4",
                "url": "https://www.youtube.com/watch?v=YXMskVfP3v4",
                "title": "Morning yoga for flexibility Flow",
                "description": "Energising morning yoga for flexibility.",
                "difficulty": "Beginner",
                "benefits": ["Energises body"],
                "duration": "15 min",
                "tags": ["Morning", "Flow"]
            },
            {
                "id": "g_tea8ZNtKc",
                "url": "https://www.youtube.com/watch?v=g_tea8ZNtKc",
                "title": "yoga for flexibility Yin Deep Stretch",
                "description": "Deep stretching yin yoga for full body flexibility and relaxation.",
                "difficulty": "Intermediate",
                "benefits": ["Deep stretch", "Full body"],
                "duration": "25 min",
                "tags": ["Yin", "Deep Stretch"]
            },
            {
                "id": "Eml2xg6PbiQ",
                "url": "https://www.youtube.com/watch?v=Eml2xg6PbiQ",
                "title": "yoga for flexibility Power Flow",
                "description": "Dynamic power yoga flow to increase flexibility and strength simultaneously.",
                "difficulty": "Advanced",
                "benefits": ["Power yoga", "Builds strength"],
                "duration": "30 min",
                "tags": ["Power", "Advanced"]
            },
            {
                "id": "4pKly2JojMw",
                "url": "https://www.youtube.com/watch?v=4pKly2JojMw",
                "title": "yoga for flexibility Meditation & Nidra",
                "description": "Calming yoga nidra with gentle flexibility movements for mind-body balance.",
                "difficulty": "Beginner",
                "benefits": ["Relaxation", "Mind-body balance"],
                "duration": "20 min",
                "tags": ["Meditation", "Nidra"]
            }
        ],
        "beginner yoga": [
            {
                "id": "v7AYKGeFi_U",
                "url": "https://www.youtube.com/watch?v=v7AYKGeFi_U",
                "title": "Yoga For Complete Beginners - 20 Min",
                "description": "A perfect foundation practice guiding you through key poses with slow alignment cues.",
                "difficulty": "Beginner",
                "benefits": ["Builds strength", "Easy starting"],
                "duration": "20 min",
                "tags": ["Foundation", "Tutorial"]
            },
            {
                "id": "K-GpDP_4C8U",
                "url": "https://www.youtube.com/watch?v=K-GpDP_4C8U",
                "title": "20 Min Easy Gentle Beginner Warmup",
                "description": "A slow, comfortable routine focusing on simple poses and steady breathing transitions.",
                "difficulty": "Beginner",
                "benefits": ["Improves breathing", "Calms the mind"],
                "duration": "20 min",
                "tags": ["Gentle", "Warmup"]
            }
        ],
        "weight loss": [
            {
                "id": "O1114-1z34c",
                "url": "https://www.youtube.com/watch?v=O1114-1z34c",
                "title": "Yoga Tone - Yoga For Weight Loss",
                "description": "A high-energy vinyasa flow to build strength, increase endurance, and tone the entire body.",
                "difficulty": "Intermediate",
                "benefits": ["High calorie burn", "Metabolic boost"],
                "duration": "25 min",
                "tags": ["Vinyasa", "Toning"]
            },
            {
                "id": "kYvH573n6v0",
                "url": "https://www.youtube.com/watch?v=kYvH573n6v0",
                "title": "Core Strength Ritual - Tone & Stabilize",
                "description": "Tone your abdominal wall, support your spine, and improve core stability with slow holds.",
                "difficulty": "Intermediate",
                "benefits": ["Core stability", "Builds strength"],
                "duration": "20 min",
                "tags": ["Core", "Strength"]
            },
            {
                "id": "UEEsdXn8oG8",
                "url": "https://www.youtube.com/watch?v=UEEsdXn8oG8",
                "title": "Morning Vinyasa Flow - Fat Burning",
                "description": "An active morning cardio-yoga hybrid that speeds up digestion and enhances posture.",
                "difficulty": "Intermediate",
                "benefits": ["Boosts energy", "Tones core"],
                "duration": "20 min",
                "tags": ["Morning", "Cardio"]
            }
        ],
        "neck pain": [
            {
                "id": "X3-gKPNyrTA",
                "url": "https://www.youtube.com/watch?v=X3-gKPNyrTA",
                "title": "Yoga for Neck, Shoulders & Upper Back",
                "description": "Melt away cervical tension and upper-body stiffness caused by screens and laptop slouching.",
                "difficulty": "Beginner",
                "benefits": ["Relieves stiffness", "Chest opener"],
                "duration": "15 min",
                "tags": ["Therapeutic", "Upper Body"]
            },
            {
                "id": "761Vnrd7LwM",
                "url": "https://www.youtube.com/watch?v=761Vnrd7LwM",
                "title": "10 Min Quick Neck Stiffness Release",
                "description": "Easy sitting stretches you can do directly from your desk for quick relief.",
                "difficulty": "Beginner",
                "benefits": ["Quick tension relief", "Improves mobility"],
                "duration": "10 min",
                "tags": ["Quick", "Office"]
            }
        ],
        "posture correction": [
            {
                "id": "B871pU4tEEY",
                "url": "https://www.youtube.com/watch?v=B871pU4tEEY",
                "title": "Yoga for Better Posture & Alignment",
                "description": "Lengthen the spine, pull back rounded shoulders, and activate stabilizing core muscles.",
                "difficulty": "Beginner",
                "benefits": ["Improves posture", "Spinal decompression"],
                "duration": "15 min",
                "tags": ["Alignment", "Tutorial"]
            },
            {
                "id": "8p25v4y9hEE",
                "url": "https://www.youtube.com/watch?v=8p25v4y9hEE",
                "title": "10 Min Daily Stretch to Fix Hunchback",
                "description": "Targeted active stretches designed to reverse office slouching and align pelvis.",
                "difficulty": "Beginner",
                "benefits": ["Chest opener", "Builds strength"],
                "duration": "10 min",
                "tags": ["Quick", "Daily"]
            }
        ],
        "meditation yoga": [
            {
                "id": "O-6f5wQXSu8",
                "url": "https://www.youtube.com/watch?v=O-6f5wQXSu8",
                "title": "Mindfulness Meditation & Seated Flow",
                "description": "Coordinate slow pranayama breathing with stabilizing sitting poses to ground yourself.",
                "difficulty": "Beginner",
                "benefits": ["Calms the mind", "Mental clarity"],
                "duration": "15 min",
                "tags": ["Mindfulness", "Breathing"]
            },
            {
                "id": "6p_yaNFSYao",
                "url": "https://www.youtube.com/watch?v=6p_yaNFSYao",
                "title": "10 Minute Quiet Breathing & Mindful Flow",
                "description": "A peaceful vinyasa segment leading into quiet seated concentration and stillness.",
                "difficulty": "Beginner",
                "benefits": ["Anxiety relief", "Mental clarity"],
                "duration": "10 min",
                "tags": ["Peaceful", "Quiet"]
            }
        ]
    }

    # Extended NLP synonyms keyword mappings for discovery matching
    SEARCH_MAPPINGS = {
        "stress": [
            "stress", "stress relief", "anxiety", "calm", "tension", "relax", "relaxation",
            "peace", "worried", "nervous", "de-stress", "mindful", "fatigue", "mental"
        ],
        "back pain": [
            "back pain", "lower back pain", "backache", "back hurts", "back hurt", "lower back",
            "spine", "sore back", "lumbar", "back stiffness", "back tension", "cat cow"
        ],
        "flexibility": [
            "flexibility", "mobility", "stretch", "stiff", "tight hips", "hip opening",
            "hamstring", "loosen up", "flexible", "groin stretch", "range of motion"
        ],
        "beginner yoga": [
            "beginner yoga", "beginner", "gentle", "easy", "starting yoga", "first time",
            "novice", "simple", "complete beginners", "easy warmup"
        ],
        "weight loss": [
            "weight loss", "fat burn", "core strength", "metabolism", "fitness", "workout",
            "toning", "stronger core", "burn fat", "active flow", "sweat", "cardio"
        ],
        "neck pain": [
            "neck pain", "neck stiffness", "neck hurts", "neck hurt", "shoulders",
            "shoulder tension", "upper back", "cervical"
        ],
        "posture correction": [
            "posture correction", "posture", "alignment", "rounded shoulders",
            "slouching", "desk posture", "hunchback", "slouch", "align"
        ],
        "meditation yoga": [
            "meditation", "meditation yoga", "mindfulness", "mindful", "breathing",
            "pranayama", "spirituality", "seated flow", "stillness"
        ]
    }

    @staticmethod
    def extract_youtube_id(url: str) -> str:
        """Robustly extracts YouTube video ID from various link formats."""
        match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
        return match.group(1) if match else "sTANio_2E0Q"

    @staticmethod
    def get_videos_for_query(query: str) -> list[dict]:
        """
        Takes a freeform query and maps it to relevant curated videos using direct keyword matching.
        """
        text = re.sub(r"[^a-z0-9\s]", " ", query.lower()).strip()
        if not text:
            # Return default high-value beginner / stress recommendations
            return YogaRecommendationEngine.VIDEO_CATALOG["beginner yoga"]

        best_category = None
        max_score = 0

        for category, keywords in YogaRecommendationEngine.SEARCH_MAPPINGS.items():
            score = 0
            for kw in keywords:
                if kw in text:
                    score += len(kw) * 2  # Boost phrase matches

            if score > max_score:
                max_score = score
                best_category = category

        if best_category and best_category in YogaRecommendationEngine.VIDEO_CATALOG:
            return YogaRecommendationEngine.VIDEO_CATALOG[best_category]

        # Default matching logic (fallback if no perfect keywords match)
        for category in YogaRecommendationEngine.VIDEO_CATALOG.keys():
            if category in text:
                return YogaRecommendationEngine.VIDEO_CATALOG[category]

        # General search fallback using basic word contains
        words = text.split()
        for word in words:
            if len(word) > 3:
                for category, keywords in YogaRecommendationEngine.SEARCH_MAPPINGS.items():
                    if any(word in kw for kw in keywords):
                        return YogaRecommendationEngine.VIDEO_CATALOG[category]

        return []
