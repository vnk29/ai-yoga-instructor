"""
Yoga Recommendation Engine based on wellness targets.
Directly maps wellness search queries to curated YouTube yoga tutorial videos with rich card metadata.
"""

import re


class YogaRecommendationEngine:
    """Curated video discovery database matching wellness goals directly."""

    # Richly annotated video catalog with difficulty, descriptions, and benefit tags
    VIDEO_CATALOG = {
        "stress": [
            {
                "id": "sTANio_2E0Q",
                "url": "https://www.youtube.com/watch?v=sTANio_2E0Q",
                "title": "Yoga For Stress Relief - Gentle Practice",
                "description": "Unwind your body, release stress hormones, and calm the nervous system with simple calming folds.",
                "difficulty": "Beginner",
                "benefits": ["Stress reduction", "Deep relaxation", "Calms the mind"]
            },
            {
                "id": "Nw2oBIrQk84",
                "url": "https://www.youtube.com/watch?v=Nw2oBIrQk84",
                "title": "15 Min Yoga for Anxiety & Deep Calm",
                "description": "Slow, grounding floor stretches to relieve daily mental pressure and physical chest tightness.",
                "difficulty": "Beginner",
                "benefits": ["Anxiety relief", "Deep relaxation", "Improves breathing"]
            },
            {
                "id": "9J7GPje3y14",
                "url": "https://www.youtube.com/watch?v=9J7GPje3y14",
                "title": "Deep Restorative Yin Yoga Flow",
                "description": "Passive floor postures held for longer durations to release tension in deep connective tissues.",
                "difficulty": "Intermediate",
                "benefits": ["Deep relaxation", "Joint mobility", "Stress reduction"]
            }
        ],
        "back pain": [
            {
                "id": "2eA2Koq6pTI",
                "url": "https://www.youtube.com/watch?v=2eA2Koq6pTI",
                "title": "Yoga for Lower Back Pain Relief",
                "description": "Specifically designed to lengthen back muscles, release spine stiffness, and build support.",
                "difficulty": "Beginner",
                "benefits": ["Relieves stiffness", "Spinal decompression", "Builds strength"]
            },
            {
                "id": "XeXz8fIZcoY",
                "url": "https://www.youtube.com/watch?v=XeXz8fIZcoY",
                "title": "10 Min Daily Stretch for Back Tension",
                "description": "Quick and highly effective floor stretches to open up compressed lumbar vertebrae.",
                "difficulty": "Beginner",
                "benefits": ["Spinal decompression", "Relieves backache", "Improves mobility"]
            },
            {
                "id": "N643bZ2QvG0",
                "url": "https://www.youtube.com/watch?v=N643bZ2QvG0",
                "title": "Bridge Pose Core & Spine Support",
                "description": "Build glute and core stability to permanently relieve lower back pressure and strain.",
                "difficulty": "Intermediate",
                "benefits": ["Builds strength", "Glute activation", "Spinal decompression"]
            }
        ],
        "flexibility": [
            {
                "id": "o0G10Z5bA54",
                "url": "https://www.youtube.com/watch?v=o0G10Z5bA54",
                "title": "Yoga for Flexibility & Joint Mobility",
                "description": "A deep full-body stretch targeting stubborn hamstring fibers, tight groins, and hips.",
                "difficulty": "Beginner",
                "benefits": ["Improves flexibility", "Deep stretch", "Joint mobility"]
            },
            {
                "id": "S6gB0QHvRD4",
                "url": "https://www.youtube.com/watch?v=S6gB0QHvRD4",
                "title": "Triangle Pose Alignment Practice",
                "description": "Incorporate lateral bends to open up the ribs, lengthen hamstrings, and test stability.",
                "difficulty": "Intermediate",
                "benefits": ["Improves flexibility", "Improves balance", "Side stretch"]
            },
            {
                "id": "YXMskVfP3v4",
                "url": "https://www.youtube.com/watch?v=YXMskVfP3v4",
                "title": "Butterfly Pose for Hip Socket Release",
                "description": "Targeted hip opener to release pelvic congestion, loosen groin, and improve sitting posture.",
                "difficulty": "Beginner",
                "benefits": ["Hip opening", "Relieves tightness", "Improves mobility"]
            }
        ],
        "beginner yoga": [
            {
                "id": "v7AYKGeFi_U",
                "url": "https://www.youtube.com/watch?v=v7AYKGeFi_U",
                "title": "Yoga For Complete Beginners - 20 Min",
                "description": "A perfect foundation practice guiding you through key poses with slow alignment cues.",
                "difficulty": "Beginner",
                "benefits": ["Builds strength", "Joint mobility", "Easy starting"]
            },
            {
                "id": "K-GpDP_4C8U",
                "url": "https://www.youtube.com/watch?v=K-GpDP_4C8U",
                "title": "20 Min Easy Gentle Beginner Warmup",
                "description": "A slow, comfortable routine focusing on simple poses and steady breathing transitions.",
                "difficulty": "Beginner",
                "benefits": ["Easy starting", "Improves breathing", "Calms the mind"]
            }
        ],
        "weight loss": [
            {
                "id": "O1114-1z34c",
                "url": "https://www.youtube.com/watch?v=O1114-1z34c",
                "title": "Yoga Tone - Yoga For Weight Loss",
                "description": "A high-energy vinyasa flow specifically created to build strength, increase endurance, and tone the entire body.",
                "difficulty": "Intermediate",
                "benefits": ["Builds strength", "High calorie burn", "Metabolic boost"]
            },
            {
                "id": "kYvH573n6v0",
                "url": "https://www.youtube.com/watch?v=kYvH573n6v0",
                "title": "Core Strength Ritual - Tone & Stabilize",
                "description": "Tone your abdominal wall, support your spine, and improve core stability with slow, deliberate holds.",
                "difficulty": "Intermediate",
                "benefits": ["Builds strength", "Core stability", "High calorie burn"]
            },
            {
                "id": "UEEsdXn8oG8",
                "url": "https://www.youtube.com/watch?v=UEEsdXn8oG8",
                "title": "Morning Vinyasa Flow - Fat Burning",
                "description": "An active morning cardio-yoga hybrid that speeds up digestion and enhances posture.",
                "difficulty": "Intermediate",
                "benefits": ["Boosts energy", "Tones core", "Builds strength"]
            }
        ],
        "neck pain": [
            {
                "id": "X3-gKPNyrTA",
                "url": "https://www.youtube.com/watch?v=X3-gKPNyrTA",
                "title": "Yoga for Neck, Shoulders & Upper Back",
                "description": "Melt away cervical tension and upper-body stiffness caused by screens and laptop slouching.",
                "difficulty": "Beginner",
                "benefits": ["Relieves stiffness", "Shoulder tension", "Chest opener"]
            },
            {
                "id": "761Vnrd7LwM",
                "url": "https://www.youtube.com/watch?v=761Vnrd7LwM",
                "title": "10 Min Quick Neck Stiffness Release",
                "description": "Easy, sitting stretches that you can do directly from your office desk for quick relief.",
                "difficulty": "Beginner",
                "benefits": ["Relieves stiffness", "Quick tension relief", "Improves mobility"]
            }
        ],
        "posture correction": [
            {
                "id": "B871pU4tEEY",
                "url": "https://www.youtube.com/watch?v=B871pU4tEEY",
                "title": "Yoga for Better Posture & Alignment",
                "description": "Lengthen the spine, pull back rounded shoulders, and activate stabilizing core muscles.",
                "difficulty": "Beginner",
                "benefits": ["Improves posture", "Shoulder tension", "Spinal decompression"]
            },
            {
                "id": "8p25v4y9hEE",
                "url": "https://www.youtube.com/watch?v=8p25v4y9hEE",
                "title": "10 Min Daily Stretch to Fix Hunchback",
                "description": "Targeted active stretches designed to reverse office slouching and align pelvis.",
                "difficulty": "Beginner",
                "benefits": ["Improves posture", "Chest opener", "Builds strength"]
            }
        ],
        "meditation yoga": [
            {
                "id": "O-6f5wQXSu8",
                "url": "https://www.youtube.com/watch?v=O-6f5wQXSu8",
                "title": "Mindfulness Meditation & Seated Flow",
                "description": "Coordinate slow pranayama breathing with stabilizing sitting poses to ground yourself.",
                "difficulty": "Beginner",
                "benefits": ["Deep relaxation", "Calms the mind", "Mental clarity"]
            },
            {
                "id": "6p_yaNFSYao",
                "url": "https://www.youtube.com/watch?v=6p_yaNFSYao",
                "title": "10 Minute Quiet Breathing & Mindful Flow",
                "description": "A peaceful vinyasa segment leading into quiet seated concentration and stillness.",
                "difficulty": "Beginner",
                "benefits": ["Deep relaxation", "Anxiety relief", "Mental clarity"]
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
