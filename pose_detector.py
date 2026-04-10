"""
MediaPipe Pose Landmarker wrapper using the Tasks API (mediapipe ≥ 0.10).

The model file is downloaded automatically on first run into the project
directory. Subsequent runs use the cached file.
"""

import os
import urllib.request
from pathlib import Path

import mediapipe as mp

_tv = mp.tasks.vision

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_MODEL_PATH = Path(__file__).parent / "pose_landmarker_full.task"


def _ensure_model() -> str:
    if not _MODEL_PATH.exists():
        print(f"Downloading MediaPipe Pose model → {_MODEL_PATH.name} …")
        try:
            import ssl
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            with urllib.request.urlopen(_MODEL_URL, context=ctx) as resp, open(_MODEL_PATH, "wb") as f:
                f.write(resp.read())
        except Exception:
            # Fallback: use curl (always available on macOS)
            import subprocess
            subprocess.run(
                ["curl", "-L", "-o", str(_MODEL_PATH), _MODEL_URL],
                check=True,
            )
        print("  Download complete.")
    return str(_MODEL_PATH)


# ---------------------------------------------------------------------------
# PoseDetector
# ---------------------------------------------------------------------------


class PoseDetector:
    """
    Thin wrapper around mediapipe.tasks.vision.PoseLandmarker.

    Landmark lists
    --------------
    After calling `process(frame_rgb)`, check::

        result = detector.process(frame_rgb)
        if result.pose_landmarks:
            lm_list = result.pose_landmarks[0]   # first (and only) person

    Then pass `lm_list` to `get_coords` / `midpoint` / `draw_landmarks`.
    """

    _PoseLandmark = _tv.PoseLandmark
    _Connections = _tv.PoseLandmarksConnections.POSE_LANDMARKS

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        model_path = _ensure_model()

        options = _tv.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=_tv.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=False,
        )
        self._landmarker = _tv.PoseLandmarker.create_from_options(options)
        self._du = _tv.drawing_utils

    # ------------------------------------------------------------------
    # Core detection
    # ------------------------------------------------------------------

    def process(self, frame_rgb):
        """
        Run pose estimation on an RGB numpy array.
        Returns a PoseLandmarkerResult; check .pose_landmarks for detections.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        return self._landmarker.detect(mp_image)

    # ------------------------------------------------------------------
    # Landmark helpers
    # ------------------------------------------------------------------

    def get_coords(
        self,
        landmark_list: list,
        name: str,
    ) -> tuple[float, float, float, float]:
        """
        Return (x, y, z, visibility) for a named landmark from a single-person
        landmark list (i.e. result.pose_landmarks[0]).
        x, y are normalised to [0, 1].
        """
        idx = self._PoseLandmark[name].value
        lm = landmark_list[idx]
        return (lm.x, lm.y, lm.z, getattr(lm, "visibility", 1.0))

    @staticmethod
    def midpoint(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Average of two (x, y, z, vis) landmark tuples."""
        return (
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2,
            (a[2] + b[2]) / 2,
            min(a[3], b[3]),
        )

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------

    def draw_landmarks(self, frame_bgr, landmark_list: list) -> None:
        """Overlay the pose skeleton onto a BGR frame (in-place)."""
        self._du.draw_landmarks(
            frame_bgr,
            landmark_list,
            self._Connections,
            landmark_drawing_spec=self._du.DrawingSpec(
                color=(50, 200, 50), thickness=2, circle_radius=3
            ),
            connection_drawing_spec=self._du.DrawingSpec(
                color=(200, 200, 50), thickness=2
            ),
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._landmarker.close()
