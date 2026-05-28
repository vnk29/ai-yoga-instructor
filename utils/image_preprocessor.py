"""
Advanced Image Preprocessing and Multi-Stage Pose Detection Retry Pipeline.
Specifically optimizes static human photos, line drawings, cartoons, sketches,
and low-contrast illustrations for robust MediaPipe landmark extraction.
"""

import cv2
import numpy as np
import os


class ImagePreprocessor:
    """Performs adaptive contrast, brightness scaling, edge sharpening, and retries for static pose analysis."""

    @staticmethod
    def safe_resize(img: np.ndarray, max_dim: int = 900) -> np.ndarray:
        """Resizes the image safely keeping aspect ratio to prevent crashes on high-res photos."""
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def apply_clahe(img: np.ndarray, clip_limit: float = 3.0) -> np.ndarray:
        """Applies Contrast Limited Adaptive Histogram Equalization on L channel of LAB representation."""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        
        # Merge back and convert to BGR
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    @staticmethod
    def apply_sharpen(img: np.ndarray) -> np.ndarray:
        """Applies a 2D sharpening convolution filter to highlight faint edges and sketches."""
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    @classmethod
    def run_detection_pipeline(cls, image_source, detector) -> tuple[bool, np.ndarray, str, any, np.ndarray]:
        """
        Executes a 4-stage preprocessing and detection retry pipeline:
        1. Stage 1: Base Preprocessed (Resized, RGB, Normalized)
        2. Stage 2: Adaptive Contrast Enhanced (CLAHE)
        3. Stage 3: Edge Sharpened
        4. Stage 4: Combined Contrast + Edge Sharpened
        
        Returns:
            success (bool): Whether landmarks were successfully detected.
            processed_image (np.ndarray): The BGR image of the successful stage.
            stage_name (str): The name of the successful stage.
            results (any): The MediaPipe detection results.
            original_resized (np.ndarray): The resized original image (for side-by-side comparison).
        """
        try:
            if isinstance(image_source, str):
                img_original = cv2.imread(image_source)
                if img_original is None:
                    raise ValueError(f"Image at {image_source} could not be read.")
            else:
                img_original = image_source

            # Safe resize and normalize initial brightness
            img_resized = cls.safe_resize(img_original, max_dim=900)
            
            # Ensure the image has valid dimensions
            if img_resized.size == 0 or len(img_resized.shape) < 2:
                raise ValueError(f"Image dimensions invalid after resize: {img_resized.shape}")
            
            # Ensure 3-channel BGR if grayscale
            if len(img_resized.shape) == 2:
                img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
            
            img_normalized = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX)

            # Prepare processed variations with error handling
            try:
                img_clahe = cls.apply_clahe(img_normalized)
            except Exception as e:
                print(f"[PREPROCESS] CLAHE failed: {e}, using normalized image")
                img_clahe = img_normalized
            
            try:
                img_sharpened = cls.apply_sharpen(img_normalized)
            except Exception as e:
                print(f"[PREPROCESS] Sharpening failed: {e}, using normalized image")
                img_sharpened = img_normalized
            
            try:
                img_combined = cls.apply_sharpen(img_clahe)
            except Exception as e:
                print(f"[PREPROCESS] Combined filtering failed: {e}, using CLAHE image")
                img_combined = img_clahe

            stages = [
                ("Stage 1: Base Normalization", img_normalized),
                ("Stage 2: Adaptive Contrast (CLAHE)", img_clahe),
                ("Stage 3: Edge Sharpening", img_sharpened),
                ("Stage 4: Contrast + Edge Sharpening", img_combined)
            ]

            # Retry detection loop
            for stage_name, stage_img in stages:
                if stage_img is None or stage_img.size == 0:
                    continue
                    
                try:
                    rgb = cv2.cvtColor(stage_img, cv2.COLOR_BGR2RGB)
                    results = detector.process(rgb)
                    
                    if results.pose_landmarks:
                        print(f"[RETRY PIPELINE] Success on {stage_name}!")
                        return True, stage_img, stage_name, results, img_resized
                except Exception as e:
                    print(f"[RETRY PIPELINE] Error on {stage_name}: {e}")
                    continue

            # If all failed, return base preprocessed image with failure state
            print("[RETRY PIPELINE] All 4 preprocessing retry stages failed.")
            return False, img_normalized, "None (All Stages Failed)", None, img_resized
            
        except Exception as e:
            print(f"[RETRY PIPELINE] Pipeline error: {e}")
            return False, None, "Error", None, None

    # ------------------------------------------------------------------
    # Convenience wrappers for batch / dataset processing
    # ------------------------------------------------------------------

    @classmethod
    def process_image_for_pose(cls, image_path: str, detector) -> list | None:
        """
        Run the full 4-stage retry pipeline on a single image file and return
        the best landmark list, or None if all stages fail.

        This is the primary entry-point used by build_pose_database.py for
        batch dataset processing.

        Parameters
        ----------
        image_path : str
            Absolute path to the image file.
        detector : PoseDetector
            The detector instance to use for landmark extraction.

        Returns
        -------
        list | None
            The first-person landmark list on success, or None.
        """
        try:
            success, _, stage, results, _ = cls.run_detection_pipeline(image_path, detector)
            if success and results and results.pose_landmarks:
                return results.pose_landmarks[0]
        except Exception as e:
            print(f"[PREPROCESS] Error processing {image_path}: {e}")
        return None

    @staticmethod
    def process_to_cv2(uploaded_file) -> "np.ndarray | None":
        """
        Convert a Streamlit UploadedFile to an OpenCV BGR numpy array.

        Returns None if the file cannot be decoded.
        """
        import numpy as np
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        return img
