import time

import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# ---------------------------------------------------------------------------
# Face / mouth detector
# ---------------------------------------------------------------------------

class FaceDetector:
    """
    Wraps the MediaPipe FaceLandmarker and tracks how long the mouth has
    been open, toggling a callback after a configurable hold duration.

    Args:
        model_path: Path to the face_landmarker.task model file.
        mouth_threshold: Normalised vertical gap between landmark 13 and 14
                         required to consider the mouth "open".
        toggle_wait: Seconds the mouth must stay open to fire the toggle.
    """

    def __init__(
        self,
        model_path: str,
        mouth_threshold: float = 0.04,
        toggle_wait: float = 3.0,
    ) -> None:
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._detector = vision.FaceLandmarker.create_from_options(opts)
        self.mouth_threshold = mouth_threshold
        self.toggle_wait = toggle_wait
        self._open_since: float | None = None

    def detect(self, mp_image, timestamp_ms: int):
        """Run face detection and return the MediaPipe result."""
        return self._detector.detect_for_video(mp_image, timestamp_ms)

    def is_mouth_open(self, face_result) -> bool:
        """True when the mouth gap exceeds the threshold."""
        if not face_result.face_landmarks:
            return False
        lms = face_result.face_landmarks[0]
        return abs(lms[13].y - lms[14].y) > self.mouth_threshold

    def update(self, face_result, frame: np.ndarray) -> bool:
        """
        Update internal timer and overlay countdown text on `frame`.

        Returns:
            True exactly once when the hold duration has been reached,
            signalling that the caller should toggle training mode.
        """
        if self.is_mouth_open(face_result):
            if self._open_since is None:
                self._open_since = time.time()
            elapsed = time.time() - self._open_since
            if elapsed >= self.toggle_wait:
                self._open_since = None
                return True  # fire the toggle
            countdown = max(0, int(self.toggle_wait - elapsed))
            cv2.putText(
                frame,
                f"HOLD MOUTH OPEN: {countdown}s",
                (30, 30),
                cv2.FONT_HERSHEY_DUPLEX,
                0.7,
                (0, 0, 0),
                2,
            )
        else:
            self._open_since = None
        return False