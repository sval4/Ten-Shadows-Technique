import cv2
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from shadow import HAND_CONNECTIONS

# ---------------------------------------------------------------------------
# Hand detector
# ---------------------------------------------------------------------------

class HandDetector:
    """
    Wraps the MediaPipe HandLandmarker and provides a convenience method
    for building a left/right hand index map.

    Args:
        model_path: Path to the hand_landmarker.task model file.
        num_hands: Maximum number of hands to detect simultaneously.
        mirrored: Whether the camera feed is horizontally flipped, which
                  requires swapping the Left/Right labels from MediaPipe.
    """

    def __init__(
        self,
        model_path: str,
        num_hands: int = 2,
        mirrored: bool = True,
    ) -> None:
        base_opts = python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base_opts,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
        )
        self._detector = vision.HandLandmarker.create_from_options(opts)
        self.mirrored = mirrored
        self.sword = cv2.imread("sword.png", cv2.IMREAD_UNCHANGED)  # Keep alpha channel
        gray = cv2.cvtColor(self.sword, cv2.COLOR_BGR2GRAY)
        # _, alpha = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)  # white → transparent
        # self.sword = cv2.cvtColor(self.sword, cv2.COLOR_BGR2BGRA)
        # self.sword[:, :, 3] = alpha

    def detect(self, mp_image, timestamp_ms: int):
        """Run hand detection and return the MediaPipe result."""
        return self._detector.detect_for_video(mp_image, timestamp_ms)

    def build_hand_map(self, hand_result) -> dict[str, int]:
        """
        Return {"left": idx, "right": idx} where idx is the index into
        `hand_result.hand_landmarks`, or -1 if that hand was not detected.
        """
        hand_map = {"left": -1, "right": -1}
        for i, handedness in enumerate(hand_result.handedness):
            label = handedness[0].category_name.lower()
            if self.mirrored:
                label = "right" if label == "left" else "left"
            hand_map[label] = i
        return hand_map

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_result,
        hand_map: dict[str, int],
        summon_status: tuple[str | None, int],
        frames_to_confirm: int,
    ) -> None:
        """Draw skeleton, fingertip dots, and index labels onto `frame`."""
        h, w = frame.shape[:2]
        line_brightness = (
            int(summon_status[1] * 255.0 / frames_to_confirm)
            if summon_status[0] is not None
            else 0
        )

        for i, landmarks in enumerate(hand_result.hand_landmarks):
            label = next(
                (k for k, v in hand_map.items() if v == i), "unknown"
            ).capitalize()

            for start_idx, end_idx in HAND_CONNECTIONS:
                p1 = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                p2 = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                cv2.line(frame, p1, p2, (0, line_brightness, 0), 2)

            for idx, lm in enumerate(landmarks):
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (255, 0, 255), -1)
                cv2.putText(
                    frame, str(idx), (cx + 10, cy + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1,
                )

            wrist = landmarks[0]
            cv2.putText(
                frame, label,
                (int(wrist.x * w), int(wrist.y * h) - 15),
                cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2,
            )
            # Leave Out the implementation of the sword for now
            # if hand_map["right"] == i:
            #     # Anchor: midpoint between wrist (0) and middle MCP (9)
            #     point_1_x = (landmarks[0].x + landmarks[17].x) / 2.0
            #     point_1_y = (landmarks[0].y + landmarks[17].y) / 2.0
            #     point_2_x = (landmarks[5].x + landmarks[2].x) / 2.0
            #     point_2_y = (landmarks[5].y + landmarks[2].y) / 2.0

            #     # Angle: point sword from wrist toward middle finger
            #     dx = (point_2_x - point_1_x) * w
            #     dy = (point_2_y - point_1_y) * h
            #     angle = -np.degrees(np.arctan2(dy, dx)) + 135

            #     # Scale: proportional to wrist-to-MCP distance
            #     hand_length = np.hypot(dx, dy)
            #     scale = hand_length / max(self.sword.shape[0], 1) * 8  # tune multiplier
            #     ux, uy = dx / hand_length, dy / hand_length
            #     handle_offset = self.sword.shape[0] * scale * 0.4 # To move the image up/down in relation to his hand
            #     anchor_x = int(point_2_x * w + ux * handle_offset)
            #     anchor_y = int(point_2_y * h + uy * handle_offset)

            #     self._overlay_image(frame, self.sword, (anchor_x, anchor_y), angle, scale)

        if summon_status[0] is not None:
            cv2.putText(
                frame, str(summon_status[1]),
                (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0, 0, 0), 2,
            )

    def _overlay_image(self, frame, overlay, center, angle_deg, scale):
        h, w = overlay.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < 1 or new_h < 1:
            return

        resized = cv2.resize(overlay, (new_w, new_h))

        # Compute the bounding box of the rotated image
        rad = np.radians(angle_deg)
        cos, sin = abs(np.cos(rad)), abs(np.sin(rad))
        canvas_w = int(new_w * cos + new_h * sin)
        canvas_h = int(new_w * sin + new_h * cos)

        # Rotate around the center of the expanded canvas
        M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle_deg, 1.0)
        M[0, 2] += (canvas_w - new_w) / 2  # shift to fit expanded canvas
        M[1, 2] += (canvas_h - new_h) / 2

        rotated = cv2.warpAffine(resized, M, (canvas_w, canvas_h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))

        cx, cy = center
        x = cx - canvas_w // 2
        y = cy - canvas_h // 2

        ox, oy = max(-x, 0), max(-y, 0)
        cx, cy = center
        x = cx - canvas_w // 2
        y = cy - canvas_h // 2

        ox, oy = max(-x, 0), max(-y, 0)
        rotated = rotated[oy:, ox:]
        x, y = max(x, 0), max(y, 0)

        rh, rw = rotated.shape[:2]
        rh = min(rh, frame.shape[0] - y)
        rw = min(rw, frame.shape[1] - x)
        if rh <= 0 or rw <= 0:
            return

        roi = rotated[:rh, :rw]
        alpha = roi[:, :, 3:4] / 255.0
        frame[y:y+rh, x:x+rw] = (
            roi[:, :, :3] * alpha + frame[y:y+rh, x:x+rw] * (1 - alpha)
        ).astype(np.uint8)