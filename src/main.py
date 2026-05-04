"""
ten_shadows.py

A class-based implementation of the Ten Shadows hand-gesture summoning system.
Uses MediaPipe for hand/face detection and OpenCV for rendering.

To add a new shadow:
    1. Subclass `Shadow` and implement `check_summon()`.
    2. Register it in `ShadowSummonerApp.SHADOWS`.
"""
import cv2
import mediapipe as mp
import numpy as np
import pyvirtualcam
import torch
from shadow import Shadow
from handDetector import HandDetector
from faceDetector import FaceDetector
TRAIN_MODE = False

# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class ShadowSummonerApp:
    """
    Orchestrates webcam capture, face/hand detection, shadow matching, and
    output to a virtual camera (for use with OBS or similar).

    Configuration:
        active_shadow:    Key from `SHADOWS` dict for the currently active spirit.
        hand_model_path:  Path to the MediaPipe hand landmark model.
        face_model_path:  Path to the MediaPipe face landmark model.
        camera_index:     Index of the capture device.
        mirrored:         Whether to flip the webcam feed horizontally.
        show_preview:     Whether to open an OpenCV preview window.
        landmark_visual:  Whether to draw hand skeleton overlays.
        virtual_cam_w/h:  Output resolution for the virtual camera.
        virtual_cam_fps:  Output framerate for the virtual camera.
    """
    NUE_CONFIG = {
        "name": "Nue",
        "frame_folder": "../nue_frames",
        "x":0,
        "y":0,
        "frame_size": (745,400),
        "num_animations_loop": 10,
        "TENSOR_FILE": "../tensors/nue_tensors.pt",
        "MAX_SAMPLES": 50,
        "LOSS_THRESHOLD": 0.01
    }
    MAHORAGA_CONFIG = {
        "name": "Mahoraga",
        "frame_folder": "../mahoraga_frames",
        "x":0,
        "y":0,
        "frame_size": (800,800),
        "num_animations_loop": 1,
        "TENSOR_FILE": "../tensors/mahoraga_tensors.pt",
        "MAX_SAMPLES": 50,
        "LOSS_THRESHOLD": 0.1
    }
    # Registry of all available shadows.
    # To add a new shadow, instantiate it here and give it a key.
    # Right now need to place the shadow in training at the top of the list
    SHADOWS: list[Shadow] = [
        Shadow(**MAHORAGA_CONFIG),
        Shadow(**NUE_CONFIG)
    ]

    def __init__(
        self,
        frames_to_confirm: int = 10,
        hand_model_path: str = "../tasks/hand_landmarker.task",
        face_model_path: str = "../tasks/face_landmarker.task",
        training_mode: bool = False,
        camera_index: int = 0,
        mirrored: bool = True,
        show_preview: bool = True,
        landmark_visual: bool = True,
        virtual_cam_w: int = 1280,
        virtual_cam_h: int = 720,
        virtual_cam_fps: int = 30,
    ) -> None:
        self.summon_status = (None,0)
        self.training_mode = training_mode
        self.show_preview = show_preview
        self.landmark_visual = landmark_visual
        self.virtual_cam_w = virtual_cam_w
        self.virtual_cam_h = virtual_cam_h
        self.virtual_cam_fps = virtual_cam_fps
        self.frames_to_confirm = frames_to_confirm
        self.currShadow = None
        self.anim_idx = 0

        # Subsystem initialisation
        self.hand_detector = HandDetector(hand_model_path, mirrored=mirrored)
        self.face_detector = FaceDetector(face_model_path)

        # Webcam
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, virtual_cam_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, virtual_cam_h)
        self.mirrored = mirrored

        # Animation playback state
        self._animating: bool = False

    # ------------------------------------------------------------------
    # Frame processing
    # ------------------------------------------------------------------

    def _process_frame(self, frame: np.ndarray, virtual_cam) -> np.ndarray:
        """Run one complete detection + render cycle on `frame`."""
        if self.mirrored:
            frame = cv2.flip(frame, 1)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
        )
        timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))

        hand_result = self.hand_detector.detect(mp_image, timestamp_ms)
        face_result = self.face_detector.detect(mp_image, timestamp_ms)

        # Mouth-open detection → toggle training mode
        if self.face_detector.update(face_result, frame):
            self.training_mode = not self.training_mode
            print(f"[App] Training mode: {self.training_mode}")

        # Build hand map and optionally draw landmarks
        hand_map = self.hand_detector.build_hand_map(hand_result)
        if self.landmark_visual and hand_result.hand_landmarks:
            self.hand_detector.draw_landmarks(
                frame, hand_result, hand_map,
                self.summon_status,
                self.frames_to_confirm,
            )

        # Summon check + animation
        if not self._animating:
            if hand_map["left"] == -1 or hand_map["right"] == -1:
                return frame
            if len(hand_result.hand_landmarks) < 2:
                return frame

            left_raw = torch.tensor(
                [[lm.x, lm.y] for lm in hand_result.hand_landmarks[hand_map["left"]]],
                dtype=torch.float64,
            )
            right_raw = torch.tensor(
                [[lm.x, lm.y] for lm in hand_result.hand_landmarks[hand_map["right"]]],
                dtype=torch.float64,
            )
            left_norm, right_norm = self._normalize_dual_hands(left_raw, right_raw)
            combined = torch.cat((left_norm, right_norm), dim=0)
            matchedStatus = self.summon_status
            allNone = True
            for shadow in self.SHADOWS:
                triggered, matchedStatus = shadow.check_summon(left_norm, right_norm, combined, self.training_mode, self.summon_status, self.frames_to_confirm)
                if matchedStatus[0] != None:
                    self.summon_status = matchedStatus
                    allNone = False
                if triggered:
                    self._animating = True
                    self.currShadow=shadow
                    allNone = False
                    break
            if allNone:
                self.summon_status = (None, 1)
        if self._animating:
            frame, self.summon_status, self.anim_idx = self.currShadow.render_frame(frame, self.summon_status, self.anim_idx, self.training_mode)
            # render_frame resets summon_status when the loop ends
            if self.anim_idx == 0:
                self.currShadow = None
                self._animating = False

        return frame
    
    def _normalize_dual_hands(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Centre both hands around their shared midpoint and scale by the
        average palm size so the representation is camera-distance invariant.
        """
        midpoint = (left[0] + right[0]) / 2.0
        left_c = left - midpoint
        right_c = right - midpoint

        scale_l = torch.linalg.norm(left[0] - left[12])
        scale_r = torch.linalg.norm(right[0] - right[12])
        scale = torch.mean(torch.stack([scale_l, scale_r]))
        scale = max(scale.item(), 1e-8)

        left_n = left_c / scale
        right_n = right_c / scale

        # Canonicalize: left hand wrist should always be at negative x.
        # If it's positive, the frame is in the opposite orientation — flip x.
        if left_n[0, 0] > 0:
            left_n[:, 0] = -left_n[:, 0]
            right_n[:, 0] = -right_n[:, 0]

        return left_n, right_n
    
    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the main capture/render/output loop. Press 'q' to quit."""
        print(f"[App] Starting")
        print("[App] Press 'q' in the preview window to quit.")

        try:
            with pyvirtualcam.Camera(
                width=self.virtual_cam_w,
                height=self.virtual_cam_h,
                fps=self.virtual_cam_fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            ) as cam:
                while self.cap.isOpened():
                    success, frame = self.cap.read()
                    if not success:
                        print("[App] Failed to read from webcam.")
                        break

                    frame = self._process_frame(frame, cam)
                    cam.send(frame)
                    cam.sleep_until_next_frame()

                    if self.show_preview:
                        cv2.imshow("Ten Shadows Preview", frame)

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        except Exception as exc:
            print(f"[App] Error: {exc}")
            raise
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("[App] Shutdown complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = ShadowSummonerApp(
        hand_model_path="../tasks/hand_landmarker.task",
        face_model_path="../tasks/face_landmarker.task",
        training_mode=TRAIN_MODE,
        mirrored=True,
        show_preview=True,
        landmark_visual=True,
        frames_to_confirm=10
    )
    app.run()