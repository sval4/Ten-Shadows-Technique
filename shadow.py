import os

import cv2
import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (5, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


# ---------------------------------------------------------------------------
# Base Shadow class
# ---------------------------------------------------------------------------

class Shadow:
    """
    Abstract base class for a Ten Shadows spirit.

    Subclasses must implement `check_summon()` and supply a name, frame
    folder, and any other shadow-specific configuration.

    Attributes:
        name (str): Display name of the shadow.
        frame_folder (str): Path to the folder of animation PNGs.
        frames_to_confirm (int): Consecutive matching frames required to trigger a summon.
        training_mode (bool): When True, incoming hand tensors are saved rather than matched.
    """

    def __init__(self,name="Unknown",training_mode=False,frame_folder="",x=0,y=0,frame_size=(745,400),num_animations_loop=10,TENSOR_FILE="",MAX_SAMPLES=100,LOSS_THRESHOLD=0.0025) -> None:
        self.name=name
        self.frame_folder=frame_folder
        self.frame_size=frame_size
        self.num_animations_loop=num_animations_loop
        self.x = x
        self.y = y
        self.TENSOR_FILE=TENSOR_FILE
        self.MAX_SAMPLES=MAX_SAMPLES
        self.LOSS_THRESHOLD=LOSS_THRESHOLD
        self._criterion = nn.MSELoss()
        self._stored_tensors: list[torch.Tensor] = []
        self._mean_tensor: torch.Tensor | None = None
        self.animation_frames: list[np.ndarray] = self._load_animation_frames(training_mode)
        self._load_mean_tensor()

    # ------------------------------------------------------------------
    # Animation helpers
    # ------------------------------------------------------------------

    def _load_animation_frames(self, training_mode) -> list[np.ndarray]:
        """Load and resize all PNG frames from `self.frame_folder`."""
        folder = self.frame_folder
        if not os.path.isdir(folder):
            print(f"[{self.name}] Warning: frame folder '{folder}' not found.")
            return []

        files = sorted([f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))])

        if not files:
            print(f"[{self.name}] Warning: No files found in '{folder}'.")
            return []

        loops = 1 if training_mode else self.num_animations_loop
        frames: list[np.ndarray] = []

        for _ in range(loops):
            for filename in files:
                path = os.path.join(folder, filename)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                
                if img is not None:
                    img = cv2.resize(img, self.frame_size)
                    frames.append(img)
                else:
                    # This catches non-image files like .txt or .DS_Store
                    print(f"[{self.name}] Warning: could not load '{path}' as an image.")

        return frames

    @staticmethod
    def overlay_transparent(
        background: np.ndarray,
        overlay: np.ndarray,
        x: int,
        y: int,
    ) -> np.ndarray:
        """Alpha-blend a 4-channel overlay onto a 3-channel background."""
        bg_h, bg_w = background.shape[:2]
        h = min(overlay.shape[0], bg_h - y)
        w = min(overlay.shape[1], bg_w - x)

        if h <= 0 or w <= 0 or x >= bg_w or y >= bg_h:
            return background

        overlay_rgb = overlay[:h, :w, :3]
        alpha = overlay[:h, :w, 3:4] / 255.0  # shape (h, w, 1) for broadcasting

        roi = background[y : y + h, x : x + w]
        background[y : y + h, x : x + w] = (1.0 - alpha) * roi + alpha * overlay_rgb
        return background

    def render_frame(self, frame: np.ndarray, summon_status, anim_idx) -> np.ndarray:
        """
        Built the current animation sprite onto `frame` (in-place) and
        advance the animation index.  Returns the modified frame.
        Resets summon status when the animation completes.
        """
        if not self.animation_frames:
            return frame

        sprite = self.animation_frames[anim_idx]
        frame = self.overlay_transparent(frame, sprite, x=self.x, y=self.y)
        anim_idx = (anim_idx + 1) % len(self.animation_frames)

        if anim_idx == 0:
            summon_status = (None, 1)

        return frame, summon_status, anim_idx

    # ------------------------------------------------------------------
    # Summon logic — must be implemented by each shadow
    # ------------------------------------------------------------------

    def _load_mean_tensor(self) -> None:
        """Compute and cache the mean pose tensor from the saved .pt file."""
        if os.path.exists(self.TENSOR_FILE):
            data: list[torch.Tensor] = torch.load(self.TENSOR_FILE)
            stacked = torch.stack(data)
            self._mean_tensor = torch.mean(stacked, dim=0)
            print(f"[{self.name}] Mean tensor loaded from '{self.TENSOR_FILE}'.")

    def _save_training_tensor(self, tensor: torch.Tensor) -> None:
        """Append a new tensor to the training set and persist to disk."""
        if not self._stored_tensors and os.path.exists(self.TENSOR_FILE):
            os.remove(self.TENSOR_FILE)
            print(f"[{self.name}] Cleared existing '{self.TENSOR_FILE}'.")

        if len(self._stored_tensors) < self.MAX_SAMPLES:
            self._stored_tensors.append(tensor)
            torch.save(self._stored_tensors, self.TENSOR_FILE)
            print(f"[{self.name}] Stored tensor {len(self._stored_tensors)}/{self.MAX_SAMPLES}.")
        else:
            print(f"[{self.name}] Storage full — {self.MAX_SAMPLES} tensors collected.")
    
    def check_summon(self, left_norm, right_norm, combined, training_mode, summon_status, frames_to_confirm) -> bool:
        """
        Returns True when both hands hold the Nue pose for enough consecutive
        frames.  In training mode, saves the current pose tensor instead.
        """
        if training_mode:
            print(f"\n[{self.name}] Training — Left:\n{left_norm}\nRight:\n{right_norm}")
            self._save_training_tensor(combined)
            return True, (None, 0)

        # Inference path
        if self._mean_tensor is None:
            return False, (None, 0)
        if combined.shape != self._mean_tensor.shape:
            return False, (None, 0)

        loss = self._criterion(combined, self._mean_tensor)
        print(f"[{self.name}] Loss: {loss.item():.6f}")

        prev_name, prev_count = summon_status

        if loss < self.LOSS_THRESHOLD:
            count = prev_count if prev_name == self.name else 0
            count = count % frames_to_confirm + 1
            summon_status = (self.name, count)
        else:
            count = prev_count if prev_name is None else 0
            count = count % frames_to_confirm + 1
            summon_status = (None, count)

        print(f"[{self.name}] Summon status: {summon_status}")
        matched = summon_status[0] == self.name
        confirmed = summon_status[1] >= frames_to_confirm
        return matched and confirmed, summon_status
