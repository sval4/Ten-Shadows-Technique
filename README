# Ten Shadows Summoning System

The **Ten Shadows Summoning System** is a class-based computer vision implementation of the technique from **Jujutsu Kaisen (JJK)**. This application uses MediaPipe and PyTorch to recognize specific hand gestures in real-time, allowing users to "summon" shikigami as animated overlays on a virtual camera feed.

New users should refer to the image below for the specific hand gestures required for each shadow:

![Jujutsu Kaisen Ten Shadows Hand Gestures](https://www.wikihow.com/images/thumb/3/3e/Megumi-Hand-Signs-Summary.jpg/v4-460px-Megumi-Hand-Signs-Summary.jpg)

---

## Technical Overview

The system captures webcam input and processes it through a detection pipeline to identify hand landmarks. By comparing real-time hand formations against pre-trained tensors, the application triggers high-quality, alpha-blended animations.

### Key Features
*   **Dual-Hand Normalization**: Ensures detection is invariant to camera distance and hand placement by centering landmarks around a shared midpoint and scaling by palm size.
*   **Gestural Training Toggle**: Users can toggle **Training Mode** without a keyboard by holding their mouth open for a set duration.
*   **Virtual Camera Integration**: Outputs the processed frame directly to a virtual camera for use in OBS, Discord, or Zoom.
*   **MSE-Based Recognition**: Uses Mean Squared Error (MSE) to calculate the "loss" between the user's current pose and the stored shadow pose.

---

## Installation and Setup

### Prerequisites
*   **Python 3.9+**
*   **PyTorch** and **OpenCV**
*   **MediaPipe**
*   **pyvirtualcam** (requires a virtual camera driver like OBS Virtual Cam)

### File Structure
To run the system, ensure your directory is organized as follows:
*   `tasks/`: Contains `hand_landmarker.task` and `face_landmarker.task`.
*   `nue_frames/`: Contains the PNG animation sequence for the "Nue" shadow.
*   `shadow.py`, `handDetector.py`, `faceDetector.py`: Component modules.
*   `ten_shadows.py`: The main application script.

---

## How it Works

### 1. Training Mode
To teach the system a new shadow gesture:
1.  Look into the camera and **hold your mouth open** for 3 seconds.
2.  Once the "Training Mode" overlay appears, perform the desired hand gesture.
3.  The system will collect 50 samples and save them as a `.pt` tensor file.
4.  Hold your mouth open again for 3 seconds to exit training and save the mean pose.

### 2. Summoning (Inference)
When the system is not in training mode, it continuously monitors for a match:
*   It calculates the difference between your hands and the **mean_tensor** of the active shadow.
*   If the **Loss** drops below the threshold (e.g., 0.01) for 10 consecutive frames, the summoning animation triggers.

---

## Project Subsystems

| Module | Responsibility |
| :--- | :--- |
| **ShadowSummonerApp** | Orchestrates webcam capture, detection cycles, and virtual camera output. |
| **Shadow Class** | Handles animation loading, alpha-blending, and tensor storage/comparison. |
| **HandDetector** | Wraps MediaPipe HandLandmarker to build hand maps and draw skeleton overlays. |
| **FaceDetector** | Uses MediaPipe FaceLandmarker to track mouth state for training mode toggles. |

---

## Roadmap

*   **Mahoraga Implementation**: Work is currently underway to implement the Eight-Handled Sword Divergent Sila Divine General, Mahoraga.