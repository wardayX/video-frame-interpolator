
---

#  Video High-FPS Interpolator (RIFE)

A production-ready Python tool that uses **RIFE (Real-Time Intermediate Flow Estimation)** to increase video frame rates using AI. Convert standard 24/30fps videos into ultra-smooth **60fps, 120fps, or even 240fps** slow-motion ready footage.

This project refactors the core RIFE algorithm into a clean, modular repository with CLI support, automated model patching, and Docker integration.

---

##  Features

* **Multi-Factor Interpolation:** Supports **2x**, **4x**, and **8x** frame multiplication.
* **Recursive Inference:** Uses a smart "divide and conquer" strategy to generate high-order frames (e.g., converting 1 frame gap into 7 intermediate frames for 8x mode).
* **Scene Detection:** (Inherited from RIFE) Effectively handles scene cuts to prevent "morphing" artifacts between unrelated shots.
* **Automated Setup:** The included setup script automatically fetches `IFNet_HDv3` weights and patches them for local compatibility.
* **Docker Ready:** Includes a `Dockerfile` for isolated, conflict-free execution on any machine with NVIDIA drivers.

##  Installation

### Option A: Local Python Setup

**Prerequisites:**

1. **NVIDIA GPU** (Required for reasonable speeds).
2. **FFmpeg** installed and added to system PATH.
* *Windows:* `winget install ffmpeg`
* *Ubuntu:* `sudo apt install ffmpeg`
* *Mac:* `brew install ffmpeg`



**Steps:**

```bash
# 1. Clone the repository
git clone https://github.com/wardayX/video-interpolator.git
cd video-interpolator

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Download and Patch Models (Run once)
python setup_models.py

```

### Option B: Docker (Recommended)

Avoid dependency hell by running in a container.

```bash
# Build the image (This automatically downloads the models)
docker build -t rife-interpolator .

# Run the container (Mounts your current folder to /data inside the container)
# Note: --gpus all is required for CUDA support
docker run --gpus all -v $(pwd):/app/data rife-interpolator --input /app/data/myvideo.mp4 --multi 4

```

---

## 🛠️ Usage

### Quick Start

To double the frame rate (e.g., 30fps -> 60fps):

```bash
python main.py --input video.mp4

```

### Advanced Usage

To create **4x** super-smooth video (e.g., 30fps -> 120fps) with higher quality encoding:

```bash
python main.py --input video.mp4 --output smooth_120fps.mp4 --multi 4 --crf 15

```

### CLI Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--input` | *Required* | Path to the source video file (mp4, avi, mov, etc.) |
| `--output` | `output.mp4` | Path for the final processed video. |
| `--multi` | `2` | Frame Multiplier. Options: `2` (Standard), `4` (High), `8` (Extreme). |
| `--crf` | `17` | FFmpeg Constant Rate Factor (Quality). Lower is better. Range 0-51. |

---

##  Project Structure

```text
video-interpolator/
├── main.py              # CLI Entry point
├── setup_models.py      # Downloader & Patcher for RIFE weights
├── Dockerfile           # Container configuration
├── src/                 # Core Logic
│   ├── inference.py     # Recursive interpolation algorithm
│   ├── warplayer.py     # Grid sampling & Optical Flow warping
│   └── loss.py          # Architecture shims
├── models/              # (Generated) Stores .pkl and .py model files
└── notebook.ipynb       # Original prototyping notebook

```

##  How It Works

1. **Flow Estimation:** The AI (IFNet) analyzes two consecutive frames () and predicts the motion (optical flow) between them.
2. **Warping:** It warps  forward and  backward to meet in the middle.
3. **Refinement:** A fusion network fixes artifacts and occlusions (areas visible in one frame but not the other) to generate the middle frame ().
4. **Recursion:** For 4x interpolation, the script takes the new  and runs the process again with  to create , and so on.

##  Credits

* **Original Algorithm:** [Practical-RIFE](https://github.com/hzwer/Practical-RIFE) by Hzwer.
* **Model Weights:** Hosted by [Isi99999](https://www.google.com/search?q=https://huggingface.co/Isi99999).
---

*Created by WardayX for educational purposes.*
