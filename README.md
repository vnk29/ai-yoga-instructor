# AI Yoga Instructor using Computer Vision

A professional, fully local Windows-compatible Streamlit application that provides real-time yoga posture analysis, joint-angle alignment checks, and offline vocal corrections using computer vision.

Everything runs completely on-device for maximum privacy: **MediaPipe Pose** for skeleton tracking, and **pyttsx3** for offline threaded text-to-speech coaching feedback.

---

## 🧘 Key Features

1. **Live Yoga Mode:**
   - Real-time webcam-based pose tracking and skeleton visualization.
   - Intelligent detection of 5 major postures: **Tree Pose**, **Warrior II Pose**, **Cobra Pose**, **T Pose**, and **Mountain Pose**.
   - Displays real-time pose name, alignment accuracy %, frame stability %, and coaching tips.
2. **AI Voice Coaching:**
   - Multi-threaded offline vocal corrections via `pyttsx3`.
   - Advanced cooldown logic to prevent repetitive speech fatigue.
3. **Advanced Posture Analysis:**
   - High-precision joint-angle calculations (Spine, Hips, Shoulders, Knees, Arms) using NumPy geometry.
4. **Image Upload Mode:**
   - Upload any static yoga posture image (`jpg`/`png`) to analyze alignment and generate a visual diagnostics report.
5. **Yoga Recommendation Mode:**
   - Choose fitness targets (Stress, Back Pain, Flexibility, Weight Loss) and get custom, tailored yoga routines with biomechanical explanations.
6. **Session Logging:**
   - Tracks posture accuracy and duration metrics, automatically logging summaries locally to `session_log.json`.

---

## 💻 Tech Stack & Windows Compatibility

- **Python 3.10+**
- **MediaPipe:** For high-fidelity pose keypoint tracking.
- **OpenCV:** For video capture, processing, and skeleton drawing.
- **NumPy:** For vector calculations.
- **Streamlit:** For a dark modern glassmorphism UI.
- **pyttsx3:** Offline cross-platform voice synthesis (safe COM initialization for Windows threads).

---

## 🚀 Running on Windows

### 1. Set Up Virtual Environment

Open your terminal or PowerShell inside the project directory:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install Dependencies

Install the updated libraries directly:

```powershell
pip install -r requirements.txt
```

### 3. Launch the Streamlit App

Run the application locally:

```powershell
streamlit run app.py
```

The app will open automatically in your default browser at `http://localhost:8501`.

---

## 📐 Supported Postures & Analysis

| Posture | Checked Joints | Focus Areas |
| :--- | :--- | :--- |
| **Tree Pose** | Standing knee, bent knee, standing hip, arms angle | Balance, standing leg straightness, leg folding |
| **Warrior II** | Front knee, back knee, arms alignment, vertical torso | Hip openings, horizontal arms, back leg tension |
| **Cobra Pose** | Elbow angle, back arch hip angle, shoulders alignment | Spinal extension, relaxed shoulders, gentle lift |
| **T Pose** | Arm horizontal extension, vertical spine alignment | Core balance, symmetrical horizontal reach |
| **Mountain Pose** | Perfect vertical spine, arms active at sides | Core posture, pelvis alignment, neutral joints |

---

## 📂 Project Directory Structure

```text
AI-Yoga-Instructor/
│
├── app.py                      # Main Streamlit web application & pages routing
├── yoga_analyzer.py            # NumPy angle computations & posture diagnostics logic
├── pose_detector.py            # MediaPipe Tasks PoseLandmarker wrapper
├── voice_coach.py              # Thread-safe offline pyttsx3 vocal alert engine
├── recommendation_engine.py    # Rule-based wellness pose recommendations
├── session_logger.py           # Session metrics collector & JSON storage
├── config.py                   # Pose angle thresholds, voice logs, and parameters
├── utils/
│   └── ui_components.py        # Glassmorphic layout custom CSS & components
├── uploads/                    # Temporary storage for uploaded images
├── assets/                     # Custom assets directory
└── requirements.txt            # Python dependencies configuration
```

---

## � Dataset Integration (Optional)

The application supports integration with the Kaggle yoga pose dataset for **improved pose detection accuracy** through dataset-derived statistics.

### Setup Instructions

1. **Download the Dataset:**
   - Download the yoga pose dataset from Kaggle
   - Extract it to the project directory as `dataset/` or any convenient location

2. **Run the Dataset Setup Script:**

```powershell
python setup_dataset.py                           # Auto-detects dataset
# OR
python setup_dataset.py --dataset-dir "C:\path\to\dataset"
```

This script will:
- Scan all pose images in the dataset
- Extract MediaPipe landmarks using multi-stage preprocessing
- Compute joint angle statistics per pose class
- Generate `data/pose_stats.json` with confidence thresholds

3. **Expected Dataset Structure:**

```text
dataset/
├── train/
│   ├── plank/
│   ├── tree_pose/
│   ├── warrior_2/
│   ├── goddess_pose/
│   └── downward_dog/
└── test/
    └── (same structure as train/)
```

### Benefits

✓ **Improved Accuracy:** Pose detection uses dataset-derived angle ranges and visibility thresholds
✓ **Adaptive Detection:** Confidence thresholds auto-tune per pose class
✓ **Better Upload Mode:** Enhanced preprocessing helps detect poses in sketches, illustrations, and low-quality photos

---

## �🔒 Privacy & Safety

All pose estimations, camera frames, and vocal guides are computed locally on your CPU/GPU. No images or stream frames are sent to external web servers, keeping your workout fully private and secure.
