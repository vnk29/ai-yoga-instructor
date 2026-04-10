# Plank Posture AI Coach

A local macOS desktop app that watches you hold a plank and gives real-time spoken corrections when your form breaks down. Everything runs on-device: MediaPipe Pose for skeleton tracking, and Gradium TTS for voice.


https://github.com/user-attachments/assets/ad97f2f0-e1ac-4f70-8333-96f31025cae1


## How it works

1. Captures frames from your camera via OpenCV.
2. Runs **MediaPipe Pose Landmarker** to detect 33 body landmarks each frame.
3. Computes three geometric checks against the detected skeleton:
   - **Hip deviation** – how far your hips stray above/below the ideal shoulder-to-ankle line.
   - **Head rise** – whether your nose is at or below shoulder level (neutral), raised (looking up), or severely drooped.
   - **Body angle** – whether the body is roughly horizontal (in plank) or not.
4. Requires a configurable number of consecutive bad frames before triggering an alert (avoids one-frame noise).
5. Plays a spoken correction via **Gradium TTS**, then enforces a cooldown so the same note isn't repeated immediately.
6. Cycles through all messages for a given issue before repeating.
7. Saves a session summary to `session_log.json` on exit.

## Camera setup (important)

For best results, place the camera **to your side at roughly torso height**, so your body is horizontal in the frame.

```
     [Camera] ──────────────────────────────────────────>
                HEAD ── SHOULDER ── HIP ── KNEE ── ANKLE
```

- If you use a laptop camera, prop the laptop on its side or use a stand next to your mat.
- Front-facing cameras will not work well because the hip-deviation check relies on a side view.

## Posture checks

| Issue | Trigger | What you'll hear |
|---|---|---|
| **Hips too high** | Hip is above the shoulder–ankle line (pike) | "Bring those hips down…" |
| **Hips sagging** | Hip is below the shoulder–ankle line | "Engage your core and lift them up…" |
| **Head up** | Nose is significantly above shoulder level | "Lower your head, keep your neck neutral…" |
| **Head down** | Nose is significantly below shoulder level | "Lift your head slightly…" |
| **Not in plank** | Body angle > 30° from horizontal | "Get into plank position…" |
| **Good form** | No issues detected for a sustained period | "Great form! Keep it up." |

## Sensitivity presets

Head thresholds are split by direction: craning the neck up is always wrong (tight threshold), while looking slightly down is normal in a plank (lenient threshold).

| Preset | Hip threshold | Head-up threshold | Head-down threshold | Consecutive frames | Cooldown |
|---|---|---|---|---|---|
| Low    | 0.04  | 0.05  | 0.18 | 12 | 6 s   |
| Medium | 0.025 | 0.035 | 0.14 | 8  | 4 s   |
| High   | 0.015 | 0.02  | 0.10 | 4  | 2.5 s |

All thresholds are normalised by the shoulder-to-ankle body length, so they scale with how far you are from the camera.

## Session log

On exit, a summary is appended to `session_log.json`:

```json
{
  "date": "2026-04-10T09:30:00",
  "duration_seconds": 120.5,
  "sensitivity": "medium",
  "total_corrections": 14,
  "corrections_by_type": {
    "hip_low": 9,
    "head_up": 5
  },
  "compliments": 2
}
```

## Running locally

### Prerequisites

- macOS (uses `afplay` for audio)
- Python 3.10+
- A Gradium API key — get one at [gradium.ai](https://gradium.ai)
- A camera positioned to your side at torso height (see [Camera setup](#camera-setup-important))

### 1. Clone the repo

```bash
git clone https://github.com/Haimantika/plank-posture.git
cd plank-posture
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Gradium API key

```bash
export GRADIUM_API_KEY=your-key-here
```

Add this line to your `~/.zshrc` (or `~/.bash_profile`) to avoid repeating it each session.

### 5. Run

```bash
bash run.sh
```

`run.sh` handles SSL certificate configuration for Gradium automatically. The MediaPipe pose model (~7 MB) will be downloaded on the first run.

Grant camera access if macOS prompts you (**System Settings → Privacy & Security → Camera**).

> **Note:** The app will refuse to start if `GRADIUM_API_KEY` is not set.

## Keyboard controls (preview window must be focused)

| Key | Action |
|---|---|
| `q` / `ESC` | Quit |
| `d` | Toggle skeleton + debug overlay |
| `p` | Pause / resume |
| `1` | Sensitivity: low |
| `2` | Sensitivity: medium (default) |
| `3` | Sensitivity: high |

## File structure

```
main.py              – camera loop, key handling, visual overlay
pose_detector.py     – MediaPipe Pose wrapper and landmark helpers
posture_analyzer.py  – geometric plank checks, issue list construction
voice_coach.py       – background thread, per-issue cooldowns, Gradium TTS
session_logger.py    – session stats, JSON persistence
config.py            – sensitivity presets, thresholds, voice messages
run.sh               – launcher: sets SSL_CERT_FILE for Gradium, then runs main.py
```
