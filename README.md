# Smart Fitness PT 🏋️

**Real-time AI squat form coach** — built with Python, OpenCV, and MediaPipe.

Point your webcam at yourself, and it counts your reps, scores your squat form, and speaks corrections out loud as you lift — all running locally, no internet or cloud processing required.

![Python](https://img.shields.io/badge/python-3.11-blue)
![OpenCV](https://img.shields.io/badge/opencv-computer%20vision-green)
![MediaPipe](https://img.shields.io/badge/mediapipe-pose%20tracking-orange)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)

---

## Demo

https://github.com/siddhanttiwari19/Smart-fitness-PT/assets/demo.mp4

*(GitHub doesn't autoplay videos in READMEs — click the link above to view, or check the repo's release/assets tab.)*

---

## What It Does

| Capability | How |
|---|---|
| **Pose tracking** | MediaPipe Pose (`model_complexity=0`) on every 2nd frame; cached skeleton replayed on skipped frames for smooth visuals at full fps |
| **Rep counting** | 4-state finite state machine — `Standing → Descending → Bottom → Ascending` |
| **Form checks** | Depth (hip below knee), knee-over-toe, knee valgus (inward collapse), back angle (forward lean) |
| **Scoring** | Per-rep score starts at 100, deducts per fault, rolls a running average across the session |
| **Voice coaching** | Real-time spoken corrections via macOS `say` |
| **Live overlay** | Rep counter, phase badge, form score bar, coaching text, sparkline of recent rep scores |
| **Session log** | Appended to `session_log.json` on exit — reps, scores, correction breakdown, duration |

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/siddhanttiwari19/Smart-fitness-PT.git
cd Smart-fitness-PT
```

### 2. Set up a virtual environment (Python 3.11 recommended)

```bash
python3 -m venv venv
source venv/bin/activate       # macOS/Linux
```

> **Note:** MediaPipe's legacy `solutions` API (used here for pose landmarks) isn't reliably available on Python 3.13+. Stick to Python 3.11 for a smooth install.

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Just three packages: `opencv-python`, `mediapipe`, `numpy`.

### 4. Run it

```bash
bash run.sh
```

or directly:

```bash
python3 main.py
```

First run downloads MediaPipe's pose model (~10 MB) — cached after that.

---

## Camera Setup

- Stand **1.5–2 m** from the camera
- Go **side-on** for accurate depth and knee-travel detection
- Keep your **full body in frame**, head to feet
- Stand still for ~1 second to let calibration finish — then start lifting

---

## Controls

| Key | Action |
|---|---|
| `q` / `ESC` | Quit and save session to `session_log.json` |
| `p` | Pause / resume |
| `r` | Reset reps, scores, and calibration |
| `v` | Toggle voice feedback |
| `d` | Toggle debug overlay (raw back/knee angles) |

---

## Project Structure

```
Smart-fitness-PT/
├── main.py              # Entry point — OpenCV capture + render loop
├── config.py             # Thresholds, cooldowns, coaching messages
├── pose_detector.py      # MediaPipe wrapper, landmark smoothing, cached draw
├── rep_counter.py         # Squat FSM + calibration logic
├── squat_analyzer.py      # Depth / knee-toe / valgus / back-angle checks + scoring
├── voice_coach.py          # Queue-based subprocess TTS worker
├── session_logger.py       # Per-session JSON logging
├── utils.py                 # Math helpers (angles, moving average, colours)
├── requirements.txt
├── run.sh                    # Launcher script
└── session_log.json          # Appended on every run (gitignored)
```

---

## Tuning

All thresholds live in `config.py`:

| Setting | Controls |
|---|---|
| `DEPTH_HIP_BELOW_KNEE_PX` | How strict depth-below-parallel detection is |
| `KNEE_TOE_SLACK_FRACTION` | How much forward knee travel is allowed |
| `VALGUS_SLACK_FRACTION` | How much inward knee collapse is allowed |
| `BACK_LEAN_MAX_FROM_VERTICAL` | Max torso lean before "chest up" cue fires |
| `VOICE_COOLDOWN_SEC` | Minimum seconds before repeating a voice cue |
| `CONSECUTIVE_FRAMES` | Bad frames in a row needed to trigger an alert |
| `ISSUE_MESSAGES` | Edit coaching lines directly |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Cannot open camera" | Close apps using the webcam (Zoom, Photo Booth), grant camera permission in **System Settings → Privacy → Camera** |
| Skeleton lags / low fps | Raise `INFERENCE_EVERY` from 2 to 3 in `config.py`, or lower `FRAME_WIDTH`/`FRAME_HEIGHT` |
| No voice output | `say` should work by default on macOS — check system volume. On Linux, install `espeak`: `sudo apt install espeak` |
| Reps mis-counting | Hold calibration stance longer, or tune `DESCENT_THRESHOLD` / `COMPLETE_THRESHOLD` in `config.py` |
| "No pose detected" | Step back so your full body is visible; avoid cluttered backgrounds |

---

## Requirements

- Python 3.11+
- Webcam
- macOS (voice coaching relies on the built-in `say` command)

---

## About

Built by [Siddhant Tiwari](https://github.com/siddhanttiwari19) — CS undergrad (AI/ML) exploring real-time computer vision applications.

This is a personal project — issues and forks are welcome for your own experimentation, but this repo isn't currently accepting external contributions.
