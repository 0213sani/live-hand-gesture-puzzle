1n# 🧩 LIVE PUZZLE — Complete Setup Guide

---

## What you're building

A hand-gesture controlled live puzzle game using your webcam.  
No mouse. No keyboard during gameplay. Just your hands.

**Flow:**
1. **Capture** — Hold both hands up, frame a rectangle with your index fingers. Pinch both hands → snap a photo.
2. **Solve** — That photo gets sliced into a 3×3 scrambled grid. Pinch a tile to pick it up, drag to another slot, release to swap. Timer is running.
3. **Win** — COMPLETE screen shows your time. Type your name, hit Enter → saved to leaderboard.

---

## Step 1 — Install Python

You need Python 3.9, 3.10, or 3.11. **Do NOT use 3.12+** (MediaPipe doesn't fully support it yet).

1. Go to https://python.org/downloads
2. Download Python 3.11.x
3. During install → **check "Add Python to PATH"** (very important!)
4. Open Terminal / Command Prompt and run:
   ```
   python --version
   ```
   You should see `Python 3.11.x`

---

## Step 2 — Install the libraries

Open Terminal and run these one by one:

```bash
pip install opencv-python
pip install mediapipe
pip install numpy
```

To verify everything installed correctly:
```bash
python -c "import cv2, mediapipe, numpy; print('All good!')"
```

If you see `All good!` — you're set.

---

## Step 3 — Run the game

1. Put `live_puzzle.py` anywhere on your computer (e.g., Desktop)
2. Open Terminal in that folder
3. Run:
   ```bash
   python live_puzzle.py
   ```
4. A window called **LIVE PUZZLE** will open with your webcam feed.

---

## Step 4 — How to play

### Phase 1: Capture

- Hold up **both hands** in front of the camera
- Your index fingertips define the corners of the capture rectangle
- Spread them to make the box bigger
- **Pinch both hands** (touch thumb to index finger on BOTH hands) → photo snaps

### Phase 2: Solve

- **Pinch** over a tile → picks it up (it floats with your hand)
- Move hand over another tile slot → **release pinch** → tiles swap
- **Make a fist** → scrambles the puzzle again (if you want a reset)
- Timer counts up from 0

### Win screen

- Type your name using keyboard
- Press **Enter** to save to leaderboard
- Press **ESC** to skip
- Leaderboard is saved in `leaderboard.json` next to the script

### To quit anytime
Press `Q`

---

## Common issues & fixes

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: mediapipe` | Run `pip install mediapipe` again |
| Black screen / no webcam | Change `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` in the code |
| Pinch too sensitive / not sensitive enough | Change `PINCH_THRESH = 0.055` (higher = easier to trigger, lower = harder) |
| Hands not detected | Make sure your hands are well-lit and clearly visible |
| Game window too large for screen | Change `1280` and `720` in the cap.set lines to `960` and `540` |
| Tiles look stretched | Make sure you're capturing a decently sized rectangle |

---

## Understanding the code (quick map)

```
live_puzzle.py
│
├── CONSTANTS          — tweak PINCH_THRESH, TILE_PX, GRID here
├── LEADERBOARD        — saves/loads leaderboard.json
├── GESTURE HELPERS    — pinch detection, fist detection
├── PUZZLE HELPERS     — slicing image, scrambling, checking if solved
├── DRAWING            — draws grid, held tile, overlays, text
└── main()
    ├── STATE: CAPTURE  — two-hand frame + pinch to snap
    ├── STATE: SOLVE    — pinch drag swap + fist reset + timer
    └── STATE: COMPLETE — win screen + name input + leaderboard
```

---

## Tweaks you can make

- **Bigger grid** → change `GRID = 3` to `GRID = 4` for a 4×4 puzzle (harder!)
- **Smaller tiles** → change `TILE_PX = 130` to `100`
- **Pinch sensitivity** → `PINCH_THRESH`: try values between `0.04` and `0.08`
- **Leaderboard entries shown** → change `[:5]` to `[:10]` in the COMPLETE section

---

## Week-by-week plan

| Week | Goal |
|---|---|
| Week 1 | Install everything. Run the game. Get comfortable with the gestures. |
| Week 2 | Customize — colors, fonts, grid size. Add your event's branding. |
| Week 3 | Stress test. Try it in different lighting. Fix any bugs. |
| Week 4 | Event setup. Put it on a laptop with a good webcam, big monitor if possible. |

---

## Tech stack recap

- **OpenCV** — webcam feed, image display, drawing
- **MediaPipe** — real-time hand tracking (21 landmark points per hand)
- **NumPy** — image manipulation (slicing, overlays)
- **json** — leaderboard persistence

---

Good luck! The hardest part is already done. 🎉
