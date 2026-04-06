import cv2
import mediapipe as mp
import numpy as np
import time
import random
import json
import os

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
WINDOW_NAME    = "LIVE PUZZLE"
GRID           = 3                # 3x3 puzzle
TILE_PX        = 130              # each tile = 130x130 pixels
PINCH_THRESH   = 0.055            # tweak if pinch is too sensitive / not sensitive enough
FIST_THRESH    = 0.10             # how closed fingers need to be for a fist
GREEN          = (50, 255, 50)
WHITE          = (255, 255, 255)
CYAN           = (0, 255, 255)
GRAY           = (160, 160, 160)
DARK           = (20, 20, 20)
LB_FILE        = "leaderboard.json"

# ─────────────────────────────────────────────────────────────────────────────
# LEADERBOARD HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def load_lb():
    if os.path.exists(LB_FILE):
        with open(LB_FILE) as f:
            return json.load(f)
    return []

def save_leaderboard(leaderboard):
    with open(LB_FILE, "w") as f:
        json.dump(lb, f)

# ─────────────────────────────────────────────────────────────────────────────
# GESTURE HELPERS  (all use MediaPipe normalized coords 0-1)
# ─────────────────────────────────────────────────────────────────────────────
def lm_px(lm, w, h):
    """Convert a single landmark to pixel coords."""
    return int(lm.x * w), int(lm.y * h)

def pinch_dist(hand):
    """Euclidean distance between thumb-tip (4) and index-tip (8), normalized."""
    t, i = hand.landmark[4], hand.landmark[8]
    return ((t.x - i.x)**2 + (t.y - i.y)**2) ** 0.5

def is_pinching(hand):
    return pinch_dist(hand) < PINCH_THRESH

def pinch_mid(hand, w, h):
    """Pixel midpoint of the pinch gesture."""
    t, i = hand.landmark[4], hand.landmark[8]
    return int((t.x + i.x) / 2 * w), int((t.y + i.y) / 2 * h)

def is_fist(hand):
    """True when all four fingers are curled (tips below their pip joints)."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    return all(hand.landmark[t].y > hand.landmark[p].y for t, p in zip(tips, pips))

# ─────────────────────────────────────────────────────────────────────────────
# PUZZLE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def slice_image(img):
    """Slice image into GRID×GRID tiles. Returns list of tile images."""
    sz = GRID * TILE_PX
    img = cv2.resize(img, (sz, sz))
    tiles = []
    for r in range(GRID):
        for c in range(GRID):
            tiles.append(img[r*TILE_PX:(r+1)*TILE_PX, c*TILE_PX:(c+1)*TILE_PX].copy())
    return tiles

def new_scramble():
    order = list(range(GRID * GRID))
    random.shuffle(order)
    while order == list(range(GRID * GRID)):
        random.shuffle(order)
    return order

def solved(order):
    return order == list(range(GRID * GRID))

def tile_at(px, py, ox, oy):
    """Return grid slot index at pixel (px, py), or None if outside grid."""
    c = (px - ox) // TILE_PX
    r = (py - oy) // TILE_PX
    if 0 <= r < GRID and 0 <= c < GRID:
        return r * GRID + c
    return None

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────────────────────────────────────
def draw_grid(frame, tiles, order, ox, oy, hover=None, held=None):
    for slot, tile_idx in enumerate(order):
        r, c = divmod(slot, GRID)
        x, y = ox + c * TILE_PX, oy + r * TILE_PX

        if slot == held:
            # show dark placeholder where tile was lifted from
            ph = np.full((TILE_PX, TILE_PX, 3), 30, dtype=np.uint8)
            cv2.rectangle(ph, (3,3), (TILE_PX-3, TILE_PX-3), (70,70,70), 2)
            frame[y:y+TILE_PX, x:x+TILE_PX] = ph
        else:
            frame[y:y+TILE_PX, x:x+TILE_PX] = tiles[tile_idx]

        # border
        if slot == hover and slot != held:
            cv2.rectangle(frame, (x,y), (x+TILE_PX, y+TILE_PX), CYAN, 3)
        else:
            cv2.rectangle(frame, (x,y), (x+TILE_PX, y+TILE_PX), (80,80,80), 1)

def draw_held_tile(frame, tiles, order, held, pinch_pos):
    """Draw the held tile floating near the pinch point."""
    if held is None or pinch_pos is None:
        return
    tile = tiles[order[held]]
    px = max(0, min(frame.shape[1] - TILE_PX, pinch_pos[0] - TILE_PX//2))
    py = max(0, min(frame.shape[0] - TILE_PX, pinch_pos[1] - TILE_PX//2))
    region = frame[py:py+TILE_PX, px:px+TILE_PX]
    frame[py:py+TILE_PX, px:px+TILE_PX] = cv2.addWeighted(region, 0.25, tile, 0.75, 0)
    cv2.rectangle(frame, (px,py), (px+TILE_PX, py+TILE_PX), CYAN, 2)

def dark_overlay(frame, alpha=0.55):
    ov = np.zeros_like(frame)
    cv2.addWeighted(ov, alpha, frame, 1 - alpha, 0, frame)

def put(frame, text, pos, scale=0.6, color=WHITE, thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    cv2.putText(frame, text, pos, font, scale, color, thickness, cv2.LINE_AA)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    mp_hands  = mp.solutions.hands
    mp_draw   = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    leaderboard = load_lb()

    # ── State machine ──────────────────────────────────────────────────────
    CAPTURE  = "capture"
    SOLVE    = "solve"
    COMPLETE = "complete"
    state    = CAPTURE

    # Shared
    tiles      = []
    tile_order = []

    # Capture phase
    snap_flash = 0          # countdown frames for snap flash effect

    # Solve phase
    start_time    = 0.0
    held          = None    # which grid slot is currently held
    prev_pinching = False

    # Complete phase
    final_time  = 0.0
    player_name = ""

    with mp_hands.Hands(
        max_num_hands          = 2,
        min_detection_confidence = 0.72,
        min_tracking_confidence  = 0.60,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)          # mirror so it feels natural
            H, W  = frame.shape[:2]
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res   = hands.process(rgb)

            hand_list = res.multi_hand_landmarks or []

            # ── Title bar ─────────────────────────────────────────────────
            cv2.rectangle(frame, (0,0), (W, 62), DARK, -1)
            put(frame, "LIVE PUZZLE",   (W//2 - 105, 36), scale=1.1, color=GREEN, thickness=2)
            put(frame, "Frame it to Snap.  Pinch & Drag to Swap.",
                (W//2 - 175, 54), scale=0.38, color=GRAY)

            # ── Draw hand skeletons ───────────────────────────────────────
            for hl in hand_list:
                mp_draw.draw_landmarks(
                    frame, hl, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # ══════════════════════════════════════════════════════════════
            # CAPTURE STATE
            # ══════════════════════════════════════════════════════════════
            if state == CAPTURE:
                # Instruction panel (top-right)
                put(frame, "PHASE 1: CAPTURE", (W-215, 88),  scale=0.44, color=GREEN)
                put(frame, "1. Form frame with both hands", (W-215, 108), scale=0.34, color=GRAY)
                put(frame, "2. Pinch both hands to SNAP",  (W-215, 122), scale=0.34, color=GRAY)

                if len(hand_list) == 2:
                    # Rectangle defined by index-fingertips of both hands
                    pts  = [lm_px(hl.landmark[8], W, H) for hl in hand_list]
                    x1   = min(pts[0][0], pts[1][0])
                    y1   = min(pts[0][1], pts[1][1])
                    x2   = max(pts[0][0], pts[1][0])
                    y2   = max(pts[0][1], pts[1][1])
                    both_pinching = all(is_pinching(hl) for hl in hand_list)

                    cv2.rectangle(frame, (x1,y1), (x2,y2), GREEN, 2)

                    if both_pinching and (x2-x1) > 60 and (y2-y1) > 60:
                        # Flash effect
                        flash_ov = frame.copy()
                        cv2.rectangle(flash_ov, (x1,y1), (x2,y2), GREEN, -1)
                        cv2.addWeighted(flash_ov, 0.35, frame, 0.65, 0, frame)
                        put(frame, "SNAPPING...", (W//2-80, H//2), scale=1.0, color=GREEN, thickness=2)

                        # Capture region, resize square
                        crop = frame[y1:y2, x1:x2].copy()
                        side = min(crop.shape[:2])
                        crop = crop[:side, :side]
                        tiles      = slice_image(crop)
                        tile_order = new_scramble()
                        start_time = time.time()
                        held       = None
                        prev_pinching = False
                        state      = SOLVE
                    else:
                        put(frame, "PINCH TO CAPTURE", (W//2-105, H-55),
                            scale=0.75, color=GRAY)
                else:
                    put(frame, "Show both hands to begin", (W//2-140, H-55),
                        scale=0.65, color=GRAY)

            # ══════════════════════════════════════════════════════════════
            # SOLVE STATE
            # ══════════════════════════════════════════════════════════════
            elif state == SOLVE:
                elapsed = time.time() - start_time
                mins, secs = divmod(int(elapsed), 60)
                timer_str  = f"0:{secs:02d}" if mins == 0 else f"{mins}:{secs:02d}"

                # Instruction panel
                put(frame, "PHASE 2: SOLVE",       (W-200, 88),  scale=0.44, color=GREEN)
                put(frame, "1. Pinch to pick up",  (W-200, 108), scale=0.34, color=GRAY)
                put(frame, "2. Drag & drop to swap",(W-200, 122), scale=0.34, color=GRAY)
                put(frame, "Hold fist to reset",   (W-200, 136), scale=0.34, color=GRAY)

                # Timer
                put(frame, timer_str, (W//2-35, 92), scale=0.85, color=GREEN, thickness=2)

                # Grid centered in frame
                puzzle_px = GRID * TILE_PX
                ox = (W - puzzle_px) // 2
                oy = (H - puzzle_px) // 2 + 25

                # ── Gesture detection (primary = first hand) ──────────────
                pinch_pos     = None
                cur_pinching  = False
                hover         = None

                if hand_list:
                    primary      = hand_list[0]
                    cur_pinching = is_pinching(primary)
                    pinch_pos    = pinch_mid(primary, W, H)
                    hover        = tile_at(pinch_pos[0], pinch_pos[1], ox, oy)

                    # Fist → reset scramble
                    if is_fist(primary):
                        tile_order = new_scramble()
                        held       = None
                        prev_pinching = False
                        start_time = time.time()

                # Pick up / drop logic
                if cur_pinching and not prev_pinching:
                    # Start of pinch — grab tile under hand
                    if hover is not None:
                        held = hover

                elif not cur_pinching and prev_pinching:
                    # Released — swap if over a different tile
                    if held is not None and hover is not None and hover != held:
                        tile_order[held], tile_order[hover] = tile_order[hover], tile_order[held]
                    held = None

                prev_pinching = cur_pinching

                # Draw grid + floating tile
                draw_grid(frame, tiles, tile_order, ox, oy, hover=hover, held=held)
                if cur_pinching and held is not None:
                    draw_held_tile(frame, tiles, tile_order, held, pinch_pos)

                # Check win
                if solved(tile_order):
                    final_time  = elapsed
                    player_name = ""
                    state       = COMPLETE

            # ══════════════════════════════════════════════════════════════
            # COMPLETE STATE
            # ══════════════════════════════════════════════════════════════
            elif state == COMPLETE:
                dark_overlay(frame, alpha=0.62)

                cx, cy = W//2, H//2
                mins, secs = divmod(int(final_time), 60)
                time_str = f"0:{secs:02d}" if mins == 0 else f"{mins}:{secs:02d}"

                # Trophy ASCII
                put(frame, "( * )", (cx-28, cy-90), scale=0.8, color=GREEN, thickness=2)
                put(frame, " | | ",  (cx-26, cy-68), scale=0.7, color=GREEN, thickness=2)
                put(frame, "COMPLETE!", (cx-100, cy-30), scale=1.3, color=GREEN, thickness=3)
                put(frame, time_str,    (cx-35,  cy+10), scale=0.95, color=GREEN, thickness=2)

                # Name input box
                put(frame, "Enter your name for the leaderboard:", (cx-175, cy+50), scale=0.48, color=GRAY)
                cv2.rectangle(frame, (cx-140, cy+58), (cx+140, cy+85), (60,60,60), -1)
                cv2.rectangle(frame, (cx-140, cy+58), (cx+140, cy+85), GRAY, 1)
                put(frame, player_name + "|", (cx-130, cy+78), scale=0.75, color=WHITE, thickness=2)
                put(frame, "Press ENTER to save, ESC to skip", (cx-155, cy+100), scale=0.38, color=GRAY)

                # Leaderboard
                if leaderboard:
                    put(frame, "── LEADERBOARD ──", (cx-100, cy+130), scale=0.5, color=GREEN)
                    for i, e in enumerate(leaderboard[:5]):
                        t = e["time"]
                        m2, s2 = divmod(int(t), 60)
                        ts = f"0:{s2:02d}" if m2==0 else f"{m2}:{s2:02d}"
                        put(frame, f"{i+1}.  {e['name']:<12}  {ts}",
                            (cx-105, cy+155+i*24), scale=0.5, color=GRAY)

            # ── Render ────────────────────────────────────────────────────
            cv2.imshow(WINDOW_NAME, frame)

            # ── Keyboard input ────────────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if state == COMPLETE:
                if key == 13:                       # ENTER — save & restart
                    if player_name.strip():
                        leaderboard.append({"name": player_name.strip(), "time": round(final_time, 2)})
                        leaderboard.sort(key=lambda x: x["time"])
                        save_leaderboard(leaderboard)
                    state = CAPTURE
                elif key == 27:                     # ESC — skip name, restart
                    state = CAPTURE
                elif key == 8 or key == 127:        # Backspace
                    player_name = player_name[:-1]
                elif 32 <= key <= 126:              # Printable ASCII
                    if len(player_name) < 15:
                        player_name += chr(key)

            if key == ord("q"):                     # Q — quit anytime
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
