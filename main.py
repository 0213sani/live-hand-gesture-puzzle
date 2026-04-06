import cv2
import mediapipe as mp
import numpy as np
import random

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def is_pinching(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = ((thumb_tip.x - index_tip.x)**2 + 
                (thumb_tip.y - index_tip.y)**2)**0.5
    return distance < 0.05

def get_index_tip(hand_landmarks, frame_w, frame_h):
    tip = hand_landmarks.landmark[8]
    return int(tip.x * frame_w), int(tip.y * frame_h)

def slice_image(image, grid_size=3):
    h, w, _ = image.shape
    tile_h = h // grid_size
    tile_w = w // grid_size
    tiles = []
    for row in range(grid_size):
        for col in range(grid_size):
            tile = image[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w]
            tiles.append(tile)
    return tiles, tile_h, tile_w

def draw_puzzle(tiles, positions, tile_h, tile_w, grid_size=3):
    puzzle_surface = np.zeros((tile_h*grid_size, tile_w*grid_size, 3), dtype=np.uint8)
    for idx, pos in enumerate(positions):
        row = pos // grid_size
        col = pos % grid_size
        puzzle_surface[row*tile_h:(row+1)*tile_h, col*tile_w:(col+1)*tile_w] = tiles[idx]
    for i in range(1, grid_size):
        cv2.line(puzzle_surface, (i*tile_w, 0), (i*tile_w, tile_h*grid_size), (255,255,255), 2)
        cv2.line(puzzle_surface, (0, i*tile_h), (tile_w*grid_size, i*tile_h), (255,255,255), 2)
    return puzzle_surface

cap = cv2.VideoCapture(0)
captured_image = None
phase = "capture"

tiles = []
positions = []
tile_h = tile_w = 0
GRID_SIZE = 3

# Track previous pinch state
was_both_pinching = False
last_rx1 = last_ry1 = last_rx2 = last_ry2 = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # Save clean frame BEFORE drawing anything on it
    clean_frame = frame.copy()

    if phase == "capture":
        both_pinching = False
        index_tips = []

        if results.multi_hand_landmarks:
            if len(results.multi_hand_landmarks) == 2:
                pinching_count = 0
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    if is_pinching(hand_landmarks):
                        pinching_count += 1
                    index_tips.append(get_index_tip(hand_landmarks, w, h))

                if pinching_count == 2 and len(index_tips) == 2:
                    both_pinching = True
                    x1, y1 = index_tips[0]
                    x2, y2 = index_tips[1]
                    rx1, ry1 = min(x1,x2), min(y1,y2)
                    rx2, ry2 = max(x1,x2), max(y1,y2)

                    # Save last known rectangle
                    last_rx1, last_ry1 = rx1, ry1
                    last_rx2, last_ry2 = rx2, ry2

                    # Draw rectangle on display frame only
                    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 255, 0), 3)
                    cv2.putText(frame, "Unpinch to snap!", (50, 50),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Was pinching, now unpinched → SNAP
                if was_both_pinching and not both_pinching:
                    # Capture from CLEAN frame (no green border)
                    captured_image = clean_frame[last_ry1:last_ry2, last_rx1:last_rx2].copy()

                    if captured_image.size > 0:
                        # Resize keeping aspect ratio, divisible by 3
                        cap_h, cap_w, _ = captured_image.shape
                        new_w = (cap_w // (GRID_SIZE * 50)) * (GRID_SIZE * 50)
                        new_h = (cap_h // (GRID_SIZE * 50)) * (GRID_SIZE * 50)
                        if new_w == 0: new_w = 300
                        if new_h == 0: new_h = 300
                        captured_image = cv2.resize(captured_image, (new_w, new_h))

                        tiles, tile_h, tile_w = slice_image(captured_image)
                        positions = list(range(GRID_SIZE * GRID_SIZE))
                        random.shuffle(positions)
                        phase = "solve"
                        print("SNAPPED!")
            else:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        was_both_pinching = both_pinching

        cv2.putText(frame, "Pinch both hands to frame, unpinch to snap!", (10, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Hand Gesture Puzzle", frame)

    elif phase == "solve":
        puzzle = draw_puzzle(tiles, positions, tile_h, tile_w)
        cv2.imshow("Puzzle!", puzzle)
        cv2.putText(frame, "Solve the puzzle!", (50, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Gesture Puzzle", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()