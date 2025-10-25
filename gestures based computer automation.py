import cv2
import mediapipe as mp
import pyautogui
import time
import numpy as np

# Safety: move mouse to any screen corner to abort
pyautogui.FAILSAFE = True  # helpful during testing [docs]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)  # real-time hand landmarks [MediaPipe Hands]

# Debounce per-gesture (for hotkey gestures)
gesture_cooldown = {}
DEBOUNCE_TIME = 1.5  # seconds

# Landmark indices
FINGER_TIPS = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP
]  # fingertip landmarks [MediaPipe Hands]

# Cursor smoothing and dwell-click config
SMOOTH = 0.35             # 0..1, higher = smoother/slower motion
DWELL_RADIUS_PX = 25      # fingertip must stay within this radius (pixels)
DWELL_TIME_S = 1.5        # CHANGED: seconds to hold still for a left-click

# Screen size for mapping camera coords to screen coords
screen_w, screen_h = pyautogui.size()  # map camera to OS cursor

# Middle-finger cursor state
prev_cursor_x, prev_cursor_y = None, None
dwell_center = None
dwell_start_t = None
dwell_clicked = False  # prevent repeated clicks until movement

# ---------- Optional: keep existing hotkey gestures ----------
def fingers_up(hand_landmarks):
    lm = hand_landmarks.landmark
    fingers = []
    fingers.append(lm[mp_hands.HandLandmark.THUMB_TIP].x < lm[mp_hands.HandLandmark.THUMB_IP].x)
    for tip in FINGER_TIPS[1:]:
        fingers.append(lm[tip].y < lm[tip - 2].y)
    return fingers  # simple finger-up rule using landmarks

def is_fist(f): return f == [False, False, False, False, False]
def is_palm(f): return f == [True, True, True, True, True]
def is_one(f): return f == [False, True, False, False, False]
def is_two(f): return f == [False, True, True, False, False]
def is_three(f): return f == [False, True, True, True, False]

def is_thumb_up(f, hand_landmarks):
    if f == [True, False, False, False, False]:
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        return thumb_tip.y < wrist.y
    return False

def perform_action(name):
    now = time.time()
    last = gesture_cooldown.get(name, 0)
    if now - last < DEBOUNCE_TIME:
        return
    gesture_cooldown[name] = now
    if name == "fist":
        pyautogui.press('space')        # Play/Pause
    elif name == "palm":
        pyautogui.hotkey('alt', 'tab')  # Switch window
    elif name == "one":
        pyautogui.press('right')        # Next
    elif name == "two":
        pyautogui.press('left')         # Previous
    elif name == "three":
        pyautogui.press('volumeup')     # Volume up
    elif name == "thumb_up":
        pyautogui.press('volumedown')   # Volume down
# -------------------------------------------------------------

def map_to_screen(ix, iy, frame_w, frame_h):
    sx = np.interp(ix, [0, frame_w], [0, screen_w])
    sy = np.interp(iy, [0, frame_h], [0, screen_h])
    return int(sx), int(sy)

cap = cv2.VideoCapture(0)

try:
    while True:
        ok, img = cap.read()
        if not ok:
            break

        img = cv2.flip(img, 1)
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)  # run landmarks

        overlay = ""

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)  # draw

            # Hotkey gestures (optional)
            f = fingers_up(hand)
            if is_fist(f):
                perform_action("fist")
                overlay = "Fist: Play/Pause"
            elif is_palm(f):
                perform_action("palm")
                overlay = "Palm: Alt+Tab"
            elif is_one(f):
                perform_action("one")
                overlay = "One: Next"
            elif is_two(f):
                perform_action("two")
                overlay = "Two: Previous"
            elif is_three(f):
                perform_action("three")
                overlay = "Three: Vol Up"
            elif is_thumb_up(f, hand):
                perform_action("thumb_up")
                overlay = "Thumb Up: Vol Down"

            # ===== Middle-finger cursor + dwell-to-click (1.5s) =====
            mid_tip = hand.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            mx_px, my_px = int(mid_tip.x * w), int(mid_tip.y * h)
            scr_x, scr_y = map_to_screen(mx_px, my_px, w, h)

            # Smooth cursor motion
            if prev_cursor_x is None:
                cur_x, cur_y = scr_x, scr_y
            else:
                cur_x = int(prev_cursor_x + (scr_x - prev_cursor_x) * (1 - SMOOTH))
                cur_y = int(prev_cursor_y + (scr_y - prev_cursor_y) * (1 - SMOOTH))
            prev_cursor_x, prev_cursor_y = cur_x, cur_y

            pyautogui.moveTo(cur_x, cur_y, duration=0)  # move OS cursor

            # Dwell-click detection
            if dwell_center is None:
                dwell_center = (mx_px, my_px)
                dwell_start_t = time.time()
                dwell_clicked = False
            else:
                dx = mx_px - dwell_center[0]
                dy = my_px - dwell_center[1]
                dist2 = dx*dx + dy*dy
                if dist2 <= DWELL_RADIUS_PX * DWELL_RADIUS_PX:
                    if not dwell_clicked and (time.time() - dwell_start_t) >= DWELL_TIME_S:
                        pyautogui.click()  # left click after 1.5s dwell
                        dwell_clicked = True
                else:
                    dwell_center = (mx_px, my_px)
                    dwell_start_t = time.time()
                    dwell_clicked = False

            # Visual overlays
            cv2.circle(img, (mx_px, my_px), 10, (0, 200, 255), -1)
            if dwell_center:
                cv2.circle(
                    img,
                    (dwell_center[0], dwell_center[1]),
                    DWELL_RADIUS_PX,
                    (0, 255, 0) if not dwell_clicked else (0, 128, 255),
                    2
                )
                elapsed = time.time() - (dwell_start_t or time.time())
                label = "CLICKED" if dwell_clicked else f"DWELL {elapsed:.1f}s/{DWELL_TIME_S:.1f}s"
                cv2.putText(img, label, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 50), 2)

        if overlay:
            cv2.putText(img, overlay, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 0), 2)

        cv2.imshow("Gesture Controlled Automation", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()