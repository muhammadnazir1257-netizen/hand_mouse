import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import urllib.request
import math
import os

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    print("downloading model, one sec...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0

# mediapipe gives 21 landmarks per hand, these are the ones i actually use
THUMB_TIP  = 4
INDEX_TIP  = 8
INDEX_MCP  = 5
MIDDLE_TIP = 12
MIDDLE_MCP = 9

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# 1 euro filter - way better than a rolling average
# the idea: move slow = smooth more, move fast = smooth less
# tuned it until it felt like an actual mouse
class OneEuroFilter:
    def __init__(self, freq=30.0, min_cutoff=1.2, beta=0.08, d_cutoff=1.0):
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.xp = None
        self.dxp = 0.0

    def _alpha(self, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau * self.freq)

    def reset(self):
        self.xp = None
        self.dxp = 0.0

    def __call__(self, x):
        if self.xp is None:
            self.xp = x
            return x
        dx = (x - self.xp) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dxp
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1.0 - a) * self.xp
        self.xp = x_hat
        self.dxp = dx_hat
        return x_hat


def tip_dist(lm, a, b, fw, fh):
    ax, ay = int(lm[a].x * fw), int(lm[a].y * fh)
    bx, by = int(lm[b].x * fw), int(lm[b].y * fh)
    return float(np.hypot(bx - ax, by - ay)), (ax, ay), (bx, by)

def is_up(lm, tip, mcp):
    return lm[tip].y < lm[mcp].y

def draw_skeleton(frame, lm, fw, fh, col=(80, 210, 80)):
    pts = [(int(p.x * fw), int(p.y * fh)) for p in lm]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], col, 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (255, 255, 255), cv2.FILLED)
        cv2.circle(frame, pt, 4, col, 1)


class HandMouse:
    PINCH_ON  = 38   # px to trigger pinch
    PINCH_OFF = 58   # px to release - gap stops flicker near the threshold
    CD        = 18   # frames before you can click again

    def __init__(self, cam=0, margin=100):
        self.margin = margin
        self.cam = cam

        opts = vision.HandLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.det = vision.HandLandmarker.create_from_options(opts)

        self.sw, self.sh = pyautogui.size()

        self.fx = OneEuroFilter()
        self.fy = OneEuroFilter()

        self.l_held = False
        self.r_held = False
        self.l_cd = 0
        self.r_cd = 0
        self.lock_x = self.sw // 2
        self.lock_y = self.sh // 2
        self.scroll_anchor = None

    def process(self, lm, frame):
        fh, fw = frame.shape[:2]
        m = self.margin

        ix = int(lm[INDEX_TIP].x * fw)
        iy = int(lm[INDEX_TIP].y * fh)
        mx = int(lm[MIDDLE_TIP].x * fw)
        my = int(lm[MIDDLE_TIP].y * fh)

        ld, la, lb = tip_dist(lm, INDEX_TIP,  THUMB_TIP, fw, fh)
        rd, ra, rb = tip_dist(lm, MIDDLE_TIP, THUMB_TIP, fw, fh)

        lp = (ld < self.PINCH_OFF) if self.l_held else (ld < self.PINCH_ON)
        rp = (rd < self.PINCH_OFF) if self.r_held else (rd < self.PINCH_ON)

        idx_up = is_up(lm, INDEX_TIP,  INDEX_MCP)
        mid_up = is_up(lm, MIDDLE_TIP, MIDDLE_MCP)

        # both fingers up and no pinch = scroll mode
        if idx_up and mid_up and not lp and not rp:
            cy = (iy + my) // 2
            if self.scroll_anchor is None:
                self.scroll_anchor = cy
            else:
                d = self.scroll_anchor - cy
                if abs(d) > 4:
                    pyautogui.scroll(int(d / 7))
            cv2.circle(frame, (ix, iy), 11, (255, 200, 0), cv2.FILLED)
            cv2.circle(frame, (mx, my), 11, (255, 200, 0), cv2.FILLED)
            return "scroll"

        self.scroll_anchor = None

        # lock cursor position the moment pinch starts
        # without this the tip moves toward the thumb and drags the cursor off target
        if not lp and not rp:
            sx = np.interp(ix, [m, fw - m], [0, self.sw])
            sy = np.interp(iy, [m, fh - m], [0, self.sh])
            nx = int(self.fx(sx))
            ny = int(self.fy(sy))
            pyautogui.moveTo(nx, ny)
            self.lock_x, self.lock_y = nx, ny
        else:
            pyautogui.moveTo(self.lock_x, self.lock_y)

        status = "move"

        if self.l_cd > 0: self.l_cd -= 1
        if lp and not self.l_held and self.l_cd == 0:
            pyautogui.click()
            self.l_cd = self.CD
            status = "left click"
        self.l_held = lp

        if self.r_cd > 0: self.r_cd -= 1
        if rp and not lp and not self.r_held and self.r_cd == 0:
            pyautogui.rightClick()
            self.r_cd = self.CD
            status = "right click"
        self.r_held = rp and not lp

        cv2.circle(frame, (ix, iy), 11, (255, 50, 220), cv2.FILLED)
        cv2.line(frame, la, lb, (0, 255, 80)  if lp else (55, 55, 55), 2)
        cv2.line(frame, ra, rb, (0, 140, 255) if rp else (55, 55, 55), 2)
        cv2.putText(frame, f"L:{int(ld)}px  R:{int(rd)}px", (10, fh - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 80) if (lp or rp) else (120, 120, 120), 1)
        return status

    def run(self):
        cap = cv2.VideoCapture(self.cam)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print("camera not found, try a different index")
            return

        print("running - q to quit, move mouse to corner to emergency stop")
        ts = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            fh, fw = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = self.det.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts
            )
            ts += 33

            m = self.margin
            cv2.rectangle(frame, (m, m), (fw - m, fh - m), (55, 55, 55), 1)

            status = "no hand"
            if result.hand_landmarks:
                lm = result.hand_landmarks[0]
                draw_skeleton(frame, lm, fw, fh)
                status = self.process(lm, frame)
            else:
                self.fx.reset()
                self.fy.reset()
                self.scroll_anchor = None
                self.l_held = self.r_held = False

            cv2.rectangle(frame, (0, 0), (fw, 52), (0, 0, 0), cv2.FILLED)
            cv2.putText(frame, f"status: {status}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 240, 80), 2)
            cv2.putText(frame, "index=move | index+thumb=click | middle+thumb=rclick | 2fingers=scroll",
                        (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150, 150, 150), 1)

            cv2.imshow("hand mouse", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.det.close()


HandMouse().run()
