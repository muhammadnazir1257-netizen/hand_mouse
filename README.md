# hand-mouse

control your mouse with just your hand and a webcam. built this because i wanted to try computer vision and thought it'd be a cool project.

uses mediapipe to track 21 points on your hand in real time, then maps your index fingertip to your screen coordinates.

## gestures

| gesture | action |
|---|---|
| point index finger | move cursor |
| touch index tip to thumb | left click |
| touch middle tip to thumb | right click |
| raise index + middle, move up/down | scroll |

## setup

python 3.9+ required, needs a webcam

```bash
pip install -r requirements.txt
python hand_mouse.py
```

first run will download the mediapipe hand model (~2mb) automatically.

keep your hand inside the grey box on screen - that's the tracking zone. press q to quit, or shove the mouse into any corner of your screen to stop it immediately (pyautogui failsafe).

## how it works

- mediapipe detects hand landmarks every frame
- index fingertip position gets mapped from the camera frame to screen coordinates
- uses a 1 euro filter for smoothing - basically adapts how much smoothing to apply based on how fast you're moving. slow = more smoothing so it's stable, fast = less smoothing so it keeps up
- pinch detection has two thresholds (38px to start, 58px to stop) so you don't get flickering clicks when your fingers are near the edge
- cursor locks in place the moment you start pinching, so the click lands where you were actually pointing instead of drifting as your fingers meet

## stack

- opencv - camera
- mediapipe - hand tracking
- pyautogui - mouse control
- numpy - math
