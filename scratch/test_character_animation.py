import cv2
import numpy as np
import math
from pathlib import Path

# Load doctor RGBA
doc_rgba = cv2.imread("data/characters/rgba/doctor.png", cv2.IMREAD_UNCHANGED)
h, w = doc_rgba.shape[:2]

# Load background
bg = cv2.imread("data/cache/backgrounds/hospital_room.jpg")
if bg is None:
    bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bg[:, :] = (235, 240, 245)
else:
    bg = cv2.resize(bg, (1920, 1080))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_path = "scratch/test_doctor_animated.mp4"
writer = cv2.VideoWriter(out_path, fourcc, 25.0, (1920, 1080))

for f_idx in range(75):  # 3 seconds at 25 fps
    t = f_idx / 25.0
    frame = bg.copy()
    
    # Audio energy simulation (talking rhythm)
    audio_energy = max(0.0, math.sin(t * 8.0) * 0.7 + math.sin(t * 14.0) * 0.3)
    if (int(t * 2) % 2) == 1 and t > 2.0:
        audio_energy = 0.0  # Pause

    sprite = doc_rgba.copy()

    # 1. Mouth animation (Coordinates on doctor: cx=0.495, cy=0.395)
    mx = int(w * 0.495)
    my = int(h * 0.395)
    if audio_energy > 0.04:
        m_open = int(audio_energy * 20)
        m_spread = int(audio_energy * 10)
        mw = 42 + m_spread
        mh = 8 + m_open
        # Dark oral cavity
        cv2.ellipse(sprite, (mx, my), (mw, mh), 0, 0, 360, (25, 20, 50, 255), -1, cv2.LINE_AA)
        # Upper teeth highlight
        if m_open > 6:
            cv2.rectangle(sprite, (mx - int(mw*0.65), my - mh + 2), (mx + int(mw*0.65), my - mh + 8), (230, 235, 240, 255), -1)
        # Lower lip contour
        cv2.ellipse(sprite, (mx, my + mh + 1), (int(mw*0.8), 5), 0, 0, 180, (80, 75, 140, 255), 2, cv2.LINE_AA)

    # 2. Eye Blinking (Blink every 3s, duration 0.15s)
    blink_phase = t % 3.0
    if blink_phase < 0.15:
        blink_ratio = (1.0 - math.cos(blink_phase / 0.15 * math.pi * 2)) / 2.0
        ey = int(h * 0.272)
        elx = int(w * 0.448)
        erx = int(w * 0.552)
        lid_h = int(14 * blink_ratio)
        skin_tone = (180, 200, 225, 255)
        for ex in [elx, erx]:
            cv2.ellipse(sprite, (ex, ey), (18, max(2, lid_h)), 0, 0, 180, skin_tone, -1, cv2.LINE_AA)
            cv2.line(sprite, (ex - 16, ey), (ex + 16, ey), (40, 40, 50, 255), 2, cv2.LINE_AA)

    # 3. Kinematic Sway & Breathing
    sway_x = int(math.sin(t * 2.2) * 4)
    sway_y = int(math.sin(t * 2.8) * 5)
    nod_y = int(math.sin(t * 7.5) * audio_energy * 5)
    
    # 4. Composite onto background (cx=960, bottom=1080)
    target_h = 920
    scale = target_h / float(h)
    target_w = int(w * scale)
    scaled_sprite = cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    cx = 960 + sway_x
    bot_y = 1080 + sway_y + nod_y
    
    x1 = cx - target_w // 2
    x2 = x1 + target_w
    y2 = bot_y
    y1 = y2 - target_h
    
    bx1 = max(0, x1)
    bx2 = min(1920, x2)
    by1 = max(0, y1)
    by2 = min(1080, y2)
    
    sx1 = bx1 - x1
    sx2 = sx1 + (bx2 - bx1)
    sy1 = by1 - y1
    sy2 = sy1 + (by2 - by1)
    
    if bx2 > bx1 and by2 > by1:
        alpha = (scaled_sprite[sy1:sy2, sx1:sx2, 3] / 255.0)[:, :, np.newaxis]
        color = scaled_sprite[sy1:sy2, sx1:sx2, :3]
        frame[by1:by2, bx1:bx2] = (color * alpha + frame[by1:by2, bx1:bx2] * (1.0 - alpha)).astype(np.uint8)
        
    writer.write(frame)

writer.release()
print("Saved", out_path)
