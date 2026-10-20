import cv2
import numpy as np
import math
import sys
sys.path.insert(0, '.')
from src.pipeline.avatar_service import avatar_service

width, height = 1280, 720
canvas = np.zeros((height, width, 3), dtype=np.uint8)
canvas[:] = (35, 30, 28)

t = 2.0
frame_idx = 45
lip_sync_energy = 0.55

bob_y = int(math.sin(t * 3.0) * 5)
cx = int(width * 0.80)
cy = int(height * 0.68) + bob_y

white_coat = (250, 250, 252)
lapel_gray = (220, 225, 230)
inner_shirt = (235, 210, 180)
tie_blue = (140, 45, 25)
steth_tube = (190, 120, 25)
steth_bell = (215, 215, 225)
skin = (195, 215, 240)

# Shadow beneath
cv2.ellipse(canvas, (cx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)

# Body / Lab Coat Torso
cv2.ellipse(canvas, (cx, cy + 170), (115, 140), 0, 0, 360, white_coat, -1, cv2.LINE_AA)
cv2.polylines(canvas, [np.array([[cx - 75, cy + 80], [cx - 75, cy + 240], [cx + 75, cy + 240], [cx + 75, cy + 80]], np.int32)], False, lapel_gray, 2, cv2.LINE_AA)

# Inner shirt & tie
shirt_pts = np.array([[cx - 28, cy + 45], [cx + 28, cy + 45], [cx + 12, cy + 115], [cx - 12, cy + 115]], np.int32)
cv2.fillPoly(canvas, [shirt_pts], inner_shirt, cv2.LINE_AA)
tie_pts = np.array([[cx - 7, cy + 55], [cx + 7, cy + 55], [cx + 10, cy + 110], [cx, cy + 125], [cx - 10, cy + 110]], np.int32)
cv2.fillPoly(canvas, [tie_pts], tie_blue, cv2.LINE_AA)

# Coat lapels & buttons
cv2.line(canvas, (cx - 40, cy + 50), (cx - 14, cy + 130), lapel_gray, 3, cv2.LINE_AA)
cv2.line(canvas, (cx + 40, cy + 50), (cx + 14, cy + 130), lapel_gray, 3, cv2.LINE_AA)
cv2.line(canvas, (cx, cy + 130), (cx, cy + 240), lapel_gray, 2, cv2.LINE_AA)
for by_btn in [cy + 150, cy + 180, cy + 210]:
    cv2.circle(canvas, (cx, by_btn), 4, (40, 40, 45), -1, cv2.LINE_AA)

# Left coat pocket with ID badge & pen
cv2.rectangle(canvas, (cx - 65, cy + 105), (cx - 30, cy + 135), (240, 240, 245), -1)
cv2.rectangle(canvas, (cx - 65, cy + 105), (cx - 30, cy + 135), lapel_gray, 1, cv2.LINE_AA)
cv2.line(canvas, (cx - 48, cy + 112), (cx - 48, cy + 122), (40, 40, 220), 2, cv2.LINE_AA)
cv2.line(canvas, (cx - 53, cy + 117), (cx - 43, cy + 117), (40, 40, 220), 2, cv2.LINE_AA)
cv2.line(canvas, (cx - 38, cy + 96), (cx - 38, cy + 112), (200, 40, 20), 3, cv2.LINE_AA)

# Stethoscope
steth_l = [(cx - 32, cy + 42), (cx - 45, cy + 65), (cx - 38, cy + 100), (cx - 20, cy + 128), (cx + 10, cy + 135)]
steth_r = [(cx + 32, cy + 42), (cx + 45, cy + 65), (cx + 38, cy + 100), (cx + 25, cy + 125), (cx + 10, cy + 135)]
for i in range(len(steth_l) - 1):
    cv2.line(canvas, steth_l[i], steth_l[i+1], steth_tube, 4, cv2.LINE_AA)
for i in range(len(steth_r) - 1):
    cv2.line(canvas, steth_r[i], steth_r[i+1], steth_tube, 4, cv2.LINE_AA)
cv2.line(canvas, (cx + 10, cy + 135), (cx + 18, cy + 165), steth_tube, 4, cv2.LINE_AA)
cv2.circle(canvas, (cx + 18, cy + 165), 14, steth_bell, -1, cv2.LINE_AA)
cv2.circle(canvas, (cx + 18, cy + 165), 14, (140, 140, 150), 2, cv2.LINE_AA)
cv2.circle(canvas, (cx + 18, cy + 165), 7, (90, 90, 100), -1, cv2.LINE_AA)

# Gesturing Arm
# Draw arm pointing left
hand_x = int(cx - 140 + math.cos(t * 3.5) * 20)
hand_y = int(cy + 30 + math.sin(t * 3.5) * 25)
cv2.line(canvas, (cx - 95, cy + 100), (hand_x, hand_y), white_coat, 22, cv2.LINE_AA)
cv2.circle(canvas, (hand_x, hand_y), 14, skin, -1, cv2.LINE_AA)

# Real Human Face Head
portrait = avatar_service.get_avatar_portrait("doctor")
head_r = 64
face = cv2.resize(portrait, (head_r * 2, head_r * 2), interpolation=cv2.INTER_LANCZOS4)

# Dynamic pulse when speaking
if lip_sync_energy > 0.05:
    for ring_i in range(2):
        cv2.circle(canvas, (cx, cy), head_r + 6 + ring_i * 6, (100, 220, 255), 2, cv2.LINE_AA)

mask = np.zeros((head_r * 2, head_r * 2), dtype=np.uint8)
cv2.circle(mask, (head_r, head_r), head_r - 2, 255, -1, cv2.LINE_AA)

y1, y2 = cy - head_r, cy + head_r
x1, x2 = cx - head_r, cx + head_r
roi = canvas[y1:y2, x1:x2]
for c in range(3):
    roi[:, :, c] = np.where(mask == 255, face[:, :, c], roi[:, :, c])

# Refined glowing rim
cv2.circle(canvas, (cx, cy), head_r, (100, 220, 255) if lip_sync_energy > 0.05 else (200, 210, 220), 3, cv2.LINE_AA)

# Animated Equalizer Speech Bubble
if lip_sync_energy > 0.04:
    bx, by = cx - 140, cy - 150
    cv2.rectangle(canvas, (bx, by), (bx + 220, by + 55), (20, 25, 35), -1)
    cv2.rectangle(canvas, (bx, by), (bx + 220, by + 55), (60, 180, 240), 2, cv2.LINE_AA)
    for bi in range(8):
        bar_x = bx + 20 + bi * 22
        jitter = math.sin(t * 18 + bi * 1.5) * 6
        bar_h = int(max(4, lip_sync_energy * 34 + jitter))
        cv2.rectangle(canvas, (bar_x, by + 44 - bar_h), (bar_x + 12, by + 44), (60, 240, 120) if bi % 2 == 0 else (60, 180, 240), -1)

cv2.imwrite('scratch/test_doctor_body.png', canvas)
print('Saved scratch/test_doctor_body.png')
