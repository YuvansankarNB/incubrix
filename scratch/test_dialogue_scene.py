import cv2
import numpy as np
import math
import sys
sys.path.insert(0, '.')
from src.pipeline.avatar_service import avatar_service

width, height = 1280, 720
canvas = np.zeros((height, width, 3), dtype=np.uint8)

# Hospital consultation room background
canvas[:] = (38, 32, 28)
# Warm hospital clinic wall gradient & wood wainscoting
cv2.rectangle(canvas, (0, 0), (width, int(height * 0.65)), (50, 42, 38), -1)
cv2.rectangle(canvas, (0, int(height * 0.65)), (width, height), (32, 28, 26), -1)
cv2.line(canvas, (0, int(height * 0.65)), (width, int(height * 0.65)), (80, 70, 60), 3)

# Medical monitor on wall
cv2.rectangle(canvas, (int(width * 0.38), 60), (int(width * 0.62), 240), (20, 20, 25), -1)
cv2.rectangle(canvas, (int(width * 0.38), 60), (int(width * 0.62), 240), (80, 90, 100), 2)
# Heart rate ECG trace
for ex in range(int(width * 0.39), int(width * 0.61), 4):
    ecg_y = 150 + int(math.sin(ex * 0.08) * 12)
    if (ex // 30) % 4 == 0:
        ecg_y -= 35
    cv2.circle(canvas, (ex, ecg_y), 1, (60, 240, 120), -1)
cv2.putText(canvas, "VITALS: NORMAL // HR 72", (int(width * 0.40), 95), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 220, 255), 1, cv2.LINE_AA)

t = 1.8
frame_idx = 35

# ── 1. PATIENT (LEFT SIDE - LISTENING) ──
pcx = int(width * 0.22)
pcy = int(height * 0.68)
p_bob = int(math.sin(t * 2.8) * 4)
pcy += p_bob

gown_blue = (225, 195, 150)
gown_trim = (190, 160, 115)
skin = (190, 210, 238)

# Patient body
cv2.ellipse(canvas, (pcx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)
cv2.ellipse(canvas, (pcx, pcy + 170), (112, 138), 0, 0, 360, gown_blue, -1, cv2.LINE_AA)
cv2.ellipse(canvas, (pcx, pcy + 52), (32, 18), 0, 0, 180, gown_trim, 3, cv2.LINE_AA)
cv2.line(canvas, (pcx, pcy + 62), (pcx, pcy + 120), gown_trim, 2, cv2.LINE_AA)

# Patient real human face
p_portrait = avatar_service.get_avatar_portrait("patient")
head_r = 64
if p_portrait is not None:
    p_face = cv2.resize(p_portrait, (head_r * 2, head_r * 2), interpolation=cv2.INTER_LANCZOS4)
    p_mask = np.zeros((head_r * 2, head_r * 2), dtype=np.uint8)
    cv2.circle(p_mask, (head_r, head_r), head_r - 2, 255, -1, cv2.LINE_AA)
    py1, py2 = pcy - head_r, pcy + head_r
    px1, px2 = pcx - head_r, pcx + head_r
    proi = canvas[py1:py2, px1:px2]
    for c in range(3):
        proi[:, :, c] = np.where(p_mask == 255, p_face[:, :, c], proi[:, :, c])
    cv2.circle(canvas, (pcx, pcy), head_r, (160, 180, 200), 2, cv2.LINE_AA)

# Patient badge
cv2.rectangle(canvas, (pcx - 85, pcy + 240), (pcx + 85, pcy + 268), (20, 25, 35), -1)
cv2.rectangle(canvas, (pcx - 85, pcy + 240), (pcx + 85, pcy + 268), (140, 170, 200), 1, cv2.LINE_AA)
cv2.putText(canvas, "PATIENT // LISTENING", (pcx - 76, pcy + 258), cv2.FONT_HERSHEY_DUPLEX, 0.38, (220, 230, 240), 1, cv2.LINE_AA)

# ── 2. DOCTOR (RIGHT SIDE - ACTIVELY SPEAKING) ──
dcx = int(width * 0.78)
dcy = int(height * 0.68)
d_bob = int(math.sin(t * 3.0) * 5)
dcy += d_bob
lip_sync_energy = 0.68

white_coat = (250, 250, 252)
lapel_gray = (220, 225, 230)
steth_tube = (190, 120, 25)
steth_bell = (215, 215, 225)

# Doctor body
cv2.ellipse(canvas, (dcx, int(height * 0.94)), (120, 26), 0, 0, 360, (10, 10, 10), -1, cv2.LINE_AA)
cv2.ellipse(canvas, (dcx, dcy + 170), (115, 140), 0, 0, 360, white_coat, -1, cv2.LINE_AA)
cv2.polylines(canvas, [np.array([[dcx - 75, dcy + 80], [dcx - 75, dcy + 240], [dcx + 75, dcy + 240], [dcx + 75, dcy + 80]], np.int32)], False, lapel_gray, 2, cv2.LINE_AA)

# Stethoscope
steth_l = [(dcx - 32, dcy + 42), (dcx - 45, dcy + 65), (dcx - 38, dcy + 100), (dcx - 20, dcy + 128), (dcx + 10, dcy + 135)]
steth_r = [(dcx + 32, dcy + 42), (dcx + 45, dcy + 65), (dcx + 38, dcy + 100), (dcx + 25, dcy + 125), (dcx + 10, dcy + 135)]
for i in range(len(steth_l) - 1):
    cv2.line(canvas, steth_l[i], steth_l[i+1], steth_tube, 4, cv2.LINE_AA)
for i in range(len(steth_r) - 1):
    cv2.line(canvas, steth_r[i], steth_r[i+1], steth_tube, 4, cv2.LINE_AA)
cv2.line(canvas, (dcx + 10, dcy + 135), (dcx + 18, dcy + 165), steth_tube, 4, cv2.LINE_AA)
cv2.circle(canvas, (dcx + 18, dcy + 165), 14, steth_bell, -1, cv2.LINE_AA)

# Doctor Gesturing Arm towards Patient
d_hand_x = int(dcx - 140 + math.cos(t * 3.5) * 18)
d_hand_y = int(dcy + 25 + math.sin(t * 3.5) * 22)
cv2.line(canvas, (dcx - 90, dcy + 100), (d_hand_x, d_hand_y), white_coat, 22, cv2.LINE_AA)
cv2.circle(canvas, (d_hand_x, d_hand_y), 14, skin, -1, cv2.LINE_AA)

# Doctor Real Human Face
d_portrait = avatar_service.get_avatar_portrait("doctor")
if d_portrait is not None:
    # Glow rings when speaking
    for ri in range(3):
        glow_r = head_r + 6 + ri * 6
        cv2.circle(canvas, (dcx, dcy), glow_r, (100, 220, 255), 2, cv2.LINE_AA)

    d_face = cv2.resize(d_portrait, (head_r * 2, head_r * 2), interpolation=cv2.INTER_LANCZOS4)
    d_mask = np.zeros((head_r * 2, head_r * 2), dtype=np.uint8)
    cv2.circle(d_mask, (head_r, head_r), head_r - 2, 255, -1, cv2.LINE_AA)
    dy1, dy2 = dcy - head_r, dcy + head_r
    dx1, dx2 = dcx - head_r, dcx + head_r
    droi = canvas[dy1:dy2, dx1:dx2]
    for c in range(3):
        droi[:, :, c] = np.where(d_mask == 255, d_face[:, :, c], droi[:, :, c])
    cv2.circle(canvas, (dcx, dcy), head_r, (100, 220, 255), 3, cv2.LINE_AA)

# Doctor Equalizer Bubble overhead
bx, by = dcx - 140, dcy - 150
cv2.rectangle(canvas, (bx, by), (bx + 220, by + 55), (20, 25, 35), -1)
cv2.rectangle(canvas, (bx, by), (bx + 220, by + 55), (60, 180, 240), 2, cv2.LINE_AA)
for bi in range(8):
    bar_x = bx + 20 + bi * 22
    jitter = math.sin(t * 18 + bi * 1.5) * 6
    bar_h = int(max(4, lip_sync_energy * 34 + jitter))
    cv2.rectangle(canvas, (bar_x, by + 44 - bar_h), (bar_x + 12, by + 44), (60, 240, 120) if bi % 2 == 0 else (60, 180, 240), -1)

# Doctor Badge
cv2.rectangle(canvas, (dcx - 85, dcy + 240), (dcx + 85, dcy + 268), (20, 25, 35), -1)
cv2.rectangle(canvas, (dcx - 85, dcy + 240), (dcx + 85, dcy + 268), (60, 180, 240), 1, cv2.LINE_AA)
cv2.putText(canvas, "DR. CHEN // SPEAKING", (dcx - 78, dcy + 258), cv2.FONT_HERSHEY_DUPLEX, 0.38, (80, 220, 255), 1, cv2.LINE_AA)

cv2.imwrite('scratch/test_dialogue_scene.png', canvas)
print('Saved scratch/test_dialogue_scene.png')
