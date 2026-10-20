import cv2
import numpy as np
import math
from src.pipeline.avatar_service import avatar_service

# Create a sample 1920x1080 canvas
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
canvas[:] = (20, 22, 28)

# Draw hospital room background
cv2.rectangle(canvas, (0, 0), (1280, 720), (35, 30, 28), -1)
# Grid lines
for y in range(0, 720, 40):
    cv2.line(canvas, (0, y), (1280, y), (45, 40, 35), 1)

# Doctor presenter on the right side
cx = int(1280 * 0.82)
cy = int(720 * 0.55)
lip_sync_energy = 0.65
t = 2.4

# Get real human portrait
portrait = avatar_service.get_avatar_portrait("doctor")
assert portrait is not None, "Portrait should not be None"

# Draw presenter card / medallion
r = 110
face = cv2.resize(portrait, (r * 2, r * 2), interpolation=cv2.INTER_LANCZOS4)

# Dynamic pulse when speaking
pulse_r = r + int(lip_sync_energy * 14)
# Outer audio glow rings
for i in range(3):
    ring_r = pulse_r + (i + 1) * 8
    alpha = max(0, int((3 - i) * 50 * lip_sync_energy))
    cv2.circle(canvas, (cx, cy), ring_r, (255, 200, 100), 2, cv2.LINE_AA)

# Mask and composite face
mask = np.zeros((r * 2, r * 2), dtype=np.uint8)
cv2.circle(mask, (r, r), r - 2, 255, -1, cv2.LINE_AA)

y1, y2 = cy - r, cy + r
x1, x2 = cx - r, cx + r
roi = canvas[y1:y2, x1:x2]
for c in range(3):
    roi[:, :, c] = np.where(mask == 255, face[:, :, c], roi[:, :, c])

# Sleek metallic gold border
cv2.circle(canvas, (cx, cy), r, (80, 210, 255), 4, cv2.LINE_AA)
cv2.circle(canvas, (cx, cy), r + 2, (30, 130, 180), 2, cv2.LINE_AA)

# Live Audio Waveform HUD around bottom
bar_count = 16
for b in range(bar_count):
    ang = 45 + b * (90 / bar_count)
    rad = math.radians(ang)
    bar_h = 10 + int(math.sin(t * 12 + b * 0.8) * 15 * lip_sync_energy + lip_sync_energy * 25)
    bx1 = int(cx + math.cos(rad) * (r + 8))
    by1 = int(cy + math.sin(rad) * (r + 8))
    bx2 = int(cx + math.cos(rad) * (r + 8 + bar_h))
    by2 = int(cy + math.sin(rad) * (r + 8 + bar_h))
    col = (100, 240, 180) if b % 2 == 0 else (60, 210, 255)
    cv2.line(canvas, (bx1, by1), (bx2, by2), col, 3, cv2.LINE_AA)

# Professional Lower-Third Nameplate Badge
badge_y = cy + r + 38
cv2.rectangle(canvas, (cx - 150, badge_y - 20), (cx + 150, badge_y + 24), (15, 18, 25), -1)
cv2.rectangle(canvas, (cx - 150, badge_y - 20), (cx + 150, badge_y + 24), (80, 210, 255), 2, cv2.LINE_AA)
# Pulsing LIVE indicator dot
live_col = (50, 220, 50) if int(t * 4) % 2 == 0 else (30, 140, 30)
cv2.circle(canvas, (cx - 130, badge_y - 3), 6, live_col, -1, cv2.LINE_AA)
cv2.putText(canvas, "DR. R. CHEN, MD", (cx - 112, badge_y + 3), cv2.FONT_HERSHEY_DUPLEX, 0.58, (245, 250, 255), 1, cv2.LINE_AA)
cv2.putText(canvas, "ATTENDING CARDIOLOGIST // LIVE", (cx - 130, badge_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (140, 200, 240), 1, cv2.LINE_AA)

cv2.imwrite('scratch/test_presenter_scene.png', canvas)
print('Saved scratch/test_presenter_scene.png')
