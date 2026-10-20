import cv2
import numpy as np
import math
import sys
from pathlib import Path

sys.path.insert(0, '.')
from src.pipeline.avatar_service import avatar_service

# Load the real hospital room photo fetched from internet API
bg_path = Path("data/cache/backgrounds/hospital_room.jpg")
if bg_path.exists():
    bg = cv2.imread(str(bg_path))
    bg = cv2.resize(bg, (1920, 1080))
else:
    bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bg[:] = (40, 35, 30)

# Apply sleek cinematic grading (65% photo, 35% dark slate overlay)
overlay = np.zeros_like(bg)
overlay[:] = (20, 24, 30)
cv2.addWeighted(bg, 0.65, overlay, 0.35, 0, bg)

# Load real human portraits from free API
doc_portrait = avatar_service.get_avatar_portrait("doctor")
pat_portrait = avatar_service.get_avatar_portrait("patient")

t = 1.8
lip_sync_energy = 0.72

# ── 1. PATIENT FEED (LEFT - LISTENING) ──
pcx, pcy = int(1920 * 0.20), int(1080 * 0.66)
pr = 110
pat_face = cv2.resize(pat_portrait, (pr * 2, pr * 2), interpolation=cv2.INTER_LANCZOS4)
p_mask = np.zeros((pr * 2, pr * 2), dtype=np.uint8)
cv2.circle(p_mask, (pr, pr), pr - 2, 255, -1, cv2.LINE_AA)

py1, py2 = pcy - pr, pcy + pr
px1, px2 = pcx - pr, pcx + pr
proi = bg[py1:py2, px1:px2]
for c in range(3):
    proi[:, :, c] = np.where(p_mask == 255, pat_face[:, :, c], proi[:, :, c])
cv2.circle(bg, (pcx, pcy), pr, (180, 190, 205), 3, cv2.LINE_AA)

# Patient Glassmorphic Nameplate
p_badge_y = pcy + pr + 36
cv2.rectangle(bg, (pcx - 145, p_badge_y - 20), (pcx + 145, p_badge_y + 24), (15, 20, 28), -1)
cv2.rectangle(bg, (pcx - 145, p_badge_y - 20), (pcx + 145, p_badge_y + 24), (140, 160, 180), 2, cv2.LINE_AA)
cv2.circle(bg, (pcx - 122, p_badge_y - 1), 6, (120, 140, 160), -1, cv2.LINE_AA)
cv2.putText(bg, "PATIENT FEED", (pcx - 105, p_badge_y + 4), cv2.FONT_HERSHEY_DUPLEX, 0.54, (230, 235, 245), 1, cv2.LINE_AA)
cv2.putText(bg, "TELEHEALTH ROOM // LISTENING", (pcx - 122, p_badge_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 160, 190), 1, cv2.LINE_AA)

# ── 2. DOCTOR FEED (RIGHT - ACTIVELY SPEAKING) ──
dcx, dcy = int(1920 * 0.80), int(1080 * 0.66)
dr = 110
doc_face = cv2.resize(doc_portrait, (dr * 2, dr * 2), interpolation=cv2.INTER_LANCZOS4)

# Active audio pulse aura
for ri in range(3):
    aura_r = dr + 6 + ri * 8 + int(lip_sync_energy * 12)
    cv2.circle(bg, (dcx, dcy), aura_r, (100, 220, 255), 2, cv2.LINE_AA)

d_mask = np.zeros((dr * 2, dr * 2), dtype=np.uint8)
cv2.circle(d_mask, (dr, dr), dr - 2, 255, -1, cv2.LINE_AA)
dy1, dy2 = dcy - dr, dcy + dr
dx1, dx2 = dcx - dr, dcx + dr
droi = bg[dy1:dy2, dx1:dx2]
for c in range(3):
    droi[:, :, c] = np.where(d_mask == 255, doc_face[:, :, c], droi[:, :, c])
cv2.circle(bg, (dcx, dcy), dr, (80, 210, 255), 4, cv2.LINE_AA)

# Radial Audio Waveform HUD around Doctor Feed
bar_count = 16
for b in range(bar_count):
    ang = 45 + b * (90 / bar_count)
    rad = math.radians(ang)
    bar_h = 10 + int(math.sin(t * 14 + b * 0.9) * 14 * lip_sync_energy + lip_sync_energy * 22)
    bx1 = int(dcx + math.cos(rad) * (dr + 8))
    by1 = int(dcy + math.sin(rad) * (dr + 8))
    bx2 = int(dcx + math.cos(rad) * (dr + 8 + bar_h))
    by2 = int(dcy + math.sin(rad) * (dr + 8 + bar_h))
    col = (100, 240, 180) if b % 2 == 0 else (60, 210, 255)
    cv2.line(bg, (bx1, by1), (bx2, by2), col, 3, cv2.LINE_AA)

# Doctor Glassmorphic Nameplate
d_badge_y = dcy + dr + 36
cv2.rectangle(bg, (dcx - 155, d_badge_y - 20), (dcx + 155, d_badge_y + 24), (15, 20, 28), -1)
cv2.rectangle(bg, (dcx - 155, d_badge_y - 20), (dcx + 155, d_badge_y + 24), (80, 210, 255), 2, cv2.LINE_AA)
cv2.circle(bg, (dcx - 132, d_badge_y - 1), 6, (50, 240, 50), -1, cv2.LINE_AA)
cv2.putText(bg, "ATTENDING PHYSICIAN", (dcx - 114, d_badge_y + 4), cv2.FONT_HERSHEY_DUPLEX, 0.54, (245, 250, 255), 1, cv2.LINE_AA)
cv2.putText(bg, "CONSULTATION PROTOCOL // SPEAKING", (dcx - 132, d_badge_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 220, 255), 1, cv2.LINE_AA)

# ── 3. CLINICAL TELEMETRY MONITOR (CENTER HUD) ──
mon_x1, mon_x2 = int(1920 * 0.30), int(1920 * 0.70)
mon_y1, mon_y2 = 70, 360
cv2.rectangle(bg, (mon_x1, mon_y1), (mon_x2, mon_y2), (12, 18, 25), -1)
cv2.rectangle(bg, (mon_x1, mon_y1), (mon_x2, mon_y2), (60, 180, 120), 2, cv2.LINE_AA)
cv2.putText(bg, "ICU CLINICAL MONITOR // REAL-TIME VITALS", (mon_x1 + 30, mon_y1 + 45), cv2.FONT_HERSHEY_DUPLEX, 0.68, (240, 245, 250), 1, cv2.LINE_AA)
cv2.putText(bg, "HR: 74 BPM    SpO2: 99%    BP: 120/80", (mon_x1 + 30, mon_y1 + 100), cv2.FONT_HERSHEY_DUPLEX, 0.88, (60, 240, 120), 2, cv2.LINE_AA)

# ECG Waveform
ecg_pts = []
for ex in range(mon_x1 + 30, mon_x2 - 30, 4):
    ey = mon_y1 + 185 + int(math.sin(ex * 0.05 + t * 4) * 10)
    if (ex // 35) % 5 == 0:
        ey -= 48
    elif (ex // 35) % 5 == 1:
        ey += 22
    ecg_pts.append((ex, ey))
for i in range(len(ecg_pts) - 1):
    cv2.line(bg, ecg_pts[i], ecg_pts[i + 1], (60, 240, 120), 3, cv2.LINE_AA)

cv2.putText(bg, "LAB STATUS: RECOVERY PROTOCOL ACTIVE // PROGNOSIS: FULL RECOVERY", (mon_x1 + 30, mon_y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 220, 240), 1, cv2.LINE_AA)

# Header Banner
cv2.rectangle(bg, (int(1920 * 0.39), 20), (int(1920 * 0.61), 58), (15, 23, 42), -1)
cv2.rectangle(bg, (int(1920 * 0.39), 20), (int(1920 * 0.61), 58), (59, 130, 246), 1, cv2.LINE_AA)
cv2.putText(bg, "DRAMATIC CONVERSATION", (int(1920 * 0.428), 46), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imwrite("scratch/test_broadcast_full.png", bg)
print("Saved scratch/test_broadcast_full.png")
