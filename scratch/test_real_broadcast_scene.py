import urllib.request
import urllib.parse
import json
import cv2
import numpy as np
import math
import sys
from pathlib import Path

cache_bg_dir = Path("./data/cache/backgrounds")
cache_bg_dir.mkdir(parents=True, exist_ok=True)

def fetch_relevant_background_from_web(query: str, default_color=(35, 30, 28)) -> np.ndarray:
    """
    Searches the internet (Wikimedia Commons Open API) for a high-res real-world photographic background
    relevant to the query, caches it locally, and returns a 1920x1080 BGR image.
    """
    clean_q = query.strip()
    safe_name = "".join(c if c.isalnum() else "_" for c in clean_q[:30]).lower()
    local_path = cache_bg_dir / f"{safe_name}.jpg"

    if local_path.exists():
        img = cv2.imread(str(local_path))
        if img is not None and img.shape[0] > 100:
            return cv2.resize(img, (1920, 1080))

    # Query Wikimedia Commons API for real photograph
    search_terms = [clean_q, clean_q.split()[0] if clean_q else "hospital"]
    for q in search_terms:
        try:
            url = (
                "https://commons.wikimedia.org/w/api.php?action=query&generator=search"
                f"&gsrsearch={urllib.parse.quote(q)}"
                "&gsrnamespace=6&gsrlimit=8&prop=imageinfo&iiprop=url&format=json"
            )
            req = urllib.request.Request(url, headers={"User-Agent": "IncuBrixStudio/1.0 (contact@incubrix.ai)"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                pages = data.get("query", {}).get("pages", {})
                for pid, pdata in pages.items():
                    info = pdata.get("imageinfo", [{}])[0]
                    img_url = info.get("url", "")
                    # Ensure it's a standard image file
                    if any(img_url.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png"]):
                        img_req = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                        with urllib.request.urlopen(img_req, timeout=6) as img_resp:
                            raw = img_resp.read()
                            nparr = np.frombuffer(raw, np.uint8)
                            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            if img is not None and img.shape[0] > 300 and img.shape[1] > 300:
                                local_path.write_bytes(raw)
                                return cv2.resize(img, (1920, 1080))
        except Exception as e:
            print(f"Failed searching '{q}': {e}")

    # Fallback to rich dark gradient
    bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bg[:] = default_color
    return bg

# Test fetching real hospital background
print("Fetching real hospital background from internet API...")
bg = fetch_relevant_background_from_web("hospital room interior")
print(f"Background obtained: shape {bg.shape}")

# Apply dark cinematic overlay (so HUD elements & text pop with clarity)
overlay = np.zeros_like(bg)
overlay[:] = (20, 22, 28)
cv2.addWeighted(bg, 0.40, overlay, 0.60, 0, bg)

# Load real human portraits
from src.pipeline.avatar_service import avatar_service
doc_portrait = avatar_service.get_avatar_portrait("doctor")
pat_portrait = avatar_service.get_avatar_portrait("patient")

t = 2.4
lip_sync_energy = 0.65

# ── 1. PATIENT FEED (LEFT - LISTENING) ──
pcx, pcy = int(1920 * 0.20), int(1080 * 0.68)
pr = 100
pat_face = cv2.resize(pat_portrait, (pr * 2, pr * 2), interpolation=cv2.INTER_LANCZOS4)
p_mask = np.zeros((pr * 2, pr * 2), dtype=np.uint8)
cv2.circle(p_mask, (pr, pr), pr - 2, 255, -1, cv2.LINE_AA)

py1, py2 = pcy - pr, pcy + pr
px1, px2 = pcx - pr, pcx + pr
proi = bg[py1:py2, px1:px2]
for c in range(3):
    proi[:, :, c] = np.where(p_mask == 255, pat_face[:, :, c], proi[:, :, c])
# Silver border
cv2.circle(bg, (pcx, pcy), pr, (180, 190, 205), 3, cv2.LINE_AA)

# Patient Lower-Third Nameplate
p_badge_y = pcy + pr + 32
cv2.rectangle(bg, (pcx - 140, p_badge_y - 18), (pcx + 140, p_badge_y + 22), (15, 20, 28), -1)
cv2.rectangle(bg, (pcx - 140, p_badge_y - 18), (pcx + 140, p_badge_y + 22), (140, 160, 180), 2, cv2.LINE_AA)
cv2.circle(bg, (pcx - 120, p_badge_y - 1), 6, (120, 140, 160), -1, cv2.LINE_AA)
cv2.putText(bg, "PATIENT FEED", (pcx - 105, p_badge_y + 4), cv2.FONT_HERSHEY_DUPLEX, 0.52, (230, 235, 245), 1, cv2.LINE_AA)
cv2.putText(bg, "TELEHEALTH ROOM // LISTENING", (pcx - 120, p_badge_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140, 160, 190), 1, cv2.LINE_AA)

# ── 2. DOCTOR FEED (RIGHT - ACTIVELY SPEAKING) ──
dcx, dcy = int(1920 * 0.80), int(1080 * 0.68)
dr = 100
doc_face = cv2.resize(doc_portrait, (dr * 2, dr * 2), interpolation=cv2.INTER_LANCZOS4)

# Active speech aura
for ri in range(3):
    aura_r = dr + 6 + ri * 8 + int(lip_sync_energy * 10)
    cv2.circle(bg, (dcx, dcy), aura_r, (100, 220, 255), 2, cv2.LINE_AA)

d_mask = np.zeros((dr * 2, dr * 2), dtype=np.uint8)
cv2.circle(d_mask, (dr, dr), dr - 2, 255, -1, cv2.LINE_AA)
dy1, dy2 = dcy - dr, dcy + dr
dx1, dx2 = dcx - dr, dcx + dr
droi = bg[dy1:dy2, dx1:dx2]
for c in range(3):
    droi[:, :, c] = np.where(d_mask == 255, doc_face[:, :, c], droi[:, :, c])
# Cyan border
cv2.circle(bg, (dcx, dcy), dr, (80, 210, 255), 4, cv2.LINE_AA)

# Audio Waveform HUD around bottom of Doctor Feed
bar_count = 14
for b in range(bar_count):
    ang = 45 + b * (90 / bar_count)
    rad = math.radians(ang)
    bar_h = 8 + int(math.sin(t * 14 + b * 0.9) * 14 * lip_sync_energy + lip_sync_energy * 20)
    bx1 = int(dcx + math.cos(rad) * (dr + 8))
    by1 = int(dcy + math.sin(rad) * (dr + 8))
    bx2 = int(dcx + math.cos(rad) * (dr + 8 + bar_h))
    by2 = int(dcy + math.sin(rad) * (dr + 8 + bar_h))
    col = (100, 240, 180) if b % 2 == 0 else (60, 210, 255)
    cv2.line(bg, (bx1, by1), (bx2, by2), col, 3, cv2.LINE_AA)

# Doctor Lower-Third Nameplate
d_badge_y = dcy + dr + 32
cv2.rectangle(bg, (dcx - 150, d_badge_y - 18), (dcx + 150, d_badge_y + 22), (15, 20, 28), -1)
cv2.rectangle(bg, (dcx - 150, d_badge_y - 18), (dcx + 150, d_badge_y + 22), (80, 210, 255), 2, cv2.LINE_AA)
cv2.circle(bg, (dcx - 130, d_badge_y - 1), 6, (50, 240, 50), -1, cv2.LINE_AA)
cv2.putText(bg, "ATTENDING PHYSICIAN", (dcx - 114, d_badge_y + 4), cv2.FONT_HERSHEY_DUPLEX, 0.52, (245, 250, 255), 1, cv2.LINE_AA)
cv2.putText(bg, "CONSULTATION PROTOCOL // SPEAKING", (dcx - 130, d_badge_y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 220, 255), 1, cv2.LINE_AA)

# ── 3. CLINICAL TELEMETRY MONITOR (CENTER HUD) ──
mon_x1, mon_x2 = int(1920 * 0.32), int(1920 * 0.68)
mon_y1, mon_y2 = 80, 360
cv2.rectangle(bg, (mon_x1, mon_y1), (mon_x2, mon_y2), (10, 15, 20), -1)
cv2.rectangle(bg, (mon_x1, mon_y1), (mon_x2, mon_y2), (60, 180, 120), 2, cv2.LINE_AA)
cv2.putText(bg, "ICU CLINICAL MONITOR // REAL-TIME VITALS", (mon_x1 + 25, mon_y1 + 40), cv2.FONT_HERSHEY_DUPLEX, 0.65, (240, 245, 250), 1, cv2.LINE_AA)
cv2.putText(bg, "HR: 74 BPM    SpO2: 99%    BP: 120/80", (mon_x1 + 25, mon_y1 + 90), cv2.FONT_HERSHEY_DUPLEX, 0.85, (60, 240, 120), 2, cv2.LINE_AA)

# ECG waveform in monitor
ecg_pts = []
for ex in range(mon_x1 + 25, mon_x2 - 25, 4):
    ey = mon_y1 + 180 + int(math.sin(ex * 0.05 + t * 4) * 10)
    if (ex // 35) % 5 == 0:
        ey -= 45
    elif (ex // 35) % 5 == 1:
        ey += 20
    ecg_pts.append((ex, ey))
for i in range(len(ecg_pts) - 1):
    cv2.line(bg, ecg_pts[i], ecg_pts[i + 1], (60, 240, 120), 3, cv2.LINE_AA)

cv2.putText(bg, "LAB STATUS: RECOVERY PROTOCOL ACTIVE // PROGNOSIS: FULL RECOVERY", (mon_x1 + 25, mon_y2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 220, 240), 1, cv2.LINE_AA)

# Header banner
cv2.rectangle(bg, (int(1920 * 0.40), 25), (int(1920 * 0.60), 65), (15, 23, 42), -1)
cv2.rectangle(bg, (int(1920 * 0.40), 25), (int(1920 * 0.60), 65), (59, 130, 246), 1, cv2.LINE_AA)
cv2.putText(bg, "DRAMATIC CONVERSATION", (int(1920 * 0.435), 52), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

cv2.imwrite("scratch/test_real_broadcast_scene.png", bg)
print("Saved scratch/test_real_broadcast_scene.png")
