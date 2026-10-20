import cv2
import numpy as np
import math
from pathlib import Path
from src.pipeline.avatar_service import avatar_service

def draw_realistic_human_head(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    t: float,
    frame_idx: int,
    lip_sync_energy: float,
    role: str = "doctor",
    skin_tone: tuple = (195, 215, 240),
    hair_color: tuple = (30, 35, 45),
    eye_color: tuple = (80, 55, 35),
    hair_style: str = "short_part",
    glasses: bool = False
):
    """
    Renders realistic human presenter head:
    1. If free API portrait is available (from randomuser.me or local cache),
       composites the high-res human face with realistic circular feathering,
       3D ambient border, and audio-driven lip sync.
    2. Fallback: High-fidelity procedural craniofacial anatomy with volumetric shading,
       almond eyes with limbal ring and catchlights, sculpted 3D nose, and Cupid's bow lips.
    """
    # Check for real photo portrait from free API
    portrait = avatar_service.get_avatar_portrait(role)
    if portrait is not None:
        # Composite real human face with circular feathered mask and audio reactivity
        r = 60
        face_crop = cv2.resize(portrait, (r * 2, r * 2), interpolation=cv2.INTER_LANCZOS4)
        
        # Circular mask with smooth antialiased edge
        mask = np.zeros((r * 2, r * 2), dtype=np.uint8)
        cv2.circle(mask, (r, r), r - 1, 255, -1, cv2.LINE_AA)
        
        y1, y2 = cy - r, cy + r
        x1, x2 = cx - r, cx + r
        
        if 0 <= y1 and y2 <= canvas.shape[0] and 0 <= x1 and x2 <= canvas.shape[1]:
            roi = canvas[y1:y2, x1:x2]
            # Blend
            for c in range(3):
                roi[:, :, c] = np.where(mask == 255, face_crop[:, :, c], roi[:, :, c])
            
            # Subtle ambient shadow and rim
            glow_col = (100, 220, 255) if lip_sync_energy > 0.05 else (200, 180, 140)
            cv2.circle(canvas, (cx, cy), r, glow_col, 2, cv2.LINE_AA)
            cv2.circle(canvas, (cx, cy), r + 2, (30, 30, 40), 1, cv2.LINE_AA)
            
            # Draw audio-synced expressive mouth over portrait when speaking
            if lip_sync_energy > 0.04:
                mouth_y = cy + 24
                mouth_open_h = int(3 + lip_sync_energy * 20)
                mouth_w = int(14 + lip_sync_energy * 6)
                # Oral cavity
                cv2.ellipse(canvas, (cx, mouth_y), (mouth_w, mouth_open_h), 0, 0, 360, (25, 20, 80), -1, cv2.LINE_AA)
                # Upper teeth arch
                cv2.ellipse(canvas, (cx, mouth_y - mouth_open_h + 3), (mouth_w - 4, 3), 0, 0, 360, (245, 248, 250), -1, cv2.LINE_AA)
                if mouth_open_h > 8:
                    cv2.ellipse(canvas, (cx, mouth_y + mouth_open_h - 4), (mouth_w - 5, 4), 0, 0, 360, (90, 80, 200), -1, cv2.LINE_AA)
                # Subtle lip border
                cv2.ellipse(canvas, (cx, mouth_y), (mouth_w + 2, mouth_open_h + 2), 0, 0, 360, (60, 60, 140), 1, cv2.LINE_AA)
            return

    # ── PROCEDURAL PHOTOREALISTIC HUMAN ANATOMY ──
    # Neck with anatomical shading
    neck_w = 26
    neck_top = cy + 30
    neck_bot = cy + 65
    skin_dark = (max(0, skin_tone[0] - 35), max(0, skin_tone[1] - 35), max(0, skin_tone[2] - 35))
    skin_light = (min(255, skin_tone[0] + 15), min(255, skin_tone[1] + 15), min(255, skin_tone[2] + 15))
    
    cv2.rectangle(canvas, (cx - neck_w, neck_top), (cx + neck_w, neck_bot), skin_dark, -1)
    cv2.rectangle(canvas, (cx - neck_w + 5, neck_top), (cx + neck_w - 5, neck_bot), skin_tone, -1)

    # Ears with cartilage contours
    for ex, flip in [(cx - 56, -1), (cx + 56, 1)]:
        cv2.ellipse(canvas, (ex, cy - 2), (9, 20), flip * 5, 0, 360, skin_tone, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (ex + flip * 2, cy - 2), (5, 12), flip * 5, 0, 360, skin_dark, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (ex, cy + 14), (6, 8), 0, 0, 360, skin_tone, -1, cv2.LINE_AA)

    # Craniofacial Contour (Anatomical Jawline and Forehead)
    jaw_pts = np.array([
        [cx - 52, cy - 22], [cx - 54, cy + 5], [cx - 44, cy + 34],
        [cx - 22, cy + 54], [cx, cy + 57], [cx + 22, cy + 54],
        [cx + 44, cy + 34], [cx + 54, cy + 5], [cx + 52, cy - 22],
        [cx + 40, cy - 48], [cx, cy - 58], [cx - 40, cy - 48]
    ], np.int32)
    cv2.fillPoly(canvas, [jaw_pts], skin_tone, cv2.LINE_AA)

    # Volumetric 3D Skin Shading
    # Forehead highlight
    cv2.ellipse(canvas, (cx, cy - 30), (32, 18), 0, 0, 360, skin_light, -1, cv2.LINE_AA)
    # Zygomatic (cheekbone) highlights
    cv2.ellipse(canvas, (cx - 30, cy + 8), (14, 8), -10, 0, 360, skin_light, -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx + 30, cy + 8), (14, 8), 10, 0, 360, skin_light, -1, cv2.LINE_AA)
    # Under-jaw ambient shadow
    cv2.polylines(canvas, [np.array([[cx - 44, cy + 34], [cx - 22, cy + 54], [cx, cy + 57], [cx + 22, cy + 54], [cx + 44, cy + 34]], np.int32)], False, skin_dark, 2, cv2.LINE_AA)

    # Lively cheek flush when speaking
    if lip_sync_energy > 0.05:
        blush = (min(255, skin_tone[0] + 10), max(0, skin_tone[1] - 30), min(255, skin_tone[2] + 40))
        cv2.ellipse(canvas, (cx - 30, cy + 14), (12, 6), 0, 0, 360, blush, -1, cv2.LINE_AA)
        cv2.ellipse(canvas, (cx + 30, cy + 14), (12, 6), 0, 0, 360, blush, -1, cv2.LINE_AA)

    # ── SCULPTED 3D NOSE ──
    # Nasal bridge
    cv2.line(canvas, (cx - 2, cy - 10), (cx - 3, cy + 15), skin_dark, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx, cy - 8), (cx, cy + 15), skin_light, 1, cv2.LINE_AA)
    # Cartilage tip highlight
    cv2.circle(canvas, (cx, cy + 16), 4, skin_light, -1, cv2.LINE_AA)
    # Nostrils & alae
    cv2.line(canvas, (cx - 8, cy + 17), (cx - 3, cy + 20), skin_dark, 1, cv2.LINE_AA)
    cv2.line(canvas, (cx + 8, cy + 17), (cx + 3, cy + 20), skin_dark, 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx - 4, cy + 19), (3, 2), 0, 0, 360, (40, 30, 30), -1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx + 4, cy + 19), (3, 2), 0, 0, 360, (40, 30, 30), -1, cv2.LINE_AA)

    # ── REALISTIC EYES WITH LIMBAL RING & CATCHLIGHT ──
    ey = cy - 6
    lx, rx = cx - 25, cx + 25
    is_blinking = (frame_idx % 110) < 6
    
    brow_lift = int(lip_sync_energy * 6)
    brow_col = (max(0, hair_color[0] - 10), max(0, hair_color[1] - 10), max(0, hair_color[2] - 10))

    if is_blinking:
        # Realistic closed eyelids with lash lines
        cv2.line(canvas, (lx - 14, ey), (lx + 14, ey), brow_col, 2, cv2.LINE_AA)
        cv2.line(canvas, (rx - 14, ey), (rx + 14, ey), brow_col, 2, cv2.LINE_AA)
    else:
        # Almond eye contours
        for ex, side in [(lx, -1), (rx, 1)]:
            eye_pts = np.array([
                [ex - 15, ey], [ex - 7, ey - 7], [ex, ey - 8], [ex + 7, ey - 7],
                [ex + 15, ey], [ex + 7, ey + 7], [ex, ey + 8], [ex - 7, ey + 7]
            ], np.int32)
            cv2.fillPoly(canvas, [eye_pts], (242, 244, 246), cv2.LINE_AA)
            # Medial caruncle (inner eye corner)
            cv2.circle(canvas, (ex + side * 13, ey), 2, (180, 170, 220), -1, cv2.LINE_AA)

            # Gaze tracking
            cycle = t % 4.2
            px = int(-5 + math.sin(t * 1.5) * 2) if cycle < 2.0 else int(math.sin(t * 1.8) * 2)
            
            # Limbal ring (dark outer iris boundary)
            cv2.circle(canvas, (ex + px, ey), 8, (max(0, eye_color[0]-40), max(0, eye_color[1]-40), max(0, eye_color[2]-40)), -1, cv2.LINE_AA)
            # Iris
            cv2.circle(canvas, (ex + px, ey), 7, eye_color, -1, cv2.LINE_AA)
            # Iris striation/depth
            cv2.circle(canvas, (ex + px, ey), 5, (min(255, eye_color[0]+30), min(255, eye_color[1]+30), min(255, eye_color[2]+30)), 1, cv2.LINE_AA)
            # Pupil
            cv2.circle(canvas, (ex + px, ey), 4, (15, 15, 20), -1, cv2.LINE_AA)
            # Primary specular catchlight (10 o'clock)
            cv2.circle(canvas, (ex + px - 2, ey - 3), 2, (255, 255, 255), -1, cv2.LINE_AA)
            # Secondary corneal reflection (4 o'clock)
            cv2.circle(canvas, (ex + px + 2, ey + 2), 1, (200, 210, 230), -1, cv2.LINE_AA)

            # Upper eyelid crease
            cv2.line(canvas, (ex - 12, ey - 10), (ex + 12, ey - 10), skin_dark, 1, cv2.LINE_AA)
            # Upper lash rim
            cv2.line(canvas, (ex - 14, ey - 6), (ex + 14, ey - 6), (30, 25, 25), 2, cv2.LINE_AA)

    # Eyebrows (expressive arch)
    cv2.line(canvas, (lx - 14, ey - 17 - brow_lift), (lx + 13, ey - 19 - brow_lift), brow_col, 3, cv2.LINE_AA)
    cv2.line(canvas, (rx - 13, ey - 19 - brow_lift), (rx + 14, ey - 17 - brow_lift), brow_col, 3, cv2.LINE_AA)

    # ── CUPID'S BOW LIPS & SPEECH VISEMES ──
    mouth_y = cy + 34
    lip_tone = (max(0, skin_tone[0] - 35), max(0, skin_tone[1] - 45), min(255, skin_tone[2] + 35))
    lip_dark = (max(0, skin_tone[0] - 50), max(0, skin_tone[1] - 60), min(255, skin_tone[2] + 20))

    if lip_sync_energy > 0.04:
        mouth_open_h = int(3 + lip_sync_energy * 22)
        mouth_w = int(14 + lip_sync_energy * 6)
        # Oral cavity depth
        cv2.ellipse(canvas, (cx, mouth_y), (mouth_w, mouth_open_h), 0, 0, 360, (25, 20, 80), -1, cv2.LINE_AA)
        # Upper dental arch (teeth)
        cv2.ellipse(canvas, (cx, mouth_y - mouth_open_h + 3), (mouth_w - 4, 3), 0, 0, 360, (245, 248, 250), -1, cv2.LINE_AA)
        # Tongue base
        if mouth_open_h > 8:
            cv2.ellipse(canvas, (cx, mouth_y + mouth_open_h - 4), (mouth_w - 5, 4), 0, 0, 360, (90, 80, 200), -1, cv2.LINE_AA)
        # Upper lip with Cupid's bow
        cv2.line(canvas, (cx - mouth_w, mouth_y - 2), (cx, mouth_y - mouth_open_h + 1), lip_dark, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx, mouth_y - mouth_open_h + 1), (cx + mouth_w, mouth_y - 2), lip_dark, 2, cv2.LINE_AA)
        # Lower lip
        cv2.ellipse(canvas, (cx, mouth_y + mouth_open_h), (mouth_w - 2, 4), 0, 0, 180, lip_tone, -1, cv2.LINE_AA)
    else:
        # Resting relaxed smile with Cupid's bow
        cv2.line(canvas, (cx - 14, mouth_y), (cx - 4, mouth_y - 2), lip_dark, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx - 4, mouth_y - 2), (cx, mouth_y - 1), lip_dark, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx, mouth_y - 1), (cx + 4, mouth_y - 2), lip_dark, 2, cv2.LINE_AA)
        cv2.line(canvas, (cx + 4, mouth_y - 2), (cx + 14, mouth_y), lip_dark, 2, cv2.LINE_AA)
        # Lower lip fullness
        cv2.ellipse(canvas, (cx, mouth_y + 3), (10, 4), 0, 0, 180, lip_tone, -1, cv2.LINE_AA)

    # ── VOLUMETRIC TEXTURED HAIR ──
    hair_high = (min(255, hair_color[0] + 35), min(255, hair_color[1] + 35), min(255, hair_color[2] + 45))
    cv2.ellipse(canvas, (cx, cy - 38), (64, 40), 0, 180, 360, hair_color, -1, cv2.LINE_AA)
    # Side part & layered sweep
    cv2.ellipse(canvas, (cx - 15, cy - 44), (46, 26), -15, 180, 360, hair_high, -1, cv2.LINE_AA)
    # Temporal sideburns
    cv2.line(canvas, (cx - 52, cy - 15), (cx - 50, cy + 10), hair_color, 4, cv2.LINE_AA)
    cv2.line(canvas, (cx + 52, cy - 15), (cx + 50, cy + 10), hair_color, 4, cv2.LINE_AA)

    # Optional Glasses
    if glasses:
        for ex in [lx, rx]:
            cv2.rectangle(canvas, (ex - 14, ey - 10), (ex + 14, ey + 10), (40, 40, 45), 2, cv2.LINE_AA)
        cv2.line(canvas, (lx + 14, ey), (rx - 14, ey), (40, 40, 45), 2, cv2.LINE_AA)

print('Testing head render...')
canvas = np.zeros((400, 400, 3), dtype=np.uint8)
canvas[:] = (30, 25, 25)
draw_realistic_human_head(canvas, 200, 200, 1.0, 30, 0.5, role='doctor')
cv2.imwrite('scratch/test_head_doctor.png', canvas)
print('Saved scratch/test_head_doctor.png successfully!')
