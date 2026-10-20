import cv2
import numpy as np

# Load background
bg_path = "data/cache/backgrounds/hospital_room.jpg"
bg = cv2.imread(bg_path)
if bg is None:
    # Create clean modern gradient clinic if file missing
    bg = np.zeros((1080, 1920, 3), dtype=np.uint8)
    bg[:, :] = (235, 240, 245)
else:
    bg = cv2.resize(bg, (1920, 1080))

canvas = bg.copy()

# Load doctor and patient RGBA
doc_rgba = cv2.imread("data/characters/rgba/doctor.png", cv2.IMREAD_UNCHANGED)
pat_rgba = cv2.imread("data/characters/rgba/patient.png", cv2.IMREAD_UNCHANGED)

def overlay_character(base, sprite_rgba, center_x, bottom_y, target_height):
    h, w = sprite_rgba.shape[:2]
    scale = target_height / float(h)
    new_w = int(w * scale)
    new_h = int(target_height)
    resized = cv2.resize(sprite_rgba, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    x1 = center_x - new_w // 2
    x2 = x1 + new_w
    y2 = bottom_y
    y1 = y2 - new_h
    
    # Clip bounds
    bx1 = max(0, x1)
    bx2 = min(base.shape[1], x2)
    by1 = max(0, y1)
    by2 = min(base.shape[0], y2)
    
    sx1 = bx1 - x1
    sx2 = sx1 + (bx2 - bx1)
    sy1 = by1 - y1
    sy2 = sy1 + (by2 - by1)
    
    if bx2 > bx1 and by2 > by1:
        alpha = (resized[sy1:sy2, sx1:sx2, 3] / 255.0)[:, :, np.newaxis]
        color = resized[sy1:sy2, sx1:sx2, :3]
        base[by1:by2, bx1:bx2] = (color * alpha + base[by1:by2, bx1:bx2] * (1.0 - alpha)).astype(np.uint8)

# Doctor on the left (cx=520, bottom=1080, height=840)
overlay_character(canvas, doc_rgba, 520, 1080, 840)

# Patient on the right (cx=1400, bottom=1080, height=840)
overlay_character(canvas, pat_rgba, 1400, 1080, 840)

cv2.imwrite("scratch/test_doctor_patient_composite.png", canvas)
print("Saved scratch/test_doctor_patient_composite.png")
