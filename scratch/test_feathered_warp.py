import cv2
import numpy as np

img = cv2.imread("data/characters/rgba/doctor.png", cv2.IMREAD_UNCHANGED)
h, w = img.shape[:2]

def animate_mouth_feathered(sprite, energy):
    out = sprite.copy()
    if energy <= 0.03:
        return out
    
    # Doctor mouth center is at y=405, x=507
    drop = int(energy * 14)
    if drop <= 0:
        return out

    # We define a region around mouth and chin:
    # y: 395 to 490, x: 430 to 585
    y1, y2 = 395, 490
    x1, x2 = 430, 585
    rw = x2 - x1
    rh = y2 - y1
    
    # Source ROI
    roi = sprite[y1:y2, x1:x2].copy()
    
    # Stretch lower half
    mid_rel = 18  # line below teeth
    top_part = roi[:mid_rel, :]
    bot_part = roi[mid_rel:, :]
    
    # Stretch bot_part vertically
    stretched_bot = cv2.resize(bot_part, (rw, bot_part.shape[0] + drop), interpolation=cv2.INTER_LANCZOS4)
    # Combined stretched ROI
    combined = np.vstack([top_part, stretched_bot])[:rh, :]
    
    # Soft inner mouth cavity between teeth and lower lip
    cv2.ellipse(combined, (rw // 2, mid_rel + drop // 2), (24, max(2, drop // 2)), 0, 0, 360, (25, 20, 50, 255), -1, cv2.LINE_AA)

    # Feathered elliptical mask for seamless blending
    mask = np.zeros((rh, rw), dtype=np.float32)
    cv2.ellipse(mask, (rw // 2, rh // 2), (rw // 2 - 4, rh // 2 - 4), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
    mask = cv2.GaussianBlur(mask, (15, 15), 5.0)[:, :, np.newaxis]
    
    # Blend onto target
    target_roi = out[y1:y2, x1:x2].astype(np.float32)
    blended = combined.astype(np.float32) * mask + target_roi * (1.0 - mask)
    out[y1:y2, x1:x2] = np.clip(blended, 0, 255).astype(np.uint8)
    
    return out

out_speaking = animate_mouth_feathered(img, 0.8)
cv2.imwrite("scratch/test_doc_feathered.png", out_speaking)
print("Saved scratch/test_doc_feathered.png")
