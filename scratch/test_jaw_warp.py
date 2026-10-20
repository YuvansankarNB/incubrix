import cv2
import numpy as np

img = cv2.imread("data/characters/rgba/doctor.png", cv2.IMREAD_UNCHANGED)
h, w = img.shape[:2]

def animate_mouth_natural(sprite, energy):
    out = sprite.copy()
    if energy <= 0.02:
        return out
    
    # Doctor mouth center is at y=405, x=507
    # Mouth height from 385 to 430, chin down to 480
    y_top = 390
    y_mid = 412
    y_bot = 480
    x_left = 440
    x_right = 574
    
    # Amount of jaw drop
    drop = int(energy * 16)
    if drop <= 0:
        return out

    # We shift/stretch the lower lip and chin downwards
    # Region 1: Inside mouth opening (expand inner mouth)
    roi_mouth = out[y_mid:y_bot, x_left:x_right]
    stretched_roi = cv2.resize(roi_mouth, (x_right - x_left, (y_bot - y_mid) + drop), interpolation=cv2.INTER_LANCZOS4)
    
    # Place stretched lower mouth and chin
    target_y2 = min(h, y_mid + stretched_roi.shape[0])
    src_h = target_y2 - y_mid
    out[y_mid:target_y2, x_left:x_right] = stretched_roi[:src_h, :]

    # Add subtle natural inner mouth shadow between lips
    cv2.ellipse(out, (507, y_mid + drop // 2), (28, max(2, drop // 2)), 0, 0, 360, (20, 15, 45, 255), -1, cv2.LINE_AA)
    
    return out

out_neutral = img.copy()
out_speaking = animate_mouth_natural(img, 0.8)

cv2.imwrite("scratch/test_doc_natural_neutral.png", out_neutral)
cv2.imwrite("scratch/test_doc_natural_speaking.png", out_speaking)
print("Saved neutral and natural speaking frames.")
