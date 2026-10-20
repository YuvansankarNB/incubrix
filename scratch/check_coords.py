import cv2
import numpy as np

chars = ["doctor", "patient", "teacher", "student", "business", "narrator", "farmer"]
for c in chars:
    img = cv2.imread(f"data/characters/rgba/{c}.png", cv2.IMREAD_UNCHANGED)
    h, w = img.shape[:2]
    print(f"{c:10s}: w={w}, h={h}")
