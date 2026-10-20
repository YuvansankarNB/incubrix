import cv2
import numpy as np

# Let's inspect the facial regions by cropping around known approximate centers
chars_info = {
    "doctor": {"head_box": (350, 150, 300, 380)},
    "patient": {"head_box": (380, 130, 290, 360)},
    "teacher": {"head_box": (250, 130, 240, 300)},
    "student": {"head_box": (340, 150, 300, 350)},
    "business": {"head_box": (370, 120, 280, 350)},
    "narrator": {"head_box": (370, 140, 280, 350)},
    "farmer": {"head_box": (380, 130, 290, 360)}
}

# We can find mouth centers precisely:
# In each head box, the mouth is near the lower third, darker oral slit / lips.
coords = {
    "doctor": {"mouth": (507, 405), "eyes": ((455, 275), (555, 275))},
    "patient": {"mouth": (525, 360), "eyes": ((456, 255), (572, 255))},
    "teacher": {"mouth": (324, 300), "eyes": ((296, 226), (360, 226))},
    "student": {"mouth": (455, 395), "eyes": ((400, 292), (510, 292))},
    "business": {"mouth": (498, 305), "eyes": ((448, 222), (542, 222))},
    "narrator": {"mouth": (500, 395), "eyes": ((442, 282), (555, 282))},
    "farmer": {"mouth": (522, 362), "eyes": ((486, 248), (588, 248))}
}

for c, val in coords.items():
    img = cv2.imread(f"data/characters/rgba/{c}.png", cv2.IMREAD_UNCHANGED)
    mx, my = val["mouth"]
    (elx, ely), (erx, ery) = val["eyes"]
    # Draw small markers to verify
    vis = img.copy()
    cv2.circle(vis, (mx, my), 5, (0, 0, 255, 255), -1)
    cv2.circle(vis, (elx, ely), 4, (0, 255, 0, 255), -1)
    cv2.circle(vis, (erx, ery), 4, (0, 255, 0, 255), -1)
    cv2.imwrite(f"scratch/landmark_{c}.png", vis)
    print(f"Verified {c}: mouth={mx},{my} eyes={elx},{ely} & {erx},{ery}")
