import cv2
import numpy as np
from pathlib import Path

def extract_alpha(img: np.ndarray, role: str) -> np.ndarray:
    """
    Extracts high quality RGBA image with smooth alpha transparency.
    Uses edge-aware flood-fill from corners and luminance difference.
    """
    h, w = img.shape[:2]
    # Check top corners
    top_left = img[0, 0].astype(float)
    top_right = img[0, w - 1].astype(float)
    bg_ref = (top_left + top_right) / 2.0

    # Color distance from background
    diff = np.linalg.norm(img.astype(float) - bg_ref, axis=2)

    # For student, teacher, doctor, etc. with near-white background
    if np.mean(bg_ref) > 230:
        # Distance thresholding with soft transition
        alpha = np.clip((diff - 8.0) / 22.0 * 255.0, 0, 255).astype(np.uint8)
    else:
        alpha = np.clip((diff - 12.0) / 28.0 * 255.0, 0, 255).astype(np.uint8)

    # Use morphological closing and opening to preserve character interior
    # Flood-fill exterior background to avoid cutting interior white areas (e.g. doctor's white coat)
    mask = (alpha < 180).astype(np.uint8) * 255
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)
    # Flood fill from top-left, top-right, and top-middle
    cv2.floodFill(mask, flood_mask, (0, 0), 128)
    cv2.floodFill(mask, flood_mask, (w - 1, 0), 128)
    cv2.floodFill(mask, flood_mask, (w // 2, 0), 128)

    # Exterior is marked with 128. Anything else is foreground!
    is_exterior = (mask == 128)

    # Final smooth alpha: 0 for exterior, 255 for interior, feathered boundary
    final_alpha = np.ones((h, w), dtype=np.uint8) * 255
    final_alpha[is_exterior] = 0

    # Smooth the boundary
    # Blur edge slightly for anti-aliasing
    edge_zone = cv2.Canny(final_alpha, 100, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edge_zone = cv2.dilate(edge_zone, kernel, iterations=1)
    
    soft_alpha = cv2.GaussianBlur(final_alpha, (5, 5), 1.0)
    final_alpha[edge_zone > 0] = soft_alpha[edge_zone > 0]

    b, g, r = cv2.split(img)
    rgba = cv2.merge([b, g, r, final_alpha])
    return rgba

out_dir = Path("data/characters/rgba")
out_dir.mkdir(parents=True, exist_ok=True)

for name in ["doctor", "patient", "teacher", "student", "business", "narrator", "farmer"]:
    p = Path(f"data/characters/{name}.jpg")
    img = cv2.imread(str(p))
    rgba = extract_alpha(img, name)
    out_p = out_dir / f"{name}.png"
    cv2.imwrite(str(out_p), rgba)
    print(f"Saved {out_p}: non-zero alpha = {np.count_nonzero(rgba[:, :, 3])} / {rgba.shape[0]*rgba.shape[1]}")
