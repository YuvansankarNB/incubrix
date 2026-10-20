import cv2
import numpy as np
from pathlib import Path

def decontaminate_sprite(rgba_path: Path):
    img = cv2.imread(str(rgba_path), cv2.IMREAD_UNCHANGED)
    if img is None or img.shape[2] != 4:
        return
    
    b, g, r, alpha = cv2.split(img)
    
    # 1. Binary core mask of solid foreground
    _, solid_mask = cv2.threshold(alpha, 220, 255, cv2.THRESH_BINARY)
    
    # 2. Erode the mask by 2 pixels to strip any white-fringed boundary pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded_mask = cv2.erode(solid_mask, kernel, iterations=2)
    
    # 3. Color decontamination:
    # Use cv2.inpaint or nearest neighbor dilation from the eroded interior to bleed true colors outward
    color_img = cv2.merge([b, g, r])
    # Inpaint region is where original alpha was partially transparent or white fringed
    inpaint_mask = ((alpha > 0) & (eroded_mask == 0)).astype(np.uint8) * 255
    # Bleed true interior color outward into the border zone
    decontaminated_color = cv2.inpaint(color_img, inpaint_mask, 3, cv2.INPAINT_TELEA)
    
    # 4. Generate clean smooth antialiased alpha from the eroded mask
    # 1.5px Gaussian blur on the boundary
    smooth_alpha = cv2.GaussianBlur(eroded_mask, (3, 3), 0.8)
    
    # Combine decontaminated color with clean halo-free alpha
    db, dg, dr = cv2.split(decontaminated_color)
    clean_rgba = cv2.merge([db, dg, dr, smooth_alpha])
    
    cv2.imwrite(str(rgba_path), clean_rgba)
    
    # Verify new border color
    new_border = clean_rgba[(smooth_alpha > 0) & (smooth_alpha < 240)]
    avg_border = np.mean(new_border[:, :3], axis=0) if len(new_border) > 0 else [0, 0, 0]
    print(f"Decontaminated {rgba_path.name}: new border RGB = {avg_border}")

chars = ["doctor", "patient", "teacher", "student", "business", "narrator", "farmer"]
for c in chars:
    p = Path(f"data/characters/rgba/{c}.png")
    if p.exists():
        decontaminate_sprite(p)
