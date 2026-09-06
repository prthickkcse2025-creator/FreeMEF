import cv2
import numpy as np
import os

NORMAL_PATH = "my_test_context/scene001/02_normal.JPG"
OVER_PATH = "my_test_context/scene001/03_over.JPG"

OUT_DIR = "production/output/diagnostic"
os.makedirs(OUT_DIR, exist_ok=True)

normal = cv2.imread(NORMAL_PATH)
over = cv2.imread(OVER_PATH)

# Ensure same resolution
over = cv2.resize(
    over,
    (normal.shape[1], normal.shape[0]),
    interpolation=cv2.INTER_CUBIC
)

normal_gray = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
over_gray = cv2.cvtColor(over, cv2.COLOR_BGR2GRAY)

# How much brighter OVER is compared to NORMAL
gain = over_gray.astype(np.float32) / (
    normal_gray.astype(np.float32) + 1.0
)

gain_vis = np.clip(gain * 80, 0, 255).astype(np.uint8)

# Shadow candidate:
# dark in NORMAL + significantly brighter in OVER
dark_mask = normal_gray < 110
gain_mask = gain > 1.20

mask = (dark_mask & gain_mask).astype(np.uint8) * 255

# Smooth the mask
mask = cv2.GaussianBlur(mask, (0, 0), 15)

# Save diagnostics
cv2.imwrite(
    os.path.join(OUT_DIR, "01_normal_gray.png"),
    normal_gray
)

cv2.imwrite(
    os.path.join(OUT_DIR, "02_over_aligned.png"),
    over
)

cv2.imwrite(
    os.path.join(OUT_DIR, "03_brightness_gain.png"),
    gain_vis
)

cv2.imwrite(
    os.path.join(OUT_DIR, "04_shadow_mask.png"),
    mask
)

print("Diagnostic complete")
print("Output:", OUT_DIR)
print("Mask coverage:", round((mask > 20).mean() * 100, 2), "%")
print("Normal mean:", round(float(normal_gray.mean()), 2))
print("Over mean:", round(float(over_gray.mean()), 2))
