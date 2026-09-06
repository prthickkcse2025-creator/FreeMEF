import cv2
import numpy as np
import os

# -------------------------------------------------
# INPUTS
# -------------------------------------------------
FUSED_PATH = "production/output/order_test/scene001.png"
NORMAL_PATH = "my_test_context/scene001/02_normal.JPG"
OVER_PATH = "my_test_context/scene001/03_over.JPG"

OUT_DIR = "production/output/exposure_replaced"
OUT_PATH = os.path.join(OUT_DIR, "scene001.png")

os.makedirs(OUT_DIR, exist_ok=True)


def read_img(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Cannot read: {path}")
    return img.astype(np.float32) / 255.0


# -------------------------------------------------
# LOAD
# -------------------------------------------------
fused = read_img(FUSED_PATH)
normal = read_img(NORMAL_PATH)
over = read_img(OVER_PATH)

h, w = fused.shape[:2]

normal = cv2.resize(normal, (w, h), interpolation=cv2.INTER_LINEAR)
over = cv2.resize(over, (w, h), interpolation=cv2.INTER_LINEAR)

# -------------------------------------------------
# 1. DETECT DARK SHADOWS FROM FUSED IMAGE
# -------------------------------------------------
gray = cv2.cvtColor(
    np.clip(fused * 255, 0, 255).astype(np.uint8),
    cv2.COLOR_BGR2GRAY
).astype(np.float32) / 255.0

# Smooth luminance to avoid noisy masks
gray_smooth = cv2.GaussianBlur(
    gray, (0, 0), sigmaX=9, sigmaY=9
)

# Shadow mask:
# 1 in dark areas
# 0 in normal/bright areas
shadow_mask = np.clip(
    (0.48 - gray_smooth) / 0.28,
    0.0,
    1.0
)

# Make transition smoother
shadow_mask = cv2.GaussianBlur(
    shadow_mask, (0, 0), sigmaX=15, sigmaY=15
)

# -------------------------------------------------
# 2. PREVENT USING OVER-EXPOSED / CLIPPED PIXELS
# -------------------------------------------------
over_gray = cv2.cvtColor(
    np.clip(over * 255, 0, 255).astype(np.uint8),
    cv2.COLOR_BGR2GRAY
).astype(np.float32) / 255.0

# Reduce replacement where OVER is clipped
valid_over = np.clip(
    (0.98 - over_gray) / 0.15,
    0.0,
    1.0
)

mask = shadow_mask * valid_over

# -------------------------------------------------
# 3. LOCAL EXPOSURE MATCHING
# -------------------------------------------------
# Estimate luminance
fused_lum = (
    0.114 * fused[:, :, 0] +
    0.587 * fused[:, :, 1] +
    0.299 * fused[:, :, 2]
)

over_lum = (
    0.114 * over[:, :, 0] +
    0.587 * over[:, :, 1] +
    0.299 * over[:, :, 2]
)

# Use a controlled ratio.
# Prevent excessive brightness/color explosion.
ratio = fused_lum / np.maximum(over_lum, 1e-4)

ratio = cv2.GaussianBlur(
    ratio, (0, 0), sigmaX=25, sigmaY=25
)

ratio = np.clip(ratio, 0.35, 1.50)

matched_over = over * ratio[:, :, None]

# -------------------------------------------------
# 4. DETAIL-PRESERVING REPLACEMENT
# -------------------------------------------------
# The OVER exposure provides actual shadow information.
# FreeMEF remains dominant outside shadow regions.
result = (
    fused * (1.0 - mask[:, :, None]) +
    matched_over * mask[:, :, None]
)

# -------------------------------------------------
# 5. PRESERVE FreeMEF HIGH-FREQUENCY DETAIL
# -------------------------------------------------
fused_blur = cv2.GaussianBlur(
    fused, (0, 0), sigmaX=2
)

detail = fused - fused_blur

result = np.clip(
    result + detail * 0.35,
    0.0,
    1.0
)

# -------------------------------------------------
# SAVE
# -------------------------------------------------
result_u8 = np.clip(
    result * 255.0,
    0,
    255
).astype(np.uint8)

cv2.imwrite(OUT_PATH, result_u8)

print("Exposure replacement complete")
print("Output:", OUT_PATH)
print("Resolution:", result_u8.shape)
print("Shadow replacement coverage:",
      round(float(mask.mean()) * 100, 2), "%")
