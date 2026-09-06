import cv2
import numpy as np
import os

BASE_PATH = "production/output/order_test/scene001.png"
OVER_PATH = "my_test_context/scene001/03_over.JPG"
NORMAL_PATH = "my_test_context/scene001/02_normal.JPG"

OUT_DIR = "production/output/improved_shadow"
os.makedirs(OUT_DIR, exist_ok=True)

# Read images
base = cv2.imread(BASE_PATH)
over = cv2.imread(OVER_PATH)
normal = cv2.imread(NORMAL_PATH)

if base is None:
    raise FileNotFoundError(f"Cannot read {BASE_PATH}")

if over is None:
    raise FileNotFoundError(f"Cannot read {OVER_PATH}")

h, w = base.shape[:2]

# Ensure same resolution
over = cv2.resize(over, (w, h), interpolation=cv2.INTER_LINEAR)

# Convert to LAB for brightness analysis
base_lab = cv2.cvtColor(base, cv2.COLOR_BGR2LAB)
over_lab = cv2.cvtColor(over, cv2.COLOR_BGR2LAB)

base_l = base_lab[:, :, 0].astype(np.float32)
over_l = over_lab[:, :, 0].astype(np.float32)

# ------------------------------------------
# SHADOW CONFIDENCE MASK
# ------------------------------------------

# Strong when FreeMEF is dark
darkness = np.clip((120 - base_l) / 120, 0, 1)

# Strong when OVER image contains brighter information
gain = np.clip((over_l - base_l) / 100, 0, 1)

# Combine both conditions
mask = darkness * gain

# Ignore weak regions
mask[mask < 0.15] = 0

# Smooth transitions
mask = cv2.GaussianBlur(mask, (0, 0), 15)

# Limit replacement strength
mask = np.clip(mask * 0.85, 0, 0.85)

# ------------------------------------------
# COLOR-CORRECT OVER IMAGE
# ------------------------------------------

base_f = base.astype(np.float32) / 255.0
over_f = over.astype(np.float32) / 255.0

# Match global color statistics
base_mean = np.mean(base_f.reshape(-1, 3), axis=0)
over_mean = np.mean(over_f.reshape(-1, 3), axis=0)

scale = base_mean / (over_mean + 1e-6)

over_corrected = np.clip(over_f * scale, 0, 1)

# ------------------------------------------
# SELECTIVE SHADOW RECOVERY
# ------------------------------------------

mask_3 = np.dstack([mask, mask, mask])

result = (
    base_f * (1.0 - mask_3)
    + over_corrected * mask_3
)

result = np.clip(result * 255, 0, 255).astype(np.uint8)

# Save output
cv2.imwrite(
    f"{OUT_DIR}/scene001.png",
    result
)

# Save mask for inspection
mask_vis = np.clip(mask * 255, 0, 255).astype(np.uint8)

cv2.imwrite(
    f"{OUT_DIR}/shadow_mask.png",
    mask_vis
)

print("Shadow recovery complete")
print("Output:", f"{OUT_DIR}/scene001.png")
print("Mask coverage:", round(float((mask > 0.05).mean()) * 100, 2), "%")
print("Mean mask strength:", round(float(mask.mean()), 4))
