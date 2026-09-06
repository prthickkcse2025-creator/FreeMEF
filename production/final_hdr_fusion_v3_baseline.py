import cv2
import numpy as np
import os

# =====================================================
# PATHS
# =====================================================

NORMAL_PATH = "my_test_context/scene001/02_normal.JPG"
UNDER_PATH  = "my_test_context/scene001/01_under.JPG"
OVER_PATH   = "my_test_context/scene001/03_over.JPG"

OUT_PATH = "production/output/final_hdr/scene001_v3.png"


# =====================================================
# LOAD
# =====================================================

normal = cv2.imread(NORMAL_PATH)
under = cv2.imread(UNDER_PATH)
over = cv2.imread(OVER_PATH)

if normal is None or under is None or over is None:
    raise FileNotFoundError("One or more input images could not be loaded")

# Ensure same resolution
h, w = normal.shape[:2]
under = cv2.resize(under, (w, h))
over = cv2.resize(over, (w, h))

normal_f = normal.astype(np.float32) / 255.0
under_f = under.astype(np.float32) / 255.0
over_f = over.astype(np.float32) / 255.0


# =====================================================
# NORMAL = COLOR / WHITE BALANCE / DETAIL ANCHOR
# =====================================================

normal_gray = cv2.cvtColor(normal_f, cv2.COLOR_BGR2GRAY)
over_gray = cv2.cvtColor(over_f, cv2.COLOR_BGR2GRAY)


# =====================================================
# SHADOW MASK
#
# Strong in dark regions.
# Smoothly falls to zero before normal midtones.
# =====================================================

shadow_mask = np.clip(
    (0.55 - normal_gray) / 0.55,
    0.0,
    1.0
)

# Keep stronger recovery in genuine shadows
shadow_mask = shadow_mask ** 1.5

# Prevent hard edges
shadow_mask = cv2.GaussianBlur(
    shadow_mask,
    (0, 0),
    sigmaX=7,
    sigmaY=7
)


# =====================================================
# SHADOW BRIGHTNESS RECOVERY
#
# Use luminance relationship from OVER.
# Preserve NORMAL color ratios.
# =====================================================

gain = over_gray / (normal_gray + 1e-4)

# Prevent extreme amplification
gain = np.clip(gain, 1.0, 2.0)

# Limit recovery to avoid washing out shadows
recovery_strength = 0.60

brightness_gain = (
    1.0
    + (gain - 1.0)
    * shadow_mask
    * recovery_strength
)

shadow_result = normal_f * brightness_gain[:, :, None]

shadow_result = np.clip(shadow_result, 0.0, 1.0)


# =====================================================
# HIGHLIGHT MASK
# =====================================================

shadow_result_gray = cv2.cvtColor(
    shadow_result.astype(np.float32),
    cv2.COLOR_BGR2GRAY
)

highlight_mask = np.clip(
    (shadow_result_gray - 0.75) / 0.25,
    0.0,
    1.0
)

highlight_mask = highlight_mask ** 1.5

highlight_mask = cv2.GaussianBlur(
    highlight_mask,
    (0, 0),
    sigmaX=5,
    sigmaY=5
)


# =====================================================
# HIGHLIGHT RECOVERY FROM UNDER
# =====================================================

highlight_strength = 0.50

highlight_weight = (
    highlight_mask
    * highlight_strength
)

result = (
    shadow_result
    * (1.0 - highlight_weight[:, :, None])
    + under_f
    * highlight_weight[:, :, None]
)

result = np.clip(result, 0.0, 1.0)


# =====================================================
# SAVE
# =====================================================

result_u8 = np.round(result * 255.0).astype(np.uint8)

os.makedirs(
    os.path.dirname(OUT_PATH),
    exist_ok=True
)

cv2.imwrite(OUT_PATH, result_u8)


# =====================================================
# REPORT
# =====================================================

print("HDR Fusion V3 complete")
print("Output:", OUT_PATH)
print("Resolution:", result_u8.shape)

print(
    "Shadow coverage:",
    round(float((shadow_mask > 0.05).mean()) * 100, 2),
    "%"
)

print(
    "Mean shadow strength:",
    round(float(shadow_mask.mean()), 4)
)

print(
    "Mean brightness gain:",
    round(float(brightness_gain.mean()), 4)
)

print(
    "Highlight coverage:",
    round(float((highlight_weight > 0.01).mean()) * 100, 2),
    "%"
)

print(
    "Mean highlight weight:",
    round(float(highlight_weight.mean()), 4)
)
