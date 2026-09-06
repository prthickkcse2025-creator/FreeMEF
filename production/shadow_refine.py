import cv2
import numpy as np
import os

# ---------------------------------------------------------
# FreeMEF Shadow-Aware Refinement
# Uses:
#   1. FreeMEF fused output
#   2. Normal exposure
#   3. Over exposure
#
# Goal:
#   Recover dark-region detail from OVER while preserving
#   the color/texture character of the FreeMEF output.
# ---------------------------------------------------------

fused_path  = "production/output/order_test/scene001.png"
normal_path = "my_test_context/scene001/02_normal.JPG"
over_path   = "my_test_context/scene001/03_over.JPG"

output_path = "production/output/shadow_refined_v2/scene001.png"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

fused  = cv2.imread(fused_path, cv2.IMREAD_COLOR)
normal = cv2.imread(normal_path, cv2.IMREAD_COLOR)
over   = cv2.imread(over_path, cv2.IMREAD_COLOR)

if fused is None:
    raise FileNotFoundError(fused_path)
if normal is None:
    raise FileNotFoundError(normal_path)
if over is None:
    raise FileNotFoundError(over_path)

h, w = fused.shape[:2]

normal = cv2.resize(normal, (w, h), interpolation=cv2.INTER_AREA)
over   = cv2.resize(over,   (w, h), interpolation=cv2.INTER_AREA)

# Convert to float [0,1]
F = fused.astype(np.float32) / 255.0
N = normal.astype(np.float32) / 255.0
O = over.astype(np.float32) / 255.0

# Luminance
def luminance(img):
    return (
        0.114 * img[:, :, 0] +
        0.587 * img[:, :, 1] +
        0.299 * img[:, :, 2]
    )

YF = luminance(F)
YN = luminance(N)
YO = luminance(O)

# ---------------------------------------------------------
# 1. Estimate how much brighter OVER is than NORMAL.
#    Use midtone pixels only so highlights don't dominate.
# ---------------------------------------------------------

valid = (
    (YN > 0.20) &
    (YN < 0.75) &
    (YO > 0.05)
)

ratio = np.median(
    YN[valid] / np.maximum(YO[valid], 1e-6)
)

# Conservative calibration
ratio = float(np.clip(ratio, 0.35, 0.95))

O_cal = np.clip(O * ratio, 0.0, 1.0)
YO_cal = luminance(O_cal)

# ---------------------------------------------------------
# 2. Build shadow mask from FreeMEF output.
#    Darker areas receive more information from OVER.
# ---------------------------------------------------------

# Start shadow recovery around 0.42 luminance.
# Full strength around 0.18.
shadow = np.clip((0.46 - YF) / (0.46 - 0.16), 0.0, 1.0)

# Don't aggressively modify very bright areas.
shadow *= np.clip((0.80 - YF) / 0.35, 0.0, 1.0)

# Smooth mask to avoid visible boundaries.
shadow = cv2.GaussianBlur(shadow, (0, 0), 18)

# Moderate maximum strength.
alpha = 0.92 * shadow

# ---------------------------------------------------------
# 3. Transfer OVER luminance while preserving FreeMEF color.
# ---------------------------------------------------------

target_luma = (
    (1.0 - alpha) * YF +
    alpha * YO_cal
)

# Prevent uncontrolled brightness.
target_luma = np.clip(
    target_luma,
    YF * 0.80,
    np.minimum(YF * 2.10 + 0.12, 0.92)
)

# Scale FreeMEF RGB by luminance ratio.
scale = target_luma / np.maximum(YF, 1e-4)

refined = F * scale[:, :, None]

# Blend only according to the shadow mask.
refined = (
    F * (1.0 - alpha[:, :, None]) +
    refined * alpha[:, :, None]
)

# ---------------------------------------------------------
# 4. Protect highlights.
# ---------------------------------------------------------

highlight = np.clip((YF - 0.78) / 0.18, 0.0, 1.0)

refined = (
    refined * (1.0 - highlight[:, :, None]) +
    F * highlight[:, :, None]
)

# ---------------------------------------------------------
# 5. Final gentle local contrast preservation.
# ---------------------------------------------------------

refined = np.clip(refined, 0.0, 1.0)

out = (refined * 255.0 + 0.5).astype(np.uint8)

cv2.imwrite(output_path, out)

print("Shadow refinement complete")
print("Calibration ratio:", ratio)
print("Output:", output_path)
print("Resolution:", out.shape)
