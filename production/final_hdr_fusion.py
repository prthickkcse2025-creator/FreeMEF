import cv2
import numpy as np
import os

# ============================================================
# INPUT: ONLY THESE 3 IMAGES FOR TESTING
# ============================================================

IMAGE_PATHS = [
    "/mnt/e/HDR_Project_Backups/HDR_BASELINE_V10_FINAL/raw_scenes/IMG_7741.jpg",
    "/mnt/e/HDR_Project_Backups/HDR_BASELINE_V10_FINAL/raw_scenes/IMG_7742.jpg",
    "/mnt/e/HDR_Project_Backups/HDR_BASELINE_V10_FINAL/raw_scenes/IMG_7743.jpg",
]

OUT_PATH = "production/output/final_hdr/test_scene_9350.png"


# ============================================================
# CALCULATE IMAGE BRIGHTNESS
# ============================================================

def get_brightness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray) / 255.0)


# ============================================================
# LOAD THE 3 IMAGES
# ============================================================

images = []

for path in IMAGE_PATHS:

    img = cv2.imread(path)

    if img is None:
        raise FileNotFoundError(f"Cannot load: {path}")

    images.append({
        "path": path,
        "image": img,
        "brightness": get_brightness(img)
    })


# ============================================================
# AUTOMATIC EXPOSURE DETECTION
#
# Darkest  -> UNDER
# Middle   -> NORMAL
# Brightest -> OVER
# ============================================================

images.sort(key=lambda x: x["brightness"])

under = images[0]["image"]
normal = images[1]["image"]
over = images[2]["image"]

print("\n============================================================")
print("HDR FUSION — AUTOMATIC EXPOSURE DETECTION")
print("============================================================")

print("\nUNDER:")
print(os.path.basename(images[0]["path"]))
print("Brightness:", round(images[0]["brightness"], 4))

print("\nNORMAL:")
print(os.path.basename(images[1]["path"]))
print("Brightness:", round(images[1]["brightness"], 4))

print("\nOVER:")
print(os.path.basename(images[2]["path"]))
print("Brightness:", round(images[2]["brightness"], 4))


# ============================================================
# ENSURE SAME RESOLUTION
# ============================================================

h, w = normal.shape[:2]

if under.shape[:2] != (h, w):
    under = cv2.resize(under, (w, h))

if over.shape[:2] != (h, w):
    over = cv2.resize(over, (w, h))


# ============================================================
# CONVERT TO FLOAT
# ============================================================

under_f = under.astype(np.float32) / 255.0
normal_f = normal.astype(np.float32) / 255.0
over_f = over.astype(np.float32) / 255.0


# ============================================================
# LUMINANCE
# ============================================================

under_gray = cv2.cvtColor(under_f, cv2.COLOR_BGR2GRAY)
normal_gray = cv2.cvtColor(normal_f, cv2.COLOR_BGR2GRAY)
over_gray = cv2.cvtColor(over_f, cv2.COLOR_BGR2GRAY)


# ============================================================
# START WITH NORMAL IMAGE
# ============================================================

result = normal_f.copy()


# ============================================================
# SHADOW RECOVERY
#
# Use OVER exposure only where NORMAL is dark
# ============================================================

shadow_mask = np.clip(
    (0.40 - normal_gray) / 0.40,
    0.0,
    1.0
)

shadow_mask = shadow_mask ** 1.8

shadow_mask = cv2.GaussianBlur(
    shadow_mask,
    (0, 0),
    sigmaX=5,
    sigmaY=5
)

shadow_weight = shadow_mask * 0.55


result = (
    result * (1.0 - shadow_weight[:, :, None])
    + over_f * shadow_weight[:, :, None]
)

result = np.clip(result, 0.0, 1.0)


# ============================================================
# HIGHLIGHT RECOVERY
#
# Use UNDER exposure only in bright regions
# ============================================================

result_gray = cv2.cvtColor(
    result.astype(np.float32),
    cv2.COLOR_BGR2GRAY
)

highlight_mask = np.clip(
    (result_gray - 0.72) / 0.25,
    0.0,
    1.0
)

highlight_mask = highlight_mask ** 1.5

highlight_mask = cv2.GaussianBlur(
    highlight_mask,
    (0, 0),
    sigmaX=4,
    sigmaY=4
)

highlight_weight = highlight_mask * 0.65


result = (
    result * (1.0 - highlight_weight[:, :, None])
    + under_f * highlight_weight[:, :, None]
)

result = np.clip(result, 0.0, 1.0)


# ============================================================
# COLOR / WHITE BALANCE ANCHOR
#
# Keep the NORMAL exposure as the main color reference
# ============================================================

color_anchor = 0.18

result = (
    result * (1.0 - color_anchor)
    + normal_f * color_anchor
)

result = np.clip(result, 0.0, 1.0)


# ============================================================
# SAVE
# ============================================================

result_u8 = np.round(result * 255.0).astype(np.uint8)

os.makedirs(
    os.path.dirname(OUT_PATH),
    exist_ok=True
)

cv2.imwrite(OUT_PATH, result_u8)


# ============================================================
# REPORT
# ============================================================

final_gray = cv2.cvtColor(
    result.astype(np.float32),
    cv2.COLOR_BGR2GRAY
)

print("\n============================================================")
print("HDR FUSION COMPLETE")
print("============================================================")

print("Output:", OUT_PATH)
print("Resolution:", result_u8.shape)

print("\nMEAN BRIGHTNESS")
print("Under :", round(float(under_gray.mean()), 4))
print("Normal:", round(float(normal_gray.mean()), 4))
print("Over  :", round(float(over_gray.mean()), 4))
print("Final :", round(float(final_gray.mean()), 4))

print("\nSHADOW RECOVERY")
print(
    "Coverage:",
    round(float((shadow_weight > 0.01).mean()) * 100, 2),
    "%"
)

print("\nHIGHLIGHT RECOVERY")
print(
    "Coverage:",
    round(float((highlight_weight > 0.01).mean()) * 100, 2),
    "%"
)

print("============================================================")
print("DONE")
