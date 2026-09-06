import cv2
import numpy as np

# ==========================================
# INPUT PATHS
# ==========================================

UNDER_PATH = "my_test_context/scene001/01_under.JPG"
NORMAL_PATH = "my_test_context/scene001/02_normal.JPG"
OVER_PATH = "my_test_context/scene001/03_over.JPG"

FINAL_PATH = "production/output/final_hdr/scene001_v6.png"

#===========================================
# LOAD IMAGES
# ==========================================

under = cv2.imread(UNDER_PATH)
normal = cv2.imread(NORMAL_PATH)
over = cv2.imread(OVER_PATH)
final = cv2.imread(FINAL_PATH)


# ==========================================
# SAFETY CHECK
# ==========================================

if under is None:
    raise FileNotFoundError(f"Cannot load: {UNDER_PATH}")

if normal is None:
    raise FileNotFoundError(f"Cannot load: {NORMAL_PATH}")

if over is None:
    raise FileNotFoundError(f"Cannot load: {OVER_PATH}")

if final is None:
    raise FileNotFoundError(f"Cannot load: {FINAL_PATH}")


# ==========================================
# MAKE SIZES MATCH
# ==========================================

h, w = normal.shape[:2]

under = cv2.resize(under, (w, h))
over = cv2.resize(over, (w, h))
final = cv2.resize(final, (w, h))


# ==========================================
# CONVERT TO FLOAT
# ==========================================

under_f = under.astype(np.float32) / 255.0
normal_f = normal.astype(np.float32) / 255.0
over_f = over.astype(np.float32) / 255.0
final_f = final.astype(np.float32) / 255.0


# ==========================================
# MEAN BRIGHTNESS
# ==========================================

under_mean = np.mean(under_f)
normal_mean = np.mean(normal_f)
over_mean = np.mean(over_f)
final_mean = np.mean(final_f)


# ==========================================
# DIFFERENCE FROM NORMAL
# ==========================================

diff_normal = np.mean(
    np.abs(final_f - normal_f)
)


# ==========================================
# SHADOW / HIGHLIGHT ANALYSIS
# ==========================================

normal_gray = cv2.cvtColor(
    normal_f,
    cv2.COLOR_BGR2GRAY
)

final_gray = cv2.cvtColor(
    final_f,
    cv2.COLOR_BGR2GRAY
)

# Dark areas in the NORMAL image
shadow_mask = normal_gray < 0.30

# Bright areas in the NORMAL image
highlight_mask = normal_gray > 0.75


if np.any(shadow_mask):

    shadow_normal = np.mean(
        normal_gray[shadow_mask]
    )

    shadow_final = np.mean(
        final_gray[shadow_mask]
    )

    shadow_change = shadow_final - shadow_normal

else:
    shadow_change = 0


if np.any(highlight_mask):

    highlight_normal = np.mean(
        normal_gray[highlight_mask]
    )

    highlight_final = np.mean(
        final_gray[highlight_mask]
    )

    highlight_change = (
        highlight_final - highlight_normal
    )

else:
    highlight_change = 0


# ==========================================
# SHARPNESS
# ==========================================

def calculate_sharpness(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    return cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()


normal_sharpness = calculate_sharpness(normal)
final_sharpness = calculate_sharpness(final)

sharpness_ratio = (
    final_sharpness /
    (normal_sharpness + 1e-6)
)


# ==========================================
# PRINT RESULTS
# ==========================================

print("\n========================================")
print("         HDR VALIDATION REPORT")
print("========================================")

print("\nIMAGE RESOLUTION")
print("Under :", under.shape)
print("Normal:", normal.shape)
print("Over  :", over.shape)
print("Final :", final.shape)


print("\nMEAN BRIGHTNESS")
print(f"Under : {under_mean:.4f}")
print(f"Normal: {normal_mean:.4f}")
print(f"Over  : {over_mean:.4f}")
print(f"Final : {final_mean:.4f}")


print("\nHDR CONTRIBUTION")
print(
    f"Difference from Normal: "
    f"{diff_normal:.6f}"
)


print("\nSHADOW RECOVERY")
print(
    f"Brightness Change: "
    f"{shadow_change:.6f}"
)

print(
    f"Shadow Pixel Coverage: "
    f"{shadow_mask.mean() * 100:.2f}%"
)


print("\nHIGHLIGHT RECOVERY")
print(
    f"Brightness Change: "
    f"{highlight_change:.6f}"
)

print(
    f"Highlight Pixel Coverage: "
    f"{highlight_mask.mean() * 100:.2f}%"
)


print("\nSHARPNESS")
print(
    f"Normal Sharpness: "
    f"{normal_sharpness:.2f}"
)

print(
    f"Final Sharpness: "
    f"{final_sharpness:.2f}"
)

print(
    f"Sharpness Ratio: "
    f"{sharpness_ratio:.4f}"
)


print("\nFINAL STATUS")

if sharpness_ratio >= 0.95:
    print("Sharpness: PASS")
else:
    print("Sharpness: WARNING")


if diff_normal > 0.01:
    print("HDR Contribution: STRONG")
elif diff_normal > 0.005:
    print("HDR Contribution: DETECTED")
else:
    print("HDR Contribution: TOO WEAK")


if shadow_change > 0.005:
    print("Shadow Recovery: DETECTED")
else:
    print("Shadow Recovery: SUBTLE")


if highlight_change < -0.005:
    print("Highlight Recovery: DETECTED")
else:
    print("Highlight Recovery: SUBTLE")


print("========================================\n")
