import cv2
import numpy as np
import os

EPS = 1e-8


# ==========================================================
# IMAGE LOADING
# ==========================================================

def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")

    return img.astype(np.float32) / 255.0


# ==========================================================
# LUMINANCE
# ==========================================================

def luminance(img):
    return (
        0.0722 * img[:, :, 0] +
        0.7152 * img[:, :, 1] +
        0.2126 * img[:, :, 2]
    )


# ==========================================================
# SMOOTHSTEP MASK
# ==========================================================

def smoothstep(x, low, high):
    x = np.clip(
        (x - low) / (high - low + EPS),
        0.0,
        1.0
    )

    return x * x * (3.0 - 2.0 * x)


# ==========================================================
# EXPOSURE QUALITY
# ==========================================================

def exposure_quality(lum):
    """
    Gives higher weight to well-exposed pixels.
    """

    sigma = 0.25

    return np.exp(
        -((lum - 0.5) ** 2) /
        (2.0 * sigma * sigma)
    )


# ==========================================================
# EDGE-AWARE SMOOTHING
# ==========================================================

def edge_smooth(weight):
    """
    Removes small local weight variations.
    """

    weight = np.clip(weight, 0.0, None)

    max_val = weight.max()

    if max_val < EPS:
        return weight

    weight_norm = weight / max_val

    weight_8 = np.uint8(
        np.clip(weight_norm * 255.0, 0, 255)
    )

    smooth = cv2.bilateralFilter(
        weight_8,
        d=11,
        sigmaColor=45,
        sigmaSpace=45
    )

    smooth = smooth.astype(np.float32) / 255.0

    return smooth * max_val


# ==========================================================
# MULTI-SCALE WEIGHT SMOOTHING
# ANTI-PATCHINESS STAGE
# ==========================================================

def multiscale_smooth(weight):
    """
    Removes patchy local exposure changes by combining
    original, medium-frequency and low-frequency weights.
    """

    h, w = weight.shape

    # ------------------------------------------------------
    # LOW-FREQUENCY WEIGHT MAP
    # ------------------------------------------------------

    small_w = max(1, w // 8)
    small_h = max(1, h // 8)

    low_freq = cv2.resize(
        weight,
        (small_w, small_h),
        interpolation=cv2.INTER_AREA
    )

    low_freq = cv2.resize(
        low_freq,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    # ------------------------------------------------------
    # MEDIUM-SCALE SMOOTHING
    # ------------------------------------------------------

    medium = cv2.GaussianBlur(
        weight,
        (0, 0),
        sigmaX=18,
        sigmaY=18
    )

    # ------------------------------------------------------
    # COMBINE SCALES
    #
    # 15% original detail
    # 35% medium smoothing
    # 50% large smooth illumination
    # ------------------------------------------------------

    smoothed = (
        0.15 * weight +
        0.35 * medium +
        0.50 * low_freq
    )

    return smoothed.astype(np.float32)


# ==========================================================
# SAVE IMAGE
# ==========================================================

def save_image(path, image):

    image = np.uint8(
        np.clip(image * 255.0, 0, 255)
    )

    cv2.imwrite(path, image)


# ==========================================================
# MAIN
# ==========================================================

def main():

    # ------------------------------------------------------
    # INPUT DIRECTORY
    # ------------------------------------------------------

    input_dir = (
        "/home/prthick/FreeMEF/"
        "my_test_small/scene001"
    )

    under_path = os.path.join(
        input_dir,
        "01_under.JPG"
    )

    normal_path = os.path.join(
        input_dir,
        "02_normal.JPG"
    )

    over_path = os.path.join(
        input_dir,
        "03_over.JPG"
    )

    # ------------------------------------------------------
    # OUTPUT DIRECTORY
    # ------------------------------------------------------

    output_dir = (
        "/home/prthick/FreeMEF/"
        "production/output/adaptive_fusion_v4"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    print("\n========================================")
    print("ADAPTIVE HDR FUSION V4")
    print("ANTI-PATCHINESS VERSION")
    print("========================================")

    # ------------------------------------------------------
    # LOAD IMAGES
    # ------------------------------------------------------

    under = read_image(under_path)
    normal = read_image(normal_path)
    over = read_image(over_path)

    # ------------------------------------------------------
    # ENSURE SAME RESOLUTION
    # ------------------------------------------------------

    h, w = normal.shape[:2]

    if under.shape[:2] != (h, w):
        under = cv2.resize(
            under,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    if over.shape[:2] != (h, w):
        over = cv2.resize(
            over,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    print(f"\nResolution: {w} x {h}")

    # ------------------------------------------------------
    # COMPUTE LUMINANCE
    # ------------------------------------------------------

    lum_under = luminance(under)
    lum_normal = luminance(normal)
    lum_over = luminance(over)

    # ------------------------------------------------------
    # HIGHLIGHT MASK
    #
    # Bright regions use the underexposed image.
    # ------------------------------------------------------

    highlight_mask = smoothstep(
        lum_normal,
        0.50,
        0.82
    )

    # ------------------------------------------------------
    # SHADOW MASK
    #
    # Dark regions use the overexposed image.
    # ------------------------------------------------------

    shadow_mask = 1.0 - smoothstep(
        lum_normal,
        0.18,
        0.55
    )

    # ------------------------------------------------------
    # EXPOSURE QUALITY
    # ------------------------------------------------------

    q_under = exposure_quality(lum_under)
    q_normal = exposure_quality(lum_normal)
    q_over = exposure_quality(lum_over)

    # ======================================================
    # INITIAL FUSION WEIGHTS
    # ======================================================

    # Underexposed image:
    # Mainly used for highlight/window recovery.
    w_under = (
        0.10 +
        2.0 * highlight_mask +
        0.50 * q_under
    )

    # Normal image:
    # Main anchor for original color and white balance.
    w_normal = (
        2.5 +
        1.5 * q_normal
    )

    # Overexposed image:
    # Mainly used for shadow/dark-area recovery.
    w_over = (
        0.10 +
        1.5 * shadow_mask +
        0.50 * q_over
    )

    # ======================================================
    # FIRST STAGE:
    # EDGE-AWARE SMOOTHING
    # ======================================================

    w_under = edge_smooth(w_under)
    w_normal = edge_smooth(w_normal)
    w_over = edge_smooth(w_over)

    # ======================================================
    # SECOND STAGE:
    # MULTI-SCALE ANTI-PATCHINESS SMOOTHING
    # ======================================================

    w_under = multiscale_smooth(w_under)
    w_normal = multiscale_smooth(w_normal)
    w_over = multiscale_smooth(w_over)

    # Prevent invalid values

    w_under = np.clip(
        w_under,
        0.0,
        None
    )

    w_normal = np.clip(
        w_normal,
        0.0,
        None
    )

    w_over = np.clip(
        w_over,
        0.0,
        None
    )

    # ======================================================
    # NORMALIZE WEIGHTS
    # ======================================================

    weight_sum = (
        w_under +
        w_normal +
        w_over +
        EPS
    )

    w_under /= weight_sum
    w_normal /= weight_sum
    w_over /= weight_sum

    # ======================================================
    # HDR FUSION
    # ======================================================

    fused = (
        under * w_under[:, :, None] +
        normal * w_normal[:, :, None] +
        over * w_over[:, :, None]
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ======================================================
    # SAVE DIAGNOSTIC MAPS
    # ======================================================

    save_image(
        os.path.join(
            output_dir,
            "01_highlight_mask.png"
        ),
        highlight_mask
    )

    save_image(
        os.path.join(
            output_dir,
            "02_shadow_mask.png"
        ),
        shadow_mask
    )

    save_image(
        os.path.join(
            output_dir,
            "03_weight_under.png"
        ),
        w_under
    )

    save_image(
        os.path.join(
            output_dir,
            "04_weight_normal.png"
        ),
        w_normal
    )

    save_image(
        os.path.join(
            output_dir,
            "05_weight_over.png"
        ),
        w_over
    )

    # ======================================================
    # SAVE FINAL OUTPUT
    # ======================================================

    final_path = os.path.join(
        output_dir,
        "FINAL_ADAPTIVE_FUSION_V4.png"
    )

    save_image(
        final_path,
        fused
    )

    # ======================================================
    # REPORT
    # ======================================================

    print("\n========================================")
    print("V4 COMPLETED SUCCESSFULLY")
    print("========================================")

    print(f"\nFinal output:")
    print(final_path)

    print("\nMean Fusion Weights:")
    print(f"Under : {w_under.mean():.4f}")
    print(f"Normal: {w_normal.mean():.4f}")
    print(f"Over  : {w_over.mean():.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
