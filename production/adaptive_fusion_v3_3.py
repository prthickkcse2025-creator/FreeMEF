import os
import cv2
import numpy as np


# ==========================================================
# CONFIGURATION
# ==========================================================

UNDER_PATH = "/home/prthick/FreeMEF/my_test/scene006/01_under.jpg"
NORMAL_PATH = "/home/prthick/FreeMEF/my_test/scene006/02_normal.jpg"
OVER_PATH = "/home/prthick/FreeMEF/my_test/scene006/03_over.jpg"

OUTPUT_DIR = (
    "/home/prthick/FreeMEF/"
    "production/output/adaptive_fusion_v3_3"
)

FINAL_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "FINAL_ADAPTIVE_FUSION_V3_3_DEPTH.jpg"
)


# ==========================================================
# IMAGE LOADING
# ==========================================================

def load_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not load image:\n{path}"
        )

    image = image.astype(
        np.float32
    ) / 255.0

    return image


# ==========================================================
# LUMINANCE
# ==========================================================

def luminance(image):

    return (
        0.2126 * image[:, :, 2] +
        0.7152 * image[:, :, 1] +
        0.0722 * image[:, :, 0]
    )


# ==========================================================
# SMOOTHSTEP
# ==========================================================

def smoothstep(x, edge0, edge1):

    if edge1 <= edge0:

        raise ValueError(
            "edge1 must be greater than edge0"
        )

    t = np.clip(
        (x - edge0) /
        (edge1 - edge0),
        0.0,
        1.0
    )

    return (
        t * t *
        (3.0 - 2.0 * t)
    )


# ==========================================================
# EXPOSURE QUALITY
#
# Pixels near middle exposure receive
# higher quality values.
# ==========================================================

def exposure_quality(lum):

    quality = (
        1.0 -
        np.abs(
            lum - 0.50
        ) / 0.50
    )

    quality = np.clip(
        quality,
        0.05,
        1.0
    )

    return quality.astype(
        np.float32
    )


# ==========================================================
# SAVE DIAGNOSTIC MAP
# ==========================================================

def save_map(path, image):

    image = np.asarray(
        image,
        dtype=np.float32
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    image_uint8 = (
        image * 255.0
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        path,
        image_uint8
    )


# ==========================================================
# MAIN
# ==========================================================

def main():

    print()
    print("====================================")
    print("ADAPTIVE HDR FUSION V3.3")
    print("DEPTH-PRESERVING SHADOW RECOVERY")
    print("====================================")
    print()

    print(f"Under : {UNDER_PATH}")
    print(f"Normal: {NORMAL_PATH}")
    print(f"Over  : {OVER_PATH}")
    print()

    # ------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # ------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ------------------------------------------------------
    # LOAD IMAGES
    # ------------------------------------------------------

    under = load_image(
        UNDER_PATH
    )

    normal = load_image(
        NORMAL_PATH
    )

    over = load_image(
        OVER_PATH
    )

    # ------------------------------------------------------
    # CHECK RESOLUTION
    # ------------------------------------------------------

    if (
        under.shape != normal.shape or
        normal.shape != over.shape
    ):

        raise ValueError(
            "All three images must have "
            "the same resolution."
        )

    height, width = normal.shape[:2]

    print(
        f"Resolution: {width} x {height}"
    )

    # ======================================================
    # LUMINANCE
    # ======================================================

    lum_under = luminance(
        under
    )

    lum_normal = luminance(
        normal
    )

    lum_over = luminance(
        over
    )

    # ======================================================
    # HIGHLIGHT DETECTION
    #
    # Only very bright regions should receive
    # significant contribution from the underexposed image.
    # ======================================================

    highlight_mask = smoothstep(
        lum_normal,
        0.75,
        0.92
    )

    # Smooth only the mask.
    # The actual image pixels are NOT blurred.

    highlight_mask = cv2.GaussianBlur(
        highlight_mask.astype(
            np.float32
        ),
        (0, 0),
        10
    )

    highlight_mask = np.clip(
        highlight_mask,
        0.0,
        1.0
    )

    # ======================================================
    # SHADOW DETECTION
    #
    # Detect dark regions in the normal exposure.
    # ======================================================

    shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.12,
            0.40
        )
    )

    shadow_mask = np.clip(
        shadow_mask,
        0.0,
        1.0
    )

    # ======================================================
    # DEEP SHADOW DETECTION
    #
    # Only genuinely crushed shadows should
    # receive significant overexposed contribution.
    # ======================================================

    deep_shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.04,
            0.18
        )
    )

    deep_shadow_mask = np.clip(
        deep_shadow_mask,
        0.0,
        1.0
    )

    # ======================================================
    # CHECK WHETHER OVEREXPOSED IMAGE
    # CONTAINS USABLE SHADOW DETAIL
    # ======================================================

    over_detail_mask = smoothstep(
        lum_over,
        0.08,
        0.25
    )

    over_detail_mask = np.clip(
        over_detail_mask,
        0.0,
        1.0
    )

    # ======================================================
    # DEPTH-PRESERVING SHADOW RECOVERY
    #
    # Shadow recovery happens only where:
    #
    # 1. Normal image is deeply dark
    # 2. Overexposed image contains usable detail
    # ======================================================

    shadow_recovery_mask = (
        deep_shadow_mask *
        over_detail_mask
    )

    shadow_recovery_mask = np.clip(
        shadow_recovery_mask,
        0.0,
        1.0
    )

    # Smooth only the recovery mask.

    shadow_recovery_mask = cv2.GaussianBlur(
        shadow_recovery_mask.astype(
            np.float32
        ),
        (0, 0),
        8
    )

    shadow_recovery_mask = np.clip(
        shadow_recovery_mask,
        0.0,
        1.0
    )

    # ======================================================
    # EXPOSURE QUALITY
    # ======================================================

    q_under = exposure_quality(
        lum_under
    )

    q_normal = exposure_quality(
        lum_normal
    )

    q_over = exposure_quality(
        lum_over
    )

    # ======================================================
    # UNDEREXPOSED WEIGHT
    #
    # Used mainly for strong highlights.
    # ======================================================

    w_under = q_under * (
        0.01 +
        0.99 * highlight_mask
    )

    # ======================================================
    # NORMAL EXPOSURE WEIGHT
    #
    # This is the main anchor.
    #
    # It preserves:
    # - Original colors
    # - White balance
    # - Material appearance
    # - Natural midtones
    # ======================================================

    w_normal = q_normal + 0.20

    # ======================================================
    # OVEREXPOSED WEIGHT
    #
    # Used only for genuine shadow recovery.
    # ======================================================

    w_over = q_over * (
        0.01 +
        0.99 * shadow_recovery_mask
    )

    # ======================================================
    # PREVENT ZERO WEIGHTS
    # ======================================================

    w_under = np.clip(
        w_under,
        0.001,
        None
    )

    w_normal = np.clip(
        w_normal,
        0.05,
        None
    )

    w_over = np.clip(
        w_over,
        0.001,
        None
    )

    # ======================================================
    # NORMALIZE WEIGHTS
    #
    # At every pixel:
    #
    # w_under + w_normal + w_over = 1
    # ======================================================

    weight_sum = (
        w_under +
        w_normal +
        w_over +
        1e-8
    )

    w_under = (
        w_under /
        weight_sum
    )

    w_normal = (
        w_normal /
        weight_sum
    )

    w_over = (
        w_over /
        weight_sum
    )

    # ======================================================
    # FINAL FUSION
    #
    # Expand weight maps to 3 channels.
    # ======================================================

    w_under_3 = np.expand_dims(
        w_under,
        axis=2
    )

    w_normal_3 = np.expand_dims(
        w_normal,
        axis=2
    )

    w_over_3 = np.expand_dims(
        w_over,
        axis=2
    )

    fused = (
        under * w_under_3 +
        normal * w_normal_3 +
        over * w_over_3
    )

    # ======================================================
    # SAFETY
    # ======================================================

    fused = np.nan_to_num(
        fused,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ======================================================
    # SAVE FINAL IMAGE
    # ======================================================

    fused_uint8 = (
        fused * 255.0
    ).astype(
        np.uint8
    )

    cv2.imwrite(
        FINAL_OUTPUT,
        fused_uint8
    )

    # ======================================================
    # SAVE DIAGNOSTICS
    # ======================================================

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "01_highlight_mask.png"
        ),
        highlight_mask
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "02_shadow_mask.png"
        ),
        shadow_mask
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "03_deep_shadow_mask.png"
        ),
        deep_shadow_mask
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "04_shadow_recovery_mask.png"
        ),
        shadow_recovery_mask
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "05_weight_under.png"
        ),
        w_under
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "06_weight_normal.png"
        ),
        w_normal
    )

    save_map(
        os.path.join(
            OUTPUT_DIR,
            "07_weight_over.png"
        ),
        w_over
    )

    # ======================================================
    # RESULTS
    # ======================================================

    print()
    print("====================================")
    print("ADAPTIVE HDR FUSION COMPLETED")
    print("====================================")
    print()

    print(
        f"Output: {FINAL_OUTPUT}"
    )

    print()
    print("Mean fusion weights:")

    print(
        f"Under : {np.mean(w_under):.4f}"
    )

    print(
        f"Normal: {np.mean(w_normal):.4f}"
    )

    print(
        f"Over  : {np.mean(w_over):.4f}"
    )

    print()


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()
