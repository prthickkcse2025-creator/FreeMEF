import os
import cv2
import numpy as np


# ============================================================
# ADAPTIVE HDR FUSION V3.5
#
# V3.5:
# - Keeps normal exposure as main anchor
# - Slightly brighter than V3.4
# - Less shadow lifting than V3.3
# - Patch-free weight smoothing
# - Original images are NEVER blurred
# ============================================================


EPS = 1e-8


# ============================================================
# INPUT / OUTPUT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_DIR = os.path.dirname(BASE_DIR)

INPUT_DIR = os.path.join(
    PROJECT_DIR,
    "my_test",
    "scene001"
)

UNDER_PATH = os.path.join(
    INPUT_DIR,
    "01_under.jpg"
)

NORMAL_PATH = os.path.join(
    INPUT_DIR,
    "02_normal.jpg"
)

OVER_PATH = os.path.join(
    INPUT_DIR,
    "03_over.jpg"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output",
    "adaptive_fusion_v3_5"
)


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {path}"
        )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    ) / 255.0

    return image


# ============================================================
# SMOOTHSTEP
# ============================================================

def smoothstep(x, edge0, edge1):

    t = (
        x - edge0
    ) / (
        edge1 - edge0 + EPS
    )

    t = np.clip(
        t,
        0.0,
        1.0
    )

    return (
        t * t *
        (
            3.0 -
            2.0 * t
        )
    )


# ============================================================
# LUMINANCE
# ============================================================

def luminance(image):

    return (
        0.2126 * image[:, :, 0] +
        0.7152 * image[:, :, 1] +
        0.0722 * image[:, :, 2]
    )


# ============================================================
# EXPOSURE QUALITY
#
# Highest weight for well-exposed pixels.
# ============================================================

def exposure_quality(lum):

    sigma = 0.25

    quality = np.exp(
        -(
            (lum - 0.50) ** 2
        ) /
        (
            2.0 *
            sigma *
            sigma +
            EPS
        )
    )

    return quality.astype(
        np.float32
    )


# ============================================================
# MAIN FUSION
# ============================================================

def adaptive_hdr_fusion():

    print()

    print(
        "===================================="
    )

    print(
        "ADAPTIVE HDR FUSION V3.5"
    )

    print(
        "PATCH-FREE WEIGHT SMOOTHING"
    )

    print(
        "===================================="
    )

    print()

    print(
        f"Under : {UNDER_PATH}"
    )

    print(
        f"Normal: {NORMAL_PATH}"
    )

    print(
        f"Over  : {OVER_PATH}"
    )

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------------

    under = load_image(
        UNDER_PATH
    )

    normal = load_image(
        NORMAL_PATH
    )

    over = load_image(
        OVER_PATH
    )

    # --------------------------------------------------------
    # MATCH RESOLUTION
    # --------------------------------------------------------

    h, w = normal.shape[:2]

    if (
        under.shape[:2] !=
        (h, w)
    ):

        under = cv2.resize(
            under,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    if (
        over.shape[:2] !=
        (h, w)
    ):

        over = cv2.resize(
            over,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    print()

    print(
        f"Resolution: {w} x {h}"
    )

    # ========================================================
    # LUMINANCE
    # ========================================================

    lum_under = luminance(
        under
    )

    lum_normal = luminance(
        normal
    )

    lum_over = luminance(
        over
    )

    # ========================================================
    # HIGHLIGHT DETECTION
    #
    # Underexposed image recovers
    # bright windows and highlights.
    # ========================================================

    highlight_mask = smoothstep(
        lum_normal,
        0.50,
        0.82
    )

    # ========================================================
    # SHADOW DETECTION
    #
    # V3.3 baseline thresholds retained.
    # ========================================================

    shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.18,
            0.55
        )
    )

    # ========================================================
    # DEEP SHADOW DETECTION
    # ========================================================

    deep_shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.05,
            0.32
        )
    )

    # ========================================================
    # COMBINE SHADOW MASKS
    # ========================================================

    shadow_recovery_mask = np.maximum(
        shadow_mask,
        deep_shadow_mask
    )

    # ========================================================
    # CHECK WHETHER OVEREXPOSED IMAGE
    # CONTAINS USABLE DETAIL
    # ========================================================

    over_detail_mask = smoothstep(
        lum_over,
        0.08,
        0.30
    )

    shadow_recovery_mask = (
        shadow_recovery_mask *
        over_detail_mask
    )

    shadow_recovery_mask = np.clip(
        shadow_recovery_mask,
        0.0,
        1.0
    )

    # ========================================================
    # SMOOTH MASKS
    #
    # Only masks are smoothed.
    # Original images remain untouched.
    # ========================================================

    highlight_mask = cv2.GaussianBlur(
        highlight_mask.astype(
            np.float32
        ),
        (0, 0),
        12
    )

    shadow_mask = cv2.GaussianBlur(
        shadow_mask.astype(
            np.float32
        ),
        (0, 0),
        15
    )

    deep_shadow_mask = cv2.GaussianBlur(
        deep_shadow_mask.astype(
            np.float32
        ),
        (0, 0),
        18
    )

    shadow_recovery_mask = cv2.GaussianBlur(
        shadow_recovery_mask.astype(
            np.float32
        ),
        (0, 0),
        18
    )

    # ========================================================
    # EXPOSURE QUALITY
    # ========================================================

    q_under = exposure_quality(
        lum_under
    )

    q_normal = exposure_quality(
        lum_normal
    )

    q_over = exposure_quality(
        lum_over
    )

    # ========================================================
    # UNDEREXPOSED IMAGE WEIGHT
    #
    # Mainly used for highlights
    # and window recovery.
    # ========================================================

    w_under = (
        0.10 +
        2.0 * highlight_mask +
        0.5 * q_under
    )

    # ========================================================
    # NORMAL IMAGE WEIGHT
    #
    # V3.5:
    # Still the main anchor, but slightly reduced
    # compared with V3.4 so the final image becomes
    # brighter without excessive shadow lifting.
    # ========================================================

    w_normal = (
        3.5 +
        1.8 * q_normal -
        0.35 * deep_shadow_mask
    )

    w_normal = np.maximum(
        w_normal,
        0.5
    )

    # ========================================================
    # OVEREXPOSED IMAGE WEIGHT
    #
    # V3.5:
    # Moderate shadow recovery.
    #
    # Brighter than V3.4.
    # More controlled than V3.3.
    # ========================================================

    w_over = (
        0.03 +
        1.20 * shadow_recovery_mask +
        0.45 * q_over
    )

    # ========================================================
    # PATCH-FREE WEIGHT SMOOTHING
    #
    # ONLY WEIGHTS ARE SMOOTHED.
    # ORIGINAL IMAGES ARE NOT BLURRED.
    # ========================================================

    w_under = cv2.GaussianBlur(
        w_under.astype(
            np.float32
        ),
        (0, 0),
        15
    )

    w_normal = cv2.GaussianBlur(
        w_normal.astype(
            np.float32
        ),
        (0, 0),
        15
    )

    w_over = cv2.GaussianBlur(
        w_over.astype(
            np.float32
        ),
        (0, 0),
        15
    )

    # ========================================================
    # PREVENT NEGATIVE WEIGHTS
    # ========================================================

    w_under = np.maximum(
        w_under,
        0.0
    )

    w_normal = np.maximum(
        w_normal,
        0.0
    )

    w_over = np.maximum(
        w_over,
        0.0
    )

    # ========================================================
    # NORMALIZE WEIGHTS
    # ========================================================

    weight_sum = (
        w_under +
        w_normal +
        w_over +
        EPS
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

    # ========================================================
    # HDR FUSION
    #
    # ORIGINAL IMAGES ARE NOT BLURRED.
    # ========================================================

    fused = (
        under *
        w_under[:, :, None] +

        normal *
        w_normal[:, :, None] +

        over *
        w_over[:, :, None]
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ========================================================
    # SAVE DIAGNOSTICS
    # ========================================================

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "01_highlight_mask.png"
        ),
        np.uint8(
            np.clip(
                highlight_mask,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "02_shadow_mask.png"
        ),
        np.uint8(
            np.clip(
                shadow_mask,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "03_deep_shadow_mask.png"
        ),
        np.uint8(
            np.clip(
                deep_shadow_mask,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "04_shadow_recovery_mask.png"
        ),
        np.uint8(
            np.clip(
                shadow_recovery_mask,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "05_weight_under.png"
        ),
        np.uint8(
            np.clip(
                w_under,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "06_weight_normal.png"
        ),
        np.uint8(
            np.clip(
                w_normal,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            OUTPUT_DIR,
            "07_weight_over.png"
        ),
        np.uint8(
            np.clip(
                w_over,
                0.0,
                1.0
            ) * 255
        )
    )

    # ========================================================
    # SAVE FINAL OUTPUT
    # ========================================================

    output_path = os.path.join(
        OUTPUT_DIR,
        "FINAL_ADAPTIVE_FUSION_V3_5.jpg"
    )

    output_bgr = cv2.cvtColor(
        np.uint8(
            fused * 255
        ),
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        output_path,
        output_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            100
        ]
    )

    # ========================================================
    # FINAL LOG
    # ========================================================

    print()

    print(
        "===================================="
    )

    print(
        "ADAPTIVE HDR FUSION V3.5 COMPLETED"
    )

    print(
        "===================================="
    )

    print()

    print(
        f"Output: {output_path}"
    )

    print()

    print(
        "Mean fusion weights:"
    )

    print(
        f"Under : {np.mean(w_under):.4f}"
    )

    print(
        f"Normal: {np.mean(w_normal):.4f}"
    )

    print(
        f"Over  : {np.mean(w_over):.4f}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    adaptive_hdr_fusion()
