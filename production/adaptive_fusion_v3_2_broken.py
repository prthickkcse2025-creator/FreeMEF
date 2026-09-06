import cv2
import numpy as np
import os

EPS = 1e-8


# ==================================================
# IMAGE FUNCTIONS
# ==================================================

def read_image(path):

    img = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise FileNotFoundError(
            f"Could not read: {path}"
        )

    return img.astype(
        np.float32
    ) / 255.0


def save_image(path, img):

    img = np.clip(
        img,
        0.0,
        1.0
    )

    img = np.uint8(
        img * 255.0
    )

    cv2.imwrite(
        path,
        img
    )


# ==================================================
# LUMINANCE
# OpenCV images are BGR
# ==================================================

def luminance(img):

    return (
        0.0722 * img[:, :, 0] +
        0.7152 * img[:, :, 1] +
        0.2126 * img[:, :, 2]
    )


# ==================================================
# SMOOTHSTEP
# ==================================================

def smoothstep(
    x,
    low,
    high
):

    x = np.clip(
        (x - low) /
        (high - low + EPS),
        0.0,
        1.0
    )

    return (
        x *
        x *
        (
            3.0 -
            2.0 * x
        )
    )


# ==================================================
# EDGE-AWARE WEIGHT SMOOTHING
# ==================================================

def edge_smooth(weight):

    weight_8 = np.uint8(
        np.clip(
            weight * 255.0,
            0,
            255
        )
    )

    smooth = cv2.bilateralFilter(
        weight_8,
        d=9,
        sigmaColor=50,
        sigmaSpace=50
    )

    smooth = smooth.astype(
        np.float32
    ) / 255.0

    return np.maximum(
        smooth,
        EPS
    )


# ==================================================
# EXPOSURE QUALITY
#
# Gives high weight to pixels close
# to good mid-tone exposure.
# ==================================================

def exposure_quality(lum):

    sigma = 0.25

    quality = np.exp(
        -(
            (lum - 0.5) ** 2
        ) /
        (
            2.0 *
            sigma *
            sigma
        )
    )

    return quality.astype(
        np.float32
    )


# ==================================================
# MAIN
# ==================================================

def main():

    # ----------------------------------------------
    # INPUT DIRECTORY
    #
    # CHANGE ONLY THIS PATH WHEN TESTING
    # DIFFERENT RAW/JPG BRACKET SCENES
    # ----------------------------------------------

    input_dir = (
        "/home/prthick/FreeMEF/"
        "my_test/scene002"
    )

    # ----------------------------------------------
    # OUTPUT DIRECTORY
    # ----------------------------------------------

    output_dir = (
        "/home/prthick/FreeMEF/"
        "production/output/"
        "adaptive_fusion_v3_2"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ----------------------------------------------
    # INPUT FILES
    # ----------------------------------------------

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

    # ----------------------------------------------
    # HEADER
    # ----------------------------------------------

    print(
        "\n===================================="
    )

    print(
        "ADAPTIVE HDR FUSION V3.2"
    )

    print(
        "SHADOW-AWARE + PATCH-FREE"
    )

    print(
        "===================================="
    )

    # ----------------------------------------------
    # LOAD IMAGES
    # ----------------------------------------------

    under = read_image(
        under_path
    )

    normal = read_image(
        normal_path
    )

    over = read_image(
        over_path
    )

    # ----------------------------------------------
    # ENSURE SAME SIZE
    # ----------------------------------------------

    h, w = normal.shape[:2]

    if under.shape[:2] != (
        h,
        w
    ):

        under = cv2.resize(
            under,
            (
                w,
                h
            ),
            interpolation=cv2.INTER_LINEAR
        )

    if over.shape[:2] != (
        h,
        w
    ):

        over = cv2.resize(
            over,
            (
                w,
                h
            ),
            interpolation=cv2.INTER_LINEAR
        )

    print(
        f"\nResolution: {w} x {h}"
    )

    # ==============================================
    # LUMINANCE
    # ==============================================

    lum_under = luminance(
        under
    )

    lum_normal = luminance(
        normal
    )

    lum_over = luminance(
        over
    )

    # ==============================================
    # HIGHLIGHT DETECTION
    #
    # Use underexposed image for bright
    # windows and highlight recovery.
    # ==============================================

    highlight_mask = smoothstep(
        lum_normal,
        0.50,
        0.82
    )

    # ==============================================
    # GENERAL SHADOW DETECTION
    #
    # Detect dark and moderately dark areas
    # ==============================================

    shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.25,
            0.60
        )
    )

    # ==============================================
    # DEEP SHADOW DETECTION
    #
    # Stronger detection for very dark regions
    # ==============================================

    deep_shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.10,
            0.38
        )
    )

    # ==============================================
    # CHECK IF OVEREXPOSED IMAGE ACTUALLY
    # PROVIDES MORE BRIGHTNESS
    #
    # This prevents blindly using the over image.
    # ==============================================

    exposure_gain = (
        lum_over -
        lum_normal
    )

    over_recovery_mask = smoothstep(
        exposure_gain,
        0.03,
        0.20
    )

    # ==============================================
    # CHECK WHETHER OVER IMAGE CONTAINS
    # USABLE DETAIL
    #
    # Avoid heavily clipped or useless regions.
    # ==============================================

    over_detail_mask = smoothstep(
        lum_over,
        0.08,
        0.38
    )

    # ==============================================
    # COMBINED SHADOW RECOVERY MASK
    #
    # A region gets strong over-exposure recovery
    # only when:
    #
    # 1. Normal image is dark
    # 2. Over image is brighter
    # 3. Over image has useful information
    # ==============================================

    shadow_recovery_mask = (
        np.maximum(
            shadow_mask,
            deep_shadow_mask
        )
        *
        over_recovery_mask
        *
        over_detail_mask
    )

    shadow_recovery_mask = np.clip(
        shadow_recovery_mask,
        0.0,
        1.0
    )

    # ==============================================
    # EXPOSURE QUALITY
    # ==============================================

    q_under = exposure_quality(
        lum_under
    )

    q_normal = exposure_quality(
        lum_normal
    )

    q_over = exposure_quality(
        lum_over
    )

    # ==============================================
    # UNDEREXPOSED IMAGE WEIGHT
    #
    # Mainly recovers:
    # - Windows
    # - Bright lights
    # - Highlights
    # ==============================================

    w_under = (
        0.10 +
        2.0 * highlight_mask +
        0.5 * q_under
    )

    # ==============================================
    # NORMAL IMAGE WEIGHT
    #
    # Main anchor of the final image.
    # ==============================================

    w_normal = (
        2.5 +
        1.5 * q_normal -
        0.8 * deep_shadow_mask
    )

    w_normal = np.maximum(
        w_normal,
        0.5
    )

    # ==============================================
    # OVEREXPOSED IMAGE WEIGHT
    #
    # Strongly used only in detected
    # recoverable shadow areas.
    # ==============================================

    w_over = (
        0.05 +
        4.0 * shadow_recovery_mask +
        0.7 * q_over
    )

    # ==============================================
    # PATCH-FREE SMOOTHING
    #
    # Smooth ONLY weight maps.
    # Image detail remains untouched.
    # ==============================================

    w_under = edge_smooth(
        w_under
    )

    w_normal = edge_smooth(
        w_normal
    )

    w_over = edge_smooth(
        w_over
    )

    # ==============================================
    # NORMALIZE WEIGHTS
    # ==============================================

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

    # ==============================================
    # HDR FUSION
    # ==============================================

    fused = (
        under *
        w_under[:, :, None]
        +
        normal *
        w_normal[:, :, None]
        +
        over *
        w_over[:, :, None]
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ==============================================
    # SAVE DIAGNOSTICS
    # ==============================================

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
            "03_deep_shadow_mask.png"
        ),
        deep_shadow_mask
    )

    save_image(
        os.path.join(
            output_dir,
            "04_shadow_recovery_mask.png"
        ),
        shadow_recovery_mask
    )

    save_image(
        os.path.join(
            output_dir,
            "05_weight_under.png"
        ),
        w_under
    )

    save_image(
        os.path.join(
            output_dir,
            "06_weight_normal.png"
        ),
        w_normal
    )

    save_image(
        os.path.join(
            output_dir,
            "07_weight_over.png"
        ),
        w_over
    )

    # ==============================================
    # FINAL OUTPUT
    # ==============================================

    final_path = os.path.join(
        output_dir,
        "FINAL_ADAPTIVE_FUSION_V3_2.png"
    )

    save_image(
        final_path,
        fused
    )

    # ==============================================
    # COMPLETE
    # ==============================================

    print(
        "\n===================================="
    )

    print(
        "V3.2 COMPLETED SUCCESSFULLY"
    )

    print(
        "===================================="
    )

    print(
        f"\nOutput: {final_path}"
    )

    print(
        "\nMean fusion weights:"
    )

    print(
        f"Under : {w_under.mean():.4f}"
    )

    print(
        f"Normal: {w_normal.mean():.4f}"
    )

    print(
        f"Over  : {w_over.mean():.4f}"
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    main()
