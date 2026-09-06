import cv2
import numpy as np
import os
import argparse


EPS = 1e-8


# ==========================================================
# IMAGE LOADING
# ==========================================================

def read_image(path):

    img = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    return (
        img.astype(np.float32) / 255.0
    )


# ==========================================================
# LUMINANCE
#
# OpenCV images are BGR
# ==========================================================

def luminance(img):

    return (
        0.0722 * img[:, :, 0] +
        0.7152 * img[:, :, 1] +
        0.2126 * img[:, :, 2]
    )


# ==========================================================
# SMOOTHSTEP
# ==========================================================

def smoothstep(x, low, high):

    x = np.clip(
        (x - low) /
        (high - low + EPS),
        0.0,
        1.0
    )

    return (
        x * x *
        (3.0 - 2.0 * x)
    )


# ==========================================================
# MULTI-SCALE WEIGHT SMOOTHING
#
# Smooths ONLY weights.
# The source images remain sharp.
# ==========================================================

def edge_smooth(weight):

    weight = weight.astype(
        np.float32
    )

    fine = weight

    medium = cv2.GaussianBlur(
        weight,
        (0, 0),
        sigmaX=8,
        sigmaY=8
    )

    large = cv2.GaussianBlur(
        weight,
        (0, 0),
        sigmaX=30,
        sigmaY=30
    )

    smooth = (
        0.25 * fine +
        0.45 * medium +
        0.30 * large
    )

    return np.maximum(
        smooth,
        EPS
    )


# ==========================================================
# EXPOSURE QUALITY
#
# Pixels near the middle of the exposure range
# receive higher confidence.
# ==========================================================

def exposure_quality(lum):

    sigma = 0.25

    quality = np.exp(
        -((lum - 0.5) ** 2) /
        (2.0 * sigma * sigma)
    )

    return quality.astype(
        np.float32
    )


# ==========================================================
# SAFE LUMINANCE REPLACEMENT
#
# Keeps the original color ratios while changing
# brightness.
# ==========================================================

def replace_luminance(img, target_lum):

    current_lum = luminance(
        img
    )

    ratio = (
        target_lum /
        (current_lum + EPS)
    )

    ratio = np.clip(
        ratio,
        0.45,
        2.20
    )

    result = (
        img *
        ratio[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ==========================================================
# CONTROLLED MIDTONE LIFT
#
# Lifts darker midtones without touching:
# - deep black excessively
# - bright highlights
# ==========================================================

def midtone_lift(lum):

    dark_start = smoothstep(
        lum,
        0.03,
        0.16
    )

    mid_end = (
        1.0 -
        smoothstep(
            lum,
            0.45,
            0.82
        )
    )

    midtone_region = (
        dark_start *
        mid_end
    )

    lift = (
        0.070 *
        midtone_region
    )

    result = (
        lum + lift
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ==========================================================
# SUBTLE LOCAL CONTRAST
#
# Operates only on luminance.
# ==========================================================

def subtle_local_contrast(img):

    lab = cv2.cvtColor(
        img.astype(np.float32),
        cv2.COLOR_BGR2LAB
    )

    L = lab[:, :, 0]

    base = cv2.GaussianBlur(
        L,
        (0, 0),
        sigmaX=18,
        sigmaY=18
    )

    detail = (
        L - base
    )

    # Very subtle enhancement.
    L_enhanced = (
        L +
        0.10 * detail
    )

    lab[:, :, 0] = np.clip(
        L_enhanced,
        0.0,
        100.0
    )

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ==========================================================
# BRIGHTNESS ANCHOR
#
# Prevents HDR fusion from drifting too far away
# from the normal exposure.
# ==========================================================

def brightness_anchor(
    fused,
    normal,
    highlight_mask,
    shadow_mask
):

    fused_lum = luminance(
        fused
    )

    normal_lum = luminance(
        normal
    )

    # In highlights:
    # preserve fusion because highlight recovery
    # is important.
    highlight_protection = (
        highlight_mask
    )

    # In dark/midtone regions:
    # allow the improved fusion to remain.
    shadow_protection = (
        shadow_mask
    )

    # Normal exposure influence.
    normal_anchor = (
        0.38 *
        (1.0 - highlight_protection)
    )

    # Reduce anchoring inside genuine shadows,
    # where recovery is needed.
    normal_anchor = (
        normal_anchor *
        (1.0 - 0.55 * shadow_protection)
    )

    target_lum = (
        fused_lum *
        (1.0 - normal_anchor) +

        normal_lum *
        normal_anchor
    )

    result = replace_luminance(
        fused,
        target_lum
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ==========================================================
# MAIN PROCESS
# ==========================================================

def process(
    under_path,
    normal_path,
    over_path,
    output_path
):

    print("\n==============================================")
    print(" ADAPTIVE HDR FUSION V3.3 PRO")
    print(" BRIGHTNESS-PRESERVING EDITION")
    print("==============================================\n")

    # ------------------------------------------------------
    # LOAD
    # ------------------------------------------------------

    print("Loading images...")

    under = read_image(
        under_path
    )

    normal = read_image(
        normal_path
    )

    over = read_image(
        over_path
    )

    # ------------------------------------------------------
    # MATCH SIZE
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

    print(
        f"Resolution: {w} x {h}"
    )

    # ------------------------------------------------------
    # LUMINANCE
    # ------------------------------------------------------

    print("Creating luminance maps...")

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
    # Underexposed image recovers windows/highlights.
    # ======================================================

    print("Creating highlight recovery mask...")

    highlight_mask = smoothstep(
        lum_normal,
        0.55,
        0.82
    )

    # ======================================================
    # GENERAL SHADOW DETECTION
    # ======================================================

    print("Creating shadow recovery mask...")

    shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.18,
            0.55
        )
    )

    # ======================================================
    # DEEP SHADOW DETECTION
    # ======================================================

    deep_shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.05,
            0.32
        )
    )

    shadow_recovery_mask = np.maximum(
        shadow_mask,
        deep_shadow_mask
    )

    # ======================================================
    # CHECK WHETHER OVER EXPOSURE HAS USABLE DETAIL
    #
    # Avoid blindly brightening clipped regions.
    # ======================================================

    over_detail_mask = smoothstep(
        lum_over,
        0.04,
        0.20
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

    # ======================================================
    # SMOOTH MASKS
    #
    # Only masks are blurred.
    # Source images remain untouched.
    # ======================================================

    highlight_mask = cv2.GaussianBlur(
        highlight_mask.astype(np.float32),
        (0, 0),
        12
    )

    shadow_mask = cv2.GaussianBlur(
        shadow_mask.astype(np.float32),
        (0, 0),
        15
    )

    deep_shadow_mask = cv2.GaussianBlur(
        deep_shadow_mask.astype(np.float32),
        (0, 0),
        18
    )

    shadow_recovery_mask = cv2.GaussianBlur(
        shadow_recovery_mask.astype(np.float32),
        (0, 0),
        18
    )

    # ======================================================
    # EXPOSURE QUALITY
    # ======================================================

    print("Calculating exposure confidence...")

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
    # UNDEREXPOSED IMAGE WEIGHT
    #
    # Used mainly for highlights and windows.
    # ======================================================

    w_under = (
        0.08 +
        2.20 * highlight_mask +
        0.45 * q_under
    )

    # ======================================================
    # NORMAL IMAGE WEIGHT
    #
    # THIS IS THE MAIN VISUAL ANCHOR.
    #
    # Increased dominance compared with aggressive
    # HDR blending to preserve the natural appearance.
    # ======================================================

    w_normal = (
        3.20 +
        1.60 * q_normal -
        0.75 * deep_shadow_mask -
        0.35 * highlight_mask
    )

    w_normal = np.maximum(
        w_normal,
        0.70
    )

    # ======================================================
    # OVEREXPOSED IMAGE WEIGHT
    #
    # Used ONLY in recoverable shadows.
    # ======================================================

    w_over = (
        0.02 +
        1.75 * shadow_recovery_mask +
        0.45 * q_over *
        shadow_recovery_mask
    )

    # ======================================================
    # SMOOTH WEIGHTS
    # ======================================================

    print("Smoothing fusion weights...")

    w_under = edge_smooth(
        w_under
    )

    w_normal = edge_smooth(
        w_normal
    )

    w_over = edge_smooth(
        w_over
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
    # HDR FUSION
    # ======================================================

    print("Performing brightness-preserving fusion...")

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

    # ======================================================
    # BRIGHTNESS ANCHOR
    #
    # Pulls the result gently toward the natural
    # brightness of the normal exposure.
    # ======================================================

    print("Anchoring natural brightness...")

    fused = brightness_anchor(
        fused,
        normal,
        highlight_mask,
        shadow_mask
    )

    # ======================================================
    # CONTROLLED MIDTONE IMPROVEMENT
    #
    # Brightens darker interior areas while
    # protecting bright windows/highlights.
    # ======================================================

    print("Improving midtone brightness...")

    fused_lum = luminance(
        fused
    )

    lifted_lum = midtone_lift(
        fused_lum
    )

    # Protect highlights from the lift.
    lift_protection = (
        1.0 -
        0.90 * highlight_mask
    )

    target_lum = (

        fused_lum *

        (1.0 - lift_protection)

        +

        lifted_lum *

        lift_protection
    )

    fused = replace_luminance(
        fused,
        target_lum
    )

    # ======================================================
    # FINAL NORMAL COLOR ANCHOR
    #
    # Prevents unnatural color drift.
    # ======================================================

    print("Protecting natural colors...")

    color_anchor = 0.10

    color_anchor_map = (

        color_anchor *

        (1.0 - highlight_mask)

        *

        (1.0 - 0.35 * shadow_recovery_mask)
    )

    fused = (

        fused *

        (1.0 - color_anchor_map[:, :, None])

        +

        normal *

        color_anchor_map[:, :, None]
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ======================================================
    # SUBTLE LOCAL CONTRAST
    # ======================================================

    print("Applying subtle local contrast...")

    fused = subtle_local_contrast(
        fused
    )

    # ======================================================
    # FINAL HIGHLIGHT SAFETY
    #
    # Keep recovered highlight regions stable.
    # ======================================================

    fused_lum = luminance(
        fused
    )

    normal_lum = luminance(
        normal
    )

    safety_mask = smoothstep(
        fused_lum,
        0.88,
        0.98
    )

    safety_target = np.minimum(
        fused_lum,
        normal_lum +
        0.05
    )

    final_lum = (

        fused_lum *
        (1.0 - safety_mask)

        +

        safety_target *
        safety_mask
    )

    fused = replace_luminance(
        fused,
        final_lum
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ======================================================
    # OUTPUT DIRECTORY
    # ======================================================

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir != "":

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    base_name = os.path.splitext(
        os.path.basename(output_path)
    )[0]

    # ======================================================
    # SAVE DIAGNOSTICS
    # ======================================================

    print("Saving diagnostics...")

    highlight_path = os.path.join(
        output_dir,
        base_name +
        "_highlight_mask.png"
    )

    shadow_path = os.path.join(
        output_dir,
        base_name +
        "_shadow_mask.png"
    )

    deep_shadow_path = os.path.join(
        output_dir,
        base_name +
        "_deep_shadow_mask.png"
    )

    recovery_path = os.path.join(
        output_dir,
        base_name +
        "_shadow_recovery_mask.png"
    )

    weight_under_path = os.path.join(
        output_dir,
        base_name +
        "_weight_under.png"
    )

    weight_normal_path = os.path.join(
        output_dir,
        base_name +
        "_weight_normal.png"
    )

    weight_over_path = os.path.join(
        output_dir,
        base_name +
        "_weight_over.png"
    )

    cv2.imwrite(
        highlight_path,
        np.uint8(
            np.clip(
                highlight_mask,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        shadow_path,
        np.uint8(
            np.clip(
                shadow_mask,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        deep_shadow_path,
        np.uint8(
            np.clip(
                deep_shadow_mask,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        recovery_path,
        np.uint8(
            np.clip(
                shadow_recovery_mask,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        weight_under_path,
        np.uint8(
            np.clip(
                w_under,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        weight_normal_path,
        np.uint8(
            np.clip(
                w_normal,
                0.0,
                1.0
            ) * 255
        )
    )

    cv2.imwrite(
        weight_over_path,
        np.uint8(
            np.clip(
                w_over,
                0.0,
                1.0
            ) * 255
        )
    )

    # ======================================================
    # SAVE FINAL IMAGE
    # ======================================================

    print("Saving final image...")

    success = cv2.imwrite(
        output_path,
        np.uint8(
            np.clip(
                fused,
                0.0,
                1.0
            ) * 255
        )
    )

    if not success:

        raise RuntimeError(
            f"Could not save output:\n{output_path}"
        )

    # ======================================================
    # RESULTS
    # ======================================================

    print("\n==============================================")
    print(" FUSION COMPLETE")
    print("==============================================")

    print(
        f"\nFinal output:\n"
        f"  {output_path}"
    )

    print(
        f"\nHighlight mask:\n"
        f"  {highlight_path}"
    )

    print(
        f"\nShadow mask:\n"
        f"  {shadow_path}"
    )

    print("\nMean fusion weights:")

    print(
        f"  Under  : "
        f"{w_under.mean():.4f}"
    )

    print(
        f"  Normal : "
        f"{w_normal.mean():.4f}"
    )

    print(
        f"  Over   : "
        f"{w_over.mean():.4f}"
    )

    print("\nPipeline:")

    print(
        "Under  -> highlight recovery"
    )

    print(
        "Normal -> brightness, color and depth anchor"
    )

    print(
        "Over   -> deep shadow recovery only"
    )


# ==========================================================
# COMMAND LINE
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Adaptive three-exposure HDR fusion"
    )

    parser.add_argument(
        "--under",
        required=True,
        help="Path to underexposed image"
    )

    parser.add_argument(
        "--normal",
        required=True,
        help="Path to normal exposure image"
    )

    parser.add_argument(
        "--over",
        required=True,
        help="Path to overexposed image"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for final output image"
    )

    args = parser.parse_args()

    process(
        args.under,
        args.normal,
        args.over,
        args.output
    )
