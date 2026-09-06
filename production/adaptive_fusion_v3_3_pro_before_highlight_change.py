#!/usr/bin/env python3

"""
ADAPTIVE EXPOSURE FUSION V3.3 PRO

Pipeline:
    UNDER  -> true highlight recovery
    NORMAL -> brightness, color and depth anchor
    OVER   -> deep shadow recovery only

Designed to preserve the natural brightness and depth of
the normal exposure while selectively recovering information
from the underexposed and overexposed images.
"""

import os
import argparse

import cv2
import numpy as np


# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-8


# ============================================================
# IMAGE I/O
# ============================================================

def load_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise FileNotFoundError(
            f"\nCould not load image:\n{path}\n"
        )

    return (
        image.astype(np.float32)
        / 255.0
    )


def save_image(path, image):

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

    output = (
        image * 255.0
    ).astype(np.uint8)

    success = cv2.imwrite(
        path,
        output
    )

    if not success:

        raise RuntimeError(
            f"Could not save:\n{path}"
        )


def save_map(path, data):

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    data = np.clip(
        data,
        0.0,
        1.0
    )

    output = (
        data * 255.0
    ).astype(np.uint8)

    cv2.imwrite(
        path,
        output
    )


# ============================================================
# LUMINANCE
# ============================================================

def luminance(image):

    b = image[:, :, 0]

    g = image[:, :, 1]

    r = image[:, :, 2]

    lum = (
        0.0722 * b
        + 0.7152 * g
        + 0.2126 * r
    )

    return np.clip(
        lum,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# SMOOTHSTEP
# ============================================================

def smoothstep(
    x,
    edge0,
    edge1
):

    t = np.clip(
        (x - edge0)
        / (edge1 - edge0 + EPS),
        0.0,
        1.0
    )

    return (
        t * t *
        (3.0 - 2.0 * t)
    ).astype(np.float32)


# ============================================================
# ALIGN IMAGE
# ============================================================

def align_to_normal(
    source,
    normal
):

    gray_source = cv2.cvtColor(
        (source * 255).astype(np.uint8),
        cv2.COLOR_BGR2GRAY
    )

    gray_normal = cv2.cvtColor(
        (normal * 255).astype(np.uint8),
        cv2.COLOR_BGR2GRAY
    )

    try:

        warp_matrix = np.eye(
            2,
            3,
            dtype=np.float32
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS
            | cv2.TERM_CRITERIA_COUNT,
            50,
            1e-6
        )

        cv2.findTransformECC(
            gray_normal,
            gray_source,
            warp_matrix,
            cv2.MOTION_EUCLIDEAN,
            criteria,
            None,
            5
        )

        h, w = normal.shape[:2]

        aligned = cv2.warpAffine(
            source,
            warp_matrix,
            (w, h),
            flags=(
                cv2.INTER_LINEAR
                | cv2.WARP_INVERSE_MAP
            ),
            borderMode=cv2.BORDER_REFLECT
        )

        return aligned

    except cv2.error:

        print(
            "Warning: alignment failed."
        )

        print(
            "Using original exposure."
        )

        return source


# ============================================================
# LOCAL DETAIL
# ============================================================

def local_detail(
    image,
    sigma=2.0
):

    lum = luminance(
        image
    )

    blurred = cv2.GaussianBlur(
        lum,
        (0, 0),
        sigma
    )

    detail = np.abs(
        lum - blurred
    )

    return detail.astype(
        np.float32
    )


# ============================================================
# DETAIL VALIDITY
# ============================================================

def detail_validity(
    source,
    normal
):

    source_detail = local_detail(
        source
    )

    normal_detail = local_detail(
        normal
    )

    ratio = (
        source_detail + EPS
    ) / (
        normal_detail + EPS
    )

    ratio = np.clip(
        ratio,
        0.0,
        2.0
    )

    detail_strength = smoothstep(
        source_detail,
        0.002,
        0.030
    )

    relative_detail = smoothstep(
        ratio,
        0.20,
        0.70
    )

    validity = np.maximum(
        detail_strength,
        relative_detail
    )

    return np.clip(
        validity,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# MAIN FUSION
# ============================================================

def adaptive_fusion(
    under,
    normal,
    over
):

    # --------------------------------------------------------
    # MATCH RESOLUTION
    # --------------------------------------------------------

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

    print()

    # --------------------------------------------------------
    # ALIGNMENT
    # --------------------------------------------------------

    print(
        "Aligning underexposed image..."
    )

    under = align_to_normal(
        under,
        normal
    )

    print(
        "Aligning overexposed image..."
    )

    over = align_to_normal(
        over,
        normal
    )

    # --------------------------------------------------------
    # LUMINANCE
    # --------------------------------------------------------

    print(
        "Creating luminance maps..."
    )

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
    # Only strong highlights use the underexposed image.
    # Medium tones remain controlled by NORMAL.
    # ========================================================

    print(
        "Creating highlight recovery mask..."
    )
    # Detect only genuinely bright regions.
    # Starting slightly later prevents normal bright wood,
    # walls and other mid-high tones from being unnecessarily
    # replaced by the underexposed image.

    highlight_mask = smoothstep(
        lum_normal,
        0.82,
        0.97
    )

    # Require the normal image to actually be losing highlight
    # information before allowing underexposure recovery.
 
    highlight_clipping = smoothstep(
        lum_normal,
        0.88,
        0.98
    )

    highlight_mask = (
        highlight_mask
        * (
            0.35
            + 0.65 * highlight_clipping
        )
    )

    highlight_mask = np.clip(
        highlight_mask,
        0.0,
        1.0
    ).astype(np.float32)

    

# --------------------------------------------------------
    # UNDER VALIDITY
    # --------------------------------------------------------

    under_improvement = smoothstep(
        lum_normal - lum_under,
        0.015,
        0.20
    )

    under_not_clipped = (
        1.0 -
        smoothstep(
            lum_under,
            0.985,
            1.0
        )
    )

    under_detail = detail_validity(
        under,
        normal
    )

    under_validity = (
        under_not_clipped
        * np.maximum(
            under_improvement,
            0.50 * under_detail
        )
    )

    under_validity = np.clip(
        under_validity,
        0.0,
        1.0
    )

    # ========================================================
    # GENERAL SHADOW MASK
    #
    # Diagnostic only.
    # ========================================================

    print(
        "Creating shadow recovery mask..."
    )

    shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.10,
            0.42
        )
    )

    # ========================================================
    # DEEP SHADOW MASK
    #
    # IMPORTANT:
    # Only genuinely dark regions are allowed to receive
    # information from the OVER exposure.
    #
    # This protects medium exposure and preserves depth.
    # ========================================================

    deep_shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.025,
            0.24
        )
    )

    # --------------------------------------------------------
    # OVER VALIDITY
    # --------------------------------------------------------

    over_lift = smoothstep(
        lum_over - lum_normal,
        0.02,
        0.25
    )

    over_not_clipped = (
        1.0 -
        smoothstep(
            lum_over,
            0.985,
            1.0
        )
    )

    over_detail = detail_validity(
        over,
        normal
    )

    over_validity = (
        over_not_clipped
        * np.maximum(
            over_lift,
            0.50 * over_detail
        )
    )

    over_validity = np.clip(
        over_validity,
        0.0,
        1.0
    )

    # ========================================================
    # FINAL SHADOW RECOVERY MASK
    # ========================================================

    shadow_recovery_mask = (
        deep_shadow_mask
        * over_validity
    )

    shadow_recovery_mask = np.clip(
        shadow_recovery_mask,
        0.0,
        1.0
    )

    # ========================================================
    # SMALL MASK SMOOTHING
    #
    # Previous versions used large blur values.
    # That allowed exposure changes to spread too far and
    # caused loss of natural depth.
    #
    # Only masks are smoothed.
    # Images remain sharp.
    # ========================================================

    MASK_SIGMA = 4.0

    highlight_mask = cv2.GaussianBlur(
        highlight_mask,
        (0, 0),
        MASK_SIGMA
    )

    shadow_mask = cv2.GaussianBlur(
        shadow_mask,
        (0, 0),
        MASK_SIGMA
    )

    deep_shadow_mask = cv2.GaussianBlur(
        deep_shadow_mask,
        (0, 0),
        MASK_SIGMA
    )

    shadow_recovery_mask = cv2.GaussianBlur(
        shadow_recovery_mask,
        (0, 0),
        MASK_SIGMA
    )

    # ========================================================
    # WEIGHTS
    #
    # NORMAL MUST DOMINATE.
    # ========================================================

    print(
        "Creating exposure weights..."
    )

    NORMAL_BASE = 4.5

    UNDER_STRENGTH = 2.5

    OVER_STRENGTH = 2.0

    # --------------------------------------------------------
    # NORMAL
    #
    # Main brightness, color and depth anchor.
    # --------------------------------------------------------

    w_normal = np.ones_like(
        lum_normal,
        dtype=np.float32
    ) * NORMAL_BASE

    # --------------------------------------------------------
    # UNDER
    #
    # Highlight recovery only.
    # --------------------------------------------------------

    w_under = (
        UNDER_STRENGTH
        * highlight_mask
        * under_validity
    )

    # --------------------------------------------------------
    # OVER
    #
    # Deep shadow recovery only.
    # --------------------------------------------------------

    w_over = (
        OVER_STRENGTH
        * shadow_recovery_mask
    )

    # ========================================================
    # SMALL WEIGHT SMOOTHING
    # ========================================================

    WEIGHT_SIGMA = 2.0

    w_under = cv2.GaussianBlur(
        w_under,
        (0, 0),
        WEIGHT_SIGMA
    )

    w_over = cv2.GaussianBlur(
        w_over,
        (0, 0),
        WEIGHT_SIGMA
    )

    # ========================================================
    # NORMALIZE WEIGHTS
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

    weight_sum = (
        w_under
        + w_normal
        + w_over
        + EPS
    )

    w_under = (
        w_under
        / weight_sum
    )

    w_normal = (
        w_normal
        / weight_sum
    )

    w_over = (
        w_over
        / weight_sum
    )

    # ========================================================
    # FUSION
    #
    # The source images themselves are NEVER blurred.
    # ========================================================

    print(
        "Performing brightness and depth preserving fusion..."
    )

    fused = (

        under
        * w_under[:, :, None]

        +

        normal
        * w_normal[:, :, None]

        +

        over
        * w_over[:, :, None]

    )

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

    # ========================================================
    # BRIGHTNESS SAFETY
    #
    # Keep final brightness close to the NORMAL exposure.
    # This prevents global brightness drift.
    # ========================================================

    print(
        "Applying brightness safety..."
    )

    fused_lum = luminance(
        fused
    )

    normal_lum = lum_normal

    brightness_ratio = (
        normal_lum + EPS
    ) / (
        fused_lum + EPS
    )

    brightness_ratio = np.clip(
        brightness_ratio,
        0.92,
        1.08
    )

    # Apply only partially.
    # This retains recovered highlights/shadows.

    correction = (
        0.35
        + 0.65 * brightness_ratio
    )

    fused = (
        fused
        * correction[:, :, None]
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ========================================================
    # DEBUG DATA
    # ========================================================

    debug = {

        "01_highlight_mask":
            highlight_mask,

        "02_shadow_mask":
            shadow_mask,

        "03_deep_shadow_mask":
            deep_shadow_mask,

        "04_shadow_recovery_mask":
            shadow_recovery_mask,

        "05_under_validity":
            under_validity,

        "06_over_validity":
            over_validity,

        "07_weight_under":
            w_under,

        "08_weight_normal":
            w_normal,

        "09_weight_over":
            w_over
    }

    return (
        fused,
        debug,
        (
            float(np.mean(w_under)),
            float(np.mean(w_normal)),
            float(np.mean(w_over))
        )
    )


# ============================================================
# PROCESS
# ============================================================

def process(
    under_path,
    normal_path,
    over_path,
    output_dir
):

    print()

    print(
        "=" * 60
    )

    print(
        "ADAPTIVE HDR FUSION V3.3 PRO"
    )

    print(
        "NORMAL-DOMINANT DEPTH-PRESERVING FUSION"
    )

    print(
        "=" * 60
    )

    print()

    print(
        f"Under : {under_path}"
    )

    print(
        f"Normal: {normal_path}"
    )

    print(
        f"Over  : {over_path}"
    )

    print()

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORY
    # --------------------------------------------------------

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    print(
        "Loading images..."
    )

    under = load_image(
        under_path
    )

    normal = load_image(
        normal_path
    )

    over = load_image(
        over_path
    )

    # --------------------------------------------------------
    # FUSE
    # --------------------------------------------------------

    fused, debug, weights = adaptive_fusion(
        under,
        normal,
        over
    )

    # --------------------------------------------------------
    # SAVE FINAL
    # --------------------------------------------------------

    final_path = os.path.join(
        output_dir,
        "FINAL_SCENE005_PRO.jpg"
    )

    print()

    print(
        "Saving final image..."
    )

    save_image(
        final_path,
        fused
    )

    # --------------------------------------------------------
    # SAVE DEBUG MAPS
    # --------------------------------------------------------

    for name, data in debug.items():

        path = os.path.join(
            output_dir,
            f"{name}.png"
        )

        save_map(
            path,
            data
        )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print()

    print(
        "=" * 60
    )

    print(
        "ADAPTIVE HDR FUSION COMPLETED"
    )

    print(
        "=" * 60
    )

    print()

    print(
        f"Output:\n{final_path}"
    )

    print()

    print(
        "Mean fusion weights:"
    )

    print(
        f"Under : {weights[0]:.4f}"
    )

    print(
        f"Normal: {weights[1]:.4f}"
    )

    print(
        f"Over  : {weights[2]:.4f}"
    )

    print()

    print(
        "Pipeline:"
    )

    print(
        "Under  -> true highlight recovery"
    )

    print(
        "Normal -> brightness, color and depth anchor"
    )

    print(
        "Over   -> deep shadow recovery only"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Adaptive HDR Fusion V3.3 PRO"
        )
    )

    parser.add_argument(
        "--under",
        default=(
            "./my_test/scene005/"
            "01_under.jpg"
        )
    )

    parser.add_argument(
        "--normal",
        default=(
            "./my_test/scene005/"
            "02_normal.jpg"
        )
    )

    parser.add_argument(
        "--over",
        default=(
            "./my_test/scene005/"
            "03_over.jpg"
        )
    )

    parser.add_argument(
        "--output",
        default=(
            "./production/output/"
            "adaptive_fusion_v3_3_pro"
        )
    )

    args = parser.parse_args()

    process(
        args.under,
        args.normal,
        args.over,
        args.output
    )


if __name__ == "__main__":

    main()
