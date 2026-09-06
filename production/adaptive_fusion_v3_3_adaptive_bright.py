#!/usr/bin/env python3

"""
ADAPTIVE HDR FUSION V3.3
ADAPTIVE BRIGHT-EXPOSURE EDITION

Pipeline:
UNDER  -> highlight recovery
NORMAL -> main color, brightness and depth anchor
OVER   -> adaptive shadow and dark-scene recovery

The amount of bright exposure used is automatically adjusted
according to the brightness characteristics of each scene.
"""

import os
import argparse

import cv2
import numpy as np


EPS = 1e-8


# ============================================================
# IMAGE I/O
# ============================================================

def read_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    return image.astype(np.float32) / 255.0


# ============================================================
# LUMINANCE
# ============================================================

def luminance(image):

    b = image[:, :, 0]
    g = image[:, :, 1]
    r = image[:, :, 2]

    lum = (
        0.0722 * b +
        0.7152 * g +
        0.2126 * r
    )

    return np.clip(
        lum,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# SMOOTHSTEP
# ============================================================

def smoothstep(x, edge0, edge1):

    t = np.clip(
        (x - edge0) /
        (edge1 - edge0 + EPS),
        0.0,
        1.0
    )

    return (
        t * t *
        (3.0 - 2.0 * t)
    ).astype(np.float32)


# ============================================================
# ALIGN IMAGE TO NORMAL
# ============================================================

def align_to_normal(source, normal):

    gray_source = cv2.cvtColor(
        np.uint8(
            np.clip(source, 0, 1) * 255
        ),
        cv2.COLOR_BGR2GRAY
    )

    gray_normal = cv2.cvtColor(
        np.uint8(
            np.clip(normal, 0, 1) * 255
        ),
        cv2.COLOR_BGR2GRAY
    )

    try:

        warp_matrix = np.eye(
            2,
            3,
            dtype=np.float32
        )

        criteria = (
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
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
                cv2.INTER_LINEAR |
                cv2.WARP_INVERSE_MAP
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
# EXPOSURE QUALITY
# ============================================================

def exposure_quality(lum):

    quality = 1.0 - np.abs(
        lum - 0.50
    ) / 0.50

    return np.clip(
        quality,
        0.0,
        1.0
    ).astype(np.float32)


# ============================================================
# EDGE-AWARE WEIGHT SMOOTHING
# ============================================================

def smooth_weight(weight):

    weight = cv2.GaussianBlur(
        weight,
        (0, 0),
        2.0
    )

    return np.clip(
        weight,
        0.0,
        None
    ).astype(np.float32)


# ============================================================
# ADAPTIVE SCENE ANALYSIS
# ============================================================

def analyze_scene(lum_normal):

    # Overall scene brightness.
    mean_lum = float(
        np.mean(lum_normal)
    )

    # Median brightness gives a more robust estimate.
    median_lum = float(
        np.median(lum_normal)
    )

    # Percentage of dark pixels.
    dark_ratio = float(
        np.mean(
            lum_normal < 0.35
        )
    )

    # Percentage of deep shadow pixels.
    deep_shadow_ratio = float(
        np.mean(
            lum_normal < 0.20
        )
    )

    # --------------------------------------------------------
    # DARK SCENE SCORE
    #
    # 0 = naturally bright scene
    # 1 = significantly dark scene
    # --------------------------------------------------------

    mean_darkness = 1.0 - smoothstep(
        np.array(mean_lum),
        0.25,
        0.60
    )

    median_darkness = 1.0 - smoothstep(
        np.array(median_lum),
        0.20,
        0.55
    )

    dark_area_score = smoothstep(
        np.array(dark_ratio),
        0.20,
        0.75
    )

    deep_shadow_score = smoothstep(
        np.array(deep_shadow_ratio),
        0.05,
        0.35
    )

    darkness_score = (

        0.35 * mean_darkness +

        0.25 * median_darkness +

        0.25 * dark_area_score +

        0.15 * deep_shadow_score
    )

    darkness_score = float(
        np.clip(
            darkness_score,
            0.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # ADAPTIVE BRIGHT-EXPOSURE STRENGTH
    #
    # Bright scenes:
    #   use less overexposed image
    #
    # Dark scenes:
    #   use more overexposed image
    # --------------------------------------------------------

    over_strength = (
        0.70 +

        1.80 * darkness_score
    )

    return {

        "mean_lum": mean_lum,

        "median_lum": median_lum,

        "dark_ratio": dark_ratio,

        "deep_shadow_ratio": deep_shadow_ratio,

        "darkness_score": darkness_score,

        "over_strength": over_strength
    }


# ============================================================
# MAIN FUSION
# ============================================================

def adaptive_fusion(
    under,
    normal,
    over
):

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

    # ========================================================
    # ALIGNMENT
    # ========================================================

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

    # ========================================================
    # LUMINANCE
    # ========================================================

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
    # ANALYZE SCENE
    # ========================================================

    print(
        "Analyzing scene brightness..."
    )

    scene = analyze_scene(
        lum_normal
    )

    print()

    print(
        "Scene analysis:"
    )

    print(
        f"  Mean brightness : "
        f"{scene['mean_lum']:.4f}"
    )

    print(
        f"  Median brightness: "
        f"{scene['median_lum']:.4f}"
    )

    print(
        f"  Dark pixel ratio : "
        f"{scene['dark_ratio']:.4f}"
    )

    print(
        f"  Darkness score   : "
        f"{scene['darkness_score']:.4f}"
    )

    print(
        f"  Adaptive bright-exposure strength: "
        f"{scene['over_strength']:.4f}"
    )

    print()

    # ========================================================
    # HIGHLIGHT MASK
    #
    # Underexposed image is mainly used for highlights.
    # ========================================================

    print(
        "Creating highlight recovery mask..."
    )

    highlight_mask = smoothstep(
        lum_normal,
        0.55,
        0.82
    )

    # ========================================================
    # SHADOW MASK
    #
    # This is intentionally not extremely restrictive.
    # The client requested more bright exposure.
    # ========================================================

    print(
        "Creating shadow recovery mask..."
    )

    shadow_mask = (

        1.0 -

        smoothstep(
            lum_normal,
            0.18,
            0.58
        )
    )

    # ========================================================
    # DEEP SHADOW MASK
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
    # RECOVERABILITY CHECK
    #
    # Only use overexposed image when it actually contains
    # useful brightness information.
    # ========================================================

    over_improvement = smoothstep(
        lum_over - lum_normal,
        0.02,
        0.20
    )

    over_not_clipped = (

        1.0 -

        smoothstep(
            lum_over,
            0.985,
            1.0
        )
    )

    # ========================================================
    # FINAL SHADOW RECOVERY MASK
    #
    # Combination of:
    # - shadow location
    # - actual brightness improvement
    # - clipping protection
    # ========================================================

    shadow_recovery_mask = (

        shadow_mask *

        over_improvement *

        over_not_clipped
    )

    shadow_recovery_mask = np.clip(
        shadow_recovery_mask,
        0.0,
        1.0
    )

    # ========================================================
    # MASK SMOOTHING
    # ========================================================

    MASK_SIGMA = 3.0

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
    # EXPOSURE QUALITY
    # ========================================================

    print(
        "Calculating exposure confidence..."
    )

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
    # Highlight recovery.
    # ========================================================

    w_under = (

        0.03 +

        1.40 *
        highlight_mask *

        q_under
    )

    # ========================================================
    # NORMAL IMAGE WEIGHT
    #
    # Main visual anchor.
    #
    # Still dominant, but reduced slightly in darker scenes
    # so the bright exposure can contribute more.
    # ========================================================

    normal_base = (

        3.60 -

        0.60 *
        scene["darkness_score"]
    )

    w_normal = (

        normal_base +

        1.50 *
        q_normal -

        0.20 *
        highlight_mask
    )

    w_normal = np.maximum(
        w_normal,
        1.20
    )

    # ========================================================
    # OVEREXPOSED IMAGE WEIGHT
    #
    # THIS IS THE ADAPTIVE PART.
    #
    # Bright scene:
    #   lower contribution
    #
    # Dark scene:
    #   stronger contribution
    #
    # But only where the overexposed image actually improves
    # the normal image.
    # ========================================================

    adaptive_over_strength = (
        scene["over_strength"]
    )

    w_over = (

        0.01 +

        adaptive_over_strength *

        (
            1.20 *
            shadow_recovery_mask +

            0.45 *
            q_over *
            shadow_recovery_mask
        )
    )

    # ========================================================
    # WEIGHT SMOOTHING
    # ========================================================

    print(
        "Smoothing fusion weights..."
    )

    w_under = smooth_weight(
        w_under
    )

    w_normal = smooth_weight(
        w_normal
    )

    w_over = smooth_weight(
        w_over
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
    # FUSION
    #
    # No artificial midtone lift.
    # No global brightness boost.
    # Brightness changes come directly from exposure blending.
    # ========================================================

    print(
        "Performing adaptive exposure fusion..."
    )

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
    # VERY LIGHT COLOR ANCHOR
    #
    # Keeps natural colors stable without changing luminance.
    # ========================================================

    print(
        "Protecting natural colors..."
    )

    color_anchor = 0.05

    fused = (

        fused *
        (1.0 - color_anchor)

        +

        normal *
        color_anchor
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ========================================================
    # DEBUG
    # ========================================================

    debug = {

        "highlight_mask":
            highlight_mask,

        "shadow_mask":
            shadow_mask,

        "deep_shadow_mask":
            deep_shadow_mask,

        "shadow_recovery_mask":
            shadow_recovery_mask,

        "weight_under":
            w_under,

        "weight_normal":
            w_normal,

        "weight_over":
            w_over
    }

    weights = (

        float(
            np.mean(w_under)
        ),

        float(
            np.mean(w_normal)
        ),

        float(
            np.mean(w_over)
        )
    )

    return (
        fused,
        debug,
        weights,
        scene
    )


# ============================================================
# SAVE MAP
# ============================================================

def save_map(path, image):

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

    cv2.imwrite(
        path,
        np.uint8(
            image * 255
        )
    )


# ============================================================
# PROCESS
# ============================================================

def process(
    under_path,
    normal_path,
    over_path,
    output_path
):

    print()

    print(
        "=" * 60
    )

    print(
        "ADAPTIVE HDR FUSION V3.3"
    )

    print(
        "ADAPTIVE BRIGHT-EXPOSURE EDITION"
    )

    print(
        "=" * 60
    )

    print()

    print(
        "Loading images..."
    )

    under = read_image(
        under_path
    )

    normal = read_image(
        normal_path
    )

    over = read_image(
        over_path
    )

    print()

    fused, debug, weights, scene = adaptive_fusion(

        under,
        normal,
        over
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # ========================================================
    # SAVE DIAGNOSTICS
    # ========================================================

    print(
        "Saving diagnostics..."
    )

    base_name = os.path.splitext(
        os.path.basename(
            output_path
        )
    )[0]

    for name, data in debug.items():

        debug_path = os.path.join(

            output_dir,

            base_name +
            "_" +
            name +
            ".png"
        )

        save_map(
            debug_path,
            data
        )

    # ========================================================
    # SAVE FINAL
    # ========================================================

    print(
        "Saving final image..."
    )

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
            f"Could not save output:\n"
            f"{output_path}"
        )

    # ========================================================
    # RESULTS
    # ========================================================

    print()

    print(
        "=" * 60
    )

    print(
        "FUSION COMPLETE"
    )

    print(
        "=" * 60
    )

    print()

    print(
        f"Final output:\n"
        f"  {output_path}"
    )

    print()

    print(
        "Adaptive scene analysis:"
    )

    print(
        f"  Mean brightness : "
        f"{scene['mean_lum']:.4f}"
    )

    print(
        f"  Darkness score  : "
        f"{scene['darkness_score']:.4f}"
    )

    print(
        f"  Bright exposure strength : "
        f"{scene['over_strength']:.4f}"
    )

    print()

    print(
        "Mean fusion weights:"
    )

    print(
        f"  Under  : "
        f"{weights[0]:.4f}"
    )

    print(
        f"  Normal : "
        f"{weights[1]:.4f}"
    )

    print(
        f"  Over   : "
        f"{weights[2]:.4f}"
    )

    print()

    print(
        "Pipeline:"
    )

    print(
        "Under  -> highlight recovery"
    )

    print(
        "Normal -> color and depth anchor"
    )

    print(
        "Over   -> adaptive bright-exposure recovery"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(

        description=(
            "Adaptive HDR Fusion "
            "with Scene-Aware Bright Exposure"
        )
    )

    parser.add_argument(
        "--under",
        required=True
    )

    parser.add_argument(
        "--normal",
        required=True
    )

    parser.add_argument(
        "--over",
        required=True
    )

    parser.add_argument(
        "--output",
        required=True
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
