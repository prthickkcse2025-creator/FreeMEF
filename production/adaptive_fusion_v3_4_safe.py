#!/usr/bin/env python3

"""
ADAPTIVE EXPOSURE FUSION V3.4 SAFE

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

def adaptive_fusion(under, normal, over):
    """
    SAFE V3.4:
    Start with NORMAL. UNDER and OVER are blended only through
    strictly limited masks. No global normalized exposure weights.
    """
    h, w = normal.shape[:2]
    under = cv2.resize(under, (w, h), interpolation=cv2.INTER_LINEAR) if under.shape[:2] != (h, w) else under
    over = cv2.resize(over, (w, h), interpolation=cv2.INTER_LINEAR) if over.shape[:2] != (h, w) else over

    print(f"Resolution: {w} x {h}")
    print("Aligning underexposed image...")
    under = align_to_normal(under, normal)
    print("Aligning overexposed image...")
    over = align_to_normal(over, normal)

    lum_under = luminance(under)
    lum_normal = luminance(normal)
    lum_over = luminance(over)

    # UNDER: only bright / clipped areas, and only if it is genuinely darker.
    highlight_mask = smoothstep(lum_normal, 0.80, 0.97)
    under_improvement = smoothstep(lum_normal - lum_under, 0.035, 0.22)
    under_safe = 1.0 - smoothstep(lum_under, 0.970, 0.998)
    highlight_alpha = highlight_mask * under_improvement * under_safe
    highlight_alpha = cv2.GaussianBlur(highlight_alpha, (0, 0), 2.0)
    highlight_alpha = np.clip(highlight_alpha, 0.0, 0.35)

    # OVER: only genuinely deep shadows, and only if it contains useful lift.
    deep_shadow_mask = 1.0 - smoothstep(lum_normal, 0.035, 0.20)
    over_improvement = smoothstep(lum_over - lum_normal, 0.035, 0.28)
    over_safe = 1.0 - smoothstep(lum_over, 0.965, 0.998)
    shadow_alpha = deep_shadow_mask * over_improvement * over_safe
    shadow_alpha = cv2.GaussianBlur(shadow_alpha, (0, 0), 2.0)
    shadow_alpha = np.clip(shadow_alpha, 0.0, 0.22)

    # CRITICAL CHANGE:
    # NORMAL is the actual base image, not merely another normalized weight.
    fused = normal.copy()
    fused = fused * (1.0 - highlight_alpha[:, :, None]) + under * highlight_alpha[:, :, None]
    fused = fused * (1.0 - shadow_alpha[:, :, None]) + over * shadow_alpha[:, :, None]
    fused = np.clip(np.nan_to_num(fused, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)

    # Keep brightness close to NORMAL without cancelling intended local recovery.
    fused_lum = luminance(fused)
    ratio = (lum_normal + EPS) / (fused_lum + EPS)
    ratio = np.clip(ratio, 0.94, 1.06)
    recovery = np.maximum(highlight_alpha, shadow_alpha)
    correction_strength = 0.85 - 0.55 * recovery
    correction = 1.0 + correction_strength * (ratio - 1.0)
    fused = np.clip(fused * correction[:, :, None], 0.0, 1.0)

    # Very mild normal-detail restoration only.
    normal_blur = cv2.GaussianBlur(lum_normal, (0, 0), 1.2)
    detail = lum_normal - normal_blur
    gain = np.clip(1.0 + 0.05 * detail, 0.97, 1.03)
    fused = np.clip(fused * gain[:, :, None], 0.0, 1.0)

    debug = {
        "01_highlight_alpha": highlight_alpha,
        "02_deep_shadow_mask": deep_shadow_mask,
        "03_shadow_alpha": shadow_alpha,
        "04_normal_protection": np.clip(1.0 - highlight_alpha - shadow_alpha, 0.0, 1.0),
        "05_under_improvement": under_improvement,
        "06_over_improvement": over_improvement,
    }

    return (
        fused,
        debug,
        (
            float(np.mean(highlight_alpha)),
            float(np.mean(1.0 - highlight_alpha - shadow_alpha)),
            float(np.mean(shadow_alpha))
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
        "ADAPTIVE EXPOSURE FUSION V3.4 SAFE"
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
        "FINAL_SCENE005_V34_SAFE.jpg"
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
            "adaptive_fusion_v3_4_safe"
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
