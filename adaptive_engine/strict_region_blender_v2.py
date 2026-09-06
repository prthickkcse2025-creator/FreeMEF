#!/usr/bin/env python3

"""
STRICT REGION BLENDER V2
========================

Purpose
-------
A conservative multi-exposure blending engine designed to
address:

    - excessive shadow lifting
    - excessive highlight lifting
    - foggy / hazy HDR appearance
    - flat-looking midtones
    - large recovery regions
    - weak natural depth

Design
------
NORMAL = primary source.

UNDER is used only when:
    1. Normal is genuinely near clipping
    2. Under is meaningfully darker
    3. Under contains useful local detail

OVER is used only when:
    1. Normal is genuinely dark
    2. Over is meaningfully brighter
    3. Over contains useful local detail

The recovery masks are deliberately conservative.

No global brightness multiplier is used.
No gamma boost is used.
No aggressive sharpening is used.

This file is independent from:
    Candidate A - V3.3
    Candidate B - Mertens
    Candidate C - MEF-Net
    Candidate D - Refiner
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
    """
    Read an image as float32 RGB-like array.

    OpenCV stores BGR. We intentionally keep the BGR ordering
    consistently throughout this file.
    """

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    return (
        image.astype(
            np.float32
        )
        / 255.0
    )


def save_image(
    path,
    image
):

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

    image8 = (
        image * 255.0
    ).astype(
        np.uint8
    )

    output_dir = os.path.dirname(
        path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    ok = cv2.imwrite(
        path,
        image8,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            100
        ]
    )

    if not ok:
        raise RuntimeError(
            f"Could not save:\n{path}"
        )


def save_mask(
    path,
    mask
):

    mask = np.clip(
        mask,
        0.0,
        1.0
    )

    image8 = (
        mask * 255.0
    ).astype(
        np.uint8
    )

    output_dir = os.path.dirname(
        path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    ok = cv2.imwrite(
        path,
        image8
    )

    if not ok:
        raise RuntimeError(
            f"Could not save mask:\n{path}"
        )


# ============================================================
# LUMINANCE
# ============================================================

def luminance(
    image
):
    """
    BGR luminance.
    """

    return (
        0.0722 * image[:, :, 0]
        +
        0.7152 * image[:, :, 1]
        +
        0.2126 * image[:, :, 2]
    ).astype(
        np.float32
    )


# ============================================================
# SIZE MATCH
# ============================================================

def match_size(
    image,
    reference
):

    h, w = reference.shape[:2]

    if image.shape[:2] == (
        h,
        w
    ):
        return image

    return cv2.resize(
        image,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# SMOOTHSTEP
# ============================================================

def smoothstep(
    x,
    edge0,
    edge1
):

    width = max(
        edge1 - edge0,
        EPS
    )

    t = np.clip(
        (x - edge0) / width,
        0.0,
        1.0
    )

    return (
        t * t *
        (3.0 - 2.0 * t)
    )


# ============================================================
# LOCAL DETAIL MEASURE
# ============================================================

def local_detail(
    image,
    sigma=1.5
):
    """
    Estimate local structural detail.

    This is NOT used as a sharpening operation.
    It is only used to decide whether an alternate exposure
    contains useful texture that Normal may be missing.
    """

    lum = luminance(
        image
    )

    blurred = cv2.GaussianBlur(
        lum,
        (0, 0),
        sigma
    )

    detail = np.abs(
        lum -
        blurred
    )

    scale = float(
        np.percentile(
            detail,
            95
        )
    )

    if scale < EPS:
        return np.zeros_like(
            detail
        )

    return np.clip(
        detail / scale,
        0.0,
        1.0
    )


# ============================================================
# DETAIL ADVANTAGE
# ============================================================

def detail_advantage(
    source,
    normal
):

    source_detail = local_detail(
        source
    )

    normal_detail = local_detail(
        normal
    )

    advantage = (
        source_detail -
        normal_detail +
        0.15
    )

    return np.clip(
        advantage,
        0.0,
        1.0
    )


# ============================================================
# HIGHLIGHT RECOVERY
# ============================================================

def highlight_recovery_mask(
    under,
    normal
):

    under_lum = luminance(
        under
    )

    normal_lum = luminance(
        normal
    )

    # --------------------------------------------------------
    # Normal genuinely needs highlight recovery.
    # --------------------------------------------------------

    clipping_need = smoothstep(
        normal_lum,
        0.90,
        0.995
    )

    # --------------------------------------------------------
    # Under must contain meaningfully darker values.
    # --------------------------------------------------------

    exposure_advantage = np.clip(
        (
            normal_lum -
            under_lum -
            0.035
        )
        /
        0.18,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Under must have local information.
    # --------------------------------------------------------

    detail_gain = detail_advantage(
        under,
        normal
    )

    # --------------------------------------------------------
    # Combined confidence.
    # --------------------------------------------------------

    mask = (
        clipping_need
        *
        exposure_advantage
        *
        (
            0.30 +
            0.70 * detail_gain
        )
    )

    # --------------------------------------------------------
    # Avoid recovering broad low-detail regions.
    # --------------------------------------------------------

    under_detail = local_detail(
        under
    )

    mask *= (
        0.35 +
        0.65 * under_detail
    )

    # --------------------------------------------------------
    # Strong global cap.
    # --------------------------------------------------------

    mask = np.minimum(
        mask,
        0.16
    )

    # --------------------------------------------------------
    # Spatial smoothing.
    # --------------------------------------------------------

    mask = cv2.GaussianBlur(
        mask.astype(
            np.float32
        ),
        (0, 0),
        2.0
    )

    return np.clip(
        mask,
        0.0,
        0.16
    )


# ============================================================
# SHADOW RECOVERY
# ============================================================

def shadow_recovery_mask(
    over,
    normal
):

    over_lum = luminance(
        over
    )

    normal_lum = luminance(
        normal
    )

    # --------------------------------------------------------
    # Normal genuinely needs shadow recovery.
    # --------------------------------------------------------

    darkness_need = smoothstep(
        0.34 -
        normal_lum,
        0.0,
        0.26
    )

    # --------------------------------------------------------
    # Over must actually provide brighter information.
    # --------------------------------------------------------

    exposure_advantage = np.clip(
        (
            over_lum -
            normal_lum -
            0.035
        )
        /
        0.20,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Over must contain useful local structure.
    # --------------------------------------------------------

    detail_gain = detail_advantage(
        over,
        normal
    )

    # --------------------------------------------------------
    # Combined confidence.
    # --------------------------------------------------------

    mask = (
        darkness_need
        *
        exposure_advantage
        *
        (
            0.25 +
            0.75 * detail_gain
        )
    )

    # --------------------------------------------------------
    # Flat-region suppression.
    # --------------------------------------------------------

    over_detail = local_detail(
        over
    )

    mask *= (
        0.30 +
        0.70 * over_detail
    )

    # --------------------------------------------------------
    # Strong cap.
    # --------------------------------------------------------

    mask = np.minimum(
        mask,
        0.14
    )

    # --------------------------------------------------------
    # Smooth transition.
    # --------------------------------------------------------

    mask = cv2.GaussianBlur(
        mask.astype(
            np.float32
        ),
        (0, 0),
        2.5
    )

    return np.clip(
        mask,
        0.0,
        0.14
    )


# ============================================================
# REMOVE MASK CONFLICT
# ============================================================

def remove_conflicting_recovery(
    highlight_mask,
    shadow_mask
):

    overlap = (
        highlight_mask *
        shadow_mask
    )

    # If both want to modify the same pixel,
    # strongly suppress both.
    highlight_mask *= (
        1.0 -
        0.90 * overlap
    )

    shadow_mask *= (
        1.0 -
        0.90 * overlap
    )

    return (
        highlight_mask,
        shadow_mask
    )


# ============================================================
# SUPPRESS LARGE CONTINUOUS RECOVERY REGIONS
# ============================================================

def suppress_broad_regions(
    mask,
    threshold=0.70
):
    """
    Prevent an entire wall/floor/ceiling region from becoming
    a recovery region.

    Recovery should mainly happen around pixels where the
    alternate exposure contains useful information.
    """

    binary = (
        mask >
        threshold * np.max(
            mask
        )
        if np.max(mask) > EPS
        else np.zeros_like(
            mask,
            dtype=bool
        )
    )

    num_labels, labels, stats, _ = (
        cv2.connectedComponentsWithStats(
            binary.astype(
                np.uint8
            ),
            connectivity=8
        )
    )

    result = mask.copy()

    total_pixels = (
        mask.shape[0] *
        mask.shape[1]
    )

    for label in range(
        1,
        num_labels
    ):

        area = stats[
            label,
            cv2.CC_STAT_AREA
        ]

        fraction = (
            float(area)
            /
            float(total_pixels)
        )

        # Large connected recovery regions are heavily reduced.
        if fraction > 0.08:

            region = (
                labels == label
            )

            result[
                region
            ] *= 0.25

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# EDGE-AWARE SMOOTHING
# ============================================================

def edge_smooth(
    mask,
    guide
):

    # Bilateral filtering gives softer transitions without
    # completely crossing strong object boundaries.
    filtered = cv2.bilateralFilter(
        mask.astype(
            np.float32
        ),
        7,
        0.08,
        5.0
    )

    filtered = cv2.GaussianBlur(
        filtered,
        (0, 0),
        1.0
    )

    return np.clip(
        filtered,
        0.0,
        1.0
    )


# ============================================================
# MICRO-CONTRAST SAFETY
# ============================================================

def protect_midtone_depth(
    fused,
    normal
):

    """
    Do NOT brighten the image.

    Preserve only a tiny amount of Normal's midtone local
    structure to avoid a flat blended appearance.
    """

    normal_lum = luminance(
        normal
    )

    blur = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        2.0
    )

    detail = (
        normal_lum -
        blur
    )

    middle_mask = np.exp(
        -(
            (
                normal_lum -
                0.50
            ) ** 2
        )
        /
        (
            2.0 *
            0.25 ** 2
        )
    )

    amount = (
        0.006 *
        middle_mask
    )

    result = (
        fused
        +
        amount[:, :, None]
        *
        detail[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# COLOR ANCHOR
# ============================================================

def color_anchor(
    fused,
    normal
):

    fused_lum = luminance(
        fused
    )

    normal_lum = luminance(
        normal
    )

    ratio = (
        normal_lum +
        EPS
    ) / (
        fused_lum +
        EPS
    )

    # Very narrow correction range.
    ratio = np.clip(
        ratio,
        0.985,
        1.015
    )

    result = (
        fused *
        ratio[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# FINAL HIGHLIGHT SAFETY
# ============================================================

def final_highlight_safety(
    fused,
    normal
):

    fused_lum = luminance(
        fused
    )

    normal_lum = luminance(
        normal
    )

    # Only very bright output pixels are constrained.
    danger = smoothstep(
        fused_lum,
        0.94,
        1.0
    )

    target = np.minimum(
        fused_lum,
        normal_lum +
        0.035
    )

    ratio = (
        target /
        (
            fused_lum +
            EPS
        )
    )

    result = (
        fused *
        (
            1.0 -
            danger[:, :, None]
        )
        +
        fused *
        ratio[:, :, None]
        *
        danger[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# CORE BLEND
# ============================================================

def strict_blend(
    under,
    normal,
    over
):

    # --------------------------------------------------------
    # Build recovery masks.
    # --------------------------------------------------------

    highlight_mask = (
        highlight_recovery_mask(
            under,
            normal
        )
    )

    shadow_mask = (
        shadow_recovery_mask(
            over,
            normal
        )
    )

    # --------------------------------------------------------
    # Remove ambiguous overlap.
    # --------------------------------------------------------

    (
        highlight_mask,
        shadow_mask
    ) = remove_conflicting_recovery(
        highlight_mask,
        shadow_mask
    )

    # --------------------------------------------------------
    # Suppress broad regions.
    # --------------------------------------------------------

    highlight_mask = (
        suppress_broad_regions(
            highlight_mask
        )
    )

    shadow_mask = (
        suppress_broad_regions(
            shadow_mask
        )
    )

    # --------------------------------------------------------
    # Edge-aware transitions.
    # --------------------------------------------------------

    highlight_mask = (
        edge_smooth(
            highlight_mask,
            normal
        )
    )

    shadow_mask = (
        edge_smooth(
            shadow_mask,
            normal
        )
    )

    # --------------------------------------------------------
    # Re-apply hard safety caps.
    # --------------------------------------------------------

    highlight_mask = np.minimum(
        highlight_mask,
        0.16
    )

    shadow_mask = np.minimum(
        shadow_mask,
        0.14
    )

    # --------------------------------------------------------
    # Normal gets everything not used for recovery.
    # --------------------------------------------------------

    normal_mask = np.clip(
        1.0
        -
        highlight_mask
        -
        shadow_mask,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Fuse.
    # --------------------------------------------------------

    fused = (
        normal *
        normal_mask[:, :, None]
        +
        under *
        highlight_mask[:, :, None]
        +
        over *
        shadow_mask[:, :, None]
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    return (
        fused,
        normal_mask,
        highlight_mask,
        shadow_mask
    )


# ============================================================
# MAIN PROCESS
# ============================================================

def process(
    under_path,
    normal_path,
    over_path,
    output_path
):

    print()
    print("=" * 70)
    print("STRICT REGION BLENDER V2")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print()
    print("Loading exposures...")

    under = read_image(
        under_path
    )

    normal = read_image(
        normal_path
    )

    over = read_image(
        over_path
    )

    # --------------------------------------------------------
    # Match dimensions.
    # --------------------------------------------------------

    under = match_size(
        under,
        normal
    )

    over = match_size(
        over,
        normal
    )

    h, w = normal.shape[:2]

    print(
        f"Resolution: {w} x {h}"
    )

    # --------------------------------------------------------
    # Core blend
    # --------------------------------------------------------

    print()
    print(
        "Detecting genuine highlight information..."
    )

    print(
        "Detecting genuine shadow information..."
    )

    (
        fused,
        normal_mask,
        highlight_mask,
        shadow_mask
    ) = strict_blend(
        under,
        normal,
        over
    )

    # --------------------------------------------------------
    # Depth preservation
    # --------------------------------------------------------

    print(
        "Preserving natural midtone depth..."
    )

    fused = protect_midtone_depth(
        fused,
        normal
    )

    # --------------------------------------------------------
    # Color
    # --------------------------------------------------------

    print(
        "Anchoring natural color..."
    )

    fused = color_anchor(
        fused,
        normal
    )

    # --------------------------------------------------------
    # Highlight safety
    # --------------------------------------------------------

    print(
        "Applying highlight safety..."
    )

    fused = final_highlight_safety(
        fused,
        normal
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Output dir
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    # --------------------------------------------------------
    # Save final
    # --------------------------------------------------------

    save_image(
        output_path,
        fused
    )

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    base = os.path.splitext(
        os.path.basename(
            output_path
        )
    )[0]

    normal_mask_path = os.path.join(
        output_dir,
        base +
        "_normal_mask.png"
    )

    highlight_mask_path = os.path.join(
        output_dir,
        base +
        "_highlight_mask.png"
    )

    shadow_mask_path = os.path.join(
        output_dir,
        base +
        "_shadow_mask.png"
    )

    save_mask(
        normal_mask_path,
        normal_mask
    )

    save_mask(
        highlight_mask_path,
        highlight_mask
    )

    save_mask(
        shadow_mask_path,
        shadow_mask
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    normal_lum = luminance(
        normal
    )

    final_lum = luminance(
        fused
    )

    print()
    print("=" * 70)
    print("STRICT REGION BLENDER V2 COMPLETE")
    print("=" * 70)

    print()

    print(
        f"Normal mean brightness : "
        f"{normal_lum.mean():.4f}"
    )

    print(
        f"Final mean brightness  : "
        f"{final_lum.mean():.4f}"
    )

    print()

    print(
        f"Normal contribution    : "
        f"{normal_mask.mean():.4f}"
    )

    print(
        f"Under contribution     : "
        f"{highlight_mask.mean():.4f}"
    )

    print(
        f"Over contribution      : "
        f"{shadow_mask.mean():.4f}"
    )

    print()

    print(
        f"Max Under contribution : "
        f"{highlight_mask.max():.4f}"
    )

    print(
        f"Max Over contribution  : "
        f"{shadow_mask.max():.4f}"
    )

    print()

    print(
        f"Final output:\n"
        f"  {output_path}"
    )

    print()

    print(
        f"Normal mask:\n"
        f"  {normal_mask_path}"
    )

    print(
        f"Highlight mask:\n"
        f"  {highlight_mask_path}"
    )

    print(
        f"Shadow mask:\n"
        f"  {shadow_mask_path}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Strict dynamic region-aware "
            "multi-exposure blender"
        )
    )

    parser.add_argument(
        "--under",
        required=True,
        help="Underexposed image"
    )

    parser.add_argument(
        "--normal",
        required=True,
        help="Normal exposure image"
    )

    parser.add_argument(
        "--over",
        required=True,
        help="Overexposed image"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Final output path"
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
