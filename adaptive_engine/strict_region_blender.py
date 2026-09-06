#!/usr/bin/env python3

"""
STRICT REGION BLENDER
=====================

A standalone dynamic exposure blender.

Design:
    Normal exposure = primary anchor.

    Under exposure:
        only for genuine highlight clipping/recovery.

    Over exposure:
        only for genuine deep-shadow recovery.

    Midtones:
        remain primarily from Normal.

Goals:
    - natural brightness
    - preserve depth from Normal exposure
    - prevent shadow lifting
    - prevent highlight lifting
    - prevent foggy/hazy blending
    - smooth exposure transitions
    - no global exposure multiplier

This file does NOT modify:
    V3.3
    Mertens
    MEF-Net
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

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
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


# ============================================================
# LUMINANCE
# ============================================================

def luminance(image):

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
# LOCAL DETAIL / TEXTURE
# ============================================================

def local_detail(
    image
):

    lum = luminance(
        image
    )

    blur = cv2.GaussianBlur(
        lum,
        (0, 0),
        2.0
    )

    detail = np.abs(
        lum -
        blur
    )

    # Robust texture measure.
    scale = float(
        np.percentile(
            detail,
            90
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
# EXPOSURE ADVANTAGE
# ============================================================

def normalized_advantage(
    source_lum,
    normal_lum,
    denominator
):

    difference = (
        source_lum -
        normal_lum
    )

    return np.clip(
        difference /
        max(
            denominator,
            EPS
        ),
        0.0,
        1.0
    )


# ============================================================
# HIGHLIGHT RECOVERY MASK
# ============================================================

def build_highlight_mask(
    normal,
    under
):

    n = luminance(
        normal
    )

    u = luminance(
        under
    )

    # --------------------------------------------------------
    # 1. Normal must genuinely be bright.
    # --------------------------------------------------------

    clipping_need = smoothstep(
        n,
        0.88,
        0.995
    )

    # --------------------------------------------------------
    # 2. Under must actually contain darker information.
    # --------------------------------------------------------

    under_advantage = normalized_advantage(
        n,
        u,
        -0.12
    )

    # --------------------------------------------------------
    # Correct advantage because normalized_advantage expects
    # source-normal.
    # --------------------------------------------------------

    under_advantage = np.clip(
        (
            n - u
        ) / 0.12,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # 3. Avoid pulling flat/noisy pixels.
    # --------------------------------------------------------

    texture = local_detail(
        under
    )

    # --------------------------------------------------------
    # Strong intersection required.
    # --------------------------------------------------------

    mask = (
        clipping_need
        *
        under_advantage
        *
        (
            0.55 +
            0.45 * texture
        )
    )

    # --------------------------------------------------------
    # Very strict cap.
    #
    # Under can never dominate.
    # --------------------------------------------------------

    mask = np.minimum(
        mask,
        0.22
    )

    # --------------------------------------------------------
    # Smooth the mask.
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
        0.22
    )


# ============================================================
# SHADOW RECOVERY MASK
# ============================================================

def build_shadow_mask(
    normal,
    over
):

    n = luminance(
        normal
    )

    o = luminance(
        over
    )

    # --------------------------------------------------------
    # 1. Normal must genuinely be dark.
    # --------------------------------------------------------

    darkness_need = smoothstep(
        0.38 - n,
        0.0,
        0.28
    )

    # --------------------------------------------------------
    # 2. Over must genuinely contain brighter information.
    # --------------------------------------------------------

    over_advantage = np.clip(
        (
            o - n
        ) / 0.20,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # 3. Don't recover completely featureless regions.
    # --------------------------------------------------------

    texture = local_detail(
        over
    )

    # --------------------------------------------------------
    # Require actual evidence from Over.
    # --------------------------------------------------------

    mask = (
        darkness_need
        *
        over_advantage
        *
        (
            0.45 +
            0.55 * texture
        )
    )

    # --------------------------------------------------------
    # Strict maximum contribution.
    # --------------------------------------------------------

    mask = np.minimum(
        mask,
        0.18
    )

    # --------------------------------------------------------
    # Smooth boundaries.
    # --------------------------------------------------------

    mask = cv2.GaussianBlur(
        mask.astype(
            np.float32
        ),
        (0, 0),
        3.0
    )

    return np.clip(
        mask,
        0.0,
        0.18
    )


# ============================================================
# REMOVE RECOVERY CONFLICT
# ============================================================

def remove_overlap(
    highlight_mask,
    shadow_mask
):

    overlap = (
        highlight_mask *
        shadow_mask
    )

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
# EDGE-AWARE MASK FILTER
# ============================================================

def edge_smooth_mask(
    mask,
    guide
):

    guide_gray = (
        luminance(
            guide
        ) * 255.0
    ).astype(
        np.uint8
    )

    # Bilateral smoothing removes small mask noise while
    # preserving major image boundaries.
    smoothed = cv2.bilateralFilter(
        mask.astype(
            np.float32
        ),
        7,
        12.0,
        12.0
    )

    # A final small blur prevents hard transition seams.
    smoothed = cv2.GaussianBlur(
        smoothed,
        (0, 0),
        1.2
    )

    return np.clip(
        smoothed,
        0.0,
        1.0
    )


# ============================================================
# BLEND
# ============================================================

def blend_regions(
    under,
    normal,
    over
):

    highlight_mask = (
        build_highlight_mask(
            normal,
            under
        )
    )

    shadow_mask = (
        build_shadow_mask(
            normal,
            over
        )
    )

    # Remove conflicting regions.
    (
        highlight_mask,
        shadow_mask
    ) = remove_overlap(
        highlight_mask,
        shadow_mask
    )

    # Edge-aware smoothing.
    highlight_mask = (
        edge_smooth_mask(
            highlight_mask,
            normal
        )
    )

    shadow_mask = (
        edge_smooth_mask(
            shadow_mask,
            normal
        )
    )

    # Re-apply strict caps after smoothing.
    highlight_mask = np.minimum(
        highlight_mask,
        0.22
    )

    shadow_mask = np.minimum(
        shadow_mask,
        0.18
    )

    # --------------------------------------------------------
    # Normal remains the base.
    #
    # Total recovery never becomes dominant.
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
# OPTIONAL VERY SMALL MIDTONE DEPTH
# ============================================================

def preserve_midtone_depth(
    fused,
    normal
):

    n = luminance(
        normal
    )

    # Middle-exposure confidence.
    midtone = np.exp(
        -(
            (n - 0.50) ** 2
        )
        /
        (
            2.0 *
            0.23 ** 2
        )
    )

    # Extract normal's local detail.
    blur = cv2.GaussianBlur(
        n,
        (0, 0),
        2.0
    )

    detail = (
        n -
        blur
    )

    # VERY small amount.
    amount = (
        0.008 *
        midtone
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
# NATURAL COLOR PROTECTION
# ============================================================

def protect_color(
    fused,
    normal
):

    f_lum = luminance(
        fused
    )

    n_lum = luminance(
        normal
    )

    ratio = (
        n_lum +
        EPS
    ) / (
        f_lum +
        EPS
    )

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
# FINAL SAFETY
# ============================================================

def final_safety(
    fused,
    normal
):

    f = luminance(
        fused
    )

    n = luminance(
        normal
    )

    # --------------------------------------------------------
    # Don't allow the final result to become globally brighter
    # than Normal just because of recovery blending.
    #
    # Only constrain the upper tail, not the complete image.
    # --------------------------------------------------------

    upper_limit = (
        n +
        0.045
    )

    danger = smoothstep(
        f,
        0.92,
        0.995
    )

    safe = np.minimum(
        f,
        upper_limit
    )

    ratio = (
        safe /
        (
            f +
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
# MAIN PROCESS
# ============================================================

def process(
    under_path,
    normal_path,
    over_path,
    output_path
):

    print()
    print("=" * 65)
    print("STRICT DYNAMIC REGION BLENDER")
    print("=" * 65)

    print()
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

    under = match_size(
        under,
        normal
    )

    over = match_size(
        over,
        normal
    )

    original_shape = normal.shape

    print(
        f"Resolution: "
        f"{original_shape[1]} x "
        f"{original_shape[0]}"
    )

    # --------------------------------------------------------
    # Dynamic regional blend
    # --------------------------------------------------------

    print(
        "Detecting genuine highlight recovery regions..."
    )

    print(
        "Detecting genuine shadow recovery regions..."
    )

    (
        fused,
        normal_mask,
        highlight_mask,
        shadow_mask
    ) = blend_regions(
        under,
        normal,
        over
    )

    print(
        "Preserving middle-exposure depth..."
    )

    fused = preserve_midtone_depth(
        fused,
        normal
    )

    print(
        "Protecting natural color..."
    )

    fused = protect_color(
        fused,
        normal
    )

    print(
        "Applying final highlight safety..."
    )

    fused = final_safety(
        fused,
        normal
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    save_image(
        output_path,
        fused
    )

    # --------------------------------------------------------
    # Diagnostic masks
    # --------------------------------------------------------

    base = os.path.splitext(
        os.path.basename(
            output_path
        )
    )[0]

    highlight_path = os.path.join(
        output_dir,
        base +
        "_highlight_mask.png"
    )

    shadow_path = os.path.join(
        output_dir,
        base +
        "_shadow_mask.png"
    )

    normal_path = os.path.join(
        output_dir,
        base +
        "_normal_mask.png"
    )

    save_mask(
        highlight_path,
        highlight_mask
    )

    save_mask(
        shadow_path,
        shadow_mask
    )

    save_mask(
        normal_path,
        normal_mask
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
    print("=" * 65)
    print("STRICT DYNAMIC BLENDING COMPLETE")
    print("=" * 65)

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
        f"Mean Normal weight     : "
        f"{normal_mask.mean():.4f}"
    )

    print(
        f"Mean Under contribution: "
        f"{highlight_mask.mean():.4f}"
    )

    print(
        f"Mean Over contribution : "
        f"{shadow_mask.mean():.4f}"
    )

    print()
    print(
        f"Maximum Under weight   : "
        f"{highlight_mask.max():.4f}"
    )

    print(
        f"Maximum Over weight    : "
        f"{shadow_mask.max():.4f}"
    )

    print()
    print(
        f"Final output:\n"
        f"  {output_path}"
    )

    print()
    print(
        f"Highlight mask:\n"
        f"  {highlight_path}"
    )

    print(
        f"Shadow mask:\n"
        f"  {shadow_path}"
    )

    print(
        f"Normal mask:\n"
        f"  {normal_path}"
    )


# ============================================================
# MASK SAVE
# ============================================================

def save_mask(
    path,
    mask
):

    image = np.uint8(
        np.clip(
            mask,
            0.0,
            1.0
        )
        * 255.0
    )

    cv2.imwrite(
        path,
        image
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Strict region-aware "
            "multi-exposure blending"
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
        help="Output image"
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
