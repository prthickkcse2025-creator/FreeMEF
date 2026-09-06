#!/usr/bin/env python3

"""
ADAPTIVE HDR FUSION V3.3
STRICT BLENDING EDITION

Purpose
-------
Fix exposure blending problems seen in the existing candidates:

    - excessive shadow lifting
    - excessive highlight lifting
    - foggy / hazy appearance
    - excessive contribution from Under / Over
    - loss of natural midtone depth

Design
------
Normal exposure is the primary anchor.

Under exposure:
    used only when the Normal image is genuinely near clipping
    and Under contains useful highlight information.

Over exposure:
    used only when the Normal image is genuinely dark
    and Over contains useful shadow information.

There is NO global baseline contribution from Under or Over.

The original V3.3 Client-Bright file is NOT modified.
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

    directory = os.path.dirname(
        path
    )

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

def luminance(
    image
):

    # OpenCV image is BGR.
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
# SIZE MATCHING
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
# EXPOSURE QUALITY
# ============================================================

def exposure_quality(
    image
):

    """
    Estimate how useful each pixel is as an exposure source.

    Stronger in the middle of the exposure range.
    We deliberately avoid using this as the final global
    blend weight. It is only used to determine whether a
    recovery exposure is actually useful.
    """

    lum = luminance(
        image
    )

    quality = np.exp(
        -(
            (lum - 0.50) ** 2
        )
        /
        (
            2.0 * 0.22 ** 2
        )
    )

    return np.clip(
        quality,
        0.0,
        1.0
    )


# ============================================================
# HIGHLIGHT RECOVERY MASK
# ============================================================

def create_highlight_mask(
    normal_lum,
    under_lum
):

    # Normal must actually be bright.
    near_clip = smoothstep(
        normal_lum,
        0.78,
        0.97
    )

    # Under must contain darker information.
    under_advantage = np.clip(
        (
            normal_lum -
            under_lum
        )
        /
        0.20,
        0.0,
        1.0
    )

    # Under itself should not be extremely dark.
    under_quality = exposure_quality(
        np.dstack(
            (
                under_lum,
                under_lum,
                under_lum
            )
        )
    )

    mask = (
        near_clip
        *
        under_advantage
        *
        under_quality
    )

    # Make transitions spatially smooth.
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
        1.0
    )


# ============================================================
# SHADOW RECOVERY MASK
# ============================================================

def create_shadow_mask(
    normal_lum,
    over_lum
):

    # Normal must actually be dark.
    dark = smoothstep(
        0.42 -
        normal_lum,
        0.0,
        0.30
    )

    # Over must actually contain brighter information.
    over_advantage = np.clip(
        (
            over_lum -
            normal_lum
        )
        /
        0.25,
        0.0,
        1.0
    )

    over_quality = exposure_quality(
        np.dstack(
            (
                over_lum,
                over_lum,
                over_lum
            )
        )
    )

    mask = (
        dark
        *
        over_advantage
        *
        over_quality
    )

    # Avoid aggressive recovery in very large regions.
    # Smooth transitions prevent halos.
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
        1.0
    )


# ============================================================
# RECOVERY SANITY LIMIT
# ============================================================

def limit_recovery(
    recovery_mask,
    maximum
):

    return np.minimum(
        recovery_mask,
        maximum
    ).astype(
        np.float32
    )


# ============================================================
# EDGE-AWARE MASK SMOOTHING
# ============================================================

def smooth_mask(
    mask,
    image
):

    guide = (
        luminance(
            image
        ) * 255.0
    ).astype(
        np.uint8
    )

    # Small bilateral smoothing preserves edges while
    # removing noisy mask transitions.
    smooth = cv2.bilateralFilter(
        mask.astype(
            np.float32
        ),
        7,
        0.08,
        5.0
    )

    # Add a light Gaussian pass for stable transitions.
    smooth = cv2.GaussianBlur(
        smooth,
        (0, 0),
        1.5
    )

    return np.clip(
        smooth,
        0.0,
        1.0
    )


# ============================================================
# COLOR ANCHOR
# ============================================================

def color_anchor(
    fused,
    normal,
    strength=0.10
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

    ratio = np.clip(
        ratio,
        0.96,
        1.04
    )

    correction = (
        1.0
        +
        strength *
        (
            ratio -
            1.0
        )
    )

    result = (
        fused *
        correction[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# NORMAL MIDTONE DETAIL
# ============================================================

def add_midtone_depth(
    fused,
    normal,
    strength=0.018
):

    normal_lum = luminance(
        normal
    )

    blurred = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        2.5
    )

    detail = (
        normal_lum -
        blurred
    )

    # Only strengthen genuine midtones.
    midtone_mask = np.exp(
        -(
            (normal_lum - 0.50) ** 2
        )
        /
        (
            2.0 * 0.24 ** 2
        )
    )

    # Avoid deep shadow/highlight enhancement.
    midtone_mask *= (
        1.0 -
        smoothstep(
            normal_lum,
            0.76,
            0.95
        )
    )

    detail_strength = (
        strength *
        midtone_mask
    )

    result = (
        fused
        +
        detail_strength[:, :, None]
        *
        detail[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# MILD ANTI-HAZE
# ============================================================

def anti_haze(
    image,
    strength=0.025
):

    lum = luminance(
        image
    )

    local = cv2.GaussianBlur(
        lum,
        (0, 0),
        5.0
    )

    detail = (
        lum -
        local
    )

    scale = float(
        np.percentile(
            np.abs(detail),
            95
        )
    )

    if scale < EPS:

        return image

    normalized = (
        detail /
        scale
    )

    result = (
        image
        +
        strength *
        normalized[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# HIGHLIGHT SAFETY
# ============================================================

def highlight_safety(
    fused,
    normal,
    under
):

    fused_lum = luminance(
        fused
    )

    normal_lum = luminance(
        normal
    )

    under_lum = luminance(
        under
    )

    danger = smoothstep(
        fused_lum,
        0.90,
        0.99
    )

    # Only permit Under to help if it is genuinely darker.
    recovery = np.clip(
        (
            normal_lum -
            under_lum
        )
        /
        0.20,
        0.0,
        1.0
    )

    amount = (
        0.18 *
        danger *
        recovery
    )

    amount = (
        amount[:, :, None]
    )

    result = (
        fused *
        (
            1.0 -
            amount
        )
        +
        under *
        amount
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# MAIN BLENDING ENGINE
# ============================================================

def strict_blend(
    under,
    normal,
    over
):

    normal_lum = luminance(
        normal
    )

    under_lum = luminance(
        under
    )

    over_lum = luminance(
        over
    )

    # --------------------------------------------------------
    # NORMAL IS THE COMPLETE BASE
    # --------------------------------------------------------

    fused = normal.copy()

    # --------------------------------------------------------
    # HIGHLIGHT RECOVERY
    # --------------------------------------------------------

    highlight_mask = create_highlight_mask(
        normal_lum,
        under_lum
    )

    highlight_mask = limit_recovery(
        highlight_mask,
        0.32
    )

    highlight_mask = smooth_mask(
        highlight_mask,
        normal
    )

    # --------------------------------------------------------
    # SHADOW RECOVERY
    # --------------------------------------------------------

    shadow_mask = create_shadow_mask(
        normal_lum,
        over_lum
    )

    shadow_mask = limit_recovery(
        shadow_mask,
        0.20
    )

    shadow_mask = smooth_mask(
        shadow_mask,
        normal
    )

    # --------------------------------------------------------
    # PREVENT BOTH RECOVERIES AT SAME LOCATION
    # --------------------------------------------------------

    overlap = (
        highlight_mask *
        shadow_mask
    )

    highlight_mask *= (
        1.0 -
        0.80 * overlap
    )

    shadow_mask *= (
        1.0 -
        0.80 * overlap
    )

    # --------------------------------------------------------
    # UNDER -> HIGHLIGHTS ONLY
    # --------------------------------------------------------

    fused = (
        fused *
        (
            1.0 -
            highlight_mask[:, :, None]
        )
        +
        under *
        highlight_mask[:, :, None]
    )

    # --------------------------------------------------------
    # OVER -> SHADOWS ONLY
    # --------------------------------------------------------

    fused = (
        fused *
        (
            1.0 -
            shadow_mask[:, :, None]
        )
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
        highlight_mask,
        shadow_mask
    )


# ============================================================
# DIAGNOSTICS
# ============================================================

def save_mask(
    path,
    mask
):

    cv2.imwrite(
        path,
        np.uint8(
            np.clip(
                mask,
                0.0,
                1.0
            )
            *
            255.0
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
    print("=" * 60)
    print(
        "V3.3 STRICT BLENDING EDITION"
    )
    print("=" * 60)

    print()
    print(
        "Loading exposures..."
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

    # --------------------------------------------------------
    # Match resolution.
    # --------------------------------------------------------

    under = match_size(
        under,
        normal
    )

    over = match_size(
        over,
        normal
    )

    print(
        "Creating strict exposure masks..."
    )

    fused, highlight_mask, shadow_mask = (
        strict_blend(
            under,
            normal,
            over
        )
    )

    print(
        "Adding middle-exposure depth..."
    )

    fused = add_midtone_depth(
        fused,
        normal,
        strength=0.018
    )

    print(
        "Applying very mild anti-haze..."
    )

    fused = anti_haze(
        fused,
        strength=0.025
    )

    print(
        "Protecting natural color..."
    )

    fused = color_anchor(
        fused,
        normal,
        strength=0.10
    )

    print(
        "Protecting highlights..."
    )

    fused = highlight_safety(
        fused,
        normal,
        under
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Output directory.
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
    # Diagnostics.
    # --------------------------------------------------------

    base = os.path.splitext(
        os.path.basename(
            output_path
        )
    )[0]

    highlight_path = os.path.join(
        output_dir,
        base +
        "_highlight_recovery.png"
    )

    shadow_path = os.path.join(
        output_dir,
        base +
        "_shadow_recovery.png"
    )

    save_mask(
        highlight_path,
        highlight_mask
    )

    save_mask(
        shadow_path,
        shadow_mask
    )

    # --------------------------------------------------------
    # Save final.
    # --------------------------------------------------------

    save_image(
        output_path,
        fused
    )

    # --------------------------------------------------------
    # Statistics.
    # --------------------------------------------------------

    final_lum = luminance(
        fused
    )

    print()
    print("=" * 60)
    print(
        "STRICT BLENDING COMPLETE"
    )
    print("=" * 60)

    print()
    print(
        f"Output:\n"
        f"  {output_path}"
    )

    print()
    print(
        "Mean brightness:"
    )

    print(
        f"  Normal : "
        f"{luminance(normal).mean():.4f}"
    )

    print(
        f"  Final  : "
        f"{final_lum.mean():.4f}"
    )

    print()
    print(
        "Mean recovery:"
    )

    print(
        f"  Under/highlight : "
        f"{highlight_mask.mean():.4f}"
    )

    print(
        f"  Over/shadow     : "
        f"{shadow_mask.mean():.4f}"
    )

    print()
    print(
        "Maximum recovery:"
    )

    print(
        f"  Under/highlight : "
        f"{highlight_mask.max():.4f}"
    )

    print(
        f"  Over/shadow     : "
        f"{shadow_mask.max():.4f}"
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


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Strict dynamically gated "
            "three-exposure blending"
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
