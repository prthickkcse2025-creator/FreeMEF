#!/usr/bin/env python3

"""
CANDIDATE D V2 — ADAPTIVE LOCAL REFINER

Fallback when A/B/C are rejected.

Design:
    Normal = base
    Under = highlight recovery
    Over = shadow recovery
    Brightness lift = scene-adaptive
    Highlight protection = local
    Color = anchored to Normal
    Contrast = mild local enhancement

No fixed global brightness multiplier.
"""

import os
import argparse
import cv2
import numpy as np


EPS = 1e-8


# ============================================================
# I/O
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

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(
            directory,
            exist_ok=True
        )

    ok = cv2.imwrite(
        path,
        output,
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
    ).astype(np.float32)


# ============================================================
# SMOOTH MASK
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
# SIZE
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
# SCENE-ADAPTIVE BRIGHTNESS TARGET
# ============================================================

def estimate_lift(
    normal_lum,
    over_lum
):
    """
    Estimate how much additional local brightness is useful.

    The brighter exposure tells us whether the scene contains
    recoverable information, while Normal determines where the
    image is genuinely too dark.
    """

    dark_ratio = float(
        np.mean(
            normal_lum < 0.35
        )
    )

    deep_ratio = float(
        np.mean(
            normal_lum < 0.20
        )
    )

    # Recoverable information available in Over.
    shadow_recoverability = float(
        np.mean(
            (
                over_lum > normal_lum
            )
            &
            (
                normal_lum < 0.40
            )
        )
    )

    # More difficult/darker scenes get more lift.
    lift = (
        0.035
        +
        0.050 * dark_ratio
        +
        0.035 * deep_ratio
        +
        0.025 * shadow_recoverability
    )

    return float(
        np.clip(
            lift,
            0.025,
            0.12
        )
    )


# ============================================================
# NORMAL COLOR ANCHOR
# ============================================================

def color_anchor(
    fused,
    normal,
    strength=0.12
):

    fused_lum = luminance(
        fused
    )

    normal_lum = luminance(
        normal
    )

    ratio = (
        normal_lum + EPS
    ) / (
        fused_lum + EPS
    )

    ratio = np.clip(
        ratio,
        0.94,
        1.06
    )

    correction = (
        1.0
        +
        strength *
        (ratio - 1.0)
    )

    return np.clip(
        fused *
        correction[:, :, None],
        0.0,
        1.0
    )


# ============================================================
# LOCAL CONTRAST
# ============================================================

def enhance_local_contrast(
    image,
    amount=0.035
):

    lum = luminance(
        image
    )

    blurred = cv2.GaussianBlur(
        lum,
        (0, 0),
        2.5
    )

    detail = (
        lum -
        blurred
    )

    result = (
        image
        +
        amount *
        detail[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# ADAPTIVE REFINEMENT
# ============================================================

def refine(
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
    # Start from Normal
    # --------------------------------------------------------

    fused = normal.copy()

    # --------------------------------------------------------
    # Estimate scene-specific lift
    # --------------------------------------------------------

    lift_amount = estimate_lift(
        normal_lum,
        over_lum
    )

    # --------------------------------------------------------
    # LOCAL SHADOW MASK
    #
    # Only lift regions that are actually dark.
    # --------------------------------------------------------

    shadow_mask = smoothstep(
        0.42 - normal_lum,
        0.0,
        0.32
    )

    # Avoid lifting already bright objects.
    shadow_mask *= (
        1.0 -
        smoothstep(
            normal_lum,
            0.45,
            0.70
        )
    )

    # --------------------------------------------------------
    # Recover shadow information from Over
    # --------------------------------------------------------

    over_gain = np.clip(
        over_lum -
        normal_lum,
        0.0,
        0.40
    )

    over_gain = (
        over_gain /
        0.40
    )

    shadow_recovery = (
        0.28 *
        shadow_mask *
        over_gain
    )

    fused = (
        fused *
        (
            1.0 -
            shadow_recovery[:, :, None]
        )
        +
        over *
        shadow_recovery[:, :, None]
    )

    # --------------------------------------------------------
    # Adaptive local brightness lift
    # --------------------------------------------------------

    local_lift = (
        lift_amount *
        shadow_mask
    )

    fused = np.clip(
        fused +
        local_lift[:, :, None],
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Highlight recovery / protection
    #
    # Use Under where Normal contains potential clipping.
    # --------------------------------------------------------

    highlight_mask = smoothstep(
        normal_lum,
        0.78,
        0.96
    )

    # Only use Under if Under is darker than Normal.
    under_advantage = np.clip(
        normal_lum -
        under_lum,
        0.0,
        0.30
    ) / 0.30

    highlight_strength = (
        0.30 *
        highlight_mask *
        under_advantage
    )

    fused = (
        fused *
        (
            1.0 -
            highlight_strength[:, :, None]
        )
        +
        under *
        highlight_strength[:, :, None]
    )

    # --------------------------------------------------------
    # NORMAL COLOR ANCHOR
    # --------------------------------------------------------

    fused = color_anchor(
        fused,
        normal,
        strength=0.12
    )

    # --------------------------------------------------------
    # HIGHLIGHT SAFETY
    # --------------------------------------------------------

    fused_lum = luminance(
        fused
    )

    highlight_safety = smoothstep(
        fused_lum,
        0.90,
        0.99
    )

    maximum_safe = (
        normal_lum +
        0.035
    )

    safe_lum = np.minimum(
        fused_lum,
        maximum_safe
    )

    ratio = (
        safe_lum /
        (
            fused_lum +
            EPS
        )
    )

    fused = (
        fused *
        (
            1.0 -
            highlight_safety[:, :, None]
        )
        +
        fused *
        ratio[:, :, None]
        *
        highlight_safety[:, :, None]
    )

    # --------------------------------------------------------
    # LOCAL DETAIL
    # --------------------------------------------------------

    fused = enhance_local_contrast(
        fused,
        amount=0.035
    )

    return np.clip(
        fused,
        0.0,
        1.0
    )


# ============================================================
# RUN
# ============================================================

def run_refiner(
    under_path,
    normal_path,
    over_path,
    output_path
):

    print()
    print("=" * 60)
    print("CANDIDATE D V2 — ADAPTIVE LOCAL REFINER")
    print("=" * 60)

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

    normal_lum = luminance(
        normal
    )

    over_lum = luminance(
        over
    )

    lift = estimate_lift(
        normal_lum,
        over_lum
    )

    print(
        f"Adaptive lift target: "
        f"{lift:.4f}"
    )

    print(
        "Refining local shadows..."
    )

    print(
        "Recovering highlight detail..."
    )

    print(
        "Anchoring color to Normal..."
    )

    print(
        "Protecting bright regions..."
    )

    result = refine(
        under,
        normal,
        over
    )

    save_image(
        output_path,
        result
    )

    print()
    print("=" * 60)
    print("CANDIDATE D V2 COMPLETE")
    print("=" * 60)

    print(
        f"Output:\n"
        f"{output_path}"
    )


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Adaptive local refinement fallback"
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

    run_refiner(
        args.under,
        args.normal,
        args.over,
        args.output
    )


if __name__ == "__main__":
    main()

