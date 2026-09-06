#!/usr/bin/env python3

"""
MERTENS CANDIDATE V2
====================

Clean experimental implementation for the adaptive FreeMEF system.

Uses the cloned Mertens repository's core concepts:

    contrast
    saturation
    well-exposedness

Then performs stable modern multi-resolution Laplacian blending.

IMPORTANT:
    - Does NOT modify the original cloned repository.
    - Does NOT modify V3.3.
    - Uses the same three exposures:
        Under / Normal / Over
"""

import os
import sys
import cv2
import numpy as np


EPS = 1e-8

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

CLONED_REPO = os.path.join(
    PROJECT_ROOT,
    "adaptive_modules",
    "mertens",
    "ExposureFusion"
)


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

    return image.astype(
        np.float32
    ) / 255.0


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
    ).astype(
        np.uint8
    )

    os.makedirs(
        os.path.dirname(path),
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
    ).astype(
        np.float32
    )


# ============================================================
# MERTENS CONTRAST
# ============================================================

def contrast_weight(image):

    gray = (
        luminance(image) * 255.0
    ).astype(
        np.uint8
    )

    lap = cv2.Laplacian(
        gray,
        cv2.CV_32F,
        ksize=3
    )

    weight = np.abs(
        lap
    )

    weight = cv2.GaussianBlur(
        weight,
        (0, 0),
        1.0
    )

    maximum = np.max(
        weight
    )

    if maximum > EPS:

        weight /= maximum

    return np.maximum(
        weight,
        EPS
    )


# ============================================================
# MERTENS SATURATION
# ============================================================

def saturation_weight(image):

    mean = np.mean(
        image,
        axis=2
    )

    saturation = np.sqrt(
        np.mean(
            (
                image
                -
                mean[:, :, None]
            ) ** 2,
            axis=2
        )
    )

    maximum = np.max(
        saturation
    )

    if maximum > EPS:

        saturation /= maximum

    return np.maximum(
        saturation,
        EPS
    )


# ============================================================
# MERTENS WELL-EXPOSEDNESS
# ============================================================

def exposedness_weight(image):

    sigma = 0.20

    red = image[:, :, 2]
    green = image[:, :, 1]
    blue = image[:, :, 0]

    red_exp = np.exp(
        -(
            (red - 0.5) ** 2
        )
        /
        (
            2.0 * sigma ** 2
        )
    )

    green_exp = np.exp(
        -(
            (green - 0.5) ** 2
        )
        /
        (
            2.0 * sigma ** 2
        )
    )

    blue_exp = np.exp(
        -(
            (blue - 0.5) ** 2
        )
        /
        (
            2.0 * sigma ** 2
        )
    )

    return np.maximum(
        red_exp *
        green_exp *
        blue_exp,
        EPS
    )


# ============================================================
# BUILD MERTENS WEIGHTS
# ============================================================

def build_weights(
    under,
    normal,
    over,
    wc=1.0,
    ws=1.0,
    we=1.0
):

    images = [
        under,
        normal,
        over
    ]

    weights = []

    for image in images:

        contrast = contrast_weight(
            image
        )

        saturation = saturation_weight(
            image
        )

        exposedness = exposedness_weight(
            image
        )

        weight = (
            contrast ** wc
            *
            saturation ** ws
            *
            exposedness ** we
        )

        weights.append(
            weight.astype(
                np.float32
            )
        )

    total = (
        weights[0]
        +
        weights[1]
        +
        weights[2]
        +
        EPS
    )

    weights = [
        weights[0] / total,
        weights[1] / total,
        weights[2] / total
    ]

    return weights


# ============================================================
# PYRAMID HELPERS
# ============================================================

def gaussian_pyramid(
    image,
    levels
):

    pyramid = [
        image.astype(
            np.float32
        )
    ]

    current = pyramid[0]

    for _ in range(
        1,
        levels
    ):

        current = cv2.pyrDown(
            current
        )

        pyramid.append(
            current
        )

    return pyramid


def laplacian_pyramid(
    image,
    levels
):

    gaussian = gaussian_pyramid(
        image,
        levels
    )

    pyramid = []

    for i in range(
        levels - 1
    ):

        expanded = cv2.pyrUp(
            gaussian[i + 1],
            dstsize=(
                gaussian[i].shape[1],
                gaussian[i].shape[0]
            )
        )

        lap = (
            gaussian[i]
            -
            expanded
        )

        pyramid.append(
            lap
        )

    pyramid.append(
        gaussian[-1]
    )

    return pyramid


# ============================================================
# MULTI-RESOLUTION FUSION
# ============================================================

def pyramid_fusion(
    images,
    weights,
    levels=6
):

    laplacians = [
        laplacian_pyramid(
            image,
            levels
        )
        for image in images
    ]

    weight_pyramids = [
        gaussian_pyramid(
            weight,
            levels
        )
        for weight in weights
    ]

    fused_levels = []

    for level in range(
        levels
    ):

        fused = np.zeros_like(
            laplacians[0][level],
            dtype=np.float32
        )

        for image_index in range(3):

            weight = (
                weight_pyramids[
                    image_index
                ][level]
            )

            fused += (
                laplacians[
                    image_index
                ][level]
                *
                weight[:, :, None]
            )

        fused_levels.append(
            fused
        )

    result = fused_levels[-1]

    for level in range(
        levels - 2,
        -1,
        -1
    ):

        result = cv2.pyrUp(
            result,
            dstsize=(
                fused_levels[
                    level
                ].shape[1],

                fused_levels[
                    level
                ].shape[0]
            )
        )

        result += fused_levels[
            level
        ]

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# NORMAL-ANCHOR SAFETY
# ============================================================

def protect_normal_brightness(
    fused,
    normal,
    strength=0.15
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
        0.92,
        1.08
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
# MAIN MERTENS CANDIDATE
# ============================================================

def run_mertens_v2(
    under_path,
    normal_path,
    over_path,
    output_path,
    working_width=1600,
    levels=6
):

    print()
    print("=" * 60)
    print("MERTENS CANDIDATE V2")
    print("STABLE MULTI-RESOLUTION FUSION")
    print("=" * 60)

    print()

    print(
        "Cloned repository reference:"
    )

    print(
        CLONED_REPO
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

    original_h, original_w = (
        normal.shape[:2]
    )

    print()

    print(
        f"Original resolution: "
        f"{original_w} x {original_h}"
    )

    # --------------------------------------------------------
    # Working resolution
    # --------------------------------------------------------

    if original_w > working_width:

        scale = (
            working_width /
            float(original_w)
        )

        work_w = int(
            round(
                original_w *
                scale
            )
        )

        work_h = int(
            round(
                original_h *
                scale
            )
        )

        under = cv2.resize(
            under,
            (work_w, work_h),
            interpolation=cv2.INTER_AREA
        )

        normal = cv2.resize(
            normal,
            (work_w, work_h),
            interpolation=cv2.INTER_AREA
        )

        over = cv2.resize(
            over,
            (work_w, work_h),
            interpolation=cv2.INTER_AREA
        )

    else:

        work_h, work_w = (
            normal.shape[:2]
        )

    print(
        f"Working resolution: "
        f"{work_w} x {work_h}"
    )

    print()

    # --------------------------------------------------------
    # Mertens weighting
    # --------------------------------------------------------

    print(
        "Calculating Mertens weights..."
    )

    weights = build_weights(
        under,
        normal,
        over
    )

    print(
        "Mertens weights calculated."
    )

    print()

    print(
        "Running stable Laplacian "
        "multi-resolution fusion..."
    )

    fused = pyramid_fusion(
        [
            under,
            normal,
            over
        ],
        weights,
        levels=levels
    )

    # --------------------------------------------------------
    # Normal protection
    # --------------------------------------------------------

    print(
        "Protecting natural normal-exposure brightness..."
    )

    fused = protect_normal_brightness(
        fused,
        normal,
        strength=0.15
    )

    # --------------------------------------------------------
    # Restore original resolution
    # --------------------------------------------------------

    if fused.shape[:2] != (
        original_h,
        original_w
    ):

        fused = cv2.resize(
            fused,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_CUBIC
        )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_image(
        output_path,
        fused
    )

    print()
    print("=" * 60)
    print("MERTENS V2 COMPLETE")
    print("=" * 60)

    print()

    print(
        f"Output:\n"
        f"{output_path}"
    )

    print()

    print(
        "Mean weights:"
    )

    print(
        f"Under  : "
        f"{weights[0].mean():.4f}"
    )

    print(
        f"Normal : "
        f"{weights[1].mean():.4f}"
    )

    print(
        f"Over   : "
        f"{weights[2].mean():.4f}"
    )

    return output_path


# ============================================================
# TEST SCENE005
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description="Mertens V2 exposure fusion"
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
        help="Output image path"
    )

    parser.add_argument(
        "--working-width",
        type=int,
        default=1600,
        help="Maximum working width"
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=6,
        help="Laplacian pyramid levels"
    )

    args = parser.parse_args()

    run_mertens_v2(
        args.under,
        args.normal,
        args.over,
        args.output,
        working_width=args.working_width,
        levels=args.levels
    )
