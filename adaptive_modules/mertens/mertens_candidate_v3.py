#!/usr/bin/env python3

"""
MERTENS CANDIDATE V3
====================

Production-oriented Mertens exposure-fusion candidate.

Goals:
    - Preserve more fine detail than V2.
    - Keep Normal exposure as the primary visual anchor.
    - Use Under mainly for highlight-side information.
    - Use Over mainly for shadow-side information.
    - Avoid the soft appearance caused by very low working resolution.
    - Avoid aggressive sharpening and artificial HDR appearance.

IMPORTANT:
    - Does NOT modify Candidate A / V3.3.
    - Does NOT modify the original cloned Mertens repository.
    - Uses the same three exposures:
        Under / Normal / Over
"""

import os
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

    folder = os.path.dirname(path)

    if folder:
        os.makedirs(
            folder,
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

    blue = image[:, :, 0]
    green = image[:, :, 1]
    red = image[:, :, 2]

    blue_exp = np.exp(
        -(
            (blue - 0.5) ** 2
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

    red_exp = np.exp(
        -(
            (red - 0.5) ** 2
        )
        /
        (
            2.0 * sigma ** 2
        )
    )

    return np.maximum(
        blue_exp
        *
        green_exp
        *
        red_exp,
        EPS
    )


# ============================================================
# NORMAL-ANCHOR PRIOR
# ============================================================

def normal_anchor_prior(
    image,
    normal,
    strength=1.0
):
    """
    Encourage the Normal exposure to remain the main
    visual anchor without forcing its exact brightness.
    """

    image_lum = luminance(
        image
    )

    normal_lum = luminance(
        normal
    )

    difference = np.abs(
        image_lum
        -
        normal_lum
    )

    # Small penalty when an exposure strongly disagrees
    # with the Normal reference.
    disagreement = np.clip(
        difference / 0.22,
        0.0,
        1.0
    )

    prior = (
        1.0
        -
        0.35
        *
        strength
        *
        disagreement
    )

    return np.clip(
        prior,
        0.55,
        1.0
    ).astype(
        np.float32
    )


# ============================================================
# BUILD BALANCED WEIGHTS
# ============================================================

def build_weights(
    under,
    normal,
    over,
    wc=1.0,
    ws=0.80,
    we=1.0,
    normal_boost=1.45
):

    images = [
        under,
        normal,
        over
    ]

    weights = []

    for index, image in enumerate(images):

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

        # ----------------------------------------------------
        # Candidate-specific exposure preference
        # ----------------------------------------------------

        lum = luminance(
            image
        )

        if index == 0:
            # Under exposure is primarily useful in bright areas.
            bright_preference = np.clip(
                (
                    lum
                    -
                    0.45
                )
                /
                0.40,
                0.20,
                1.0
            )

            weight *= (
                0.80
                +
                0.40
                *
                bright_preference
            )

        elif index == 1:
            # Normal is the primary visual anchor.
            normal_prior = normal_anchor_prior(
                image,
                normal,
                strength=1.0
            )

            weight *= (
                normal_boost
                *
                normal_prior
            )

        else:
            # Over exposure is primarily useful in dark areas.
            dark_preference = np.clip(
                (
                    0.55
                    -
                    lum
                )
                /
                0.45,
                0.20,
                1.0
            )

            weight *= (
                0.80
                +
                0.40
                *
                dark_preference
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

    # --------------------------------------------------------
    # Final Normal protection
    # --------------------------------------------------------

    normal_weight = weights[1]

    normal_weight = np.maximum(
        normal_weight,
        0.10
    )

    total_after_floor = (
        weights[0]
        +
        normal_weight
        +
        weights[2]
        +
        EPS
    )

    weights = [
        weights[0] / total_after_floor,
        normal_weight / total_after_floor,
        weights[2] / total_after_floor
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
    levels=7
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
# NORMAL-ANCHOR BRIGHTNESS PROTECTION
# ============================================================

def protect_normal_brightness(
    fused,
    normal,
    strength=0.20
):

    fused_lum = luminance(
        fused
    )

    normal_lum = luminance(
        normal
    )

    ratio = (
        normal_lum
        +
        EPS
    ) / (
        fused_lum
        +
        EPS
    )

    ratio = np.clip(
        ratio,
        0.94,
        1.06
    )

    correction = (
        1.0
        +
        strength
        *
        (
            ratio
            -
            1.0
        )
    )

    return np.clip(
        fused
        *
        correction[:, :, None],
        0.0,
        1.0
    )


# ============================================================
# NORMAL DETAIL PRESERVATION
# ============================================================

def preserve_normal_detail(
    fused,
    normal,
    strength=0.06
):
    """
    Restore a very small amount of fine detail from the
    original Normal exposure.

    This is intentionally weak. Its purpose is to prevent
    pyramid fusion from appearing unnecessarily soft, not
    to sharpen the final image aggressively.
    """

    normal_blur = cv2.GaussianBlur(
        normal,
        (0, 0),
        1.2
    )

    detail = (
        normal
        -
        normal_blur
    )

    # Reduce contribution in extremely bright pixels.
    lum = luminance(
        fused
    )

    protection = (
        1.0
        -
        0.50
        *
        np.clip(
            (
                lum
                -
                0.82
            )
            /
            0.16,
            0.0,
            1.0
        )
    )

    result = (
        fused
        +
        detail
        *
        (
            strength
            *
            protection[:, :, None]
        )
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# FINAL TONE PROTECTION
# ============================================================

def protect_extreme_tones(
    result,
    normal
):

    result_lum = luminance(
        result
    )

    normal_lum = luminance(
        normal
    )

    target = result_lum.copy()

    # --------------------------------------------------------
    # Protect very dark regions from unnatural lifting.
    # --------------------------------------------------------

    dark_mask = np.clip(
        (
            0.22
            -
            normal_lum
        )
        /
        0.22,
        0.0,
        1.0
    )

    target = (
        target
        *
        (
            1.0
            -
            0.10
            *
            dark_mask
        )
        +
        normal_lum
        *
        (
            0.10
            *
            dark_mask
        )
    )

    # --------------------------------------------------------
    # Protect very bright regions from aggressive fusion.
    # --------------------------------------------------------

    bright_mask = np.clip(
        (
            normal_lum
            -
            0.84
        )
        /
        0.16,
        0.0,
        1.0
    )

    target = (
        target
        *
        (
            1.0
            -
            0.08
            *
            bright_mask
        )
        +
        normal_lum
        *
        (
            0.08
            *
            bright_mask
        )
    )

    current = luminance(
        result
    )

    ratio = (
        target
        +
        EPS
    ) / (
        current
        +
        EPS
    )

    ratio = np.clip(
        ratio,
        0.97,
        1.03
    )

    return np.clip(
        result
        *
        ratio[:, :, None],
        0.0,
        1.0
    )


# ============================================================
# MAIN CANDIDATE
# ============================================================

def run_mertens_v3(
    under_path,
    normal_path,
    over_path,
    output_path,
    working_width=2400,
    levels=7
):

    print()
    print("=" * 60)
    print("MERTENS CANDIDATE V3")
    print("HIGHER-RESOLUTION NORMAL-ANCHORED FUSION")
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
    # Ensure all exposures match Normal dimensions before
    # working-resolution processing.
    # --------------------------------------------------------

    if under.shape[:2] != normal.shape[:2]:
        under = cv2.resize(
            under,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_LINEAR
        )

    if over.shape[:2] != normal.shape[:2]:
        over = cv2.resize(
            over,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_LINEAR
        )

    # --------------------------------------------------------
    # Working resolution
    # --------------------------------------------------------

    if original_w > working_width:

        scale = (
            working_width
            /
            float(original_w)
        )

        work_w = int(
            round(
                original_w
                *
                scale
            )
        )

        work_h = int(
            round(
                original_h
                *
                scale
            )
        )

        interpolation = cv2.INTER_AREA

        under = cv2.resize(
            under,
            (work_w, work_h),
            interpolation=interpolation
        )

        normal = cv2.resize(
            normal,
            (work_w, work_h),
            interpolation=interpolation
        )

        over = cv2.resize(
            over,
            (work_w, work_h),
            interpolation=interpolation
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
    # Mertens weights
    # --------------------------------------------------------

    print(
        "Calculating balanced Mertens weights..."
    )

    weights = build_weights(
        under,
        normal,
        over,
        wc=1.0,
        ws=0.80,
        we=1.0,
        normal_boost=1.45
    )

    print(
        "Balanced Mertens weights calculated."
    )

    print()

    print(
        "Running higher-resolution "
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
    # Normal brightness anchor
    # --------------------------------------------------------

    print(
        "Protecting Normal-exposure brightness..."
    )

    fused = protect_normal_brightness(
        fused,
        normal,
        strength=0.20
    )

    # --------------------------------------------------------
    # Fine-detail preservation
    # --------------------------------------------------------

    print(
        "Preserving fine Normal-exposure detail..."
    )

    fused = preserve_normal_detail(
        fused,
        normal,
        strength=0.06
    )

    # --------------------------------------------------------
    # Extreme-tone protection
    # --------------------------------------------------------

    print(
        "Protecting extreme tones..."
    )

    fused = protect_extreme_tones(
        fused,
        normal
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
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
            interpolation=cv2.INTER_LANCZOS4
        )

    # --------------------------------------------------------
    # Final tiny normalization
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_image(
        output_path,
        fused
    )

    print()
    print("=" * 60)
    print("MERTENS V3 COMPLETE")
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
# CLI
# ============================================================

def build_parser():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Mertens Candidate V3 "
            "for FreeMEF"
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

    parser.add_argument(
        "--working-width",
        type=int,
        default=2400
    )

    parser.add_argument(
        "--levels",
        type=int,
        default=7
    )

    return parser


def main():

    parser = build_parser()

    args = parser.parse_args()

    run_mertens_v3(
        under_path=args.under,
        normal_path=args.normal,
        over_path=args.over,
        output_path=args.output,
        working_width=args.working_width,
        levels=args.levels
    )


if __name__ == "__main__":
    main()
