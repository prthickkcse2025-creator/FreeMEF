#!/usr/bin/env python3

"""
FreeMEF INPUT VALIDATION + EXPOSURE ALIGNMENT

Purpose
-------
Production validation before candidate generation.

Checks:
    - exactly three exposure files
    - readable images
    - minimum dimensions
    - compatible dimensions
    - BGR image loading
    - exposure alignment relative to Normal
    - alignment quality measurement

Design
------
Normal exposure is the geometric reference.

Under and Over are aligned to Normal using ECC.
If alignment is poor or fails, the original image is retained.

This module does NOT modify Candidate A/B/C implementations.
"""

import os
from typing import Dict, Tuple

import cv2
import numpy as np


# ============================================================
# CONFIG
# ============================================================

MIN_WIDTH = 256
MIN_HEIGHT = 256

ECC_MAX_ITERATIONS = 60
ECC_EPSILON = 1e-6

# Maximum accepted translation magnitude in pixels at the
# validation working scale. Large movements are suspicious.
MAX_TRANSLATION_PIXELS = 80.0

# Minimum normalized correlation improvement / quality.
MIN_CORRELATION = 0.60

# Validation is performed on a reduced image for speed.
VALIDATION_WIDTH = 1200


# ============================================================
# BASIC IMAGE VALIDATION
# ============================================================

def validate_image_file(path: str) -> Dict:
    """
    Validate a single image path.
    """

    if not path:
        raise ValueError("Image path is empty.")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Image file does not exist:\n{path}"
        )

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError(
            f"Could not decode image:\n{path}"
        )

    height, width = image.shape[:2]

    if width < MIN_WIDTH or height < MIN_HEIGHT:
        raise ValueError(
            f"Image is too small: "
            f"{width}x{height}. "
            f"Minimum is "
            f"{MIN_WIDTH}x{MIN_HEIGHT}."
        )

    return {
        "path": path,
        "width": int(width),
        "height": int(height),
        "channels": int(image.shape[2]),
        "dtype": str(image.dtype),
        "image": image,
    }


def validate_exposure_set(
    under_path: str,
    normal_path: str,
    over_path: str,
) -> Dict:
    """
    Validate the complete three-exposure set.
    """

    under_info = validate_image_file(
        under_path
    )

    normal_info = validate_image_file(
        normal_path
    )

    over_info = validate_image_file(
        over_path
    )

    normal_shape = (
        normal_info["height"],
        normal_info["width"],
    )

    under_shape = (
        under_info["height"],
        under_info["width"],
    )

    over_shape = (
        over_info["height"],
        over_info["width"],
    )

    same_dimensions = (
        under_shape == normal_shape
        and
        over_shape == normal_shape
    )

    return {
        "under": under_info,
        "normal": normal_info,
        "over": over_info,
        "same_dimensions": same_dimensions,
    }


# ============================================================
# RESIZE FOR VALIDATION
# ============================================================

def resize_for_validation(
    image: np.ndarray,
    target_width: int = VALIDATION_WIDTH,
) -> np.ndarray:

    height, width = image.shape[:2]

    if width <= target_width:
        return image.copy()

    scale = (
        target_width
        /
        float(width)
    )

    target_height = int(
        round(
            height * scale
        )
    )

    return cv2.resize(
        image,
        (
            target_width,
            target_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# GRAYSCALE NORMALIZATION
# ============================================================

def grayscale_float(
    image: np.ndarray
) -> np.ndarray:

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = gray.astype(
        np.float32
    ) / 255.0

    return gray


# ============================================================
# ECC ALIGNMENT
# ============================================================

def align_to_normal(
    source: np.ndarray,
    normal: np.ndarray,
) -> Tuple[np.ndarray, Dict]:

    source_work = resize_for_validation(
        source
    )

    normal_work = resize_for_validation(
        normal
    )

    if source_work.shape != normal_work.shape:
        source_work = cv2.resize(
            source_work,
            (
                normal_work.shape[1],
                normal_work.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

    source_gray = grayscale_float(
        source_work
    )

    normal_gray = grayscale_float(
        normal_work
    )

    warp_matrix = np.eye(
        2,
        3,
        dtype=np.float32
    )

    criteria = (
        cv2.TERM_CRITERIA_EPS
        |
        cv2.TERM_CRITERIA_COUNT,
        ECC_MAX_ITERATIONS,
        ECC_EPSILON
    )

    try:

        correlation, warp_matrix = (
            cv2.findTransformECC(
                normal_gray,
                source_gray,
                warp_matrix,
                cv2.MOTION_EUCLIDEAN,
                criteria,
                None,
                5
            )
        )

        aligned = cv2.warpAffine(
            source_work,
            warp_matrix,
            (
                normal_work.shape[1],
                normal_work.shape[0]
            ),
            flags=(
                cv2.INTER_LINEAR
                |
                cv2.WARP_INVERSE_MAP
            ),
            borderMode=cv2.BORDER_REFLECT
        )

        translation_x = float(
            warp_matrix[0, 2]
        )

        translation_y = float(
            warp_matrix[1, 2]
        )

        translation_magnitude = float(
            np.sqrt(
                translation_x ** 2
                +
                translation_y ** 2
            )
        )

        result = {
            "success": True,
            "correlation": float(
                correlation
            ),
            "translation_x": translation_x,
            "translation_y": translation_y,
            "translation_magnitude": (
                translation_magnitude
            ),
            "warp_matrix": (
                warp_matrix.tolist()
            ),
        }

        return aligned, result

    except cv2.error as exc:

        return source_work.copy(), {
            "success": False,
            "correlation": 0.0,
            "translation_x": 0.0,
            "translation_y": 0.0,
            "translation_magnitude": 0.0,
            "warp_matrix": None,
            "error": str(exc),
        }


# ============================================================
# ALIGNMENT QUALITY
# ============================================================

def normalized_correlation(
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    a = a.astype(
        np.float32
    ).reshape(-1)

    b = b.astype(
        np.float32
    ).reshape(-1)

    a = a - np.mean(a)
    b = b - np.mean(b)

    denominator = float(
        np.sqrt(
            np.sum(a * a)
            *
            np.sum(b * b)
        )
    )

    if denominator < 1e-12:
        return 0.0

    return float(
        np.sum(a * b)
        /
        denominator
    )


def alignment_quality(
    normal: np.ndarray,
    original_source: np.ndarray,
    aligned_source: np.ndarray,
) -> Dict:

    normal_work = resize_for_validation(
        normal
    )

    source_work = resize_for_validation(
        original_source
    )

    aligned_work = aligned_source

    if source_work.shape != normal_work.shape:

        source_work = cv2.resize(
            source_work,
            (
                normal_work.shape[1],
                normal_work.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

    if aligned_work.shape != normal_work.shape:

        aligned_work = cv2.resize(
            aligned_work,
            (
                normal_work.shape[1],
                normal_work.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

    normal_gray = grayscale_float(
        normal_work
    )

    original_gray = grayscale_float(
        source_work
    )

    aligned_gray = grayscale_float(
        aligned_work
    )

    before_corr = normalized_correlation(
        normal_gray,
        original_gray
    )

    after_corr = normalized_correlation(
        normal_gray,
        aligned_gray
    )

    return {
        "before_correlation": before_corr,
        "after_correlation": after_corr,
        "improvement": (
            after_corr
            -
            before_corr
        ),
    }


# ============================================================
# FULL ALIGNMENT VALIDATION
# ============================================================

def validate_and_align_exposures(
    under_path: str,
    normal_path: str,
    over_path: str,
) -> Dict:

    validation = validate_exposure_set(
        under_path,
        normal_path,
        over_path
    )

    under = validation["under"]["image"]
    normal = validation["normal"]["image"]
    over = validation["over"]["image"]

    under_aligned, under_alignment = (
        align_to_normal(
            under,
            normal
        )
    )

    over_aligned, over_alignment = (
        align_to_normal(
            over,
            normal
        )
    )

    under_quality = alignment_quality(
        normal,
        under,
        under_aligned
    )

    over_quality = alignment_quality(
        normal,
        over,
        over_aligned
    )

    under_ok = (
        under_alignment["success"]
        and
        under_alignment["correlation"]
        >= MIN_CORRELATION
        and
        under_alignment["translation_magnitude"]
        <= MAX_TRANSLATION_PIXELS
        and
        under_quality["after_correlation"]
        +
        0.002
        >=
        under_quality["before_correlation"]
    )

    over_ok = (
        over_alignment["success"]
        and
        over_alignment["correlation"]
        >= MIN_CORRELATION
        and
        over_alignment["translation_magnitude"]
        <= MAX_TRANSLATION_PIXELS
        and
        over_quality["after_correlation"]
        +
        0.002
        >=
        over_quality["before_correlation"]
    )

    return {
        "validation": validation,
        "under_alignment": under_alignment,
        "over_alignment": over_alignment,
        "under_quality": under_quality,
        "over_quality": over_quality,
        "under_ok": bool(under_ok),
        "over_ok": bool(over_ok),
        "overall_ok": bool(
            under_ok
            and
            over_ok
        ),
        "same_dimensions": (
            validation["same_dimensions"]
        ),
    }


# ============================================================
# COMMAND-LINE TEST
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Validate and measure alignment "
            "of three FreeMEF exposures."
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

    args = parser.parse_args()

    result = validate_and_align_exposures(
        args.under,
        args.normal,
        args.over
    )

    print()
    print("=" * 60)
    print("FREE MEF INPUT VALIDATION")
    print("=" * 60)
    print()

    print(
        f"Same dimensions : "
        f"{result['same_dimensions']}"
    )

    print()

    print("UNDER ALIGNMENT")
    print(
        f"  ECC success       : "
        f"{result['under_alignment']['success']}"
    )
    print(
        f"  ECC correlation   : "
        f"{result['under_alignment']['correlation']:.4f}"
    )
    print(
        f"  Translation       : "
        f"{result['under_alignment']['translation_magnitude']:.2f}px"
    )
    print(
        f"  Before correlation: "
        f"{result['under_quality']['before_correlation']:.4f}"
    )
    print(
        f"  After correlation : "
        f"{result['under_quality']['after_correlation']:.4f}"
    )
    print(
        f"  Accepted           : "
        f"{result['under_ok']}"
    )

    print()

    print("OVER ALIGNMENT")
    print(
        f"  ECC success       : "
        f"{result['over_alignment']['success']}"
    )
    print(
        f"  ECC correlation   : "
        f"{result['over_alignment']['correlation']:.4f}"
    )
    print(
        f"  Translation       : "
        f"{result['over_alignment']['translation_magnitude']:.2f}px"
    )
    print(
        f"  Before correlation: "
        f"{result['over_quality']['before_correlation']:.4f}"
    )
    print(
        f"  After correlation : "
        f"{result['over_quality']['after_correlation']:.4f}"
    )
    print(
        f"  Accepted           : "
        f"{result['over_ok']}"
    )

    print()

    print(
        f"Overall alignment OK: "
        f"{result['overall_ok']}"
    )
