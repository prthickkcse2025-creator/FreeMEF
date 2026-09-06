#!/usr/bin/env python3

"""
FREE MEF - CORRECTION ENGINE V5

Reference-guided image revision engine.

Inputs:
    selected : customer-selected candidate
    normal   : normal exposure / visual anchor
    under    : under exposure / highlight-detail reference
    over     : over exposure / shadow-detail reference

Controls:
    brightness
    shadow
    highlight
    depth
    dehaze
    contrast
    saturation
    color
    blending

Important:
    This engine never selects a candidate.
    It only revises the customer-selected image.
"""

import os
import cv2
import argparse
import numpy as np


EPS = 1e-6


# ============================================================
# IMAGE IO
# ============================================================

def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(
            f"Cannot read image: {path}"
        )

    return image.astype(np.float32) / 255.0


def write_image(path, image, quality=98):
    folder = os.path.dirname(path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    output = np.clip(
        image * 255.0,
        0.0,
        255.0
    ).astype(np.uint8)

    ext = os.path.splitext(path)[1].lower()

    if ext in (".jpg", ".jpeg"):
        ok = cv2.imwrite(
            path,
            output,
            [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
        )
    else:
        ok = cv2.imwrite(
            path,
            output
        )

    if not ok:
        raise RuntimeError(
            f"Cannot write output: {path}"
        )


def match_size(image, reference):
    h, w = reference.shape[:2]

    if image.shape[:2] == (h, w):
        return image

    return cv2.resize(
        image,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# BASIC OPERATIONS
# ============================================================

def luminance(image):
    return (
        0.0722 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.2126 * image[:, :, 2]
    )


def smoothstep(x, edge0, edge1):
    t = np.clip(
        (x - edge0) /
        max(edge1 - edge0, EPS),
        0.0,
        1.0
    )

    return t * t * (3.0 - 2.0 * t)


def midtone_map(lum):
    return np.clip(
        smoothstep(lum, 0.08, 0.30)
        *
        (
            1.0
            -
            smoothstep(lum, 0.60, 0.92)
        ),
        0.0,
        1.0
    )


def shadow_map(lum):
    return (
        1.0
        -
        smoothstep(lum, 0.10, 0.45)
    )


def highlight_map(lum):
    return smoothstep(
        lum,
        0.55,
        0.92
    )


def replace_luminance(image, target_lum):
    current_lum = luminance(image)

    ratio = target_lum / (
        current_lum + EPS
    )

    ratio = np.clip(
        ratio,
        0.40,
        2.50
    )

    return np.clip(
        image * ratio[:, :, None],
        0.0,
        1.0
    )


# ============================================================
# BRIGHTNESS
# ============================================================

def apply_brightness(image, value):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    lum = luminance(image)

    if value >= 0.0:

        target = (
            lum
            +
            value
            * 0.28
            *
            (
                1.0
                -
                smoothstep(
                    lum,
                    0.78,
                    0.98
                )
            )
        )

    else:

        target = (
            lum
            *
            (
                1.0
                +
                0.38 * value
            )
        )

    return replace_luminance(
        image,
        np.clip(
            target,
            0.0,
            1.0
        )
    )


# ============================================================
# SHADOW
# ============================================================

def apply_shadow(
    image,
    normal,
    over,
    value
):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    current_lum = luminance(image)
    normal_lum = luminance(normal)
    over_lum = luminance(over)

    mask = shadow_map(
        normal_lum
    )

    if value < 0.0:

        amount = (
            0.95
            *
            abs(value)
            *
            mask
        )

        target = (
            current_lum * (1.0 - amount)
            +
            normal_lum * amount
        )

    else:

        # Positive shadow correction should
        # recover useful information from the
        # over-exposed reference rather than
        # simply adding global brightness.

        over_advantage = np.clip(
            over_lum - current_lum,
            0.0,
            0.35
        )

        recovery = (
            0.65 * over_advantage
            +
            0.35
            *
            np.clip(
                normal_lum - current_lum,
                0.0,
                0.25
            )
        )

        amount = (
            1.20
            *
            value
            *
            mask
        )

        target = (
            current_lum
            +
            recovery * amount
        )

        target = np.minimum(
            target,
            normal_lum + 0.12
        )

    return replace_luminance(
        image,
        np.clip(
            target,
            0.0,
            1.0
        )
    )


# ============================================================
# HIGHLIGHT
# ============================================================

def apply_highlight(
    image,
    normal,
    under,
    over,
    value
):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    current_lum = luminance(image)
    normal_lum = luminance(normal)
    under_lum = luminance(under)
    over_lum = luminance(over)

    mask = highlight_map(
        current_lum
    )

    if value > 0.0:

        reference = np.minimum(
            normal_lum,
            over_lum
        )

        amount = (
            0.85
            *
            value
            *
            mask
        )

        target = (
            current_lum * (1.0 - amount)
            +
            reference * amount
        )

    else:

        # Under exposure is the important
        # reference for recovering highlight
        # information.

        recoverable = np.clip(
            current_lum - under_lum,
            0.0,
            0.40
        )

        amount = (
            0.75
            *
            abs(value)
            *
            mask
        )

        target = (
            current_lum
            -
            recoverable * amount
        )

    return replace_luminance(
        image,
        np.clip(
            target,
            0.0,
            1.0
        )
    )


# ============================================================
# DEPTH
# ============================================================

def apply_depth(
    image,
    normal,
    value
):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    current_lum = luminance(image)
    normal_lum = luminance(normal)

    cur_small = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        2.5
    )

    cur_large = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        8.0
    )

    norm_small = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        2.5
    )

    norm_large = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        8.0
    )

    current_structure = (
        cur_small - cur_large
    )

    normal_structure = (
        norm_small - norm_large
    )

    missing = np.clip(
        normal_structure
        -
        current_structure,
        -0.07,
        0.07
    )

    gx = cv2.Sobel(
        normal_lum,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        normal_lum,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(
        gx * gx
        +
        gy * gy
    )

    confidence = np.clip(
        gradient / 0.12,
        0.20,
        1.0
    )

    protection = (
        smoothstep(
            normal_lum,
            0.03,
            0.15
        )
        *
        (
            1.0
            -
            smoothstep(
                normal_lum,
                0.84,
                0.98
            )
        )
    )

    correction = (
        1.60
        *
        value
        *
        missing
        *
        (
            0.45
            +
            0.55
            *
            midtone_map(
                normal_lum
            )
        )
        *
        confidence
        *
        protection
    )

    target = np.clip(
        current_lum + correction,
        0.0,
        1.0
    )

    return replace_luminance(
        image,
        target
    )


# ============================================================
# DEHAZE
# ============================================================

def apply_dehaze(
    image,
    normal,
    value
):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    current_lum = luminance(image)
    normal_lum = luminance(normal)

    cur_s = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        1.8
    )

    cur_m = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        7.0
    )

    cur_l = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        24.0
    )

    norm_s = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        1.8
    )

    norm_m = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        7.0
    )

    norm_l = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        24.0
    )

    cur_fine = (
        current_lum - cur_s
    )

    norm_fine = (
        normal_lum - norm_s
    )

    cur_mid = (
        cur_s - cur_m
    )

    norm_mid = (
        norm_s - norm_m
    )

    cur_broad = (
        cur_m - cur_l
    )

    norm_broad = (
        norm_m - norm_l
    )

    diff_fine = np.clip(
        norm_fine - cur_fine,
        -0.045,
        0.045
    )

    diff_mid = np.clip(
        norm_mid - cur_mid,
        -0.075,
        0.075
    )

    diff_broad = np.clip(
        norm_broad - cur_broad,
        -0.090,
        0.090
    )

    reference_difference = (
        0.20 * diff_fine
        +
        0.55 * diff_mid
        +
        0.25 * diff_broad
    )

    current_strength = (
        0.20 * np.abs(cur_fine)
        +
        0.55 * np.abs(cur_mid)
        +
        0.25 * np.abs(cur_broad)
    )

    normal_strength = (
        0.20 * np.abs(norm_fine)
        +
        0.55 * np.abs(norm_mid)
        +
        0.25 * np.abs(norm_broad)
    )

    deficit = (
        normal_strength
        -
        current_strength
    )

    deficit_mask = np.clip(
        (deficit + 0.003) / 0.020,
        0.0,
        1.0
    )

    adaptive_mask = (
        0.40
        +
        0.60 * deficit_mask
    )

    midtone = midtone_map(
        current_lum
    )

    shadow_protection = smoothstep(
        current_lum,
        0.03,
        0.16
    )

    highlight_protection = (
        1.0
        -
        smoothstep(
            current_lum,
            0.82,
            0.98
        )
    )

    mask = (
        (
            0.35
            +
            0.65 * midtone
        )
        *
        shadow_protection
        *
        highlight_protection
        *
        adaptive_mask
    )

    if value > 0.0:

        correction = (
            3.20
            *
            value
            *
            reference_difference
            *
            mask
        )

    else:

        correction = (
            -1.80
            *
            abs(value)
            *
            reference_difference
            *
            mask
        )

    target = np.clip(
        current_lum + correction,
        0.0,
        1.0
    )

    return replace_luminance(
        image,
        target
    )


# ============================================================
# CONTRAST
# ============================================================

def apply_contrast(
    image,
    value
):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    lum = luminance(image)

    amount = (
        0.40 * value
    )

    contrast_lum = (
        0.5
        +
        (lum - 0.5)
        *
        (1.0 + amount)
    )

    protection = (
        smoothstep(
            lum,
            0.03,
            0.18
        )
        *
        (
            1.0
            -
            smoothstep(
                lum,
                0.82,
                0.98
            )
        )
    )

    target = (
        lum
        +
        (
            contrast_lum
            -
            lum
        )
        *
        (
            0.35
            +
            0.65 * protection
        )
    )

    return replace_luminance(
        image,
        np.clip(
            target,
            0.0,
            1.0
        )
    )


# ============================================================
# SATURATION
# ============================================================

def apply_saturation(
    image,
    value
):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    hsv = cv2.cvtColor(
        image.astype(np.float32),
        cv2.COLOR_BGR2HSV
    )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1]
        *
        (
            1.0
            +
            0.60 * value
        ),
        0.0,
        1.0
    )

    return np.clip(
        cv2.cvtColor(
            hsv,
            cv2.COLOR_HSV2BGR
        ),
        0.0,
        1.0
    )


# ============================================================
# COLOR TEMPERATURE
# ============================================================

def apply_color(
    image,
    value
):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    result = image.copy()

    amount = (
        0.10 * value
    )

    # BGR
    # Positive = warmer
    # Negative = cooler

    result[:, :, 2] *= (
        1.0 + amount
    )

    result[:, :, 0] *= (
        1.0 - amount
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# GENERAL BLENDING CORRECTION
# ============================================================

def apply_blending(
    image,
    normal,
    under,
    over,
    value
):
    """
    Stronger local blending correction.

    Goal:
        Correct unnatural transitions created when
        different exposure images contribute to the
        final candidate.

    It does NOT globally brighten or darken the image.

    It detects:
        - disagreement with Normal
        - exposure-transition regions
        - local Normal gradients
        - under/normal exposure differences
        - over/normal exposure differences
        - bright-side mismatch
        - dark-side mismatch

    Positive value:
        strengthen local transition correction.

    Negative value:
        reduce excessive transition correction.
    """

    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    current = luminance(image)
    norm = luminance(normal)
    under_lum = luminance(under)
    over_lum = luminance(over)

    # --------------------------------------------------------
    # 1. Local exposure differences
    # --------------------------------------------------------

    under_diff = np.abs(
        under_lum - norm
    )

    over_diff = np.abs(
        over_lum - norm
    )

    exposure_difference = np.maximum(
        under_diff,
        over_diff
    )

    exposure_difference = cv2.GaussianBlur(
        exposure_difference,
        (0, 0),
        3.0
    )

    exposure_mask = np.clip(
        (
            exposure_difference
            - 0.025
        )
        /
        0.16,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # 2. Normal exposure transitions
    # --------------------------------------------------------

    gx = cv2.Sobel(
        norm,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        norm,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(
        gx * gx
        +
        gy * gy
    )

    gradient = cv2.GaussianBlur(
        gradient,
        (0, 0),
        1.5
    )

    gradient_mask = np.clip(
        gradient / 0.18,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # 3. Candidate disagreement with Normal
    # --------------------------------------------------------

    candidate_difference = np.abs(
        current - norm
    )

    candidate_difference = cv2.GaussianBlur(
        candidate_difference,
        (0, 0),
        2.0
    )

    disagreement_mask = np.clip(
        (
            candidate_difference
            - 0.008
        )
        /
        0.12,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # 4. Bright-side and dark-side mismatch
    # --------------------------------------------------------

    bright_mismatch = np.maximum(
        current - under_lum,
        0.0
    )

    dark_mismatch = np.maximum(
        over_lum - current,
        0.0
    )

    bright_mismatch = cv2.GaussianBlur(
        bright_mismatch,
        (0, 0),
        2.5
    )

    dark_mismatch = cv2.GaussianBlur(
        dark_mismatch,
        (0, 0),
        2.5
    )

    bright_mask = np.clip(
        bright_mismatch / 0.18,
        0.0,
        1.0
    )

    dark_mask = np.clip(
        dark_mismatch / 0.18,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # 5. Combine transition confidence
    # --------------------------------------------------------

    transition_confidence = (
        0.35 * exposure_mask
        +
        0.25 * gradient_mask
        +
        0.25 * disagreement_mask
        +
        0.15
        *
        np.maximum(
            bright_mask,
            dark_mask
        )
    )

    transition_confidence = np.clip(
        transition_confidence,
        0.0,
        1.0
    )

    # Smooth the mask to avoid hard seams.

    transition_confidence = cv2.GaussianBlur(
        transition_confidence,
        (0, 0),
        2.0
    )

    # --------------------------------------------------------
    # 6. Protect important regions
    # --------------------------------------------------------

    shadow_protection = (
        0.35
        +
        0.65
        *
        smoothstep(
            current,
            0.02,
            0.18
        )
    )

    highlight_protection = (
        0.35
        +
        0.65
        *
        (
            1.0
            -
            smoothstep(
                current,
                0.84,
                0.99
            )
        )
    )

    protection = (
        shadow_protection
        *
        highlight_protection
    )

    # --------------------------------------------------------
    # 7. Build local reference
    # --------------------------------------------------------

    # Normal remains the main anchor.

    normal_target = norm.copy()

    # Where current is darker than Normal, use a controlled
    # blend toward Normal and Over.

    dark_reference = (
        0.70 * norm
        +
        0.30 * over_lum
    )

    # Where current is brighter than Normal, use Normal and
    # Under as the highlight-side reference.

    bright_reference = (
        0.70 * norm
        +
        0.30 * under_lum
    )

    dark_target = np.clip(
        dark_reference,
        0.0,
        1.0
    )

    bright_target = np.clip(
        bright_reference,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # 8. Determine correction direction
    # --------------------------------------------------------

    darker_than_normal = np.clip(
        norm - current,
        0.0,
        0.30
    )

    brighter_than_normal = np.clip(
        current - norm,
        0.0,
        0.30
    )

    dark_correction = (
        dark_target
        -
        current
    )

    bright_correction = (
        bright_target
        -
        current
    )

    directional_correction = (
        dark_correction
        *
        (
            darker_than_normal
            /
            (
                darker_than_normal
                +
                brighter_than_normal
                +
                EPS
            )
        )
        +
        bright_correction
        *
        (
            brighter_than_normal
            /
            (
                darker_than_normal
                +
                brighter_than_normal
                +
                EPS
            )
        )
    )

    # --------------------------------------------------------
    # 9. Add local transition smoothing
    # --------------------------------------------------------

    local_current = cv2.GaussianBlur(
        current,
        (0, 0),
        4.0
    )

    local_normal = cv2.GaussianBlur(
        norm,
        (0, 0),
        4.0
    )

    local_difference = (
        local_normal
        -
        local_current
    )

    # Combine direct exposure-reference correction with
    # local transition correction.

    correction = (
        0.72
        *
        directional_correction
        +
        0.28
        *
        local_difference
    )

    # --------------------------------------------------------
    # 10. Stronger but controlled strength
    # --------------------------------------------------------

    strength = (
        2.40
        *
        abs(value)
    )

    correction = (
        correction
        *
        transition_confidence
        *
        protection
        *
        strength
    )

    # --------------------------------------------------------
    # 11. Limit individual-pixel movement
    # --------------------------------------------------------

    correction = np.clip(
        correction,
        -0.085,
        0.085
    )

    # --------------------------------------------------------
    # 12. Smooth correction field
    # --------------------------------------------------------

    correction = cv2.GaussianBlur(
        correction,
        (0, 0),
        1.25
    )

    # Negative blending should reverse only a controlled
    # portion of the correction.

    if value < 0.0:
        correction *= 0.65

    target = np.clip(
        current + correction,
        0.0,
        1.0
    )

    return replace_luminance(
        image,
        target
    )


# ============================================================
# METRICS
# ============================================================

def image_metrics(image):
    lum = luminance(image)

    base = cv2.GaussianBlur(
        lum,
        (0, 0),
        5.0
    )

    detail = (
        lum - base
    )

    return {
        "mean_brightness":
            float(np.mean(lum)),

        "median_brightness":
            float(np.median(lum)),

        "shadow_ratio":
            float(np.mean(lum < 0.30)),

        "deep_shadow_ratio":
            float(np.mean(lum < 0.18)),

        "highlight_ratio":
            float(np.mean(lum > 0.85)),

        "local_detail":
            float(
                np.mean(
                    np.abs(detail)
                )
            )
    }


def reference_metrics(
    image,
    normal
):
    lum = luminance(image)
    normal_lum = luminance(normal)

    shadow_mask = (
        normal_lum < 0.35
    )

    if np.any(shadow_mask):

        shadow_distance = float(
            np.mean(
                np.abs(
                    lum[shadow_mask]
                    -
                    normal_lum[shadow_mask]
                )
            )
        )

    else:
        shadow_distance = 0.0

    image_small = cv2.GaussianBlur(
        lum,
        (0, 0),
        2.5
    )

    image_large = cv2.GaussianBlur(
        lum,
        (0, 0),
        8.0
    )

    normal_small = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        2.5
    )

    normal_large = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        8.0
    )

    image_structure = (
        image_small
        -
        image_large
    )

    normal_structure = (
        normal_small
        -
        normal_large
    )

    depth_error = float(
        np.mean(
            np.abs(
                image_structure
                -
                normal_structure
            )
        )
    )

    image_base = cv2.GaussianBlur(
        lum,
        (0, 0),
        7.0
    )

    normal_base = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        7.0
    )

    image_local = float(
        np.mean(
            np.abs(
                lum - image_base
            )
        )
    )

    normal_local = float(
        np.mean(
            np.abs(
                normal_lum
                -
                normal_base
            )
        )
    )

    haze_gap = abs(
        normal_local
        -
        image_local
    )

    return {
        "shadow_distance":
            shadow_distance,

        "depth_structure_error":
            depth_error,

        "haze_gap":
            float(haze_gap)
    }


def blending_metrics(
    image,
    normal,
    under,
    over
):
    """
    Diagnostics for exposure-transition quality.

    Lower transition error is better.
    Higher blend structure means more local
    structural variation.
    """

    lum = luminance(image)
    norm = luminance(normal)
    under_lum = luminance(under)
    over_lum = luminance(over)

    gx = cv2.Sobel(
        norm,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    gy = cv2.Sobel(
        norm,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    gradient = np.sqrt(
        gx * gx
        +
        gy * gy
    )

    transition_mask = (
        gradient
        >
        0.035
    )

    exposure_gap = np.maximum(
        np.abs(
            under_lum - norm
        ),
        np.abs(
            over_lum - norm
        )
    )

    transition_mask &= (
        exposure_gap
        >
        0.025
    )

    if np.any(transition_mask):

        transition_error = float(
            np.mean(
                np.abs(
                    lum[transition_mask]
                    -
                    norm[transition_mask]
                )
            )
        )

    else:

        transition_error = float(
            np.mean(
                np.abs(
                    lum - norm
                )
            )
        )

    small = cv2.GaussianBlur(
        lum,
        (0, 0),
        2.0
    )

    large = cv2.GaussianBlur(
        lum,
        (0, 0),
        7.0
    )

    structure = (
        small - large
    )

    blend_structure = float(
        np.mean(
            np.abs(structure)
        )
    )

    return {
        "transition_error":
            transition_error,

        "blend_structure":
            blend_structure
    }


# ============================================================
# MAIN API
# ============================================================

def refine_with_values(
    selected_path,
    normal_path,
    under_path,
    over_path,
    values,
    output_path
):
    selected = read_image(
        selected_path
    )

    normal = match_size(
        read_image(normal_path),
        selected
    )

    under = match_size(
        read_image(under_path),
        selected
    )

    over = match_size(
        read_image(over_path),
        selected
    )

    def get_value(name):
        return float(
            np.clip(
                values.get(name, 0.0),
                -1.0,
                1.0
            )
        )

    parameters = {
        "brightness":
            get_value("brightness"),

        "shadow":
            get_value("shadow"),

        "highlight":
            get_value("highlight"),

        "depth":
            get_value("depth"),

        "dehaze":
            get_value("dehaze"),

        "contrast":
            get_value("contrast"),

        "saturation":
            get_value("saturation"),

        "color":
            get_value("color"),

        "blending":
            get_value("blending")
    }

    before = image_metrics(
        selected
    )

    before_reference = reference_metrics(
        selected,
        normal
    )

    before_blending = blending_metrics(
        selected,
        normal,
        under,
        over
    )

    result = selected.copy()

    # --------------------------------------------------------
    # Existing correction controls
    # --------------------------------------------------------

    result = apply_brightness(
        result,
        parameters["brightness"]
    )

    result = apply_shadow(
        result,
        normal,
        over,
        parameters["shadow"]
    )

    result = apply_highlight(
        result,
        normal,
        under,
        over,
        parameters["highlight"]
    )

    result = apply_depth(
        result,
        normal,
        parameters["depth"]
    )

    result = apply_dehaze(
        result,
        normal,
        parameters["dehaze"]
    )

    result = apply_contrast(
        result,
        parameters["contrast"]
    )

    result = apply_saturation(
        result,
        parameters["saturation"]
    )

    result = apply_color(
        result,
        parameters["color"]
    )

    # --------------------------------------------------------
    # General blending correction
    # --------------------------------------------------------

    result = apply_blending(
        result,
        normal,
        under,
        over,
        parameters["blending"]
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )

    after = image_metrics(
        result
    )

    after_reference = reference_metrics(
        result,
        normal
    )

    after_blending = blending_metrics(
        result,
        normal,
        under,
        over
    )

    write_image(
        output_path,
        result
    )

    return {
        "feedback_values":
            dict(values),

        "before":
            before,

        "after":
            after,

        "before_reference":
            before_reference,

        "after_reference":
            after_reference,

        "before_blending":
            before_blending,

        "after_blending":
            after_blending,

        "brightness_difference":
            (
                after["mean_brightness"]
                -
                before["mean_brightness"]
            ),

        "shadow_ratio_difference":
            (
                after["shadow_ratio"]
                -
                before["shadow_ratio"]
            ),

        "highlight_ratio_difference":
            (
                after["highlight_ratio"]
                -
                before["highlight_ratio"]
            ),

        "local_detail_difference":
            (
                after["local_detail"]
                -
                before["local_detail"]
            ),

        "shadow_distance_change":
            (
                after_reference["shadow_distance"]
                -
                before_reference["shadow_distance"]
            ),

        "depth_error_change":
            (
                after_reference[
                    "depth_structure_error"
                ]
                -
                before_reference[
                    "depth_structure_error"
                ]
            ),

        "haze_gap_change":
            (
                after_reference["haze_gap"]
                -
                before_reference["haze_gap"]
            ),

        "transition_error_change":
            (
                after_blending[
                    "transition_error"
                ]
                -
                before_blending[
                    "transition_error"
                ]
            ),

        "blend_structure_change":
            (
                after_blending[
                    "blend_structure"
                ]
                -
                before_blending[
                    "blend_structure"
                ]
            ),

        "output":
            output_path
    }


# ============================================================
# CLI
# ============================================================

def build_parser():

    parser = argparse.ArgumentParser(
        description=(
            "FREE MEF - "
            "Reference-Guided Correction Engine V5"
        )
    )

    parser.add_argument(
        "--selected",
        required=True
    )

    parser.add_argument(
        "--normal",
        required=True
    )

    parser.add_argument(
        "--under",
        required=True
    )

    parser.add_argument(
        "--over",
        required=True
    )

    parser.add_argument(
        "--brightness",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--shadow",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--highlight",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--depth",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--dehaze",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--contrast",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--saturation",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--color",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--blending",
        type=float,
        default=0.0
    )

    parser.add_argument(
        "--output",
        required=True
    )

    return parser


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(result):

    values = result[
        "feedback_values"
    ]

    before = result["before"]
    after = result["after"]

    before_reference = result[
        "before_reference"
    ]

    after_reference = result[
        "after_reference"
    ]

    before_blending = result[
        "before_blending"
    ]

    after_blending = result[
        "after_blending"
    ]

    print()
    print("=" * 70)
    print(
        "FREE MEF - CORRECTION ENGINE V5"
    )
    print("=" * 70)

    print()
    print("PARAMETERS")

    for key in (
        "brightness",
        "shadow",
        "highlight",
        "depth",
        "dehaze",
        "contrast",
        "saturation",
        "color",
        "blending"
    ):

        print(
            f"{key:13s}: "
            f"{float(values.get(key, 0.0)):+.3f}"
        )

    print()
    print("BASIC METRICS")

    print(
        f"Brightness       : "
        f"{before['mean_brightness']:.6f} -> "
        f"{after['mean_brightness']:.6f}"
    )

    print(
        f"Median           : "
        f"{before['median_brightness']:.6f} -> "
        f"{after['median_brightness']:.6f}"
    )

    print(
        f"Shadow ratio     : "
        f"{before['shadow_ratio']:.6f} -> "
        f"{after['shadow_ratio']:.6f}"
    )

    print(
        f"Deep shadow      : "
        f"{before['deep_shadow_ratio']:.6f} -> "
        f"{after['deep_shadow_ratio']:.6f}"
    )

    print(
        f"Highlight ratio  : "
        f"{before['highlight_ratio']:.6f} -> "
        f"{after['highlight_ratio']:.6f}"
    )

    print(
        f"Local detail     : "
        f"{before['local_detail']:.6f} -> "
        f"{after['local_detail']:.6f}"
    )

    print()
    print(
        "NORMAL-REFERENCE METRICS"
    )

    print(
        f"Shadow distance  : "
        f"{before_reference['shadow_distance']:.6f} -> "
        f"{after_reference['shadow_distance']:.6f}"
    )

    print(
        f"Depth error      : "
        f"{before_reference['depth_structure_error']:.6f} -> "
        f"{after_reference['depth_structure_error']:.6f}"
    )

    print(
        f"Haze gap         : "
        f"{before_reference['haze_gap']:.6f} -> "
        f"{after_reference['haze_gap']:.6f}"
    )

    print()
    print("BLENDING METRICS")

    print(
        f"Transition error : "
        f"{before_blending['transition_error']:.6f} -> "
        f"{after_blending['transition_error']:.6f}"
    )

    print(
        f"Blend structure  : "
        f"{before_blending['blend_structure']:.6f} -> "
        f"{after_blending['blend_structure']:.6f}"
    )

    print()
    print("CHANGES")

    print(
        f"Brightness       : "
        f"{result['brightness_difference']:+.6f}"
    )

    print(
        f"Shadow distance  : "
        f"{result['shadow_distance_change']:+.6f}"
    )

    print(
        f"Depth error      : "
        f"{result['depth_error_change']:+.6f}"
    )

    print(
        f"Haze gap         : "
        f"{result['haze_gap_change']:+.6f}"
    )

    print(
        f"Local detail     : "
        f"{result['local_detail_difference']:+.6f}"
    )

    print(
        f"Transition error : "
        f"{result['transition_error_change']:+.6f}"
    )

    print(
        f"Blend structure  : "
        f"{result['blend_structure_change']:+.6f}"
    )

    print()
    print("OUTPUT:")
    print(
        f"  {result['output']}"
    )

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = build_parser()

    args = parser.parse_args()

    values = {

        "brightness":
            args.brightness,

        "shadow":
            args.shadow,

        "highlight":
            args.highlight,

        "depth":
            args.depth,

        "dehaze":
            args.dehaze,

        "contrast":
            args.contrast,

        "saturation":
            args.saturation,

        "color":
            args.color,

        "blending":
            args.blending
    }

    result = refine_with_values(
        args.selected,
        args.normal,
        args.under,
        args.over,
        values,
        args.output
    )

    print_result(
        result
    )


if __name__ == "__main__":
    main()
