#!/usr/bin/env python3

"""
FREE MEF - CORRECTION ENGINE V4

Reference-guided image revision engine.

Existing controls:
    brightness
    shadow
    highlight
    depth
    dehaze
    contrast
    saturation
    color

New control:
    blending

The blending correction is intentionally conservative.
It uses the selected candidate together with the original
Under / Normal / Over exposures to improve local exposure
transitions without globally averaging the image.
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
# LUMINANCE / MASKS
# ============================================================

def luminance(image):
    return (
        0.0722 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.2126 * image[:, :, 2]
    )


def smoothstep(x, edge0, edge1):
    t = np.clip(
        (x - edge0)
        / max(edge1 - edge0, EPS),
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
            *
            0.28
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

def apply_shadow(image, normal, value):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    current_lum = luminance(image)
    normal_lum = luminance(normal)

    mask = shadow_map(normal_lum)

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

        lift = (
            value
            *
            0.22
            *
            mask
        )

        target = current_lum + lift

        max_target = normal_lum + 0.10

        target = np.minimum(
            target,
            max_target
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

    mask = highlight_map(current_lum)

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

        target = (
            current_lum
            -
            abs(value)
            *
            0.12
            *
            mask
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

def apply_depth(image, normal, value):
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
        cur_small
        -
        cur_large
    )

    normal_structure = (
        norm_small
        -
        norm_large
    )

    missing = np.clip(
        normal_structure
        -
        current_structure,
        -0.070,
        0.070
    )

    midtone = midtone_map(
        normal_lum
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
            midtone
        )
        *
        confidence
        *
        protection
    )

    target = np.clip(
        current_lum
        +
        correction,
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
    """
    Multi-scale reference-guided tonal
    separation restoration.

    Positive values restore missing local
    structure relative to Normal.

    Negative values reduce that separation.
    """

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
        current_lum
        -
        cur_s
    )

    norm_fine = (
        normal_lum
        -
        norm_s
    )

    cur_mid = (
        cur_s
        -
        cur_m
    )

    norm_mid = (
        norm_s
        -
        norm_m
    )

    cur_broad = (
        cur_m
        -
        cur_l
    )

    norm_broad = (
        norm_m
        -
        norm_l
    )

    diff_fine = np.clip(
        norm_fine
        -
        cur_fine,
        -0.045,
        0.045
    )

    diff_mid = np.clip(
        norm_mid
        -
        cur_mid,
        -0.075,
        0.075
    )

    diff_broad = np.clip(
        norm_broad
        -
        cur_broad,
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
        (deficit + 0.003)
        / 0.020,
        0.0,
        1.0
    )

    adaptive_mask = (
        0.40
        +
        0.60
        *
        deficit_mask
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
            0.65
            *
            midtone
        )
        *
        shadow_protection
        *
        highlight_protection
        *
        adaptive_mask
    )

    if value > 0.0:

        strength = 3.20 * value

        correction = (
            reference_difference
            *
            mask
        )

    else:

        strength = 1.80 * abs(value)

        correction = (
            -0.55
            *
            reference_difference
            *
            mask
        )

    target = np.clip(
        current_lum
        +
        strength
        *
        correction,
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

def apply_contrast(image, value):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    lum = luminance(image)

    amount = 0.40 * value

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
            0.65
            *
            protection
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

def apply_saturation(image, value):
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

def apply_color(image, value):
    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    result = image.copy()

    amount = 0.10 * value

    # BGR:
    # positive = warmer
    # negative = cooler

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
# BLENDING CORRECTION
# ============================================================

def apply_blending(
    image,
    normal,
    under,
    over,
    value
):
    """
    General reference-guided blending correction.

    Purpose:
        Improve unnatural local exposure transitions
        in an already-fused candidate.

    Inputs:
        image  = customer-selected candidate
        normal = normal exposure
        under  = under exposure
        over   = over exposure
        value  = blending correction strength

    Design principles:
        - Do NOT globally average exposures.
        - Normal exposure remains the main visual anchor.
        - Under exposure contributes information in
          bright/highlight transition regions.
        - Over exposure contributes information in
          dark/shadow transition regions.
        - Corrections are local and conservative.
        - Strong highlights and deep shadows are protected.
        - Smoothness is encouraged near exposure boundaries.
    """

    value = float(
        np.clip(value, -1.0, 1.0)
    )

    if abs(value) < EPS:
        return image.copy()

    current_lum = luminance(image)
    normal_lum = luminance(normal)
    under_lum = luminance(under)
    over_lum = luminance(over)

    # --------------------------------------------------------
    # Exposure consistency maps
    # --------------------------------------------------------

    # Under exposure is useful mainly where the current
    # candidate becomes too bright relative to Normal.
    bright_difference = np.maximum(
        current_lum - normal_lum,
        0.0
    )

    # Over exposure is useful mainly where the current
    # candidate becomes too dark relative to Normal.
    dark_difference = np.maximum(
        normal_lum - current_lum,
        0.0
    )

    # --------------------------------------------------------
    # Exposure transition maps
    # --------------------------------------------------------

    normal_gradient_x = cv2.Sobel(
        normal_lum,
        cv2.CV_32F,
        1,
        0,
        ksize=3
    )

    normal_gradient_y = cv2.Sobel(
        normal_lum,
        cv2.CV_32F,
        0,
        1,
        ksize=3
    )

    normal_gradient = np.sqrt(
        normal_gradient_x
        *
        normal_gradient_x
        +
        normal_gradient_y
        *
        normal_gradient_y
    )

    gradient_confidence = np.clip(
        normal_gradient / 0.10,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Local exposure disagreement
    # --------------------------------------------------------

    under_difference = np.abs(
        under_lum - normal_lum
    )

    over_difference = np.abs(
        over_lum - normal_lum
    )

    exposure_disagreement = np.maximum(
        under_difference,
        over_difference
    )

    disagreement_mask = np.clip(
        (
            exposure_disagreement
            -
            0.015
        )
        /
        0.20,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Local smoothness
    # --------------------------------------------------------

    current_blur_small = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        2.0
    )

    current_blur_large = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        8.0
    )

    normal_blur_small = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        2.0
    )

    normal_blur_large = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        8.0
    )

    current_local_structure = (
        current_blur_small
        -
        current_blur_large
    )

    normal_local_structure = (
        normal_blur_small
        -
        normal_blur_large
    )

    structure_difference = np.abs(
        current_local_structure
        -
        normal_local_structure
    )

    structure_mask = np.clip(
        (
            structure_difference
            -
            0.002
        )
        /
        0.035,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Candidate/reference color consistency
    # --------------------------------------------------------

    current_smooth = cv2.GaussianBlur(
        image,
        (0, 0),
        3.0
    )

    normal_smooth = cv2.GaussianBlur(
        normal,
        (0, 0),
        3.0
    )

    color_difference = np.mean(
        np.abs(
            current_smooth
            -
            normal_smooth
        ),
        axis=2
    )

    color_mask = np.clip(
        (
            color_difference
            -
            0.008
        )
        /
        0.10,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Transition protection
    # --------------------------------------------------------

    shadow_protection = (
        1.0
        -
        smoothstep(
            current_lum,
            0.00,
            0.12
        )
    )

    highlight_protection = (
        1.0
        -
        smoothstep(
            current_lum,
            0.88,
            1.00
        )
    )

    natural_range = (
        shadow_protection
        *
        highlight_protection
    )

    # --------------------------------------------------------
    # Main local blending mask
    # --------------------------------------------------------

    local_mask = (
        0.30
        +
        0.70
        *
        disagreement_mask
    )

    local_mask *= (
        0.35
        +
        0.65
        *
        gradient_confidence
    )

    local_mask *= (
        0.40
        +
        0.60
        *
        structure_mask
    )

    local_mask *= (
        0.50
        +
        0.50
        *
        color_mask
    )

    local_mask *= natural_range

    # --------------------------------------------------------
    # Reference selection
    # --------------------------------------------------------

    # For bright transition regions:
    # Under exposure is normally the safer source.
    bright_reference = np.minimum(
        under_lum,
        normal_lum
    )

    # For dark transition regions:
    # Over exposure is normally the safer source.
    dark_reference = np.maximum(
        over_lum,
        normal_lum
    )

    # Blend both reference directions according to the
    # candidate's local relationship with Normal.

    bright_amount = np.clip(
        bright_difference / 0.20,
        0.0,
        1.0
    )

    dark_amount = np.clip(
        dark_difference / 0.20,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Bright-side correction
    # --------------------------------------------------------

    bright_correction = (
        bright_reference
        -
        current_lum
    )

    # --------------------------------------------------------
    # Dark-side correction
    # --------------------------------------------------------

    dark_correction = (
        dark_reference
        -
        current_lum
    )

    # --------------------------------------------------------
    # Combine corrections
    # --------------------------------------------------------

    correction = (
        bright_correction
        *
        bright_amount
        +
        dark_correction
        *
        dark_amount
    )

    # --------------------------------------------------------
    # Keep the correction tied to Normal
    # --------------------------------------------------------

    correction_limit = (
        0.06
        *
        abs(value)
        *
        local_mask
    )

    correction = np.clip(
        correction,
        -correction_limit,
        correction_limit
    )

    # --------------------------------------------------------
    # Final strength
    # --------------------------------------------------------

    strength = (
        0.85
        *
        value
    )

    target = (
        current_lum
        +
        strength
        *
        correction
    )

    target = np.clip(
        target,
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
        lum
        -
        base
    )

    return {
        "mean_brightness":
            float(
                np.mean(lum)
            ),

        "median_brightness":
            float(
                np.median(lum)
            ),

        "shadow_ratio":
            float(
                np.mean(
                    lum < 0.30
                )
            ),

        "deep_shadow_ratio":
            float(
                np.mean(
                    lum < 0.18
                )
            ),

        "highlight_ratio":
            float(
                np.mean(
                    lum > 0.85
                )
            ),

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
                lum
                -
                image_base
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
            float(
                haze_gap
            )
    }


# ============================================================
# BLENDING DIAGNOSTIC METRICS
# ============================================================

def blending_metrics(
    image,
    normal,
    under,
    over
):
    """
    Diagnostic metrics for the new blending correction.

    These metrics do NOT select a candidate.

    They only measure how much the selected image differs
    from local exposure/reference behavior.
    """

    current_lum = luminance(image)
    normal_lum = luminance(normal)
    under_lum = luminance(under)
    over_lum = luminance(over)

    # Local smoothness of current candidate.
    current_small = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        2.0
    )

    current_large = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        8.0
    )

    normal_small = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        2.0
    )

    normal_large = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        8.0
    )

    current_structure = (
        current_small
        -
        current_large
    )

    normal_structure = (
        normal_small
        -
        normal_large
    )

    structure_error = float(
        np.mean(
            np.abs(
                current_structure
                -
                normal_structure
            )
        )
    )

    # Local exposure disagreement.
    under_difference = np.abs(
        under_lum
        -
        normal_lum
    )

    over_difference = np.abs(
        over_lum
        -
        normal_lum
    )

    exposure_difference = np.maximum(
        under_difference,
        over_difference
    )

    # Candidate disagreement with Normal.
    candidate_difference = np.abs(
        current_lum
        -
        normal_lum
    )

    transition_mask = np.clip(
        (
            exposure_difference
            -
            0.015
        )
        /
        0.20,
        0.0,
        1.0
    )

    if np.any(
        transition_mask > 0.10
    ):

        weighted_difference = (
            candidate_difference
            *
            transition_mask
        )

        transition_error = float(
            np.sum(
                weighted_difference
            )
            /
            (
                np.sum(
                    transition_mask
                )
                +
                EPS
            )
        )

    else:

        transition_error = 0.0

    return {
        "transition_error":
            transition_error,

        "blending_structure_error":
            structure_error
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
        read_image(
            normal_path
        ),
        selected
    )

    under = match_size(
        read_image(
            under_path
        ),
        selected
    )

    over = match_size(
        read_image(
            over_path
        ),
        selected
    )

    def get_value(name):
        return float(
            np.clip(
                values.get(
                    name,
                    0.0
                ),
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
    # Existing corrections
    # --------------------------------------------------------

    result = apply_brightness(
        result,
        parameters["brightness"]
    )

    result = apply_shadow(
        result,
        normal,
        parameters["shadow"]
    )

    result = apply_highlight(
        result,
        normal,
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
    # NEW: GENERAL BLENDING
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
                after_reference[
                    "shadow_distance"
                ]
                -
                before_reference[
                    "shadow_distance"
                ]
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
                after_reference[
                    "haze_gap"
                ]
                -
                before_reference[
                    "haze_gap"
                ]
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

        "blending_structure_error_change":
            (
                after_blending[
                    "blending_structure_error"
                ]
                -
                before_blending[
                    "blending_structure_error"
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
            "FREE MEF - Correction Engine V4"
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

    # --------------------------------------------------------
    # NEW BLENDING CONTROL
    # --------------------------------------------------------

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

    before = result[
        "before"
    ]

    after = result[
        "after"
    ]

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
        "FREE MEF - CORRECTION ENGINE V4"
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
    print("NORMAL-REFERENCE METRICS")

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
        f"{before_blending['blending_structure_error']:.6f} -> "
        f"{after_blending['blending_structure_error']:.6f}"
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
        f"{result['blending_structure_error_change']:+.6f}"
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
