#!/usr/bin/env python3

"""
FREE MEF - CORRECTION ENGINE V2
===============================

Purpose
-------
Apply numeric correction values to a selected candidate image.

Typical pipeline:

    Under + Normal + Over
            |
        A / B / C
            |
      customer selects
            |
     customer feedback
            |
          Gemini
            |
     numeric parameters
            |
    this correction engine
            |
        revised image


Important design principles
---------------------------
1. Selected candidate remains the visual base.
2. Normal exposure is the structural reference.
3. Shadow correction is Normal-aware.
4. Highlight correction is Normal-aware.
5. Depth restoration uses missing Normal structure.
6. Dehaze is Normal-aware and does NOT simply add generic
   contrast.
7. Color remains conservative.
8. Corrections are deterministic once Gemini values are supplied.
"""

import os
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
        image.astype(np.float32)
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

    success = cv2.imwrite(
        path,
        image8,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            100
        ]
    )

    if not success:
        raise RuntimeError(
            f"Could not save image:\n{path}"
        )


# ============================================================
# BASIC HELPERS
# ============================================================

def luminance(
    image
):
    """
    OpenCV image is BGR.
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


def resize_like(
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


def gaussian(
    image,
    sigma
):
    return cv2.GaussianBlur(
        image.astype(
            np.float32
        ),
        (0, 0),
        sigma
    )


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
        (
            x - edge0
        )
        /
        width,
        0.0,
        1.0
    )

    return (
        t * t *
        (
            3.0 - 2.0 * t
        )
    ).astype(
        np.float32
    )


# ============================================================
# COLOR SPACE
# ============================================================

def bgr_to_lab(
    image
):
    image8 = (
        np.clip(
            image,
            0.0,
            1.0
        )
        * 255.0
    ).astype(
        np.uint8
    )

    return cv2.cvtColor(
        image8,
        cv2.COLOR_BGR2LAB
    ).astype(
        np.float32
    )


def lab_to_bgr(
    lab
):
    lab = np.clip(
        lab,
        0.0,
        255.0
    ).astype(
        np.uint8
    )

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return (
        result.astype(
            np.float32
        )
        / 255.0
    )


# ============================================================
# REGION MAPS
# ============================================================

def shadow_map(
    normal_lum
):
    """
    Detect lifted shadow / lower-mid regions.

    Deep blacks are intentionally protected.
    """

    rising = smoothstep(
        normal_lum,
        0.16,
        0.28
    )

    falling = (
        1.0
        -
        smoothstep(
            normal_lum,
            0.38,
            0.52
        )
    )

    base = (
        rising *
        falling
    )

    deep_black_protection = (
        1.0
        -
        smoothstep(
            normal_lum,
            0.04,
            0.18
        )
    )

    result = (
        base
        *
        (
            1.0
            -
            0.75 *
            deep_black_protection
        )
    )

    result = cv2.GaussianBlur(
        result.astype(
            np.float32
        ),
        (0, 0),
        3.0
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


def highlight_map(
    normal_lum
):
    """
    Detect bright regions where highlight correction may be
    appropriate.
    """

    result = smoothstep(
        normal_lum,
        0.70,
        0.96
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


def midtone_map(
    normal_lum
):
    """
    Natural midtone region for depth and dehaze.
    """

    left = smoothstep(
        normal_lum,
        0.20,
        0.38
    )

    right = (
        1.0
        -
        smoothstep(
            normal_lum,
            0.70,
            0.86
        )
    )

    return np.clip(
        left * right,
        0.0,
        1.0
    )


# ============================================================
# BRIGHTNESS
# ============================================================

def apply_brightness(
    image,
    value
):
    """
    Controlled global brightness.

    value:
        -1 = darker
        +1 = brighter
    """

    value = float(
        np.clip(
            value,
            -1.0,
            1.0
        )
    )

    if abs(value) < EPS:
        return image

    lum = luminance(
        image
    )

    # Controlled gamma response.
    gamma = (
        1.0
        -
        0.18 *
        value
    )

    gamma = max(
        gamma,
        0.72
    )

    new_lum = np.power(
        np.clip(
            lum,
            0.0,
            1.0
        ),
        gamma
    )

    ratio = (
        new_lum
        /
        (
            lum + EPS
        )
    )

    result = (
        image *
        ratio[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# SHADOW REDUCTION
# ============================================================

def reduce_lifted_shadows(
    image,
    normal,
    value
):
    """
    Negative shadow value.

    Moves lifted Candidate shadows toward Normal rather than
    indiscriminately darkening every dark pixel.
    """

    if value >= 0:
        return image

    value = float(
        np.clip(
            value,
            -1.0,
            0.0
        )
    )

    current_lum = luminance(
        image
    )

    normal_lum = luminance(
        normal
    )

    mask = shadow_map(
        normal_lum
    )

    # Only move toward Normal.
    amount = (
        0.85
        *
        abs(value)
        *
        mask
    )

    new_lum = (
        current_lum *
        (1.0 - amount)
        +
        normal_lum *
        amount
    )

    new_lum = np.clip(
        new_lum,
        0.0,
        1.0
    )

    ratio = (
        new_lum
        /
        (
            current_lum + EPS
        )
    )

    result = (
        image *
        ratio[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# SHADOW RECOVERY
# ============================================================

def recover_shadows(
    image,
    normal,
    over,
    value
):
    """
    Positive shadow value.

    Uses Over only where it actually provides additional
    shadow information.
    """

    if value <= 0:
        return image

    value = float(
        np.clip(
            value,
            0.0,
            1.0
        )
    )

    current_lum = luminance(
        image
    )

    normal_lum = luminance(
        normal
    )

    over_lum = luminance(
        over
    )

    mask = shadow_map(
        normal_lum
    )

    over_gain = np.clip(
        (
            over_lum
            -
            current_lum
            -
            0.025
        )
        /
        0.25,
        0.0,
        1.0
    )

    amount = (
        0.60
        *
        value
        *
        mask
        *
        over_gain
    )

    new_lum = (
        current_lum *
        (1.0 - amount)
        +
        over_lum *
        amount
    )

    new_lum = np.clip(
        new_lum,
        0.0,
        1.0
    )

    ratio = (
        new_lum
        /
        (
            current_lum + EPS
        )
    )

    return np.clip(
        image *
        ratio[:, :, None],
        0.0,
        1.0
    )


# ============================================================
# HIGHLIGHT REDUCTION
# ============================================================

def reduce_lifted_highlights(
    image,
    normal,
    value
):
    """
    Negative highlight value.

    Moves bright lifted areas toward Normal.
    """

    if value >= 0:
        return image

    value = float(
        np.clip(
            value,
            -1.0,
            0.0
        )
    )

    current_lum = luminance(
        image
    )

    normal_lum = luminance(
        normal
    )

    mask = highlight_map(
        normal_lum
    )

    amount = (
        0.78
        *
        abs(value)
        *
        mask
    )

    new_lum = (
        current_lum *
        (1.0 - amount)
        +
        normal_lum *
        amount
    )

    new_lum = np.clip(
        new_lum,
        0.0,
        1.0
    )

    ratio = (
        new_lum
        /
        (
            current_lum + EPS
        )
    )

    return np.clip(
        image *
        ratio[:, :, None],
        0.0,
        1.0
    )


# ============================================================
# HIGHLIGHT RECOVERY
# ============================================================

def recover_highlights(
    image,
    normal,
    under,
    value
):
    """
    Positive highlight value.

    Uses Under only where it has useful darker highlight
    information.
    """

    if value <= 0:
        return image

    value = float(
        np.clip(
            value,
            0.0,
            1.0
        )
    )

    current_lum = luminance(
        image
    )

    normal_lum = luminance(
        normal
    )

    under_lum = luminance(
        under
    )

    mask = highlight_map(
        normal_lum
    )

    under_gain = np.clip(
        (
            current_lum
            -
            under_lum
            -
            0.025
        )
        /
        0.25,
        0.0,
        1.0
    )

    amount = (
        0.50
        *
        value
        *
        mask
        *
        under_gain
    )

    new_lum = (
        current_lum *
        (1.0 - amount)
        +
        under_lum *
        amount
    )

    new_lum = np.clip(
        new_lum,
        0.0,
        1.0
    )

    ratio = (
        new_lum
        /
        (
            current_lum + EPS
        )
    )

    return np.clip(
        image *
        ratio[:, :, None],
        0.0,
        1.0
    )


# ============================================================
# DEPTH
# ============================================================

def apply_depth(
    image,
    normal,
    value
):
    """
    Restore missing local structure from Normal exposure.

    This is not ordinary sharpening.

    It compares local structure in the candidate with local
    structure in Normal and restores only part of the missing
    structure.
    """

    if abs(value) < EPS:
        return image

    value = float(
        np.clip(
            value,
            -1.0,
            1.0
        )
    )

    current_lum = luminance(
        image
    )

    normal_lum = luminance(
        normal
    )

    current_base = gaussian(
        current_lum,
        4.0
    )

    normal_base = gaussian(
        normal_lum,
        4.0
    )

    current_detail = (
        current_lum -
        current_base
    )

    normal_detail = (
        normal_lum -
        normal_base
    )

    missing = (
        normal_detail -
        current_detail
    )

    mask = midtone_map(
        normal_lum
    )

    # Prevent extreme edge boosting.
    missing = np.clip(
        missing,
        -0.035,
        0.035
    )

    strength = (
        0.38 *
        value
    )

    correction = (
        strength
        *
        mask
        *
        missing
    )

    new_lum = np.clip(
        current_lum +
        correction,
        0.0,
        1.0
    )

    ratio = (
        new_lum
        /
        (
            current_lum + EPS
        )
    )

    result = (
        image *
        ratio[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# NORMAL-REFERENCE DEHAZE V2
# ============================================================

def apply_dehaze(
    image,
    normal,
    value
):
    """
    Normal-reference dehaze.

    Positive value:
        reduce haze

    Negative value:
        intentionally move toward a softer / hazier result.

    The purpose is NOT to add generic sharpening.

    Instead:

        1. Compare candidate local structure to Normal.
        2. Detect where the candidate is flatter than Normal.
        3. Recover only part of the missing tonal separation.
        4. Protect deep shadows and bright highlights.
        5. Avoid halos and exaggerated HDR contrast.

    Normal exposure is used as the structural reference.
    """

    value = float(
        np.clip(
            value,
            -1.0,
            1.0
        )
    )

    if abs(value) < EPS:
        return image

    current_lum = luminance(
        image
    )

    normal_lum = luminance(
        normal
    )

    # --------------------------------------------------------
    # Multi-scale local structure
    # --------------------------------------------------------

    current_small = gaussian(
        current_lum,
        3.0
    )

    current_large = gaussian(
        current_lum,
        14.0
    )

    normal_small = gaussian(
        normal_lum,
        3.0
    )

    normal_large = gaussian(
        normal_lum,
        14.0
    )

    current_detail = (
        current_lum -
        current_small
    )

    normal_detail = (
        normal_lum -
        normal_small
    )

    current_separation = (
        current_lum -
        current_large
    )

    normal_separation = (
        normal_lum -
        normal_large
    )

    # --------------------------------------------------------
    # How much structure is missing relative to Normal?
    # --------------------------------------------------------

    missing_detail = (
        normal_detail -
        current_detail
    )

    missing_separation = (
        normal_separation -
        current_separation
    )

    # --------------------------------------------------------
    # Only positive missing structure is recovered.
    #
    # This is the key difference from the previous dehaze
    # implementation.
    # --------------------------------------------------------

    positive_detail = np.maximum(
        missing_detail,
        0.0
    )

    positive_separation = np.maximum(
        missing_separation,
        0.0
    )

    # --------------------------------------------------------
    # Midtone protection.
    #
    # Do not aggressively alter deep blacks or near-clipping
    # highlights.
    # --------------------------------------------------------

    midtones = midtone_map(
        normal_lum
    )

    # --------------------------------------------------------
    # Flatness detector.
    #
    # When current image already has equal or greater local
    # structure than Normal, little or no correction occurs.
    # --------------------------------------------------------

    current_strength = np.abs(
        current_detail
    )

    normal_strength = np.abs(
        normal_detail
    )

    flatness = np.clip(
        (
            normal_strength
            -
            current_strength
            +
            0.002
        )
        /
        0.018,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Limit the correction.
    # --------------------------------------------------------

    detail_component = np.clip(
        positive_detail,
        0.0,
        0.025
    )

    separation_component = np.clip(
        positive_separation,
        0.0,
        0.045
    )

    # --------------------------------------------------------
    # Combine.
    # --------------------------------------------------------

    correction = (
        0.65 *
        detail_component
        +
        0.35 *
        separation_component
    )

    correction *= (
        midtones *
        flatness
    )

    # --------------------------------------------------------
    # Final restrained strength.
    # --------------------------------------------------------

    strength = (
        0.85 *
        value
    )

    new_lum = (
        current_lum
        +
        strength *
        correction
    )

    new_lum = np.clip(
        new_lum,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # Convert luminance change to RGB/BGR scaling.
    # --------------------------------------------------------

    ratio = (
        new_lum
        /
        (
            current_lum + EPS
        )
    )

    result = (
        image *
        ratio[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# CONTRAST
# ============================================================

def apply_contrast(
    image,
    value
):

    value = float(
        np.clip(
            value,
            -1.0,
            1.0
        )
    )

    if abs(value) < EPS:
        return image

    lum = luminance(
        image
    )

    factor = (
        1.0
        +
        0.16 *
        value
    )

    new_lum = (
        0.5
        +
        factor *
        (
            lum -
            0.5
        )
    )

    new_lum = np.clip(
        new_lum,
        0.0,
        1.0
    )

    ratio = (
        new_lum
        /
        (
            lum + EPS
        )
    )

    return np.clip(
        image *
        ratio[:, :, None],
        0.0,
        1.0
    )


# ============================================================
# SATURATION
# ============================================================

def apply_saturation(
    image,
    value
):

    value = float(
        np.clip(
            value,
            -1.0,
            1.0
        )
    )

    if abs(value) < EPS:
        return image

    gray = (
        0.114 * image[:, :, 0]
        +
        0.587 * image[:, :, 1]
        +
        0.299 * image[:, :, 2]
    )

    factor = (
        1.0
        +
        0.16 *
        value
    )

    result = (
        gray[:, :, None]
        +
        factor *
        (
            image -
            gray[:, :, None]
        )
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# COLOR
# ============================================================

def apply_color(
    image,
    normal,
    value
):
    """
    Positive value:
        gently anchor chroma toward Normal.

    Negative value:
        reduce the correction.
    """

    value = float(
        np.clip(
            value,
            -1.0,
            1.0
        )
    )

    if abs(value) < EPS:
        return image

    image_lab = bgr_to_lab(
        image
    )

    normal_lab = bgr_to_lab(
        normal
    )

    amount = (
        0.35 *
        max(
            value,
            0.0
        )
    )

    image_lab[:, :, 1] = (
        (
            1.0 -
            amount
        )
        *
        image_lab[:, :, 1]
        +
        amount *
        normal_lab[:, :, 1]
    )

    image_lab[:, :, 2] = (
        (
            1.0 -
            amount
        )
        *
        image_lab[:, :, 2]
        +
        amount *
        normal_lab[:, :, 2]
    )

    return lab_to_bgr(
        image_lab
    )


# ============================================================
# METRICS
# ============================================================

def metrics(
    image
):

    lum = luminance(
        image
    )

    blur = gaussian(
        lum,
        2.0
    )

    detail = (
        lum -
        blur
    )

    return {

        "mean_brightness":
            float(
                lum.mean()
            ),

        "median_brightness":
            float(
                np.median(
                    lum
                )
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
                    np.abs(
                        detail
                    )
                )
            )
    }


# ============================================================
# MAIN REFINEMENT
# ============================================================

def refine_with_values(
    selected_image_path,
    normal_path,
    under_path,
    over_path,
    values,
    output_path
):
    """
    Apply Gemini numeric values to the selected candidate.
    """

    selected = read_image(
        selected_image_path
    )

    normal = resize_like(
        read_image(
            normal_path
        ),
        selected
    )

    under = resize_like(
        read_image(
            under_path
        ),
        selected
    )

    over = resize_like(
        read_image(
            over_path
        ),
        selected
    )

    # --------------------------------------------------------
    # Read parameters
    # --------------------------------------------------------

    brightness = float(
        np.clip(
            values.get(
                "brightness",
                0.0
            ),
            -1.0,
            1.0
        )
    )

    shadow = float(
        np.clip(
            values.get(
                "shadow",
                0.0
            ),
            -1.0,
            1.0
        )
    )

    highlight = float(
        np.clip(
            values.get(
                "highlight",
                0.0
            ),
            -1.0,
            1.0
        )
    )

    depth = float(
        np.clip(
            values.get(
                "depth",
                0.0
            ),
            -1.0,
            1.0
        )
    )

    dehaze = float(
        np.clip(
            values.get(
                "dehaze",
                0.0
            ),
            -1.0,
            1.0
        )
    )

    contrast = float(
        np.clip(
            values.get(
                "contrast",
                0.0
            ),
            -1.0,
            1.0
        )
    )

    saturation = float(
        np.clip(
            values.get(
                "saturation",
                0.0
            ),
            -1.0,
            1.0
        )
    )

    color = float(
        np.clip(
            values.get(
                "color",
                0.0
            ),
            -1.0,
            1.0
        )
    )

    # --------------------------------------------------------
    # BEFORE
    # --------------------------------------------------------

    before = metrics(
        selected
    )

    result = selected.copy()

    # ========================================================
    # 1. BRIGHTNESS
    # ========================================================

    result = apply_brightness(
        result,
        brightness
    )

    # ========================================================
    # 2. SHADOWS
    # ========================================================

    if shadow < 0:

        result = reduce_lifted_shadows(
            result,
            normal,
            shadow
        )

    elif shadow > 0:

        result = recover_shadows(
            result,
            normal,
            over,
            shadow
        )

    # ========================================================
    # 3. HIGHLIGHTS
    # ========================================================

    if highlight < 0:

        result = reduce_lifted_highlights(
            result,
            normal,
            highlight
        )

    elif highlight > 0:

        result = recover_highlights(
            result,
            normal,
            under,
            highlight
        )

    # ========================================================
    # 4. DEPTH
    # ========================================================

    result = apply_depth(
        result,
        normal,
        depth
    )

    # ========================================================
    # 5. DEHAZE
    # ========================================================

    result = apply_dehaze(
        result,
        normal,
        dehaze
    )

    # ========================================================
    # 6. CONTRAST
    # ========================================================

    result = apply_contrast(
        result,
        contrast
    )

    # ========================================================
    # 7. SATURATION
    # ========================================================

    result = apply_saturation(
        result,
        saturation
    )

    # ========================================================
    # 8. COLOR
    # ========================================================

    result = apply_color(
        result,
        normal,
        color
    )

    # ========================================================
    # FINAL SAFETY
    # ========================================================

    result = np.nan_to_num(
        result,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # AFTER
    # --------------------------------------------------------

    after = metrics(
        result
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_image(
        output_path,
        result
    )

    diagnostics = {

        "feedback_values":
            values,

        "before":
            before,

        "after":
            after,

        "brightness_difference":
            (
                after[
                    "mean_brightness"
                ]
                -
                before[
                    "mean_brightness"
                ]
            ),

        "shadow_ratio_difference":
            (
                after[
                    "shadow_ratio"
                ]
                -
                before[
                    "shadow_ratio"
                ]
            ),

        "highlight_ratio_difference":
            (
                after[
                    "highlight_ratio"
                ]
                -
                before[
                    "highlight_ratio"
                ]
            ),

        "local_detail_difference":
            (
                after[
                    "local_detail"
                ]
                -
                before[
                    "local_detail"
                ]
            ),

        "output":
            output_path
    }

    return diagnostics


# ============================================================
# GEMINI TEXT COMPATIBILITY
# ============================================================

def refine(
    selected_image_path,
    normal_path,
    under_path,
    over_path,
    feedback,
    output_path
):
    """
    Convenience wrapper:

        customer text
             ↓
        Gemini translator
             ↓
        numeric values
             ↓
        correction engine
    """

    from adaptive_engine.gemini_feedback import (
        translate_feedback
    )

    values = translate_feedback(
        feedback
    )

    return refine_with_values(
        selected_image_path,
        normal_path,
        under_path,
        over_path,
        values,
        output_path
    )


# ============================================================
# CLI
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "FreeMEF Normal-reference "
            "correction engine V2"
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
        "--output",
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
            args.color
    }

    result = refine_with_values(
        args.selected,
        args.normal,
        args.under,
        args.over,
        values,
        args.output
    )

    print()
    print("=" * 70)
    print("CORRECTION ENGINE V2")
    print("=" * 70)

    print()
    print("PARAMETERS")

    for key, value in values.items():

        print(
            f"{key:<14}: {value:+.3f}"
        )

    print()
    print("METRICS")

    print(
        f"Brightness       : "
        f"{result['before']['mean_brightness']:.6f}"
        f" -> "
        f"{result['after']['mean_brightness']:.6f}"
    )

    print(
        f"Shadow ratio     : "
        f"{result['before']['shadow_ratio']:.6f}"
        f" -> "
        f"{result['after']['shadow_ratio']:.6f}"
    )

    print(
        f"Highlight ratio  : "
        f"{result['before']['highlight_ratio']:.6f}"
        f" -> "
        f"{result['after']['highlight_ratio']:.6f}"
    )

    print(
        f"Local detail     : "
        f"{result['before']['local_detail']:.6f}"
        f" -> "
        f"{result['after']['local_detail']:.6f}"
    )

    print()
    print(
        f"Output:\n"
        f"  {result['output']}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
