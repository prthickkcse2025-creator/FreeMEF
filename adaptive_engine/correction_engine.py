#!/usr/bin/env python3

"""
FREE MEF CORRECTION ENGINE
==========================

Human/client-guided image refinement engine.

Pipeline:

    Candidate A/B/C
          |
          v
    Customer feedback
          |
          v
        Gemini
          |
          v
    numeric parameters
          |
          v
    this correction engine
          |
          v
      revised image

Design principles
-----------------
1. Never replace the selected candidate globally with another
   exposure just because a correction was requested.

2. Normal exposure is used primarily as a structural/color
   reference.

3. Each correction is spatially targeted.

4. Corrections are intentionally visible but controlled.

5. Revisions can be chained:
       revision 1 -> revision 2 -> revision 3
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
            f"Could not save image:\n{path}"
        )


# ============================================================
# BASIC IMAGE HELPERS
# ============================================================

def luminance(image):
    """
    OpenCV BGR luminance.
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


def normalize_map(x):
    x = np.nan_to_num(
        x,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    lo = float(
        np.percentile(
            x,
            2
        )
    )

    hi = float(
        np.percentile(
            x,
            98
        )
    )

    if hi - lo < EPS:
        return np.zeros_like(
            x,
            dtype=np.float32
        )

    return np.clip(
        (
            x - lo
        )
        /
        (
            hi - lo
        ),
        0.0,
        1.0
    ).astype(
        np.float32
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
        (3.0 - 2.0 * t)
    ).astype(
        np.float32
    )


# ============================================================
# COLOR SPACE HELPERS
# ============================================================

def bgr_to_lab(image):
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

    lab = cv2.cvtColor(
        image8,
        cv2.COLOR_BGR2LAB
    ).astype(
        np.float32
    )

    return lab


def lab_to_bgr(lab):
    lab = np.clip(
        lab,
        0.0,
        255.0
    ).astype(
        np.uint8
    )

    bgr = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return (
        bgr.astype(
            np.float32
        )
        / 255.0
    )


# ============================================================
# TONE CURVE
# ============================================================

def apply_luminance_curve(
    image,
    amount
):
    """
    Controlled luminance-only adjustment.

    amount:
        negative -> darker
        positive -> brighter

    Uses a mild power curve instead of raw addition.
    """

    amount = float(
        np.clip(
            amount,
            -1.0,
            1.0
        )
    )

    if abs(amount) < EPS:
        return image

    lum = luminance(
        image
    )

    # Deliberately controlled response.
    #
    # -1 -> gamma about 1.22
    # +1 -> gamma about 0.82
    gamma = (
        1.0
        -
        0.20 * amount
    )

    gamma = max(
        gamma,
        0.65
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
# SHADOW REGION MAP
# ============================================================

def shadow_region(
    lum
):
    """
    Stronger in lifted shadows.
    Very dark pixels are protected from excessive crushing.

    0.0 = not shadow
    1.0 = likely lifted shadow
    """

    # Main target: approximately 0.20 - 0.42
    mid_shadow = (
        smoothstep(
            lum,
            0.18,
            0.28
        )
        *
        (
            1.0
            -
            smoothstep(
                lum,
                0.36,
                0.48
            )
        )
    )

    # Protect very deep blacks.
    deep_protection = (
        1.0
        -
        smoothstep(
            lum,
            0.05,
            0.18
        )
    )

    result = (
        mid_shadow
        *
        (
            1.0
            -
            0.65 *
            deep_protection
        )
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# HIGHLIGHT REGION MAP
# ============================================================

def highlight_region(
    lum
):
    """
    Targets bright regions.
    """

    result = smoothstep(
        lum,
        0.68,
        0.96
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# MIDTONE REGION
# ============================================================

def midtone_region(
    lum
):
    """
    Focuses depth enhancement around natural midtones.
    """

    result = (
        smoothstep(
            lum,
            0.20,
            0.36
        )
        *
        (
            1.0
            -
            smoothstep(
                lum,
                0.68,
                0.82
            )
        )
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# LOCAL CONTRAST
# ============================================================

def local_contrast(
    lum,
    sigma=4.0
):
    """
    High-pass luminance structure.
    """

    base = gaussian(
        lum,
        sigma
    )

    detail = (
        lum -
        base
    )

    return detail.astype(
        np.float32
    )


# ============================================================
# APPLY SHADOW CORRECTION
# ============================================================

def apply_shadow_change(
    image,
    direction
):
    """
    direction:

        < 0:
            reduce lifted/open shadows

        > 0:
            recover/open shadows

    The correction is localized to lifted shadows.
    """

    direction = float(
        np.clip(
            direction,
            -1.0,
            1.0
        )
    )

    if abs(direction) < EPS:
        return image

    lum = luminance(
        image
    )

    mask = shadow_region(
        lum
    )

    # --------------------------------------------------------
    # Reduce lifted shadows
    # --------------------------------------------------------

    if direction < 0:

        strength = (
            0.28 *
            abs(direction)
        )

        # Darken through luminance scaling.
        factor = (
            1.0
            -
            strength * mask
        )

        new_lum = (
            lum *
            factor
        )

    # --------------------------------------------------------
    # Recover shadows
    # --------------------------------------------------------

    else:

        strength = (
            0.22 *
            direction
        )

        # Smooth lift instead of simple addition.
        new_lum = (
            lum
            +
            strength *
            mask *
            (
                1.0 -
                lum
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
# APPLY HIGHLIGHT CORRECTION
# ============================================================

def apply_highlight_change(
    image,
    direction
):
    """
    direction:

        < 0:
            reduce bright/lifted highlights

        > 0:
            recover highlights conservatively
    """

    direction = float(
        np.clip(
            direction,
            -1.0,
            1.0
        )
    )

    if abs(direction) < EPS:
        return image

    lum = luminance(
        image
    )

    mask = highlight_region(
        lum
    )

    if direction < 0:

        strength = (
            0.20 *
            abs(direction)
        )

        factor = (
            1.0
            -
            strength *
            mask
        )

        new_lum = (
            lum *
            factor
        )

    else:

        strength = (
            0.08 *
            direction
        )

        new_lum = (
            lum
            +
            strength *
            mask *
            (
                1.0 -
                lum
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
# APPLY DEPTH
# ============================================================

def apply_depth(
    image,
    normal,
    direction
):
    """
    Adds natural midtone structure from the Normal exposure.

    This is deliberately not a sharpening filter.

    The Normal exposure provides the local luminance structure.
    """

    direction = float(
        np.clip(
            direction,
            -1.0,
            1.0
        )
    )

    if abs(direction) < EPS:
        return image

    current_lum = luminance(
        image
    )

    normal_lum = luminance(
        normal
    )

    mask = midtone_region(
        normal_lum
    )

    # --------------------------------------------------------
    # Structural difference from Normal.
    # --------------------------------------------------------

    normal_local = local_contrast(
        normal_lum,
        sigma=5.0
    )

    # Normalize conservatively.
    structural = np.tanh(
        normal_local *
        10.0
    )

    # --------------------------------------------------------
    # Moderate depth strength.
    # --------------------------------------------------------

    strength = (
        0.16 *
        direction
    )

    correction = (
        strength
        *
        mask
        *
        structural
    )

    new_lum = (
        current_lum
        +
        correction
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
# APPLY DEHAZE
# ============================================================

def apply_dehaze(
    image,
    direction
):
    """
    Local haze reduction.

    This does NOT globally brighten or darken the image.

    It increases tonal separation mainly in midtone regions.
    """

    direction = float(
        np.clip(
            direction,
            -1.0,
            1.0
        )
    )

    if abs(direction) < EPS:
        return image

    lum = luminance(
        image
    )

    # --------------------------------------------------------
    # Large-scale illumination.
    # --------------------------------------------------------

    large = gaussian(
        lum,
        15.0
    )

    local = gaussian(
        lum,
        3.0
    )

    # Local structural component.
    detail = (
        lum -
        local
    )

    # Large-scale tonal separation.
    separation = (
        lum -
        large
    )

    # Midtone targeting.
    mask = midtone_region(
        lum
    )

    # Avoid making already-dark regions artificially darker.
    safe_positive = (
        np.clip(
            detail,
            -0.10,
            0.10
        )
    )

    tonal_component = np.clip(
        separation,
        -0.20,
        0.20
    )

    strength = (
        0.18 *
        direction
    )

    correction = (
        strength
        *
        mask
        *
        (
            0.65 *
            safe_positive
            +
            0.35 *
            tonal_component
        )
    )

    new_lum = (
        lum
        +
        correction
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
# APPLY CONTRAST
# ============================================================

def apply_contrast(
    image,
    direction
):

    direction = float(
        np.clip(
            direction,
            -1.0,
            1.0
        )
    )

    if abs(direction) < EPS:
        return image

    lum = luminance(
        image
    )

    factor = (
        1.0
        +
        0.15 *
        direction
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
# APPLY SATURATION
# ============================================================

def apply_saturation(
    image,
    direction
):

    direction = float(
        np.clip(
            direction,
            -1.0,
            1.0
        )
    )

    if abs(direction) < EPS:
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
        direction
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
# COLOR CORRECTION
# ============================================================

def apply_color_correction(
    image,
    normal,
    direction
):
    """
    Conservative color anchoring.

    The Normal image is the reference.
    """

    direction = float(
        np.clip(
            direction,
            -1.0,
            1.0
        )
    )

    if abs(direction) < EPS:
        return image

    # --------------------------------------------------------
    # Convert to LAB.
    # --------------------------------------------------------

    image_lab = bgr_to_lab(
        image
    )

    normal_lab = bgr_to_lab(
        normal
    )

    # --------------------------------------------------------
    # Blend chroma toward Normal.
    # --------------------------------------------------------

    amount = (
        0.30 *
        max(
            direction,
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

    result = lab_to_bgr(
        image_lab
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# NATURAL COLOR ANCHOR
# ============================================================

def preserve_normal_chroma(
    image,
    normal,
    strength=0.08
):
    """
    Tiny chroma protection step.

    Does not force brightness to Normal.
    Only gently stabilizes color.
    """

    image_lab = bgr_to_lab(
        image
    )

    normal_lab = bgr_to_lab(
        normal
    )

    strength = float(
        np.clip(
            strength,
            0.0,
            1.0
        )
    )

    image_lab[:, :, 1] = (
        (
            1.0 -
            strength
        )
        *
        image_lab[:, :, 1]
        +
        strength *
        normal_lab[:, :, 1]
    )

    image_lab[:, :, 2] = (
        (
            1.0 -
            strength
        )
        *
        image_lab[:, :, 2]
        +
        strength *
        normal_lab[:, :, 2]
    )

    return lab_to_bgr(
        image_lab
    )


# ============================================================
# FINAL SAFETY
# ============================================================

def final_safety(
    image,
    normal
):
    """
    Prevent extreme tonal drift.

    This does NOT force the whole image back to Normal.
    """

    image_lum = luminance(
        image
    )

    normal_lum = luminance(
        normal
    )

    # --------------------------------------------------------
    # Prevent excessively bright highlights.
    # --------------------------------------------------------

    highlight = (
        image_lum >
        0.97
    )

    safe_high = np.minimum(
        image_lum,
        normal_lum +
        0.04
    )

    high_ratio = (
        safe_high
        /
        (
            image_lum + EPS
        )
    )

    image = np.where(
        highlight[:, :, None],
        image *
        high_ratio[:, :, None],
        image
    )

    # --------------------------------------------------------
    # Prevent very large global drift.
    #
    # Compare mean luminance but only correct a portion of
    # the difference.
    # --------------------------------------------------------

    current_mean = float(
        image_lum.mean()
    )

    normal_mean = float(
        normal_lum.mean()
    )

    delta = (
        current_mean -
        normal_mean
    )

    # Only extremely large drifts are partially corrected.
    if abs(delta) > 0.14:

        correction = (
            0.25 *
            delta
        )

        new_lum = luminance(
            image
        ) - correction

        ratio = (
            new_lum
            /
            (
                luminance(image)
                + EPS
            )
        )

        image = (
            image *
            ratio[:, :, None]
        )

    return np.clip(
        image,
        0.0,
        1.0
    )


# ============================================================
# METRICS
# ============================================================

def image_metrics(
    image
):

    lum = luminance(
        image
    )

    detail = local_contrast(
        lum,
        sigma=2.0
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
# MAIN NUMERIC REFINEMENT
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
    Apply Gemini-generated numeric correction values.
    """

    selected = read_image(
        selected_image_path
    )

    normal = read_image(
        normal_path
    )

    under = read_image(
        under_path
    )

    over = read_image(
        over_path
    )

    normal = resize_like(
        normal,
        selected
    )

    under = resize_like(
        under,
        selected
    )

    over = resize_like(
        over,
        selected
    )

    # --------------------------------------------------------
    # Safe parameters
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
    # BEFORE metrics
    # --------------------------------------------------------

    before_metrics = image_metrics(
        selected
    )

    result = selected.copy()

    # ========================================================
    # 1. OVERALL BRIGHTNESS
    # ========================================================

    result = apply_luminance_curve(
        result,
        brightness
    )

    # ========================================================
    # 2. SHADOWS
    # ========================================================

    result = apply_shadow_change(
        result,
        shadow
    )

    # ========================================================
    # 3. HIGHLIGHTS
    # ========================================================

    result = apply_highlight_change(
        result,
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

    result = apply_color_correction(
        result,
        normal,
        color
    )

    # ========================================================
    # 9. SMALL NORMAL COLOR ANCHOR
    # ========================================================

    result = preserve_normal_chroma(
        result,
        normal,
        strength=0.05
    )

    # ========================================================
    # 10. FINAL SAFETY
    # ========================================================

    result = final_safety(
        result,
        normal
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # AFTER metrics
    # --------------------------------------------------------

    after_metrics = image_metrics(
        result
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_image(
        output_path,
        result
    )

    diagnostics = {

        "feedback_values":
            values,

        "before":
            before_metrics,

        "after":
            after_metrics,

        "brightness_difference":
            (
                after_metrics[
                    "mean_brightness"
                ]
                -
                before_metrics[
                    "mean_brightness"
                ]
            ),

        "local_detail_difference":
            (
                after_metrics[
                    "local_detail"
                ]
                -
                before_metrics[
                    "local_detail"
                ]
            ),

        "shadow_ratio_difference":
            (
                after_metrics[
                    "shadow_ratio"
                ]
                -
                before_metrics[
                    "shadow_ratio"
                ]
            ),

        "highlight_ratio_difference":
            (
                after_metrics[
                    "highlight_ratio"
                ]
                -
                before_metrics[
                    "highlight_ratio"
                ]
            ),

        "output":
            output_path
    }

    return diagnostics


# ============================================================
# BACKWARD-COMPATIBLE TEXT REFINEMENT
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
    Compatibility wrapper.

    This function first attempts the local parser from
    gemini_feedback.py only when needed.

    Preferred path:
        Gemini -> refine_with_values()
    """

    try:

        from adaptive_engine.gemini_feedback import (
            translate_feedback
        )

    except Exception as exc:

        raise RuntimeError(
            "Could not import Gemini feedback translator."
        ) from exc

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
# CLI TEST
# ============================================================

def main():

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "FreeMEF correction engine"
        )
    )

    parser.add_argument(
        "--selected",
        required=True,
        help="Selected candidate image"
    )

    parser.add_argument(
        "--normal",
        required=True,
        help="Normal exposure"
    )

    parser.add_argument(
        "--under",
        required=True,
        help="Under exposure"
    )

    parser.add_argument(
        "--over",
        required=True,
        help="Over exposure"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output image"
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
    print("CORRECTION ENGINE")
    print("=" * 70)

    print()

    print("Parameters:")

    for key, value in values.items():

        print(
            f"  {key:<12}: "
            f"{value:+.3f}"
        )

    print()

    print(
        "Before mean brightness : "
        f"{result['before']['mean_brightness']:.6f}"
    )

    print(
        "After mean brightness  : "
        f"{result['after']['mean_brightness']:.6f}"
    )

    print(
        "Brightness difference  : "
        f"{result['brightness_difference']:+.6f}"
    )

    print()

    print(
        "Before local detail    : "
        f"{result['before']['local_detail']:.6f}"
    )

    print(
        "After local detail     : "
        f"{result['after']['local_detail']:.6f}"
    )

    print(
        "Detail difference      : "
        f"{result['local_detail_difference']:+.6f}"
    )

    print()

    print(
        "Before shadow ratio    : "
        f"{result['before']['shadow_ratio']:.6f}"
    )

    print(
        "After shadow ratio     : "
        f"{result['after']['shadow_ratio']:.6f}"
    )

    print(
        "Shadow ratio change    : "
        f"{result['shadow_ratio_difference']:+.6f}"
    )

    print()

    print(
        "Before highlight ratio : "
        f"{result['before']['highlight_ratio']:.6f}"
    )

    print(
        "After highlight ratio  : "
        f"{result['after']['highlight_ratio']:.6f}"
    )

    print(
        "Highlight ratio change : "
        f"{result['highlight_ratio_difference']:+.6f}"
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
