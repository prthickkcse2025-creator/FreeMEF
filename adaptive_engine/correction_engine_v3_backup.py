#!/usr/bin/env python3

"""
FREE MEF - CORRECTION ENGINE V3
===============================

Client-guided image refinement engine.

Pipeline:

    Candidate A / B / C
            |
            v
      Customer selects
            |
            v
      Customer feedback
            |
            v
          Gemini
            |
            v
    Numeric correction values
            |
            v
    Correction Engine V3
            |
            v
        Revised image


Design
------
The selected candidate remains the base image.

The Normal exposure is used as a reference for:
    - lifted shadow correction
    - highlight correction
    - depth restoration
    - dehaze / tonal separation

Under exposure is used for:
    - highlight recovery

Over exposure is used for:
    - shadow recovery

The engine does not choose A/B/C.
The engine does not interpret customer text.
Gemini performs text -> numeric translation.
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
        image.astype(
            np.float32
        )
        /
        255.0
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


def luminance(
    image
):
    """
    BGR -> luminance.
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
            3.0 -
            2.0 * t
        )
    ).astype(
        np.float32
    )


# ============================================================
# COLOR SPACE HELPERS
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

    image = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return (
        image.astype(
            np.float32
        )
        /
        255.0
    )


# ============================================================
# REGION MAPS
# ============================================================

def shadow_map(
    normal_lum
):
    """
    Detect lifted shadow / lower-mid areas.

    Deep blacks are protected.
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

    deep_protection = (
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
            deep_protection
        )
    )

    return np.clip(
        cv2.GaussianBlur(
            result.astype(
                np.float32
            ),
            (0, 0),
            3.0
        ),
        0.0,
        1.0
    )


def highlight_map(
    normal_lum
):
    """
    Detect bright regions.
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
    Natural midtone region.
    """

    lower = smoothstep(
        normal_lum,
        0.20,
        0.38
    )

    upper = (
        1.0
        -
        smoothstep(
            normal_lum,
            0.70,
            0.86
        )
    )

    return np.clip(
        lower * upper,
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
    Controlled global luminance change.

    value:
        -1 darker
        +1 brighter
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
            lum +
            EPS
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
    Negative shadow correction.

    Move lifted candidate shadows toward Normal.

    Important:
        already-dark pixels are protected.
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

    # Only pull part of the difference toward Normal.
    amount = (
        0.90
        *
        abs(value)
        *
        mask
    )

    new_lum = (
        current_lum
        *
        (
            1.0 -
            amount
        )
        +
        normal_lum
        *
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
            current_lum +
            EPS
        )
    )

    return np.clip(
        image *
        ratio[:, :, None],
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
    Positive shadow correction.

    Uses Over only where it contains useful extra shadow
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

    over_lum = luminance(
        over
    )

    mask = shadow_map(
        normal_lum
    )

    gain = np.clip(
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
        gain
    )

    new_lum = (
        current_lum
        *
        (
            1.0 -
            amount
        )
        +
        over_lum
        *
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
            current_lum +
            EPS
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
    Negative highlight correction.

    Moves bright/lifted regions toward Normal.
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
        0.80
        *
        abs(value)
        *
        mask
    )

    new_lum = (
        current_lum
        *
        (
            1.0 -
            amount
        )
        +
        normal_lum
        *
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
            current_lum +
            EPS
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
    Positive highlight recovery.

    Uses Under only where it contains useful darker
    highlight information.
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

    gain = np.clip(
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
        gain
    )

    new_lum = (
        current_lum
        *
        (
            1.0 -
            amount
        )
        +
        under_lum
        *
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
            current_lum +
            EPS
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
    Normal-reference depth restoration.

    We calculate the difference between:
        Normal local structure
        Candidate local structure

    Then restore only a controlled portion of the missing
    structure in natural midtones.
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

    # --------------------------------------------------------
    # Multi-scale structures
    # --------------------------------------------------------

    current_small = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        2.5
    )

    current_large = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        6.0
    )

    normal_small = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        2.5
    )

    normal_large = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        6.0
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

    missing = (
        normal_structure
        -
        current_structure
    )

    # Prevent large edge overshoot.
    missing = np.clip(
        missing,
        -0.040,
        0.040
    )

    # --------------------------------------------------------
    # Midtone mask
    # --------------------------------------------------------

    midtone = midtone_map(
        normal_lum
    )

    # --------------------------------------------------------
    # Gradient confidence
    #
    # Encourage real structure, but don't create giant halos.
    # --------------------------------------------------------

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
        gx * gx +
        gy * gy
    )

    confidence = np.clip(
        gradient /
        0.20,
        0.20,
        1.0
    )

    # --------------------------------------------------------
    # Apply depth
    # --------------------------------------------------------

    strength = (
        0.55 *
        value
    )

    correction = (
        strength
        *
        midtone
        *
        confidence
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
            current_lum +
            EPS
        )
    )

    return np.clip(
        image *
        ratio[:, :, None],
        0.0,
        1.0
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
    Reference-guided dehaze / local contrast restoration.

    Positive value:
        Restores lost local and broad tonal separation
        using the Normal exposure as a reference.

    Value range:
        0.0 -> no correction
        0.2 -> subtle correction
        0.5 -> natural visible correction
        1.0 -> strong correction

    Shadows and highlights are protected to avoid
    crushing blacks or clipping bright regions.
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

    # ========================================================
    # MULTI-SCALE LUMINANCE
    # ========================================================

    current_small = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        2.0
    )

    current_medium = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        8.0
    )

    current_large = cv2.GaussianBlur(
        current_lum,
        (0, 0),
        24.0
    )

    normal_small = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        2.0
    )

    normal_medium = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        8.0
    )

    normal_large = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        24.0
    )

    # ========================================================
    # LOCAL DETAIL BANDS
    # ========================================================

    current_fine = (
        current_lum
        -
        current_small
    )

    normal_fine = (
        normal_lum
        -
        normal_small
    )

    current_local = (
        current_small
        -
        current_medium
    )

    normal_local = (
        normal_small
        -
        normal_medium
    )

    current_broad = (
        current_medium
        -
        current_large
    )

    normal_broad = (
        normal_medium
        -
        normal_large
    )

    # ========================================================
    # REFERENCE-GUIDED SEPARATION DIFFERENCE
    # ========================================================

    fine_difference = (
        normal_fine
        -
        current_fine
    )

    local_difference = (
        normal_local
        -
        current_local
    )

    broad_difference = (
        normal_broad
        -
        current_broad
    )

    # Prevent extreme reference corrections
    fine_difference = np.clip(
        fine_difference,
        -0.060,
        0.060
    )

    local_difference = np.clip(
        local_difference,
        -0.080,
        0.080
    )

    broad_difference = np.clip(
        broad_difference,
        -0.100,
        0.100
    )

    # ========================================================
    # COMBINE MULTI-SCALE CORRECTION
    # ========================================================

    correction = (
        0.20
        *
        fine_difference
        +
        0.50
        *
        local_difference
        +
        0.30
        *
        broad_difference
    )

    # ========================================================
    # MIDTONE PRIORITY
    # ========================================================

    midtone = midtone_map(
        current_lum
    )

    # ========================================================
    # SHADOW PROTECTION
    # ========================================================

    shadow_protection = smoothstep(
        current_lum,
        0.05,
        0.22
    )

    # ========================================================
    # HIGHLIGHT PROTECTION
    # ========================================================

    highlight_protection = (
        1.0
        -
        smoothstep(
            current_lum,
            0.78,
            0.96
        )
    )

    protection = (
        shadow_protection
        *
        highlight_protection
    )

    # ========================================================
    # FLAT-AREA EMPHASIS
    #
    # Dehaze should primarily affect areas that have
    # lost tonal separation.
    # ========================================================

    local_strength = np.abs(
        current_local
    )

    broad_strength = np.abs(
        current_broad
    )

    texture_strength = (
        local_strength
        +
        0.50
        *
        broad_strength
    )

    flatness = 1.0 - np.clip(
        texture_strength
        /
        0.080,
        0.0,
        1.0
    )

    # Keep some correction in textured regions.
    adaptive_mask = (
        0.45
        +
        0.55
        *
        flatness
    )

    # ========================================================
    # FINAL CORRECTION MASK
    # ========================================================

    correction *= (
        midtone
        *
        protection
        *
        adaptive_mask
    )

    # ========================================================
    # USER CONTROL
    # ========================================================

    strength = (
        1.80
        *
        value
    )

    new_lum = (
        current_lum
        +
        strength
        *
        correction
    )

    new_lum = np.clip(
        new_lum,
        0.0,
        1.0
    )

    # ========================================================
    # APPLY LUMINANCE CHANGE TO RGB
    # ========================================================

    ratio = (
        new_lum
        /
        (
            current_lum
            +
            EPS
        )
    )

    result = (
        image
        *
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
            lum +
            EPS
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
# COLOR CORRECTION
# ============================================================

def apply_color(
    image,
    normal,
    value
):
    """
    Conservative chroma correction toward Normal.
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
        0.30 *
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

    blur = cv2.GaussianBlur(
        lum,
        (0, 0),
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
# NORMAL-REFERENCE DISTANCE METRICS
# ============================================================

def reference_metrics(
    image,
    normal
):
    """
    Measure how close the selected/revised image is to the
    Normal exposure in relevant regions.
    """

    img_lum = luminance(
        image
    )

    normal_lum = luminance(
        normal
    )

    # --------------------------------------------------------
    # Shadow region
    # --------------------------------------------------------

    shadow = (
        (normal_lum >= 0.18)
        &
        (normal_lum < 0.48)
    )

    if np.any(shadow):

        shadow_error = float(
            np.mean(
                np.abs(
                    img_lum[shadow]
                    -
                    normal_lum[shadow]
                )
            )
        )

    else:

        shadow_error = 0.0

    # --------------------------------------------------------
    # Midtone structure
    # --------------------------------------------------------

    img_blur = cv2.GaussianBlur(
        img_lum,
        (0, 0),
        4.0
    )

    normal_blur = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        4.0
    )

    img_detail = (
        img_lum -
        img_blur
    )

    normal_detail = (
        normal_lum -
        normal_blur
    )

    midtone = (
        (normal_lum >= 0.30)
        &
        (normal_lum <= 0.75)
    )

    if np.any(midtone):

        depth_error = float(
            np.mean(
                np.abs(
                    img_detail[midtone]
                    -
                    normal_detail[midtone]
                )
            )
        )

    else:

        depth_error = 0.0

    # --------------------------------------------------------
    # Local contrast / haze gap
    # --------------------------------------------------------

    img_local = np.abs(
        img_lum -
        cv2.GaussianBlur(
            img_lum,
            (0, 0),
            6.0
        )
    )

    normal_local = np.abs(
        normal_lum -
        cv2.GaussianBlur(
            normal_lum,
            (0, 0),
            6.0
        )
    )

    haze_region = (
        (normal_lum >= 0.25)
        &
        (normal_lum <= 0.80)
    )

    if np.any(haze_region):

        img_contrast = float(
            img_local[haze_region].mean()
        )

        normal_contrast = float(
            normal_local[haze_region].mean()
        )

        haze_gap = abs(
            img_contrast -
            normal_contrast
        )

    else:

        haze_gap = 0.0

    return {

        "shadow_distance_to_normal":
            shadow_error,

        "depth_structure_error":
            depth_error,

        "haze_contrast_gap":
            haze_gap
    }


# ============================================================
# COMPLETE REFINEMENT
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
    Apply Gemini numeric values to selected candidate.
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
    # Read numeric values
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

    before_basic = metrics(
        selected
    )

    before_reference = reference_metrics(
        selected,
        normal
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

    after_basic = metrics(
        result
    )

    after_reference = reference_metrics(
        result,
        normal
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_image(
        output_path,
        result
    )

    # ========================================================
    # DIAGNOSTICS
    # ========================================================

    diagnostics = {

        "feedback_values":
            values,

        "before":
            before_basic,

        "after":
            after_basic,

        "before_reference":
            before_reference,

        "after_reference":
            after_reference,

        "brightness_difference":
            (
                after_basic[
                    "mean_brightness"
                ]
                -
                before_basic[
                    "mean_brightness"
                ]
            ),

        "shadow_distance_change":
            (
                after_reference[
                    "shadow_distance_to_normal"
                ]
                -
                before_reference[
                    "shadow_distance_to_normal"
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
                    "haze_contrast_gap"
                ]
                -
                before_reference[
                    "haze_contrast_gap"
                ]
            ),

        "local_detail_difference":
            (
                after_basic[
                    "local_detail"
                ]
                -
                before_basic[
                    "local_detail"
                ]
            ),

        "output":
            output_path
    }

    return diagnostics


# ============================================================
# TEXT -> GEMINI -> ENGINE
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
    Convenience wrapper.

    Customer text
        ↓
    Gemini
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
            "FreeMEF Correction Engine V3"
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
    print("FREE MEF - CORRECTION ENGINE V3")
    print("=" * 70)

    print()
    print("PARAMETERS")

    for key, value in values.items():

        print(
            f"{key:<14}: "
            f"{value:+.3f}"
        )

    print()
    print("BASIC METRICS")

    print(
        f"Brightness       : "
        f"{result['before']['mean_brightness']:.6f}"
        f" -> "
        f"{result['after']['mean_brightness']:.6f}"
    )

    print(
        f"Median           : "
        f"{result['before']['median_brightness']:.6f}"
        f" -> "
        f"{result['after']['median_brightness']:.6f}"
    )

    print(
        f"Shadow ratio     : "
        f"{result['before']['shadow_ratio']:.6f}"
        f" -> "
        f"{result['after']['shadow_ratio']:.6f}"
    )

    print(
        f"Deep shadow      : "
        f"{result['before']['deep_shadow_ratio']:.6f}"
        f" -> "
        f"{result['after']['deep_shadow_ratio']:.6f}"
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
    print("NORMAL-REFERENCE METRICS")

    print(
        f"Shadow distance  : "
        f"{result['before_reference']['shadow_distance_to_normal']:.6f}"
        f" -> "
        f"{result['after_reference']['shadow_distance_to_normal']:.6f}"
    )

    print(
        f"Depth error      : "
        f"{result['before_reference']['depth_structure_error']:.6f}"
        f" -> "
        f"{result['after_reference']['depth_structure_error']:.6f}"
    )

    print(
        f"Haze gap         : "
        f"{result['before_reference']['haze_contrast_gap']:.6f}"
        f" -> "
        f"{result['after_reference']['haze_contrast_gap']:.6f}"
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

    print()
    print(
        f"OUTPUT:\n"
        f"  {result['output']}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
