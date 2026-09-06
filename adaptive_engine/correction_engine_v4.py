import argparse
import os
import cv2
import numpy as np

EPS = 1e-6


# ============================================================
# BASIC HELPERS
# ============================================================

def clamp(x):
    return float(np.clip(float(x), -1.0, 1.0))


def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise RuntimeError(f"Could not read: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.float32) / 255.0


def save_image(path, image):

    directory = os.path.dirname(path)

    if directory:
        os.makedirs(directory, exist_ok=True)

    output = np.clip(
        image * 255.0,
        0,
        255
    ).astype(np.uint8)

    output = cv2.cvtColor(
        output,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        path,
        output,
        [cv2.IMWRITE_JPEG_QUALITY, 100]
    )


def luminance(image):

    return (
        0.2126 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.0722 * image[:, :, 2]
    ).astype(np.float32)


def smoothstep(x, a, b):

    t = np.clip(
        (x - a) / (b - a + EPS),
        0.0,
        1.0
    )

    return t * t * (3.0 - 2.0 * t)


def midtone_map(lum):

    return np.clip(
        1.0 - np.abs(lum - 0.5) / 0.5,
        0.0,
        1.0
    )


def shadow_map(lum):

    return 1.0 - smoothstep(
        lum,
        0.10,
        0.55
    )


def highlight_map(lum):

    return smoothstep(
        lum,
        0.55,
        0.95
    )


def apply_luminance(
    image,
    old_lum,
    new_lum
):

    ratio = new_lum / (
        old_lum + EPS
    )

    result = image * ratio[:, :, None]

    return np.clip(
        result,
        0.0,
        1.0
    )


def structure(
    lum,
    small,
    large
):

    a = cv2.GaussianBlur(
        lum,
        (0, 0),
        small
    )

    b = cv2.GaussianBlur(
        lum,
        (0, 0),
        large
    )

    return a - b


# ============================================================
# BRIGHTNESS
# ============================================================

def apply_brightness(
    image,
    value
):

    value = clamp(value)

    if abs(value) < EPS:
        return image.copy()

    lum = luminance(image)

    if value > 0:

        new_lum = (
            lum
            + 0.18
            * value
            * (1.0 - lum)
        )

    else:

        new_lum = (
            lum
            * (
                1.0
                - 0.18
                * abs(value)
            )
        )

    return apply_luminance(
        image,
        lum,
        np.clip(new_lum, 0, 1)
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

    value = clamp(value)

    if abs(value) < EPS:
        return image.copy()

    current = luminance(image)

    normal_lum = luminance(normal)

    over_lum = luminance(over)

    mask = shadow_map(
        normal_lum
    )

    if value < 0:

        target = normal_lum

        amount = (
            0.95
            * abs(value)
            * mask
        )

    else:

        target = over_lum

        amount = (
            0.75
            * value
            * mask
        )

    new_lum = (
        current
        * (1.0 - amount)
        + target
        * amount
    )

    return apply_luminance(
        image,
        current,
        np.clip(new_lum, 0, 1)
    )


# ============================================================
# HIGHLIGHT
# ============================================================

def apply_highlight(
    image,
    normal,
    under,
    value
):

    value = clamp(value)

    if abs(value) < EPS:
        return image.copy()

    current = luminance(image)

    normal_lum = luminance(normal)

    under_lum = luminance(under)

    mask = highlight_map(
        normal_lum
    )

    if value > 0:

        target = under_lum

        amount = (
            0.85
            * value
            * mask
        )

    else:

        target = normal_lum

        amount = (
            0.55
            * abs(value)
            * mask
        )

    new_lum = (
        current
        * (1.0 - amount)
        + target
        * amount
    )

    return apply_luminance(
        image,
        current,
        np.clip(new_lum, 0, 1)
    )


# ============================================================
# DEPTH - IMPROVED
# ============================================================

def apply_depth(
    image,
    normal,
    value
):

    value = clamp(value)

    if abs(value) < EPS:
        return image.copy()

    current = luminance(image)

    normal_lum = luminance(normal)

    # Negative depth softens structure
    if value < 0:

        blur = cv2.GaussianBlur(
            current,
            (0, 0),
            3.0
        )

        strength = (
            0.70
            * abs(value)
        )

        new_lum = (
            current
            * (1.0 - strength)
            + blur
            * strength
        )

        return apply_luminance(
            image,
            current,
            np.clip(new_lum, 0, 1)
        )

    # Fine structure
    current_fine = structure(
        current,
        0.8,
        2.5
    )

    normal_fine = structure(
        normal_lum,
        0.8,
        2.5
    )

    # Medium structure
    current_medium = structure(
        current,
        2.5,
        8.0
    )

    normal_medium = structure(
        normal_lum,
        2.5,
        8.0
    )

    fine_difference = np.clip(
        normal_fine
        - current_fine,
        -0.10,
        0.10
    )

    medium_difference = np.clip(
        normal_medium
        - current_medium,
        -0.14,
        0.14
    )

    midtone = midtone_map(
        normal_lum
    )

    protection = (
        0.30
        + 0.70
        * midtone
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
        + gy * gy
    )

    confidence = np.clip(
        gradient / 0.10,
        0.25,
        1.0
    )

    correction = (
        0.55
        * fine_difference
        +
        0.95
        * medium_difference
    )

    strength = (
        1.20
        * value
    )

    new_lum = (
        current
        +
        strength
        * correction
        * protection
        * confidence
    )

    return apply_luminance(
        image,
        current,
        np.clip(new_lum, 0, 1)
    )


# ============================================================
# DEHAZE - IMPROVED
# ============================================================

def apply_dehaze(
    image,
    normal,
    value
):

    value = clamp(value)

    if abs(value) < EPS:
        return image.copy()

    current = luminance(image)

    normal_lum = luminance(normal)

    # Negative = intentionally softer image
    if value < 0:

        blur = cv2.GaussianBlur(
            current,
            (0, 0),
            5.0
        )

        strength = (
            0.75
            * abs(value)
        )

        new_lum = (
            current
            * (1.0 - strength)
            + blur
            * strength
        )

        return apply_luminance(
            image,
            current,
            np.clip(new_lum, 0, 1)
        )

    # Local tonal separation
    current_local = structure(
        current,
        1.5,
        5.0
    )

    normal_local = structure(
        normal_lum,
        1.5,
        5.0
    )

    # Broad tonal separation
    current_broad = structure(
        current,
        5.0,
        18.0
    )

    normal_broad = structure(
        normal_lum,
        5.0,
        18.0
    )

    local_difference = np.clip(
        normal_local
        - current_local,
        -0.12,
        0.12
    )

    broad_difference = np.clip(
        normal_broad
        - current_broad,
        -0.16,
        0.16
    )

    correction = (
        0.70
        * local_difference
        +
        1.00
        * broad_difference
    )

    shadow_protection = smoothstep(
        normal_lum,
        0.04,
        0.18
    )

    highlight_protection = (
        1.0
        -
        smoothstep(
            normal_lum,
            0.80,
            0.98
        )
    )

    protection = (
        shadow_protection
        * highlight_protection
    )

    midtone = midtone_map(
        normal_lum
    )

    mask = (
        0.25
        +
        0.75
        * midtone
        * protection
    )

    meaningful = np.clip(
        np.abs(correction)
        / 0.008,
        0.15,
        1.0
    )

    strength = (
        1.35
        * value
    )

    new_lum = (
        current
        +
        strength
        * correction
        * mask
        * meaningful
    )

    return apply_luminance(
        image,
        current,
        np.clip(new_lum, 0, 1)
    )


# ============================================================
# CONTRAST
# ============================================================

def apply_contrast(
    image,
    value
):

    value = clamp(value)

    if abs(value) < EPS:
        return image.copy()

    lum = luminance(image)

    if value >= 0:

        factor = (
            1.0
            + 0.45
            * value
        )

    else:

        factor = (
            1.0
            /
            (
                1.0
                + 0.65
                * abs(value)
            )
        )

    new_lum = (
        (lum - 0.5)
        * factor
        + 0.5
    )

    return apply_luminance(
        image,
        lum,
        np.clip(new_lum, 0, 1)
    )


# ============================================================
# SATURATION
# ============================================================

def apply_saturation(
    image,
    value
):

    value = clamp(value)

    if abs(value) < EPS:
        return image.copy()

    hsv = cv2.cvtColor(
        image.astype(np.float32),
        cv2.COLOR_RGB2HSV
    )

    if value >= 0:

        factor = (
            1.0
            + 0.60
            * value
        )

    else:

        factor = (
            1.0
            - 0.70
            * abs(value)
        )

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1]
        * factor,
        0,
        1
    )

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    )


# ============================================================
# COLOR
# ============================================================

def apply_color(
    image,
    normal,
    value
):

    value = clamp(value)

    if abs(value) < EPS:
        return image.copy()

    if value < 0:

        return apply_saturation(
            image,
            value * 0.60
        )

    hsv = cv2.cvtColor(
        image.astype(np.float32),
        cv2.COLOR_RGB2HSV
    )

    normal_hsv = cv2.cvtColor(
        normal.astype(np.float32),
        cv2.COLOR_RGB2HSV
    )

    amount = (
        0.50
        * value
    )

    dh = (
        normal_hsv[:, :, 0]
        - hsv[:, :, 0]
    )

    dh = (
        (dh + 90.0)
        % 180.0
        - 90.0
    )

    hsv[:, :, 0] = (
        hsv[:, :, 0]
        + amount
        * dh
    ) % 180.0

    hsv[:, :, 1] = np.clip(
        hsv[:, :, 1]
        * (1.0 - amount)
        +
        normal_hsv[:, :, 1]
        * amount,
        0,
        1
    )

    return cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2RGB
    )


# ============================================================
# METRICS
# ============================================================

def basic_metrics(image):

    lum = luminance(image)

    detail = (
        lum
        -
        cv2.GaussianBlur(
            lum,
            (0, 0),
            2.0
        )
    )

    return {

        "mean_brightness":
            float(np.mean(lum)),

        "median_brightness":
            float(np.median(lum)),

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

    shadow_mask_ref = shadow_map(
        normal_lum
    )

    shadow_distance = float(
        np.mean(
            np.abs(
                lum
                - normal_lum
            )
            * shadow_mask_ref
        )
        /
        (
            np.mean(shadow_mask_ref)
            + EPS
        )
    )

    structure_a = structure(
        lum,
        2.5,
        8.0
    )

    structure_b = structure(
        normal_lum,
        2.5,
        8.0
    )

    depth_error = float(
        np.mean(
            np.abs(
                structure_a
                - structure_b
            )
        )
    )

    local_a = structure(
        lum,
        1.5,
        5.0
    )

    broad_a = structure(
        lum,
        5.0,
        18.0
    )

    local_b = structure(
        normal_lum,
        1.5,
        5.0
    )

    broad_b = structure(
        normal_lum,
        5.0,
        18.0
    )

    contrast_a = (
        float(
            np.mean(
                np.abs(local_a)
            )
        )
        +
        float(
            np.mean(
                np.abs(broad_a)
            )
        )
    )

    contrast_b = (
        float(
            np.mean(
                np.abs(local_b)
            )
        )
        +
        float(
            np.mean(
                np.abs(broad_b)
            )
        )
    )

    haze_gap = abs(
        contrast_a
        - contrast_b
    )

    return {

        "shadow_distance":
            shadow_distance,

        "depth_structure_error":
            depth_error,

        "haze_gap":
            haze_gap
    }


# ============================================================
# MAIN ENGINE
# ============================================================

def refine_with_values(
    selected_path,
    normal_path,
    under_path,
    over_path,
    values,
    output_path
):

    selected = load_image(
        selected_path
    )

    normal = load_image(
        normal_path
    )

    under = load_image(
        under_path
    )

    over = load_image(
        over_path
    )

    h, w = selected.shape[:2]

    normal = cv2.resize(
        normal,
        (w, h)
    )

    under = cv2.resize(
        under,
        (w, h)
    )

    over = cv2.resize(
        over,
        (w, h)
    )

    controls = {

        "brightness":
            clamp(
                values.get(
                    "brightness",
                    0.0
                )
            ),

        "shadow":
            clamp(
                values.get(
                    "shadow",
                    0.0
                )
            ),

        "highlight":
            clamp(
                values.get(
                    "highlight",
                    0.0
                )
            ),

        "depth":
            clamp(
                values.get(
                    "depth",
                    0.0
                )
            ),

        "dehaze":
            clamp(
                values.get(
                    "dehaze",
                    0.0
                )
            ),

        "contrast":
            clamp(
                values.get(
                    "contrast",
                    0.0
                )
            ),

        "saturation":
            clamp(
                values.get(
                    "saturation",
                    0.0
                )
            ),

        "color":
            clamp(
                values.get(
                    "color",
                    0.0
                )
            )
    }

    before = basic_metrics(
        selected
    )

    before_reference = reference_metrics(
        selected,
        normal
    )

    result = selected.copy()

    result = apply_brightness(
        result,
        controls["brightness"]
    )

    result = apply_shadow(
        result,
        normal,
        over,
        controls["shadow"]
    )

    result = apply_highlight(
        result,
        normal,
        under,
        controls["highlight"]
    )

    result = apply_depth(
        result,
        normal,
        controls["depth"]
    )

    result = apply_dehaze(
        result,
        normal,
        controls["dehaze"]
    )

    result = apply_contrast(
        result,
        controls["contrast"]
    )

    result = apply_saturation(
        result,
        controls["saturation"]
    )

    result = apply_color(
        result,
        normal,
        controls["color"]
    )

    result = np.clip(
        result,
        0.0,
        1.0
    )

    after = basic_metrics(
        result
    )

    after_reference = reference_metrics(
        result,
        normal
    )

    save_image(
        output_path,
        result
    )

    return {

        "feedback_values":
            values,

        "before":
            before,

        "after":
            after,

        "before_reference":
            before_reference,

        "after_reference":
            after_reference,

        "brightness_difference":
            after["mean_brightness"]
            -
            before["mean_brightness"],

        "shadow_distance_change":
            after_reference["shadow_distance"]
            -
            before_reference["shadow_distance"],

        "depth_error_change":
            after_reference["depth_structure_error"]
            -
            before_reference["depth_structure_error"],

        "haze_gap_change":
            after_reference["haze_gap"]
            -
            before_reference["haze_gap"],

        "local_detail_difference":
            after["local_detail"]
            -
            before["local_detail"],

        "output":
            output_path
    }


# ============================================================
# GEMINI COMPATIBILITY
# ============================================================

def refine(
    selected_path,
    normal_path,
    under_path,
    over_path,
    feedback,
    output_path
):

    try:

        from adaptive_engine.gemini_feedback import (
            translate_feedback
        )

    except ImportError:

        from gemini_feedback import (
            translate_feedback
        )

    values = translate_feedback(
        feedback
    )

    return refine_with_values(
        selected_path,
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

    parser = argparse.ArgumentParser()

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
        "--output",
        required=True
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

    print()
    print("=" * 70)
    print(
        "FREE MEF - CORRECTION ENGINE V4"
    )
    print("=" * 70)

    print("\nPARAMETERS")

    for key, value in values.items():

        print(
            f"{key:14s}: "
            f"{value:+.3f}"
        )

    result = refine_with_values(

        args.selected,

        args.normal,

        args.under,

        args.over,

        values,

        args.output
    )

    before = result["before"]

    after = result["after"]

    before_ref = result[
        "before_reference"
    ]

    after_ref = result[
        "after_reference"
    ]

    print("\nBASIC METRICS")

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

    print("\nNORMAL-REFERENCE METRICS")

    print(
        f"Shadow distance  : "
        f"{before_ref['shadow_distance']:.6f} -> "
        f"{after_ref['shadow_distance']:.6f}"
    )

    print(
        f"Depth error      : "
        f"{before_ref['depth_structure_error']:.6f} -> "
        f"{after_ref['depth_structure_error']:.6f}"
    )

    print(
        f"Haze gap         : "
        f"{before_ref['haze_gap']:.6f} -> "
        f"{after_ref['haze_gap']:.6f}"
    )

    print("\nCHANGES")

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

    print("\nOUTPUT:")

    print(
        result["output"]
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
