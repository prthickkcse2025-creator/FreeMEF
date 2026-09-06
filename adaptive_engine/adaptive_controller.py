#!/usr/bin/env python3

"""
Adaptive Multi-Method Fusion Controller

Phase 1:
    Analyze each scene and prepare adaptive decisions.

Future candidates:
    1. V3.3 PRO baseline
    2. Mertens exposure fusion
    3. MEF-Net
"""

import os
import cv2
import numpy as np


EPS = 1e-8


# ============================================================
# IMAGE LOADING
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


# ============================================================
# LUMINANCE
# ============================================================

def luminance(image):

    return (
        0.0722 * image[:, :, 0]
        + 0.7152 * image[:, :, 1]
        + 0.2126 * image[:, :, 2]
    ).astype(np.float32)


# ============================================================
# SCENE ANALYSIS
# ============================================================

def analyze_scene(normal):

    lum = luminance(
        normal
    )

    mean_brightness = float(
        np.mean(lum)
    )

    median_brightness = float(
        np.median(lum)
    )

    shadow_ratio = float(
        np.mean(
            lum < 0.30
        )
    )

    deep_shadow_ratio = float(
        np.mean(
            lum < 0.18
        )
    )

    highlight_ratio = float(
        np.mean(
            lum > 0.85
        )
    )

    return {
        "mean_brightness":
            mean_brightness,

        "median_brightness":
            median_brightness,

        "shadow_ratio":
            shadow_ratio,

        "deep_shadow_ratio":
            deep_shadow_ratio,

        "highlight_ratio":
            highlight_ratio
    }


# ============================================================
# SCENE CLASSIFICATION
# ============================================================

def classify_scene(features):

    brightness = features[
        "mean_brightness"
    ]

    shadows = features[
        "shadow_ratio"
    ]

    highlights = features[
        "highlight_ratio"
    ]

    if (
        brightness < 0.40
        and shadows > 0.45
    ):
        return "dark"

    if (
        brightness > 0.55
        and highlights > 0.20
    ):
        return "bright"

    if shadows > 0.35:
        return "shadow-heavy"

    return "balanced"


# ============================================================
# CANDIDATE EVALUATION
# ============================================================

def evaluate_candidate(
    fused,
    normal
):

    lum_fused = luminance(
        fused
    )

    lum_normal = luminance(
        normal
    )

    # --------------------------------------------------------
    # BRIGHTNESS CONSISTENCY
    # --------------------------------------------------------

    brightness_error = float(
        np.mean(
            np.abs(
                lum_fused
                - lum_normal
            )
        )
    )

    brightness_score = np.clip(
        1.0
        - 3.0 * brightness_error,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # STRUCTURE / EDGE SIMILARITY
    # --------------------------------------------------------

    normal_gray = (
        lum_normal * 255.0
    ).astype(np.uint8)

    fused_gray = (
        lum_fused * 255.0
    ).astype(np.uint8)

    normal_edges = cv2.Canny(
        normal_gray,
        50,
        150
    )

    fused_edges = cv2.Canny(
        fused_gray,
        50,
        150
    )

    edge_difference = float(
        np.mean(
            np.abs(
                normal_edges.astype(
                    np.float32
                )
                -
                fused_edges.astype(
                    np.float32
                )
            )
        )
        / 255.0
    )

    structure_score = np.clip(
        1.0 - edge_difference,
        0.0,
        1.0
    )

    # --------------------------------------------------------
    # HIGHLIGHT PROTECTION
    # --------------------------------------------------------

    normal_highlights = (
        lum_normal > 0.90
    )

    if np.any(
        normal_highlights
    ):

        clipped_ratio = float(
            np.mean(
                lum_fused[
                    normal_highlights
                ] > 0.99
            )
        )

        highlight_score = (
            1.0
            - clipped_ratio
        )

    else:

        highlight_score = 1.0

    # --------------------------------------------------------
    # SHADOW NATURALNESS
    # --------------------------------------------------------

    normal_shadows = (
        lum_normal < 0.25
    )

    if np.any(
        normal_shadows
    ):

        shadow_change = float(
            np.mean(
                lum_fused[
                    normal_shadows
                ]
                -
                lum_normal[
                    normal_shadows
                ]
            )
        )

        # Allow useful recovery,
        # penalize excessive lifting.

        excessive_lift = max(
            0.0,
            shadow_change - 0.08
        )

        shadow_score = np.clip(
            1.0
            - 4.0 * excessive_lift,
            0.0,
            1.0
        )

    else:

        shadow_score = 1.0

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    score = (

        0.35 * brightness_score

        +

        0.30 * structure_score

        +

        0.20 * shadow_score

        +

        0.15 * highlight_score

    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# ANALYZE ONE SCENE
# ============================================================

def analyze(
    under_path,
    normal_path,
    over_path
):

    under = read_image(
        under_path
    )

    normal = read_image(
        normal_path
    )

    over = read_image(
        over_path
    )

    features = analyze_scene(
        normal
    )

    profile = classify_scene(
        features
    )

    print()
    print("=" * 60)
    print("SCENE ANALYSIS")
    print("=" * 60)

    print(
        f"Mean brightness   : "
        f"{features['mean_brightness']:.4f}"
    )

    print(
        f"Median brightness : "
        f"{features['median_brightness']:.4f}"
    )

    print(
        f"Shadow ratio      : "
        f"{features['shadow_ratio']:.4f}"
    )

    print(
        f"Deep shadow ratio : "
        f"{features['deep_shadow_ratio']:.4f}"
    )

    print(
        f"Highlight ratio   : "
        f"{features['highlight_ratio']:.4f}"
    )

    print(
        f"Scene profile     : "
        f"{profile}"
    )

    return {
        "under": under,
        "normal": normal,
        "over": over,
        "features": features,
        "profile": profile
    }


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    BASE = os.path.expanduser(
        "~/FreeMEF"
    )

    under_path = os.path.join(
        BASE,
        "my_test",
        "scene005",
        "01_under.jpg"
    )

    normal_path = os.path.join(
        BASE,
        "my_test",
        "scene005",
        "02_normal.jpg"
    )

    over_path = os.path.join(
        BASE,
        "my_test",
        "scene005",
        "03_over.jpg"
    )

    analyze(
        under_path,
        normal_path,
        over_path
    )
