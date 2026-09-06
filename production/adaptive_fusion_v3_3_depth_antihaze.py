#!/usr/bin/env python3

"""
V3.3 CLIENT-BRIGHT + DEPTH + ANTI-HAZE

Original V3.3 is used unchanged as the base.
This file adds:
    - normal-exposure midtone detail
    - mild anti-haze/local contrast
    - color protection
    - highlight protection
"""

import os
import sys
import cv2
import argparse
import subprocess
import numpy as np


EPS = 1e-8


# ============================================================
# PROJECT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

V33_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "production",
    "adaptive_fusion_v3_3_client_bright.py"
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

    output = (
        image * 255.0
    ).astype(
        np.uint8
    )

    output_dir = os.path.dirname(
        path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    success = cv2.imwrite(
        path,
        output,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            100
        ]
    )

    if not success:
        raise RuntimeError(
            f"Could not save output:\n{path}"
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
# SIZE
# ============================================================

def match_size(
    image,
    reference
):

    h, w = reference.shape[:2]

    if image.shape[:2] == (h, w):
        return image

    return cv2.resize(
        image,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )


# ============================================================
# NORMAL EXPOSURE DETAIL
# ============================================================

def extract_normal_detail(
    normal
):

    gray = luminance(
        normal
    )

    smooth = cv2.GaussianBlur(
        gray,
        (0, 0),
        2.0
    )

    detail = (
        gray -
        smooth
    )

    scale = float(
        np.percentile(
            np.abs(detail),
            95
        )
    )

    if scale < EPS:

        return np.zeros_like(
            detail
        )

    detail = (
        detail /
        scale
    )

    return np.clip(
        detail,
        -1.0,
        1.0
    )


# ============================================================
# NORMAL MIDTONE CONFIDENCE
# ============================================================

def structure_confidence(
    normal_lum
):

    middle = np.exp(
        -(
            (
                normal_lum -
                0.50
            ) ** 2
        )
        /
        (
            2.0 * 0.24 ** 2
        )
    )

    shadow_protection = (
        1.0 -
        np.clip(
            (
                0.22 -
                normal_lum
            ) / 0.22,
            0.0,
            1.0
        )
    )

    highlight_protection = (
        1.0 -
        np.clip(
            (
                normal_lum -
                0.82
            ) / 0.18,
            0.0,
            1.0
        )
    )

    confidence = (
        middle
        *
        shadow_protection
        *
        highlight_protection
    )

    confidence = cv2.GaussianBlur(
        confidence.astype(
            np.float32
        ),
        (0, 0),
        2.5
    )

    return np.clip(
        confidence,
        0.0,
        1.0
    )


# ============================================================
# DEPTH DETAIL
# ============================================================

def apply_depth_detail(
    fused,
    normal,
    strength=0.055
):

    normal_lum = luminance(
        normal
    )

    detail = extract_normal_detail(
        normal
    )

    confidence = structure_confidence(
        normal_lum
    )

    # Convert the 2-D scalar map into a 3-D map explicitly.
    effective_strength = (
        strength *
        confidence
    )[:, :, None]

    detail_3d = (
        detail[:, :, None]
    )

    result = (
        fused +
        effective_strength *
        detail_3d
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# ANTI-HAZE
# ============================================================

def anti_haze(
    image,
    strength=0.10
):

    lab = cv2.cvtColor(
        np.clip(
            image,
            0.0,
            1.0
        ).astype(
            np.float32
        ),
        cv2.COLOR_BGR2LAB
    )

    L = lab[:, :, 0]

    local = cv2.GaussianBlur(
        L,
        (0, 0),
        7.0
    )

    local_detail = (
        L -
        local
    )

    scale = float(
        np.percentile(
            np.abs(local_detail),
            95
        )
    )

    if scale < EPS:

        return image

    normalized = (
        local_detail /
        scale
    )

    L_new = (
        L +
        strength *
        normalized *
        20.0
    )

    lab[:, :, 0] = np.clip(
        L_new,
        0.0,
        100.0
    )

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# COLOR PROTECTION
# ============================================================

def protect_color(
    refined,
    normal,
    strength=0.10
):

    normal_lum = luminance(
        normal
    )

    refined_lum = luminance(
        refined
    )

    ratio = (
        normal_lum +
        EPS
    ) / (
        refined_lum +
        EPS
    )

    ratio = np.clip(
        ratio,
        0.95,
        1.05
    )

    correction = (
        1.0 +
        strength *
        (
            ratio -
            1.0
        )
    )

    result = (
        refined *
        correction[:, :, None]
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# HIGHLIGHT PROTECTION
# ============================================================

def protect_highlights(
    refined,
    normal,
    under
):

    refined_lum = luminance(
        refined
    )

    normal_lum = luminance(
        normal
    )

    under_lum = luminance(
        under
    )

    highlight_mask = np.clip(
        (
            refined_lum -
            0.86
        ) / 0.14,
        0.0,
        1.0
    )

    under_advantage = np.clip(
        (
            normal_lum -
            under_lum
        ) / 0.25,
        0.0,
        1.0
    )

    recovery = (
        0.22 *
        highlight_mask *
        under_advantage
    )

    recovery = (
        recovery[:, :, None]
    )

    result = (
        refined *
        (1.0 - recovery)
        +
        under *
        recovery
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# REFINEMENT
# ============================================================

def refine_output(
    v33,
    normal,
    under
):

    result = v33.copy()

    print(
        "Extracting middle-exposure structure..."
    )

    result = apply_depth_detail(
        result,
        normal,
        strength=0.055
    )

    print(
        "Applying mild anti-haze correction..."
    )

    result = anti_haze(
        result,
        strength=0.10
    )

    print(
        "Protecting natural colors..."
    )

    result = protect_color(
        result,
        normal,
        strength=0.10
    )

    print(
        "Protecting highlights..."
    )

    result = protect_highlights(
        result,
        normal,
        under
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# ORIGINAL V3.3
# ============================================================

def run_original_v33(
    under_path,
    normal_path,
    over_path,
    temporary_output
):

    print(
        "Running original V3.3 Client-Bright..."
    )

    command = [
        sys.executable,
        V33_SCRIPT,
        "--under",
        under_path,
        "--normal",
        normal_path,
        "--over",
        over_path,
        "--output",
        temporary_output
    ]

    result = subprocess.run(
        command,
        check=False
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Original V3.3 process failed."
        )

    if not os.path.isfile(
        temporary_output
    ):
        raise RuntimeError(
            "Original V3.3 did not create "
            "the temporary output."
        )


# ============================================================
# PROCESS
# ============================================================

def process(
    under_path,
    normal_path,
    over_path,
    output_path
):

    print()
    print("=" * 60)
    print(
        "V3.3 DEPTH + ANTI-HAZE EDITION"
    )
    print("=" * 60)

    if not os.path.isfile(
        V33_SCRIPT
    ):
        raise FileNotFoundError(
            f"Original V3.3 not found:\n{V33_SCRIPT}"
        )

    normal = read_image(
        normal_path
    )

    under = read_image(
        under_path
    )

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:
        os.makedirs(
            output_dir,
            exist_ok=True
        )

    temporary_output = os.path.join(
        output_dir if output_dir else ".",
        "_v33_base_for_depth_antihaze.jpg"
    )

    try:

        run_original_v33(
            under_path,
            normal_path,
            over_path,
            temporary_output
        )

        v33 = read_image(
            temporary_output
        )

        normal = match_size(
            normal,
            v33
        )

        under = match_size(
            under,
            v33
        )

        refined = refine_output(
            v33,
            normal,
            under
        )

        save_image(
            output_path,
            refined
        )

    finally:

        try:
            if os.path.isfile(
                temporary_output
            ):
                os.remove(
                    temporary_output
                )
        except OSError:
            pass

    print()
    print("=" * 60)
    print(
        "DEPTH + ANTI-HAZE COMPLETE"
    )
    print("=" * 60)

    print(
        f"\nFinal output:\n"
        f"  {output_path}"
    )

    return output_path


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "V3.3 Client-Bright with "
            "middle-exposure depth and anti-haze"
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

    args = parser.parse_args()

    process(
        args.under,
        args.normal,
        args.over,
        args.output
    )


if __name__ == "__main__":
    main()
