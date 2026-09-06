#!/usr/bin/env python3

"""
CANDIDATE QUALITY GATE
======================

Checks a fusion candidate against the NORMAL exposure.

Purpose:
    Decide whether A/B/C is acceptable before the selector
    chooses a final method.

Checks:
    1. Shadow lift
    2. Highlight lift
    3. Midtone/depth preservation
    4. Haze / local-contrast loss

This does NOT modify A, B, C, or D.
"""

import os
import argparse

import cv2
import numpy as np


EPS = 1e-8


# ============================================================
# IMAGE
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
        / 255.0
    )


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
# LOCAL CONTRAST
# ============================================================

def local_contrast_map(image):

    lum = luminance(image)

    blur = cv2.GaussianBlur(
        lum,
        (0, 0),
        2.0
    )

    return np.abs(
        lum - blur
    )


# ============================================================
# BASIC REGION MASKS
# ============================================================

def shadow_mask(lum):

    return lum < 0.30


def deep_shadow_mask(lum):

    return lum < 0.18


def highlight_mask(lum):

    return lum > 0.85


def midtone_mask(lum):

    return (
        (lum >= 0.30)
        &
        (lum <= 0.75)
    )


# ============================================================
# SHADOW LIFT
# ============================================================

def measure_shadow_lift(
    normal,
    candidate
):

    n = luminance(
        normal
    )

    c = luminance(
        candidate
    )

    mask = shadow_mask(
        n
    )

    if np.sum(mask) == 0:
        return 0.0

    delta = (
        c[mask] -
        n[mask]
    )

    # Only positive lift counts as a problem.
    positive = np.maximum(
        delta,
        0.0
    )

    return float(
        np.mean(
            positive
        )
    )


# ============================================================
# HIGHLIGHT LIFT
# ============================================================

def measure_highlight_lift(
    normal,
    candidate
):

    n = luminance(
        normal
    )

    c = luminance(
        candidate
    )

    mask = highlight_mask(
        n
    )

    if np.sum(mask) == 0:
        return 0.0

    delta = (
        c[mask] -
        n[mask]
    )

    positive = np.maximum(
        delta,
        0.0
    )

    return float(
        np.mean(
            positive
        )
    )


# ============================================================
# MIDTONE DEPTH
# ============================================================

def measure_midtone_depth(
    normal,
    candidate
):

    n_lum = luminance(
        normal
    )

    c_lum = luminance(
        candidate
    )

    mask = midtone_mask(
        n_lum
    )

    if np.sum(mask) == 0:
        return 0.0

    n_detail = local_contrast_map(
        normal
    )

    c_detail = local_contrast_map(
        candidate
    )

    n_mean = float(
        np.mean(
            n_detail[mask]
        )
    )

    c_mean = float(
        np.mean(
            c_detail[mask]
        )
    )

    if n_mean < EPS:
        return 1.0

    ratio = (
        c_mean /
        n_mean
    )

    return float(
        np.clip(
            ratio,
            0.0,
            2.0
        )
    )


# ============================================================
# HAZE / FLATNESS
# ============================================================

def measure_haze_loss(
    normal,
    candidate
):

    n_lum = luminance(
        normal
    )

    c_lum = luminance(
        candidate
    )

    mask = midtone_mask(
        n_lum
    )

    if np.sum(mask) == 0:
        return 0.0

    n_detail = local_contrast_map(
        normal
    )

    c_detail = local_contrast_map(
        candidate
    )

    n_value = float(
        np.mean(
            n_detail[mask]
        )
    )

    c_value = float(
        np.mean(
            c_detail[mask]
        )
    )

    if n_value < EPS:
        return 0.0

    # Positive means the candidate lost local contrast.
    loss = (
        n_value -
        c_value
    )

    return float(
        max(
            loss,
            0.0
        )
    )


# ============================================================
# GLOBAL MEASUREMENTS
# ============================================================

def compute_statistics(
    normal,
    candidate
):

    normal_lum = luminance(
        normal
    )

    candidate_lum = luminance(
        candidate
    )

    return {

        "normal_mean":
            float(
                np.mean(
                    normal_lum
                )
            ),

        "candidate_mean":
            float(
                np.mean(
                    candidate_lum
                )
            ),

        "brightness_delta":
            float(
                np.mean(
                    candidate_lum
                )
                -
                np.mean(
                    normal_lum
                )
            ),

        "shadow_lift":
            measure_shadow_lift(
                normal,
                candidate
            ),

        "highlight_lift":
            measure_highlight_lift(
                normal,
                candidate
            ),

        "midtone_depth":
            measure_midtone_depth(
                normal,
                candidate
            ),

        "haze_loss":
            measure_haze_loss(
                normal,
                candidate
            ),
    }


# ============================================================
# ACCEPTANCE
# ============================================================

def evaluate_candidate(
    normal,
    candidate
):

    s = compute_statistics(
        normal,
        candidate
    )

    # --------------------------------------------------------
    # Thresholds intentionally conservative.
    # These are PROVISIONAL and should be calibrated using
    # client-labeled scenes.
    # --------------------------------------------------------

    shadow_ok = (
        s["shadow_lift"] <= 0.035
    )

    highlight_ok = (
        s["highlight_lift"] <= 0.025
    )

    depth_ok = (
        s["midtone_depth"] >= 0.82
    )

    haze_ok = (
        s["haze_loss"] <= 0.012
    )

    accepted = (
        shadow_ok
        and highlight_ok
        and depth_ok
        and haze_ok
    )

    # --------------------------------------------------------
    # Quality score
    # --------------------------------------------------------

    shadow_penalty = min(
        s["shadow_lift"] / 0.035,
        2.0
    )

    highlight_penalty = min(
        s["highlight_lift"] / 0.025,
        2.0
    )

    depth_score = np.clip(
        s["midtone_depth"],
        0.0,
        1.0
    )

    haze_penalty = min(
        s["haze_loss"] / 0.012,
        2.0
    )

    score = (
        1.0
        - 0.30 * shadow_penalty
        - 0.25 * highlight_penalty
        + 0.30 * depth_score
        - 0.15 * haze_penalty
    )

    score = float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )

    s["shadow_ok"] = shadow_ok
    s["highlight_ok"] = highlight_ok
    s["depth_ok"] = depth_ok
    s["haze_ok"] = haze_ok
    s["accepted"] = accepted
    s["score"] = score

    return s


# ============================================================
# PRINT
# ============================================================

def print_result(
    name,
    result
):

    print()
    print(
        "-" * 65
    )

    print(
        name
    )

    print(
        "-" * 65
    )

    print(
        f"Normal mean       : "
        f"{result['normal_mean']:.4f}"
    )

    print(
        f"Candidate mean    : "
        f"{result['candidate_mean']:.4f}"
    )

    print(
        f"Brightness delta  : "
        f"{result['brightness_delta']:+.4f}"
    )

    print(
        f"Shadow lift       : "
        f"{result['shadow_lift']:.4f} "
        f"{'OK' if result['shadow_ok'] else 'FAIL'}"
    )

    print(
        f"Highlight lift    : "
        f"{result['highlight_lift']:.4f} "
        f"{'OK' if result['highlight_ok'] else 'FAIL'}"
    )

    print(
        f"Midtone depth     : "
        f"{result['midtone_depth']:.4f} "
        f"{'OK' if result['depth_ok'] else 'FAIL'}"
    )

    print(
        f"Haze loss         : "
        f"{result['haze_loss']:.4f} "
        f"{'OK' if result['haze_ok'] else 'FAIL'}"
    )

    print(
        f"Quality score     : "
        f"{result['score']:.4f}"
    )

    print(
        f"STATUS            : "
        f"{'ACCEPT' if result['accepted'] else 'REJECT'}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate fusion candidates "
            "against a Normal exposure"
        )
    )

    parser.add_argument(
        "--normal",
        required=True
    )

    parser.add_argument(
        "--candidate-a",
        required=True
    )

    parser.add_argument(
        "--candidate-b",
        required=True
    )

    parser.add_argument(
        "--candidate-c",
        required=True
    )

    args = parser.parse_args()

    normal = read_image(
        args.normal
    )

    candidates = {

        "Candidate A - V3.3":
            args.candidate_a,

        "Candidate B - Mertens":
            args.candidate_b,

        "Candidate C - MEF-Net":
            args.candidate_c,
    }

    print()
    print(
        "=" * 65
    )

    print(
        "CANDIDATE QUALITY GATE"
    )

    print(
        "=" * 65
    )

    results = {}

    for name, path in candidates.items():

        if not os.path.isfile(path):

            print()
            print(
                f"ERROR: Missing candidate:\n"
                f"{path}"
            )

            continue

        candidate = read_image(
            path
        )

        candidate = match_size(
            candidate,
            normal
        )

        result = evaluate_candidate(
            normal,
            candidate
        )

        results[name] = result

        print_result(
            name,
            result
        )

    print()
    print(
        "=" * 65
    )

    print(
        "QUALITY GATE SUMMARY"
    )

    print(
        "=" * 65
    )

    accepted = []

    for name, result in results.items():

        status = (
            "ACCEPT"
            if result["accepted"]
            else "REJECT"
        )

        print(
            f"{name:<30} "
            f"{status:<8} "
            f"{result['score']:.4f}"
        )

        if result["accepted"]:
            accepted.append(
                (
                    name,
                    result["score"]
                )
            )

    print()

    if accepted:

        accepted.sort(
            key=lambda x: x[1],
            reverse=True
        )

        print(
            f"Best accepted candidate: "
            f"{accepted[0][0]}"
        )

    else:

        print(
            "NO ACCEPTABLE CANDIDATE"
        )

        print(
            "Fallback refinement required."
        )

    print(
        "=" * 65
    )


if __name__ == "__main__":

    main()
