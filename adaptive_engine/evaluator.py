#!/usr/bin/env python3

"""
FREE MEF - THREE-CANDIDATE MULTI-REFERENCE EVALUATOR

Candidates:
    A = V3.3 Client-Bright
    B = Mertens
    C = MEF-Net

References:
    UNDER  -> highlight information
    NORMAL -> natural exposure / color / structure
    OVER   -> shadow information

The evaluator does NOT treat the Normal exposure as ground truth.

It evaluates every candidate using all three exposures and produces:
    - brightness score
    - shadow recovery score
    - highlight recovery score
    - color score
    - structure score
    - clipping score
    - artificial-lift penalty
    - final score
    - automatic winner
"""

import os
import argparse
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
# RESIZE REFERENCE
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
# SCENE STATISTICS
# ============================================================

def scene_statistics(image):

    lum = luminance(image)

    return {

        "mean":
            float(np.mean(lum)),

        "median":
            float(np.median(lum)),

        "shadow_ratio":
            float(np.mean(lum < 0.30)),

        "deep_shadow_ratio":
            float(np.mean(lum < 0.18)),

        "highlight_ratio":
            float(np.mean(lum > 0.85)),

        "clipped_ratio":
            float(np.mean(lum > 0.99)),

        "dark_clipped_ratio":
            float(np.mean(lum < 0.01)),

        "std":
            float(np.std(lum))
    }


# ============================================================
# BRIGHTNESS SCORE
# ============================================================

def brightness_score(
    candidate,
    normal
):

    candidate_lum = luminance(
        candidate
    )

    normal_lum = luminance(
        normal
    )

    delta = abs(
        float(np.mean(candidate_lum))
        -
        float(np.mean(normal_lum))
    )

    # Small difference from normal is preferred.
    score = np.exp(
        -delta * 5.0
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# MEDIAN EXPOSURE SCORE
# ============================================================

def median_score(
    candidate,
    normal
):

    candidate_lum = luminance(
        candidate
    )

    normal_lum = luminance(
        normal
    )

    delta = abs(
        float(np.median(candidate_lum))
        -
        float(np.median(normal_lum))
    )

    score = np.exp(
        -delta * 5.0
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# SHADOW RECOVERY
# ============================================================

def shadow_score(
    candidate,
    normal,
    over
):

    candidate_lum = luminance(
        candidate
    )

    normal_lum = luminance(
        normal
    )

    over_lum = luminance(
        over
    )

    # Regions where Normal is dark but Over contains
    # potentially recoverable information.
    mask = (
        (normal_lum < 0.30)
        &
        (over_lum > normal_lum + 0.04)
    )

    if np.count_nonzero(mask) < 100:
        return 1.0

    candidate_values = candidate_lum[mask]
    normal_values = normal_lum[mask]
    over_values = over_lum[mask]

    available_recovery = (
        over_values
        -
        normal_values
    )

    candidate_recovery = (
        candidate_values
        -
        normal_values
    )

    available = float(
        np.mean(
            np.maximum(
                available_recovery,
                0.0
            )
        )
    )

    recovered = float(
        np.mean(
            np.maximum(
                candidate_recovery,
                0.0
            )
        )
    )

    if available < EPS:
        return 1.0

    recovery_ratio = (
        recovered /
        (available + EPS)
    )

    # We don't want to demand 100% recovery.
    # Around 25-75% of available recovery is considered natural.
    target = 0.50

    score = np.exp(
        -abs(
            recovery_ratio -
            target
        ) * 2.5
    )

    # Strong penalty if candidate crushes more shadows
    # than the normal exposure.
    crushed = float(
        np.mean(
            candidate_values < 0.08
        )
    )

    normal_crushed = float(
        np.mean(
            normal_values < 0.08
        )
    )

    extra_crushed = max(
        0.0,
        crushed -
        normal_crushed -
        0.02
    )

    score *= np.exp(
        -extra_crushed * 6.0
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# HIGHLIGHT RECOVERY
# ============================================================

def highlight_score(
    candidate,
    normal,
    under
):

    candidate_lum = luminance(
        candidate
    )

    normal_lum = luminance(
        normal
    )

    under_lum = luminance(
        under
    )

    # Regions where Normal is bright and Under retains
    # useful information below clipping.
    mask = (
        (normal_lum > 0.70)
        &
        (under_lum < normal_lum - 0.04)
        &
        (under_lum > 0.20)
    )

    if np.count_nonzero(mask) < 100:
        return 1.0

    candidate_values = candidate_lum[mask]
    normal_values = normal_lum[mask]
    under_values = under_lum[mask]

    available_recovery = (
        normal_values -
        under_values
    )

    candidate_recovery = (
        normal_values -
        candidate_values
    )

    available = float(
        np.mean(
            np.maximum(
                available_recovery,
                0.0
            )
        )
    )

    recovered = float(
        np.mean(
            np.maximum(
                candidate_recovery,
                0.0
            )
        )
    )

    if available < EPS:
        return 1.0

    recovery_ratio = (
        recovered /
        (available + EPS)
    )

    # Moderate highlight recovery is preferred.
    target = 0.45

    score = np.exp(
        -abs(
            recovery_ratio -
            target
        ) * 2.5
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# CLIPPING SCORE
# ============================================================

def clipping_score(
    candidate,
    under,
    normal,
    over
):

    candidate_lum = luminance(
        candidate
    )

    under_lum = luminance(
        under
    )

    normal_lum = luminance(
        normal
    )

    over_lum = luminance(
        over
    )

    # Candidate should not clip regions where another exposure
    # contains usable information.

    recoverable_highlights = (
        (normal_lum > 0.80)
        &
        (under_lum < 0.90)
    )

    if np.any(
        recoverable_highlights
    ):

        candidate_clipped = float(
            np.mean(
                candidate_lum[
                    recoverable_highlights
                ] > 0.99
            )
        )

    else:

        candidate_clipped = 0.0

    recoverable_shadows = (
        (normal_lum < 0.20)
        &
        (over_lum > 0.25)
    )

    if np.any(
        recoverable_shadows
    ):

        candidate_black = float(
            np.mean(
                candidate_lum[
                    recoverable_shadows
                ] < 0.01
            )
        )

    else:

        candidate_black = 0.0

    penalty = (
        0.65 *
        candidate_clipped
        +
        0.35 *
        candidate_black
    )

    score = np.exp(
        -penalty * 8.0
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# COLOR CONSISTENCY
# ============================================================

def color_score(
    candidate,
    normal,
    under,
    over
):

    candidate_mean = np.mean(
        candidate,
        axis=(0, 1)
    )

    normal_mean = np.mean(
        normal,
        axis=(0, 1)
    )

    under_mean = np.mean(
        under,
        axis=(0, 1)
    )

    over_mean = np.mean(
        over,
        axis=(0, 1)
    )

    # Exposure images naturally differ in brightness.
    # Compare chromatic ratios instead.

    def chroma(v):

        total = (
            float(np.sum(v))
            +
            EPS
        )

        return (
            v /
            total
        )

    c_candidate = chroma(
        candidate_mean
    )

    c_normal = chroma(
        normal_mean
    )

    c_under = chroma(
        under_mean
    )

    c_over = chroma(
        over_mean
    )

    source_center = (
        c_normal
        +
        0.25 * c_under
        +
        0.25 * c_over
    ) / 1.5

    error = float(
        np.mean(
            np.abs(
                c_candidate -
                source_center
            )
        )
    )

    score = np.exp(
        -error * 35.0
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# STRUCTURE SCORE
# ============================================================

def structure_score(
    candidate,
    under,
    normal,
    over
):

    candidate_lum = luminance(
        candidate
    )

    under_lum = luminance(
        under
    )

    normal_lum = luminance(
        normal
    )

    over_lum = luminance(
        over
    )

    def gradient_strength(
        lum
    ):

        gx = cv2.Sobel(
            lum,
            cv2.CV_32F,
            1,
            0,
            ksize=3
        )

        gy = cv2.Sobel(
            lum,
            cv2.CV_32F,
            0,
            1,
            ksize=3
        )

        return np.sqrt(
            gx * gx +
            gy * gy
        )

    candidate_grad = (
        gradient_strength(
            candidate_lum
        )
    )

    source_grad = np.maximum.reduce(
        [
            gradient_strength(
                under_lum
            ),
            gradient_strength(
                normal_lum
            ),
            gradient_strength(
                over_lum
            )
        ]
    )

    # Only meaningful edges.
    mask = (
        source_grad >
        np.percentile(
            source_grad,
            60
        )
    )

    if np.count_nonzero(mask) < 100:
        return 1.0

    candidate_value = (
        candidate_grad[mask]
    )

    source_value = (
        source_grad[mask]
    )

    ratio = (
        candidate_value
        /
        (
            source_value +
            EPS
        )
    )

    # Avoid rewarding excessive sharpening.
    error = float(
        np.mean(
            np.abs(
                ratio -
                0.85
            )
        )
    )

    score = np.exp(
        -error * 3.0
    )

    return float(
        np.clip(
            score,
            0.0,
            1.0
        )
    )


# ============================================================
# ARTIFICIAL LIFT PENALTY
# ============================================================

def artificial_lift_penalty(
    candidate,
    normal
):

    candidate_lum = luminance(
        candidate
    )

    normal_lum = luminance(
        normal
    )

    dark_mask = (
        normal_lum < 0.30
    )

    if np.count_nonzero(
        dark_mask
    ) < 100:

        return 0.0

    lift = (
        candidate_lum[dark_mask]
        -
        normal_lum[dark_mask]
    )

    excessive = np.maximum(
        lift - 0.12,
        0.0
    )

    penalty = float(
        np.mean(
            excessive
        )
    )

    return float(
        np.clip(
            penalty,
            0.0,
            1.0
        )
    )


# ============================================================
# EVALUATE ONE CANDIDATE
# ============================================================

def evaluate_candidate(
    candidate,
    under,
    normal,
    over
):

    candidate = match_size(
        candidate,
        normal
    )

    under = match_size(
        under,
        normal
    )

    over = match_size(
        over,
        normal
    )

    scores = {}

    scores["brightness"] = (
        brightness_score(
            candidate,
            normal
        )
    )

    scores["median"] = (
        median_score(
            candidate,
            normal
        )
    )

    scores["shadow"] = (
        shadow_score(
            candidate,
            normal,
            over
        )
    )

    scores["highlight"] = (
        highlight_score(
            candidate,
            normal,
            under
        )
    )

    scores["clipping"] = (
        clipping_score(
            candidate,
            under,
            normal,
            over
        )
    )

    scores["color"] = (
        color_score(
            candidate,
            normal,
            under,
            over
        )
    )

    scores["structure"] = (
        structure_score(
            candidate,
            under,
            normal,
            over
        )
    )

    scores["artificial_lift"] = (
        artificial_lift_penalty(
            candidate,
            normal
        )
    )

    # ========================================================
    # FINAL WEIGHTED SCORE
    # ========================================================

    base_score = (

        0.20 *
        scores["brightness"]

        +

        0.08 *
        scores["median"]

        +

        0.18 *
        scores["shadow"]

        +

        0.18 *
        scores["highlight"]

        +

        0.12 *
        scores["clipping"]

        +

        0.10 *
        scores["color"]

        +

        0.14 *
        scores["structure"]
    )

    final_score = (
        base_score
        -
        0.12 *
        scores["artificial_lift"]
    )

    scores["final"] = float(
        np.clip(
            final_score,
            0.0,
            1.0
        )
    )

    # ========================================================
    # STATISTICS
    # ========================================================

    normal_stats = scene_statistics(
        normal
    )

    candidate_stats = scene_statistics(
        candidate
    )

    scores["normal_mean"] = (
        normal_stats["mean"]
    )

    scores["candidate_mean"] = (
        candidate_stats["mean"]
    )

    scores["brightness_delta"] = (
        candidate_stats["mean"]
        -
        normal_stats["mean"]
    )

    scores["candidate_shadow_ratio"] = (
        candidate_stats["shadow_ratio"]
    )

    scores["candidate_highlight_ratio"] = (
        candidate_stats["highlight_ratio"]
    )

    scores["candidate_clipped_ratio"] = (
        candidate_stats["clipped_ratio"]
    )

    return scores


# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    name,
    scores
):

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    print(
        f"Brightness          : "
        f"{scores['brightness']:.4f}"
    )

    print(
        f"Median              : "
        f"{scores['median']:.4f}"
    )

    print(
        f"Shadow recovery     : "
        f"{scores['shadow']:.4f}"
    )

    print(
        f"Highlight recovery  : "
        f"{scores['highlight']:.4f}"
    )

    print(
        f"Clipping protection : "
        f"{scores['clipping']:.4f}"
    )

    print(
        f"Color               : "
        f"{scores['color']:.4f}"
    )

    print(
        f"Structure           : "
        f"{scores['structure']:.4f}"
    )

    print(
        f"Artificial lift     : "
        f"{scores['artificial_lift']:.4f}"
    )

    print()
    print(
        f"Normal mean         : "
        f"{scores['normal_mean']:.4f}"
    )

    print(
        f"Candidate mean      : "
        f"{scores['candidate_mean']:.4f}"
    )

    print(
        f"Brightness delta    : "
        f"{scores['brightness_delta']:+.4f}"
    )

    print(
        f"Candidate shadows   : "
        f"{scores['candidate_shadow_ratio']:.4f}"
    )

    print(
        f"Candidate highlights: "
        f"{scores['candidate_highlight_ratio']:.4f}"
    )

    print(
        f"Candidate clipped   : "
        f"{scores['candidate_clipped_ratio']:.4f}"
    )

    print()
    print(
        f"FINAL SCORE         : "
        f"{scores['final']:.4f}"
    )


# ============================================================
# EVALUATE THREE CANDIDATES
# ============================================================

def evaluate_three(
    under_path,
    normal_path,
    over_path,
    candidate_a_path,
    candidate_b_path,
    candidate_c_path
):

    print()
    print("=" * 70)
    print("FREE MEF - THREE CANDIDATE EVALUATION")
    print("=" * 70)

    print()
    print("References:")
    print(
        f"  Under  : {under_path}"
    )
    print(
        f"  Normal : {normal_path}"
    )
    print(
        f"  Over   : {over_path}"
    )

    under = read_image(
        under_path
    )

    normal = read_image(
        normal_path
    )

    over = read_image(
        over_path
    )

    candidates = {

        "A": (
            "V3.3 CLIENT-BRIGHT",
            candidate_a_path
        ),

        "B": (
            "MERTENS",
            candidate_b_path
        ),

        "C": (
            "MEF-NET",
            candidate_c_path
        )
    }

    results = {}

    for label, (
        name,
        path
    ) in candidates.items():

        print()
        print(
            f"Evaluating Candidate {label}..."
        )

        candidate = read_image(
            path
        )

        results[label] = evaluate_candidate(
            candidate,
            under,
            normal,
            over
        )

        print_result(
            f"CANDIDATE {label} - {name}",
            results[label]
        )

    # ========================================================
    # SORT
    # ========================================================

    ranking = sorted(
        results.items(),
        key=lambda item:
            item[1]["final"],
        reverse=True
    )

    winner_label = ranking[0][0]
    winner_score = ranking[0][1]["final"]

    if len(ranking) >= 2:

        second_score = (
            ranking[1][1]["final"]
        )

    else:

        second_score = 0.0

    confidence = (
        max(
            0.0,
            winner_score -
            second_score
        )
        /
        max(
            winner_score,
            EPS
        )
    )

    confidence = float(
        np.clip(
            0.50 +
            confidence,
            0.0,
            1.0
        )
    )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CANDIDATE RANKING")
    print("=" * 70)

    for index, (
        label,
        score
    ) in enumerate(
        ranking,
        start=1
    ):

        names = {
            "A": "V3.3 CLIENT-BRIGHT",
            "B": "MERTENS",
            "C": "MEF-NET"
        }

        print(
            f"{index}. Candidate {label} "
            f"({names[label]})"
            f" -> {score['final']:.4f}"
        )

    print()
    print(
        "=========================================="
    )

    print(
        f"WINNER: CANDIDATE {winner_label}"
    )

    print(
        f"SCORE : {winner_score:.4f}"
    )

    print(
        f"CONFIDENCE: {confidence * 100.0:.2f}%"
    )

    print(
        "=========================================="
    )

    return {
        "results": results,
        "ranking": ranking,
        "winner": winner_label,
        "confidence": confidence
    }


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "FreeMEF three-candidate "
            "multi-reference evaluator"
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

    for label, path in [

        ("Under", args.under),

        ("Normal", args.normal),

        ("Over", args.over),

        ("Candidate A", args.candidate_a),

        ("Candidate B", args.candidate_b),

        ("Candidate C", args.candidate_c)

    ]:

        if not os.path.isfile(path):

            raise FileNotFoundError(
                f"{label} file not found:\n{path}"
            )

    evaluate_three(
        args.under,
        args.normal,
        args.over,
        args.candidate_a,
        args.candidate_b,
        args.candidate_c
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
