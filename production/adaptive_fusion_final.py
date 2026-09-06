#!/usr/bin/env python3
"""
Adaptive Production HDR Exposure Fusion

Designed for:
    Underexposed + Normal + Overexposed bracket images

Main goals:
    - NORMAL exposure remains the brightness and color anchor
    - Underexposed image recovers highlights and windows
    - Overexposed image recovers shadow detail
    - No aggressive tone mapping
    - Preserve white balance
    - Preserve natural real-estate brightness
    - Avoid halos, gray appearance, and over-processing

Usage:
    python production/adaptive_fusion_final.py \
        --under input/under.jpg \
        --normal input/normal.jpg \
        --over input/over.jpg \
        --output output/final.jpg
"""

import os
import cv2
import argparse
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

EPS = 1e-6

# The normal exposure is intentionally dominant.
BASE_NORMAL_WEIGHT = 0.72

# Maximum influence allowed from auxiliary exposures.
MAX_UNDER_WEIGHT = 0.55
MAX_OVER_WEIGHT = 0.45

# Blur sizes for smooth masks.
MASK_BLUR_SIGMA = 15
DETAIL_SIGMA = 2.0

# Highlight / shadow thresholds.
HIGHLIGHT_START = 0.68
SHADOW_START = 0.34

# Normal exposure brightness protection.
NORMAL_BRIGHTNESS_PROTECTION = 0.85

# Small local contrast enhancement.
LOCAL_CONTRAST_AMOUNT = 0.08


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(
            f"Could not load image:\n{path}"
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    return img


def save_image(path, img):
    os.makedirs(
        os.path.dirname(path) if os.path.dirname(path) else ".",
        exist_ok=True
    )

    img = np.clip(img, 0.0, 1.0)

    img8 = (img * 255.0 + 0.5).astype(np.uint8)
    img8 = cv2.cvtColor(img8, cv2.COLOR_RGB2BGR)

    cv2.imwrite(path, img8)


# ============================================================
# ALIGNMENT
# ============================================================

def align_to_reference(reference, image):
    """
    Align exposure image to the normal exposure using ECC.

    Falls back to original image if alignment fails.
    """

    ref_gray = cv2.cvtColor(
        (reference * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    img_gray = cv2.cvtColor(
        (image * 255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    ref_gray = ref_gray.astype(np.float32) / 255.0
    img_gray = img_gray.astype(np.float32) / 255.0

    warp_matrix = np.eye(3, 3, dtype=np.float32)

    try:
        criteria = (
            cv2.TERM_CRITERIA_EPS |
            cv2.TERM_CRITERIA_COUNT,
            100,
            1e-6
        )

        cv2.findTransformECC(
            ref_gray,
            img_gray,
            warp_matrix,
            cv2.MOTION_HOMOGRAPHY,
            criteria,
            None,
            5
        )

        h, w = reference.shape[:2]

        aligned = cv2.warpPerspective(
            image,
            warp_matrix,
            (w, h),
            flags=cv2.INTER_LINEAR +
                  cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE
        )

        return aligned

    except cv2.error:
        print("Warning: alignment failed. Using original exposure.")
        return image


# ============================================================
# LUMINANCE
# ============================================================

def luminance(img):
    return (
        0.2126 * img[:, :, 0] +
        0.7152 * img[:, :, 1] +
        0.0722 * img[:, :, 2]
    )


# ============================================================
# SMOOTH MASK
# ============================================================

def smooth_mask(mask, sigma=15):
    mask = cv2.GaussianBlur(
        mask,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    return np.clip(mask, 0.0, 1.0)


# ============================================================
# EDGE AWARE MASK
# ============================================================

def edge_aware_smooth(mask, guide):

    guide8 = (
        np.clip(guide, 0, 1) * 255
    ).astype(np.uint8)

    mask8 = (
        np.clip(mask, 0, 1) * 255
    ).astype(np.uint8)

    try:
        filtered = cv2.bilateralFilter(
            mask8,
            d=9,
            sigmaColor=50,
            sigmaSpace=50
        )

        filtered = filtered.astype(np.float32) / 255.0

    except cv2.error:
        filtered = mask

    filtered = smooth_mask(
        filtered,
        MASK_BLUR_SIGMA
    )

    return np.clip(filtered, 0.0, 1.0)


# ============================================================
# HIGHLIGHT MASK
# ============================================================

def create_highlight_mask(normal_lum, under_lum):
    """
    Detect regions where normal exposure is too bright.

    The underexposed image contributes mainly here.
    """

    mask = (
        normal_lum - HIGHLIGHT_START
    ) / (
        1.0 - HIGHLIGHT_START + EPS
    )

    mask = np.clip(mask, 0.0, 1.0)

    # Stronger preference if underexposure contains useful detail.
    detail_gain = np.clip(
        (normal_lum - under_lum) * 3.0,
        0.0,
        1.0
    )

    mask = mask * (
        0.45 + 0.55 * detail_gain
    )

    return mask


# ============================================================
# SHADOW MASK
# ============================================================

def create_shadow_mask(normal_lum, over_lum):
    """
    Detect dark areas where the overexposed image
    contains useful shadow information.
    """

    mask = (
        SHADOW_START - normal_lum
    ) / (
        SHADOW_START + EPS
    )

    mask = np.clip(mask, 0.0, 1.0)

    # Only recover shadows where overexposure actually reveals detail.
    recovery = np.clip(
        (over_lum - normal_lum) * 3.0,
        0.0,
        1.0
    )

    mask = mask * (
        0.40 + 0.60 * recovery
    )

    return mask


# ============================================================
# NORMAL EXPOSURE PROTECTION
# ============================================================

def create_normal_protection(normal_lum):
    """
    Strongly protects the normal exposure
    in midtones and normal brightness regions.
    """

    distance = np.abs(
        normal_lum - 0.5
    )

    protection = 1.0 - distance * 1.5

    protection = np.clip(
        protection,
        0.35,
        1.0
    )

    return protection


# ============================================================
# MULTISCALE DETAIL EXTRACTION
# ============================================================

def detail_layer(img, sigma=2.0):

    low = cv2.GaussianBlur(
        img,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma
    )

    detail = img - low

    return detail


# ============================================================
# ADAPTIVE EXPOSURE FUSION
# ============================================================

def adaptive_fusion(under, normal, over):

    print("\nCreating luminance maps...")

    lum_under = luminance(under)
    lum_normal = luminance(normal)
    lum_over = luminance(over)

    print("Creating highlight recovery mask...")

    highlight_mask = create_highlight_mask(
        lum_normal,
        lum_under
    )

    highlight_mask = edge_aware_smooth(
        highlight_mask,
        normal
    )

    print("Creating shadow recovery mask...")

    shadow_mask = create_shadow_mask(
        lum_normal,
        lum_over
    )

    shadow_mask = edge_aware_smooth(
        shadow_mask,
        normal
    )

    print("Creating normal exposure protection...")

    normal_protection = create_normal_protection(
        lum_normal
    )

    # --------------------------------------------------------
    # Adaptive weights
    # --------------------------------------------------------

    under_weight = (
        highlight_mask *
        MAX_UNDER_WEIGHT
    )

    over_weight = (
        shadow_mask *
        MAX_OVER_WEIGHT
    )

    # Strong normal anchor.
    normal_weight = (
        BASE_NORMAL_WEIGHT +
        normal_protection *
        NORMAL_BRIGHTNESS_PROTECTION
    )

    # Prevent auxiliary exposures from dominating.
    under_weight = np.minimum(
        under_weight,
        0.55
    )

    over_weight = np.minimum(
        over_weight,
        0.45
    )

    # Expand dimensions.
    uw = under_weight[:, :, None]
    nw = normal_weight[:, :, None]
    ow = over_weight[:, :, None]

    total = uw + nw + ow + EPS

    uw = uw / total
    nw = nw / total
    ow = ow / total

    print("Performing brightness-preserving fusion...")

    fused = (
        under * uw +
        normal * nw +
        over * ow
    )

    # --------------------------------------------------------
    # Preserve NORMAL exposure brightness
    # --------------------------------------------------------

    fused_lum = luminance(fused)

    # Only allow brightness correction in extreme regions.
    extreme_high = np.clip(
        (lum_normal - 0.72) / 0.28,
        0.0,
        1.0
    )

    extreme_shadow = np.clip(
        (0.25 - lum_normal) / 0.25,
        0.0,
        1.0
    )

    extreme_mask = np.maximum(
        extreme_high,
        extreme_shadow
    )

    # Midtones stay extremely close to normal image.
    preserve_strength = (
        1.0 - extreme_mask
    ) * 0.92

    preserve_strength = preserve_strength[:, :, None]

    fused = (
        fused * (1.0 - preserve_strength) +
        normal * preserve_strength
    )

    return np.clip(fused, 0.0, 1.0), highlight_mask, shadow_mask


# ============================================================
# LOCAL DETAIL RESTORATION
# ============================================================

def restore_detail(fused, normal):

    print("Restoring normal exposure detail...")

    normal_detail = detail_layer(
        normal,
        DETAIL_SIGMA
    )

    fused = (
        fused +
        normal_detail * 0.18
    )

    return np.clip(
        fused,
        0.0,
        1.0
    )


# ============================================================
# VERY SUBTLE LOCAL CONTRAST
# ============================================================

def subtle_local_contrast(img):

    print("Applying subtle local contrast...")

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    )

    L = lab[:, :, 0]

    base = cv2.GaussianBlur(
        L,
        (0, 0),
        sigmaX=25,
        sigmaY=25
    )

    detail = L - base

    L = L + (
        detail *
        LOCAL_CONTRAST_AMOUNT
    )

    lab[:, :, 0] = np.clip(
        L,
        0,
        100
    )

    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# COLOR / WHITE BALANCE PROTECTION
# ============================================================

def preserve_normal_color(fused, normal):

    print("Protecting original white balance and colors...")

    fused_lab = cv2.cvtColor(
        fused,
        cv2.COLOR_RGB2LAB
    )

    normal_lab = cv2.cvtColor(
        normal,
        cv2.COLOR_RGB2LAB
    )

    # Keep chroma extremely close to the normal exposure.
    fused_lab[:, :, 1] = (
        fused_lab[:, :, 1] * 0.25 +
        normal_lab[:, :, 1] * 0.75
    )

    fused_lab[:, :, 2] = (
        fused_lab[:, :, 2] * 0.25 +
        normal_lab[:, :, 2] * 0.75
    )

    result = cv2.cvtColor(
        fused_lab,
        cv2.COLOR_LAB2RGB
    )

    return np.clip(
        result,
        0.0,
        1.0
    )


# ============================================================
# FINAL TONE SAFETY
# ============================================================

def final_tone_safety(fused, normal):

    print("Applying final brightness safety...")

    fused_lum = luminance(fused)
    normal_lum = luminance(normal)

    ratio = normal_lum / (
        fused_lum + EPS
    )

    # Only small correction.
    ratio = np.clip(
        ratio,
        0.92,
        1.08
    )

    ratio = ratio[:, :, None]

    corrected = fused * ratio

    # Blend slightly with normal to guarantee natural output.
    corrected = (
        corrected * 0.90 +
        normal * 0.10
    )

    return np.clip(
        corrected,
        0.0,
        1.0
    )


# ============================================================
# MAIN PIPELINE
# ============================================================

def process(under_path, normal_path, over_path, output_path):

    print("\n==============================================")
    print(" ADAPTIVE PRODUCTION HDR FUSION")
    print("==============================================")

    print("\nLoading images...")

    under = load_image(under_path)
    normal = load_image(normal_path)
    over = load_image(over_path)

    print(
        f"Normal image size: "
        f"{normal.shape[1]} x {normal.shape[0]}"
    )

    # --------------------------------------------------------
    # Resize exposures if required
    # --------------------------------------------------------

    h, w = normal.shape[:2]

    if under.shape[:2] != (h, w):
        print("Resizing underexposed image...")
        under = cv2.resize(
            under,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    if over.shape[:2] != (h, w):
        print("Resizing overexposed image...")
        over = cv2.resize(
            over,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    # --------------------------------------------------------
    # Alignment
    # --------------------------------------------------------

    print("\nAligning exposures to normal exposure...")

    under = align_to_reference(
        normal,
        under
    )

    over = align_to_reference(
        normal,
        over
    )

    # --------------------------------------------------------
    # Adaptive fusion
    # --------------------------------------------------------

    fused, highlight_mask, shadow_mask = adaptive_fusion(
        under,
        normal,
        over
    )

    # --------------------------------------------------------
    # Restore detail
    # --------------------------------------------------------

    fused = restore_detail(
        fused,
        normal
    )

    # --------------------------------------------------------
    # Preserve original colors
    # --------------------------------------------------------

    fused = preserve_normal_color(
        fused,
        normal
    )

    # --------------------------------------------------------
    # Very subtle local contrast
    # --------------------------------------------------------

    fused = subtle_local_contrast(
        fused
    )

    # --------------------------------------------------------
    # Final brightness protection
    # --------------------------------------------------------

    fused = final_tone_safety(
        fused,
        normal
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    print("\nSaving final image...")

    save_image(
        output_path,
        fused
    )

    # Optional diagnostic masks.
    base_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(
        os.path.basename(output_path)
    )[0]

    highlight_path = os.path.join(
        base_dir,
        base_name + "_highlight_mask.png"
    )

    shadow_path = os.path.join(
        base_dir,
        base_name + "_shadow_mask.png"
    )

    cv2.imwrite(
        highlight_path,
        (
            np.clip(highlight_mask, 0, 1) *
            255
        ).astype(np.uint8)
    )

    cv2.imwrite(
        shadow_path,
        (
            np.clip(shadow_mask, 0, 1) *
            255
        ).astype(np.uint8)
    )

    print("\n==============================================")
    print(" FUSION COMPLETE")
    print("==============================================")

    print(f"\nFinal output:")
    print(f"  {output_path}")

    print("\nHighlight mask:")
    print(f"  {highlight_path}")

    print("\nShadow mask:")
    print(f"  {shadow_path}")

    print("\nPipeline:")
    print("  Under  -> highlight/window recovery")
    print("  Normal -> brightness + color anchor")
    print("  Over   -> controlled shadow recovery")

    print("\nNormal exposure remains dominant.")
    print("White balance and natural brightness are protected.")


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=
        "Production Adaptive HDR Exposure Fusion"
    )

    parser.add_argument(
        "--under",
        required=True,
        help="Path to underexposed image"
    )

    parser.add_argument(
        "--normal",
        required=True,
        help="Path to normal exposure image"
    )

    parser.add_argument(
        "--over",
        required=True,
        help="Path to overexposed image"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Path for final output"
    )

    args = parser.parse_args()

    process(
        args.under,
        args.normal,
        args.over,
        args.output
    )
