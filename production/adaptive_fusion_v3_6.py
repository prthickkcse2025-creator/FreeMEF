import os
import cv2
import numpy as np


# ============================================================
# ADAPTIVE HDR FUSION V3.6
#
# NORMAL EXPOSURE = MAIN COLOR / WHITE BALANCE ANCHOR
#
# UNDEREXPOSED IMAGE:
#     Used only for clipped / near-clipped highlights.
#
# OVEREXPOSED IMAGE:
#     Used only for genuinely dark regions where useful
#     shadow detail exists.
#
# IMPORTANT:
#     - No blur is applied to the original images.
#     - Only fusion weight maps are smoothed.
#     - Scene thresholds are calculated automatically.
# ============================================================


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(
            f"\nCould not read image:\n{path}\n"
        )

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.astype(np.float32) / 255.0


def write_image(path, img):
    img = np.clip(img, 0.0, 1.0)

    img = (img * 255.0 + 0.5).astype(np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    cv2.imwrite(
        path,
        img,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            98
        ]
    )


def luminance(img):
    return (
        0.2126 * img[:, :, 0] +
        0.7152 * img[:, :, 1] +
        0.0722 * img[:, :, 2]
    )


def smoothstep(x, edge0, edge1):

    if edge1 <= edge0:
        edge1 = edge0 + 1e-6

    t = np.clip(
        (x - edge0) / (edge1 - edge0),
        0.0,
        1.0
    )

    return t * t * (3.0 - 2.0 * t)


def local_detail(img):

    gray = luminance(img)

    blur = cv2.GaussianBlur(
        gray,
        (0, 0),
        2.0
    )

    detail = np.abs(gray - blur)

    return detail


def ensure_same_size(under, normal, over):

    h, w = normal.shape[:2]

    if under.shape[:2] != (h, w):
        under = cv2.resize(
            under,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    if over.shape[:2] != (h, w):
        over = cv2.resize(
            over,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )

    return under, normal, over


def main():

    # ========================================================
    # INPUT DIRECTORY
    # ========================================================

    input_dir = (
        "/home/prthick/FreeMEF/"
        "my_test/scene005"
    )

    under_path = os.path.join(
        input_dir,
        "01_under.jpg"
    )

    normal_path = os.path.join(
        input_dir,
        "02_normal.jpg"
    )

    over_path = os.path.join(
        input_dir,
        "03_over.jpg"
    )

    # ========================================================
    # OUTPUT DIRECTORY
    # ========================================================

    output_dir = (
        "/home/prthick/FreeMEF/"
        "production/output/"
        "adaptive_fusion_v3_6"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        "FINAL_ADAPTIVE_FUSION_V3_6.jpg"
    )

    # ========================================================
    # PRINT INFO
    # ========================================================

    print()
    print("=" * 60)
    print("ADAPTIVE HDR FUSION V3.6")
    print("SCENE-ADAPTIVE NORMAL-ANCHOR FUSION")
    print("=" * 60)
    print()

    print("Under :", under_path)
    print("Normal:", normal_path)
    print("Over  :", over_path)
    print()

    # ========================================================
    # LOAD IMAGES
    # ========================================================

    under = read_image(under_path)
    normal = read_image(normal_path)
    over = read_image(over_path)

    under, normal, over = ensure_same_size(
        under,
        normal,
        over
    )

    h, w = normal.shape[:2]

    print(
        f"Resolution: {w} x {h}"
    )

    # ========================================================
    # LUMINANCE
    # ========================================================

    lum_under = luminance(under)
    lum_normal = luminance(normal)
    lum_over = luminance(over)

    # ========================================================
    # SCENE-ADAPTIVE THRESHOLDS
    #
    # Thresholds are calculated from the NORMAL exposure.
    # This prevents one fixed setting from being used on
    # every scene.
    # ========================================================

    p01 = float(
        np.percentile(lum_normal, 1)
    )

    p05 = float(
        np.percentile(lum_normal, 5)
    )

    p10 = float(
        np.percentile(lum_normal, 10)
    )

    p90 = float(
        np.percentile(lum_normal, 90)
    )

    p95 = float(
        np.percentile(lum_normal, 95)
    )

    p99 = float(
        np.percentile(lum_normal, 99)
    )

    print()
    print("Scene luminance statistics:")
    print(f"P01: {p01:.4f}")
    print(f"P05: {p05:.4f}")
    print(f"P10: {p10:.4f}")
    print(f"P90: {p90:.4f}")
    print(f"P95: {p95:.4f}")
    print(f"P99: {p99:.4f}")

    # ========================================================
    # SHADOW THRESHOLDS
    #
    # Conservative on purpose.
    # Do NOT lift medium-dark areas unnecessarily.
    # ========================================================

    shadow_start = np.clip(
        max(
            p01,
            0.04
        ),
        0.03,
        0.12
    )

    shadow_end = np.clip(
        max(
            p10,
            shadow_start + 0.08
        ),
        0.14,
        0.32
    )

    # ========================================================
    # HIGHLIGHT THRESHOLDS
    #
    # Underexposure should only contribute near bright /
    # clipped areas.
    # ========================================================

    highlight_start = np.clip(
        min(
            p90,
            0.85
        ),
        0.60,
        0.88
    )

    highlight_end = np.clip(
        max(
            p99,
            highlight_start + 0.06
        ),
        highlight_start + 0.02,
        0.98
    )

    # ========================================================
    # SHADOW MASK
    #
    # 1 in dark regions
    # 0 outside dark regions
    # ========================================================

    shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            shadow_start,
            shadow_end
        )
    )

    # ========================================================
    # HIGHLIGHT MASK
    #
    # 1 in very bright regions
    # ========================================================

    highlight_mask = smoothstep(
        lum_normal,
        highlight_start,
        highlight_end
    )

    # ========================================================
    # EXPOSURE DIFFERENCE
    #
    # Only use another exposure when it contains a meaningful
    # amount of additional information.
    # ========================================================

    shadow_gain = np.clip(
        lum_over - lum_normal,
        0.0,
        1.0
    )

    highlight_gain = np.clip(
        lum_normal - lum_under,
        0.0,
        1.0
    )

    # ========================================================
    # DETAIL ANALYSIS
    #
    # Avoid replacing good normal pixels with flat pixels from
    # another exposure.
    # ========================================================

    detail_normal = local_detail(normal)
    detail_under = local_detail(under)
    detail_over = local_detail(over)

    eps = 1e-6

    shadow_detail_ratio = (
        detail_over /
        (detail_normal + eps)
    )

    highlight_detail_ratio = (
        detail_under /
        (detail_normal + eps)
    )

    shadow_detail_bonus = np.clip(
        (shadow_detail_ratio - 0.75) / 0.75,
        0.0,
        1.0
    )

    highlight_detail_bonus = np.clip(
        (highlight_detail_ratio - 0.75) / 0.75,
        0.0,
        1.0
    )

    # ========================================================
    # SHADOW RECOVERY CONFIDENCE
    #
    # Conservative:
    # overexposed image should not dominate shadows.
    # ========================================================

    shadow_recovery = (
        shadow_mask *
        np.clip(
            shadow_gain * 3.0,
            0.0,
            1.0
        ) *
        (
            0.65 +
            0.35 * shadow_detail_bonus
        )
    )

    # ========================================================
    # HIGHLIGHT RECOVERY CONFIDENCE
    # ========================================================

    highlight_recovery = (
        highlight_mask *
        np.clip(
            highlight_gain * 3.0,
            0.0,
            1.0
        ) *
        (
            0.65 +
            0.35 * highlight_detail_bonus
        )
    )

    # ========================================================
    # FUSION WEIGHTS
    #
    # NORMAL IMAGE IS THE PRIMARY IMAGE.
    # ========================================================

    w_normal = (
        5.0 +
        1.0 * np.clip(
            1.0 - shadow_recovery,
            0.0,
            1.0
        ) +
        1.0 * np.clip(
            1.0 - highlight_recovery,
            0.0,
            1.0
        )
    )

    # Under exposure:
    # Only highlight recovery.
    w_under = (
        0.02 +
        3.0 * highlight_recovery
    )

    # Over exposure:
    # Conservative shadow recovery.
    w_over = (
        0.02 +
        1.25 * shadow_recovery
    )

    # ========================================================
    # SMOOTH WEIGHT MAPS ONLY
    #
    # Original images remain untouched.
    # ========================================================

    w_under = cv2.GaussianBlur(
        w_under.astype(np.float32),
        (0, 0),
        8
    )

    w_normal = cv2.GaussianBlur(
        w_normal.astype(np.float32),
        (0, 0),
        8
    )

    w_over = cv2.GaussianBlur(
        w_over.astype(np.float32),
        (0, 0),
        8
    )

    # ========================================================
    # PREVENT NEGATIVE WEIGHTS
    # ========================================================

    w_under = np.maximum(
        w_under,
        0.0
    )

    w_normal = np.maximum(
        w_normal,
        0.01
    )

    w_over = np.maximum(
        w_over,
        0.0
    )

    # ========================================================
    # NORMALIZE WEIGHTS
    # ========================================================

    weight_sum = (
        w_under +
        w_normal +
        w_over +
        1e-8
    )

    w_under = (
        w_under /
        weight_sum
    )

    w_normal = (
        w_normal /
        weight_sum
    )

    w_over = (
        w_over /
        weight_sum
    )

    # ========================================================
    # HDR FUSION
    # ========================================================

    fused = (
        under *
        w_under[:, :, None] +

        normal *
        w_normal[:, :, None] +

        over *
        w_over[:, :, None]
    )

    # ========================================================
    # COLOR PROTECTION
    #
    # Keep final chromatic appearance close to NORMAL.
    #
    # This is NOT a global brightness change.
    # ========================================================

    normal_safe = (
        normal +
        1e-6
    )

    fused_safe = (
        fused +
        1e-6
    )

    normal_lum = luminance(
        normal_safe
    )

    fused_lum = luminance(
        fused_safe
    )

    # Preserve normal chromaticity.
    normal_chroma = (
        normal_safe /
        (
            normal_lum[:, :, None] +
            1e-6
        )
    )

    # Use fused luminance with normal color ratios.
    chroma_protected = (
        normal_chroma *
        fused_lum[:, :, None]
    )

    # Blend protection conservatively.
    fused = (
        0.70 * fused +
        0.30 * chroma_protected
    )

    # ========================================================
    # FINAL SAFETY CLIP
    # ========================================================

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ========================================================
    # SAVE OUTPUT
    # ========================================================

    write_image(
        output_path,
        fused
    )

    # ========================================================
    # SAVE DIAGNOSTIC WEIGHT MAPS
    # ========================================================

    cv2.imwrite(
        os.path.join(
            output_dir,
            "weight_under.png"
        ),
        (
            np.clip(
                w_under,
                0.0,
                1.0
            ) * 255
        ).astype(np.uint8)
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "weight_normal.png"
        ),
        (
            np.clip(
                w_normal,
                0.0,
                1.0
            ) * 255
        ).astype(np.uint8)
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "weight_over.png"
        ),
        (
            np.clip(
                w_over,
                0.0,
                1.0
            ) * 255
        ).astype(np.uint8)
    )

    print()
    print("=" * 60)
    print("ADAPTIVE HDR FUSION V3.6 COMPLETED")
    print("=" * 60)
    print()

    print(
        f"Output: {output_path}"
    )

    print()
    print("Mean fusion weights:")
    print(
        f"Under : {np.mean(w_under):.4f}"
    )
    print(
        f"Normal: {np.mean(w_normal):.4f}"
    )
    print(
        f"	Over  : {np.mean(w_over):.4f}"
    )

    print()


if __name__ == "__main__":
    main()
