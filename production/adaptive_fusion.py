import cv2
import numpy as np
import os

EPS = 1e-8


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)

    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")

    return img.astype(np.float32) / 255.0


def luminance(img):
    return (
        0.2126 * img[:, :, 2] +
        0.7152 * img[:, :, 1] +
        0.0722 * img[:, :, 0]
    )


def soft_mask(x, low, high):
    x = np.clip(
        (x - low) / (high - low + EPS),
        0.0,
        1.0
    )

    return x * x * (3.0 - 2.0 * x)


def smooth_weight(mask):
    """
    Smooth weight maps while preserving their range.
    """

    min_val = mask.min()
    max_val = mask.max()

    mask_norm = (
        (mask - min_val) /
        (max_val - min_val + EPS)
    )

    mask_8 = np.uint8(
        np.clip(mask_norm * 255.0, 0, 255)
    )

    smooth = cv2.bilateralFilter(
        mask_8,
        d=9,
        sigmaColor=50,
        sigmaSpace=50
    )

    smooth = smooth.astype(np.float32) / 255.0

    return smooth * (max_val - min_val) + min_val


def main():

    # =====================================================
    # INPUT
    # =====================================================

    UNDER_PATH = (
        "/mnt/e/HDR_Project_Backups/"
        "HDR_BASELINE_V10_FINAL/raw_scenes/"
        "0J0A9350.jpg"
    )

    NORMAL_PATH = (
        "/mnt/e/HDR_Project_Backups/"
        "HDR_BASELINE_V10_FINAL/raw_scenes/"
        "0J0A9351.jpg"
    )

    OVER_PATH = (
        "/mnt/e/HDR_Project_Backups/"
        "HDR_BASELINE_V10_FINAL/raw_scenes/"
        "0J0A9352.jpg"
    )

    output_dir = (
        "/home/prthick/FreeMEF/"
        "production/output/shadow_aware_v2"
    )

    os.makedirs(output_dir, exist_ok=True)

    # =====================================================
    # LOAD
    # =====================================================

    under = read_image(UNDER_PATH)
    normal = read_image(NORMAL_PATH)
    over = read_image(OVER_PATH)

    h, w = normal.shape[:2]

    if under.shape[:2] != (h, w):
        under = cv2.resize(
            under, (w, h),
            interpolation=cv2.INTER_AREA
        )

    if over.shape[:2] != (h, w):
        over = cv2.resize(
            over, (w, h),
            interpolation=cv2.INTER_AREA
        )

    # =====================================================
    # LUMINANCE
    # =====================================================

    lum_normal = luminance(normal)

    # =====================================================
    # HIGHLIGHT DETECTION
    #
    # Use UNDER exposure only for bright areas.
    # =====================================================

    highlight_mask = soft_mask(
        lum_normal,
        low=0.70,
        high=0.92
    )

    # =====================================================
    # SHADOW DETECTION
    #
    # Stronger recovery in genuinely dark areas.
    # =====================================================

    shadow_mask = 1.0 - soft_mask(
        lum_normal,
        low=0.12,
        high=0.55
    )

    # Prevent tiny dark details/noise from receiving
    # excessive overexposure contribution.
    shadow_mask = np.clip(
        shadow_mask,
        0.0,
        1.0
    )

    # =====================================================
    # WEIGHTS
    # =====================================================

    # Underexposed image:
    # primarily windows and clipped highlights.
    w_under = 1.8 * highlight_mask

    # Overexposed image:
    # stronger contribution for dark furniture,
    # carpet and shadow regions.
    w_over = 1.5 * shadow_mask

    # Normal image remains the anchor.
    w_normal = (
        1.5
        + 2.0 * (
            1.0
            - np.maximum(
                highlight_mask,
                shadow_mask
            )
        )
    )

    # =====================================================
    # SMOOTH WEIGHTS
    # =====================================================

    w_under = smooth_weight(w_under)
    w_normal = smooth_weight(w_normal)
    w_over = smooth_weight(w_over)

    # =====================================================
    # NORMALIZE
    # =====================================================

    weight_sum = (
        w_under +
        w_normal +
        w_over +
        EPS
    )

    w_under /= weight_sum
    w_normal /= weight_sum
    w_over /= weight_sum

    # =====================================================
    # FUSION
    # =====================================================

    fused = (
        under * w_under[:, :, None] +
        normal * w_normal[:, :, None] +
        over * w_over[:, :, None]
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # =====================================================
    # SAVE DIAGNOSTICS
    # =====================================================

    cv2.imwrite(
        os.path.join(
            output_dir,
            "01_highlight_mask.png"
        ),
        np.uint8(highlight_mask * 255)
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "02_shadow_mask.png"
        ),
        np.uint8(shadow_mask * 255)
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "03_weight_under.png"
        ),
        np.uint8(w_under * 255)
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "04_weight_normal.png"
        ),
        np.uint8(w_normal * 255)
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "05_weight_over.png"
        ),
        np.uint8(w_over * 255)
    )

    # =====================================================
    # FINAL OUTPUT
    # =====================================================

    output_path = os.path.join(
        output_dir,
        "FINAL_SHADOW_AWARE_V2.png"
    )

    cv2.imwrite(
        output_path,
        np.uint8(fused * 255)
    )

    print("\n====================================")
    print("SHADOW-AWARE HDR FUSION COMPLETED")
    print("====================================")

    print(f"\nOutput: {output_path}")

    print("\nMean fusion weights:")
    print(f"Under : {w_under.mean():.4f}")
    print(f"Normal: {w_normal.mean():.4f}")
    print(f"Over  : {w_over.mean():.4f}")


if __name__ == "__main__":
    main()
