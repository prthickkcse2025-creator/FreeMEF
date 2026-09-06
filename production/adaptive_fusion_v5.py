import os
import sys
import cv2
import numpy as np

EPS = 1e-8


def read_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read: {path}")
    return img.astype(np.float32) / 255.0


def save_image(path, img):
    img = np.uint8(np.clip(img, 0.0, 1.0) * 255.0)
    cv2.imwrite(path, img)


def luminance(img):
    return (
        0.0722 * img[:, :, 0] +
        0.7152 * img[:, :, 1] +
        0.2126 * img[:, :, 2]
    )


def smoothstep(x, low, high):
    x = np.clip((x - low) / (high - low + EPS), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def multiscale_smooth(weight):
    h, w = weight.shape

    medium = cv2.GaussianBlur(
        weight,
        (0, 0),
        sigmaX=12,
        sigmaY=12
    )

    small = cv2.resize(
        weight,
        (max(1, w // 8), max(1, h // 8)),
        interpolation=cv2.INTER_AREA
    )

    large = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_CUBIC
    )

    result = (
        0.20 * weight +
        0.35 * medium +
        0.45 * large
    )

    return np.maximum(result, EPS)


def illumination_refinement(fused, normal):
    fused_lum = luminance(fused)
    normal_lum = luminance(normal)

    fused_base = cv2.GaussianBlur(
        fused_lum,
        (0, 0),
        sigmaX=35,
        sigmaY=35
    )

    normal_base = cv2.GaussianBlur(
        normal_lum,
        (0, 0),
        sigmaX=35,
        sigmaY=35
    )

    ratio = (
        normal_base + EPS
    ) / (
        fused_base + EPS
    )

    ratio = np.clip(ratio, 0.88, 1.12)

    strength = 0.18

    correction = (
        1.0 +
        strength * (ratio - 1.0)
    )

    result = fused * correction[:, :, None]

    return np.clip(result, 0.0, 1.0)


def main():

    input_dir = "/home/prthick/FreeMEF/my_test_small/scene001"

    output_dir = (
        "/home/prthick/FreeMEF/"
        "production/output/adaptive_fusion_v5"
    )

    os.makedirs(output_dir, exist_ok=True)

    under_path = os.path.join(
        input_dir,
        "01_under.JPG"
    )

    normal_path = os.path.join(
        input_dir,
        "02_normal.JPG"
    )

    over_path = os.path.join(
        input_dir,
        "03_over.JPG"
    )

    print("\n====================================")
    print("ADAPTIVE HDR FUSION V5")
    print("ILLUMINATION REFINEMENT")
    print("====================================")

    under = read_image(under_path)
    normal = read_image(normal_path)
    over = read_image(over_path)

    h, w = normal.shape[:2]

    if under.shape[:2] != (h, w):
        under = cv2.resize(under, (w, h))

    if over.shape[:2] != (h, w):
        over = cv2.resize(over, (w, h))

    print(f"\nResolution: {w} x {h}")

    lum_normal = luminance(normal)

    highlight_mask = smoothstep(
        lum_normal,
        0.50,
        0.82
    )

    shadow_mask = 1.0 - smoothstep(
        lum_normal,
        0.18,
        0.55
    )

    w_under = (
        0.10 +
        2.0 * highlight_mask
    )

    w_normal = (
        2.8 +
        1.0 * (
            1.0 -
            np.maximum(
                highlight_mask,
                shadow_mask
            )
        )
    )

    w_over = (
        0.10 +
        1.8 * shadow_mask
    )

    # Smooth ONLY the weight maps
    w_under = multiscale_smooth(w_under)
    w_normal = multiscale_smooth(w_normal)
    w_over = multiscale_smooth(w_over)

    weight_sum = (
        w_under +
        w_normal +
        w_over +
        EPS
    )

    w_under /= weight_sum
    w_normal /= weight_sum
    w_over /= weight_sum

    fused = (
        under * w_under[:, :, None] +
        normal * w_normal[:, :, None] +
        over * w_over[:, :, None]
    )

    fused = np.clip(fused, 0.0, 1.0)

    # V5 gentle illumination refinement
    final = illumination_refinement(
        fused,
        normal
    )

    save_image(
        os.path.join(
            output_dir,
            "01_highlight_mask.png"
        ),
        highlight_mask
    )

    save_image(
        os.path.join(
            output_dir,
            "02_shadow_mask.png"
        ),
        shadow_mask
    )

    save_image(
        os.path.join(
            output_dir,
            "03_weight_under.png"
        ),
        w_under
    )

    save_image(
        os.path.join(
            output_dir,
            "04_weight_normal.png"
        ),
        w_normal
    )

    save_image(
        os.path.join(
            output_dir,
            "05_weight_over.png"
        ),
        w_over
    )

    final_path = os.path.join(
        output_dir,
        "FINAL_ADAPTIVE_FUSION_V5.png"
    )

    save_image(final_path, final)

    print("\n====================================")
    print("V5 COMPLETED SUCCESSFULLY")
    print("====================================")

    print(f"\nOutput: {final_path}")

    print("\nMean fusion weights:")
    print(f"Under : {w_under.mean():.4f}")
    print(f"Normal: {w_normal.mean():.4f}")
    print(f"Over  : {w_over.mean():.4f}")


if __name__ == "__main__":
    main()

