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
        0.0722 * img[:, :, 0] +
        0.7152 * img[:, :, 1] +
        0.2126 * img[:, :, 2]
    )


def smoothstep(x, low, high):
    x = np.clip(
        (x - low) / (high - low + EPS),
        0.0,
        1.0
    )

    return x * x * (3.0 - 2.0 * x)


def edge_smooth(weight):
    weight = weight.astype(np.float32)

    fine = weight

    medium = cv2.GaussianBlur(
        weight,
        (0, 0),
        sigmaX=8,
        sigmaY=8
    )

    large = cv2.GaussianBlur(
        weight,
        (0, 0),
        sigmaX=35,
        sigmaY=35
    )

    smooth = (
        0.20 * fine +
        0.45 * medium +
        0.35 * large
    )

    return np.maximum(smooth, EPS)


def exposure_quality(lum):
    sigma = 0.25

    quality = np.exp(
        -((lum - 0.5) ** 2) /
        (2 * sigma * sigma)
    )

    return quality.astype(np.float32)


def main():

    input_dir = "/home/prthick/FreeMEF/my_test/scene03"

    output_dir = (
        "/home/prthick/FreeMEF/production/output/"
        "test_scene03"
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
    print("ADAPTIVE HDR FUSION V3.2")
    print("SHADOW-AWARE + PATCH-FREE")
    print("====================================")

    under = read_image(under_path)
    normal = read_image(normal_path)
    over = read_image(over_path)

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

    print(f"\nResolution: {w} x {h}")

    # ==============================================
    # LUMINANCE
    # ==============================================

    lum_under = luminance(under)
    lum_normal = luminance(normal)
    lum_over = luminance(over)

    # ==============================================
    # HIGHLIGHT MASK
    # ==============================================

    highlight_mask = smoothstep(
        lum_normal,
        0.50,
        0.82
    )

    # ==============================================
    # GENERAL SHADOW MASK
    # ==============================================

    shadow_mask = 1.0 - smoothstep(
        lum_normal,
        0.20,
        0.55
    )

    # ==============================================
    # DEEP SHADOW MASK
    # ==============================================

    deep_shadow_mask = 1.0 - smoothstep(
        lum_normal,
        0.08,
        0.35
    )

    # ==============================================
    # OVEREXPOSED IMAGE USABLE DETAIL
    # ==============================================

    over_detail_mask = smoothstep(
        lum_over,
        0.08,
        0.30
    )

    # ==============================================
    # SHADOW RECOVERY MASK
    # ==============================================

    shadow_recovery_mask = (
        np.maximum(
            shadow_mask,
            deep_shadow_mask
        )
        *
        over_detail_mask
    )

    # ==============================================
    # EXPOSURE QUALITY
    # ==============================================

    q_under = exposure_quality(lum_under)
    q_normal = exposure_quality(lum_normal)
    q_over = exposure_quality(lum_over)

    # ==============================================
    # UNDER WEIGHT
    # ==============================================

    w_under = (
        0.10 +
        2.0 * highlight_mask +
        0.5 * q_under
    )

    # ==============================================
    # NORMAL WEIGHT
    # ==============================================

    w_normal = (
        2.5 +
        1.5 * q_normal -
        0.8 * deep_shadow_mask
    )

    w_normal = np.maximum(
        w_normal,
        0.5
    )

    # ==============================================
    # OVER WEIGHT
    # ==============================================

    w_over = (
        0.05 +
        2.5 * shadow_recovery_mask +
        0.7 * q_over
    )

    # ==============================================
    # SMOOTH WEIGHTS
    # ==============================================

    w_under = edge_smooth(w_under)
    w_normal = edge_smooth(w_normal)
    w_over = edge_smooth(w_over)

    # ==============================================
    # NORMALIZE WEIGHTS
    # ==============================================

    weight_sum = (
        w_under +
        w_normal +
        w_over +
        EPS
    )

    w_under /= weight_sum
    w_normal /= weight_sum
    w_over /= weight_sum

    # ==============================================
    # FUSION
    # ==============================================

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

    # ==============================================
    # SAVE DIAGNOSTICS
    # ==============================================

    cv2.imwrite(
        os.path.join(
            output_dir,
            "01_highlight_mask.png"
        ),
        np.uint8(
            np.clip(
                highlight_mask,
                0,
                1
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "02_shadow_mask.png"
        ),
        np.uint8(
            np.clip(
                shadow_mask,
                0,
                1
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "03_deep_shadow_mask.png"
        ),
        np.uint8(
            np.clip(
                deep_shadow_mask,
                0,
                1
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "04_shadow_recovery_mask.png"
        ),
        np.uint8(
            np.clip(
                shadow_recovery_mask,
                0,
                1
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "05_weight_under.png"
        ),
        np.uint8(
            np.clip(
                w_under,
                0,
                1
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "06_weight_normal.png"
        ),
        np.uint8(
            np.clip(
                w_normal,
                0,
                1
            ) * 255
        )
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "07_weight_over.png"
        ),
        np.uint8(
            np.clip(
                w_over,
                0,
                1
            ) * 255
        )
    )

    # ==============================================
    # FINAL OUTPUT
    # ==============================================

    final_path = os.path.join(
        output_dir,
        "FINAL_SCENE03.png"
    )

    cv2.imwrite(
        final_path,
        np.uint8(
            np.clip(
                fused,
                0,
                1
            ) * 255
        )
    )

    print("\n====================================")
    print("V3.2 COMPLETED SUCCESSFULLY")
    print("====================================")

    print(f"\nOutput: {final_path}")

    print("\nMean fusion weights:")
    print(f"Under : {w_under.mean():.4f}")
    print(f"Normal: {w_normal.mean():.4f}")
    print(f"Over  : {w_over.mean():.4f}")


if __name__ == "__main__":
    main()
