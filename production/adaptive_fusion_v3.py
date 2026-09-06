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
    x = np.clip((x - low) / (high - low + EPS), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def edge_smooth(weight):
    weight_8 = np.uint8(np.clip(weight * 255, 0, 255))

    smooth = cv2.bilateralFilter(
        weight_8,
        d=9,
        sigmaColor=50,
        sigmaSpace=50
    )

    return smooth.astype(np.float32) / 255.0


def exposure_quality(lum):
    """
    Gives higher weight to well-exposed pixels.
    """
    sigma = 0.25

    quality = np.exp(
        -((lum - 0.5) ** 2) /
        (2 * sigma * sigma)
    )

    return quality


def main():

    # --------------------------------------------------
    # INPUT
    # --------------------------------------------------

    input_dir = "/home/prthick/FreeMEF/my_test_small/scene001"

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

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------

    output_dir = (
        "/home/prthick/FreeMEF/production/output/"
        "adaptive_fusion_v3"
    )

    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # LOAD IMAGES
    # --------------------------------------------------

    under = read_image(under_path)
    normal = read_image(normal_path)
    over = read_image(over_path)

    # --------------------------------------------------
    # ENSURE SAME SIZE
    # --------------------------------------------------

    h, w = normal.shape[:2]

    under = cv2.resize(under, (w, h))
    over = cv2.resize(over, (w, h))

    # --------------------------------------------------
    # LUMINANCE
    # --------------------------------------------------

    lum_under = luminance(under)
    lum_normal = luminance(normal)
    lum_over = luminance(over)

    # --------------------------------------------------
    # HIGHLIGHT MASK
    #
    # Start earlier so underexposed image contributes
    # more than before.
    # --------------------------------------------------

    highlight_mask = smoothstep(
        lum_normal,
        0.50,
        0.82
    )

    # --------------------------------------------------
    # SHADOW MASK
    # --------------------------------------------------

    shadow_mask = 1.0 - smoothstep(
        lum_normal,
        0.18,
        0.55
    )

    # --------------------------------------------------
    # EXPOSURE QUALITY
    # --------------------------------------------------

    q_under = exposure_quality(lum_under)
    q_normal = exposure_quality(lum_normal)
    q_over = exposure_quality(lum_over)

    # --------------------------------------------------
    # INITIAL WEIGHTS
    #
    # Normal remains the anchor.
    # --------------------------------------------------

    w_under = (
        0.10 +
        2.0 * highlight_mask +
        0.5 * q_under
    )

    w_normal = (
        2.5 +
        1.5 * q_normal
    )

    w_over = (
        0.10 +
        1.5 * shadow_mask +
        0.5 * q_over
    )

    # --------------------------------------------------
    # SMOOTH WEIGHTS
    # --------------------------------------------------

    w_under = edge_smooth(w_under)
    w_normal = edge_smooth(w_normal)
    w_over = edge_smooth(w_over)

    # --------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------

    weight_sum = (
        w_under +
        w_normal +
        w_over +
        EPS
    )

    w_under /= weight_sum
    w_normal /= weight_sum
    w_over /= weight_sum

    # --------------------------------------------------
    # FUSION
    # --------------------------------------------------

    fused = (
        under * w_under[:, :, None] +
        normal * w_normal[:, :, None] +
        over * w_over[:, :, None]
    )

    fused = np.clip(fused, 0.0, 1.0)

    # --------------------------------------------------
    # SAVE DIAGNOSTICS
    # --------------------------------------------------

    cv2.imwrite(
        os.path.join(output_dir, "01_highlight_mask.png"),
        np.uint8(highlight_mask * 255)
    )

    cv2.imwrite(
        os.path.join(output_dir, "02_shadow_mask.png"),
        np.uint8(shadow_mask * 255)
    )

    cv2.imwrite(
        os.path.join(output_dir, "03_weight_under.png"),
        np.uint8(w_under * 255)
    )

    cv2.imwrite(
        os.path.join(output_dir, "04_weight_normal.png"),
        np.uint8(w_normal * 255)
    )

    cv2.imwrite(
        os.path.join(output_dir, "05_weight_over.png"),
        np.uint8(w_over * 255)
    )

    cv2.imwrite(
        os.path.join(
            output_dir,
            "FINAL_ADAPTIVE_FUSION_V3.png"
        ),
        np.uint8(fused * 255)
    )

    print("\n====================================")
    print("ADAPTIVE HDR FUSION V3 COMPLETED")
    print("====================================")

    print(f"\nOutput: {output_dir}")

    print("\nMean fusion weights:")
    print(f"Under : {w_under.mean():.4f}")
    print(f"Normal: {w_normal.mean():.4f}")
    print(f"Over  : {w_over.mean():.4f}")


if __name__ == "__main__":
    main()
