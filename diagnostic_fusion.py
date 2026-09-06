import os
import cv2
import numpy as np
import sys


def find_image(folder, names):
    for name in names:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    return None


def exposure_weight(img, sigma=0.20):
    x = img.astype(np.float32) / 255.0
    gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)

    weight = np.exp(
        -((gray - 0.5) ** 2) /
        (2.0 * sigma * sigma)
    )

    return weight


def main(scene_path):

    output_dir = os.path.join(
        "diagnostic_output",
        os.path.basename(scene_path)
    )

    os.makedirs(output_dir, exist_ok=True)

    # Find images supporting both .jpg and .JPG
    under_path = find_image(
        scene_path,
        ["01_under.jpg", "01_under.JPG"]
    )

    normal_path = find_image(
        scene_path,
        ["02_normal.jpg", "02_normal.JPG"]
    )

    over_path = find_image(
        scene_path,
        ["03_over.jpg", "03_over.JPG"]
    )

    if not all([under_path, normal_path, over_path]):
        print("\nERROR: Could not find all three exposure images.")
        print("Under :", under_path)
        print("Normal:", normal_path)
        print("Over  :", over_path)
        return

    under = cv2.imread(under_path)
    normal = cv2.imread(normal_path)
    over = cv2.imread(over_path)

    if under is None or normal is None or over is None:
        print("ERROR: Failed to load images.")
        return

    # Ensure identical dimensions
    h, w = normal.shape[:2]

    under = cv2.resize(under, (w, h))
    over = cv2.resize(over, (w, h))

    print("\n================================")
    print("DIAGNOSTIC FUSION")
    print("================================")

    print("Scene:", scene_path)
    print("Resolution:", w, "x", h)

    # -----------------------------------------
    # TEST A — NORMAL ONLY
    # -----------------------------------------

    print("\nGenerating TEST A: Normal image only...")

    cv2.imwrite(
        os.path.join(output_dir, "TEST_A_NORMAL_ONLY.png"),
        normal
    )

    # -----------------------------------------
    # TEST B — EQUAL FUSION
    # -----------------------------------------

    print("Generating TEST B: Equal fusion...")

    equal_fusion = (
        under.astype(np.float32) +
        normal.astype(np.float32) +
        over.astype(np.float32)
    ) / 3.0

    equal_fusion = np.clip(
        equal_fusion,
        0,
        255
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(output_dir, "TEST_B_EQUAL_FUSION.png"),
        equal_fusion
    )

    # -----------------------------------------
    # TEST C — EXPOSURE WEIGHT FUSION
    # -----------------------------------------

    print("Generating TEST C: Exposure-only fusion...")

    wu = exposure_weight(under)
    wn = exposure_weight(normal)
    wo = exposure_weight(over)

    eps = 1e-12

    weight_sum = wu + wn + wo + eps

    wu = wu / weight_sum
    wn = wn / weight_sum
    wo = wo / weight_sum

    exposure_fusion = (
        under.astype(np.float32) * wu[..., None] +
        normal.astype(np.float32) * wn[..., None] +
        over.astype(np.float32) * wo[..., None]
    )

    exposure_fusion = np.clip(
        exposure_fusion,
        0,
        255
    ).astype(np.uint8)

    cv2.imwrite(
        os.path.join(
            output_dir,
            "TEST_C_EXPOSURE_WEIGHT_FUSION.png"
        ),
        exposure_fusion
    )

    # Save weight maps for debugging
    cv2.imwrite(
        os.path.join(output_dir, "WEIGHT_UNDER.png"),
        (wu * 255).astype(np.uint8)
    )

    cv2.imwrite(
        os.path.join(output_dir, "WEIGHT_NORMAL.png"),
        (wn * 255).astype(np.uint8)
    )

    cv2.imwrite(
        os.path.join(output_dir, "WEIGHT_OVER.png"),
        (wo * 255).astype(np.uint8)
    )

    print("\nMean weights:")

    print(f"Under : {wu.mean():.4f}")
    print(f"Normal: {wn.mean():.4f}")
    print(f"Over  : {wo.mean():.4f}")

    print("\n================================")
    print("DONE")
    print("================================")

    print("\nOutput folder:")
    print(output_dir)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("\nUsage:")
        print("python diagnostic_fusion.py /path/to/scene")
        sys.exit(1)

    main(sys.argv[1])
