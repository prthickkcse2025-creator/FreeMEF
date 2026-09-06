import cv2
import numpy as np
import os
import glob

EPS = 1e-8


# ==================================================
# IMAGE LOADING
# ==================================================

def read_image(path):

    img = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if img is None:
        raise FileNotFoundError(
            f"Could not read image: {path}"
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    return img.astype(np.float32) / 255.0


# ==================================================
# FIND IMAGE
# ==================================================

def find_image(input_dir, names):

    for name in names:

        path = os.path.join(
            input_dir,
            name
        )

        if os.path.exists(path):
            return path

    # Case-insensitive fallback
    all_files = glob.glob(
        os.path.join(
            input_dir,
            "*"
        )
    )

    name_bases = [
        os.path.splitext(name)[0].lower()
        for name in names
    ]

    for path in all_files:

        filename = os.path.basename(path)

        base = os.path.splitext(
            filename
        )[0].lower()

        ext = os.path.splitext(
            filename
        )[1].lower()

        if ext in [".jpg", ".jpeg", ".png", ".tif", ".tiff"]:

            if base in name_bases:
                return path

    # Generic fallback
    for name in names:

        base = os.path.splitext(
            name
        )[0]

        matches = glob.glob(
            os.path.join(
                input_dir,
                base + ".*"
            )
        )

        if matches:
            return matches[0]

    raise FileNotFoundError(
        f"\nCould not find image in:\n{input_dir}\n\n"
        f"Tried:\n{names}\n"
    )


# ==================================================
# LUMINANCE
# ==================================================

def luminance(img):

    return (
        0.2126 * img[:, :, 0] +
        0.7152 * img[:, :, 1] +
        0.0722 * img[:, :, 2]
    )


# ==================================================
# SMOOTHSTEP
# ==================================================

def smoothstep(x, edge0, edge1):

    t = np.clip(
        (x - edge0) /
        (edge1 - edge0 + EPS),
        0.0,
        1.0
    )

    return (
        t * t *
        (3.0 - 2.0 * t)
    )


# ==================================================
# EXPOSURE QUALITY
# ==================================================

def exposure_quality(lum):

    sigma = 0.25

    quality = np.exp(
        -(
            (lum - 0.5) ** 2
        ) /
        (
            2.0 * sigma * sigma
        )
    )

    return quality.astype(
        np.float32
    )


# ==================================================
# SAVE MAP
# ==================================================

def save_map(path, img):

    img = np.clip(
        img,
        0.0,
        1.0
    )

    img = np.uint8(
        img * 255.0
    )

    cv2.imwrite(
        path,
        img
    )


# ==================================================
# MAIN
# ==================================================

def main():

    print()
    print("====================================")
    print("ADAPTIVE HDR FUSION V3.8")
    print("V3.3 BRIGHTNESS + DEPTH PRESERVATION")
    print("====================================")
    print()

    # ==============================================
    # INPUT DIRECTORY
    #
    # CHANGE THIS TO YOUR CURRENT SCENE
    # ==============================================

    input_dir = (
        "/home/prthick/FreeMEF/"
        "my_test/scene005"
    )

    # ==============================================
    # OUTPUT DIRECTORY
    # ==============================================

    output_dir = (
        "/home/prthick/FreeMEF/"
        "production/output/"
        "adaptive_fusion_v3_8"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # ==============================================
    # FIND UNDEREXPOSED IMAGE
    # ==============================================

    under_path = find_image(
        input_dir,
        [
            "01_under.JPG",
            "01_under.jpg",
            "01_under.JPEG",
            "01_under.jpeg",
            "01_under.PNG",
            "01_under.png",
            "under.JPG",
            "under.jpg",
            "under.jpeg",
            "under.png"
        ]
    )

    # ==============================================
    # FIND NORMAL IMAGE
    # ==============================================

    normal_path = find_image(
        input_dir,
        [
            "02_normal.JPG",
            "02_normal.jpg",
            "02_normal.JPEG",
            "02_normal.jpeg",
            "02_normal.PNG",
            "02_normal.png",
            "normal.JPG",
            "normal.jpg",
            "normal.jpeg",
            "normal.png"
        ]
    )

    # ==============================================
    # FIND OVEREXPOSED IMAGE
    # ==============================================

    over_path = find_image(
        input_dir,
        [
            "03_over.JPG",
            "03_over.jpg",
            "03_over.JPEG",
            "03_over.jpeg",
            "03_over.PNG",
            "03_over.png",
            "over.JPG",
            "over.jpg",
            "over.jpeg",
            "over.png"
        ]
    )

    print("Under :", under_path)
    print("Normal:", normal_path)
    print("Over  :", over_path)
    print()

    # ==============================================
    # LOAD IMAGES
    # ==============================================

    under = read_image(
        under_path
    )

    normal = read_image(
        normal_path
    )

    over = read_image(
        over_path
    )

    # ==============================================
    # MATCH RESOLUTION
    # ==============================================

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

    print(
        f"Resolution: {w} x {h}"
    )

    # ==============================================
    # LUMINANCE
    # ==============================================

    lum_under = luminance(
        under
    )

    lum_normal = luminance(
        normal
    )

    lum_over = luminance(
        over
    )

    # ==============================================
    # HIGHLIGHT DETECTION
    #
    # SAME AS V3.3
    # ==============================================

    highlight_mask = smoothstep(
        lum_normal,
        0.55,
        0.82
    )

    # ==============================================
    # SHADOW DETECTION
    #
    # KEPT FOR DIAGNOSTICS
    #
    # V3.8 DOES NOT USE THIS BROAD MASK
    # FOR STRONG EXPOSURE RECOVERY.
    # ==============================================

    shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.18,
            0.55
        )
    )

    # ==============================================
    # V3.8 DEPTH-PRESERVING DEEP SHADOW DETECTION
    #
    # IMPORTANT:
    #
    # V3.3 allowed recovery into medium-dark areas.
    #
    # This narrower mask activates mainly in genuinely
    # deep shadows, preserving normal-exposure depth.
    # ==============================================

    deep_shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.04,
            0.24
        )
    )

    # ==============================================
    # OVEREXPOSED DETAIL VALIDITY
    #
    # Only use the overexposed image if it actually
    # contains usable information.
    # ==============================================

    over_detail_mask = smoothstep(
        lum_over,
        0.04,
        0.20
    )

    # ==============================================
    # V3.8 SHADOW RECOVERY
    #
    # ONLY DEEP SHADOWS ARE RECOVERED.
    #
    # Medium shadows remain anchored to the normal image.
    # ==============================================

    shadow_recovery_mask = (
        deep_shadow_mask *
        over_detail_mask
    )

    shadow_recovery_mask = np.clip(
        shadow_recovery_mask,
        0.0,
        1.0
    )

    # ==============================================
    # PATCH-FREE MASK SMOOTHING
    #
    # ONLY MASKS ARE SMOOTHED.
    # ORIGINAL IMAGES REMAIN SHARP.
    # ==============================================

    highlight_mask = cv2.GaussianBlur(
        highlight_mask.astype(np.float32),
        (0, 0),
        12
    )

    shadow_mask = cv2.GaussianBlur(
        shadow_mask.astype(np.float32),
        (0, 0),
        15
    )

    deep_shadow_mask = cv2.GaussianBlur(
        deep_shadow_mask.astype(np.float32),
        (0, 0),
        12
    )

    shadow_recovery_mask = cv2.GaussianBlur(
        shadow_recovery_mask.astype(np.float32),
        (0, 0),
        12
    )

    shadow_recovery_mask = np.clip(
        shadow_recovery_mask,
        0.0,
        1.0
    )

    # ==============================================
    # EXPOSURE QUALITY
    # ==============================================

    q_under = exposure_quality(
        lum_under
    )

    q_normal = exposure_quality(
        lum_normal
    )

    q_over = exposure_quality(
        lum_over
    )

    # ==============================================
    # UNDEREXPOSED IMAGE WEIGHT
    #
    # SAME V3.3 BEHAVIOR
    # ==============================================

    w_under = (
        0.10 +
        2.0 * highlight_mask +
        0.5 * q_under
    )

    # ==============================================
    # V3.8 NORMAL IMAGE WEIGHT
    #
    # NORMAL IMAGE REMAINS THE MAIN ANCHOR.
    #
    # V3.3 reduced normal contribution too strongly
    # in dark areas:
    #
    # -0.8 * deep_shadow_mask
    #
    # This is reduced substantially so the natural
    # depth and contrast of the medium exposure remain.
    # ==============================================

    w_normal = (
        2.5 +
        1.5 * q_normal -
        0.25 * deep_shadow_mask
    )

    w_normal = np.maximum(
        w_normal,
        0.75
    )

    # ==============================================
    # V3.8 OVEREXPOSED IMAGE WEIGHT
    #
    # CRITICAL CORRECTION:
    #
    # q_over is no longer allowed to increase the
    # overexposed contribution across the entire image.
    #
    # The overexposed image contributes mainly where
    # genuine deep-shadow recovery is needed.
    #
    # This preserves medium-shadow depth.
    # ==============================================

    w_over = (
        0.03 +
        1.20 * shadow_recovery_mask +
        0.20 * q_over *
        shadow_recovery_mask
    )

    # ==============================================
    # PATCH-FREE WEIGHT SMOOTHING
    #
    # ONLY WEIGHTS ARE SMOOTHED.
    # IMAGES ARE NOT BLURRED.
    # ==============================================

    w_under = cv2.GaussianBlur(
        w_under.astype(np.float32),
        (0, 0),
        15
    )

    w_normal = cv2.GaussianBlur(
        w_normal.astype(np.float32),
        (0, 0),
        15
    )

    w_over = cv2.GaussianBlur(
        w_over.astype(np.float32),
        (0, 0),
        12
    )

    # ==============================================
    # PREVENT NEGATIVE WEIGHTS
    # ==============================================

    w_under = np.maximum(
        w_under,
        0.0
    )

    w_normal = np.maximum(
        w_normal,
        0.0
    )

    w_over = np.maximum(
        w_over,
        0.0
    )

    # ==============================================
    # NORMALIZE WEIGHTS
    # ==============================================

    weight_sum = (
        w_under +
        w_normal +
        w_over +
        EPS
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

    # ==============================================
    # HDR FUSION
    #
    # ORIGINAL IMAGES ARE NOT BLURRED.
    # ==============================================

    fused = (
        under *
        w_under[:, :, None] +

        normal *
        w_normal[:, :, None] +

        over *
        w_over[:, :, None]
    )

    fused = np.clip(
        fused,
        0.0,
        1.0
    )

    # ==============================================
    # SAVE DIAGNOSTIC MAPS
    # ==============================================

    save_map(
        os.path.join(
            output_dir,
            "01_highlight_mask.png"
        ),
        highlight_mask
    )

    save_map(
        os.path.join(
            output_dir,
            "02_shadow_mask.png"
        ),
        shadow_mask
    )

    save_map(
        os.path.join(
            output_dir,
            "03_deep_shadow_mask.png"
        ),
        deep_shadow_mask
    )

    save_map(
        os.path.join(
            output_dir,
            "04_shadow_recovery_mask.png"
        ),
        shadow_recovery_mask
    )

    save_map(
        os.path.join(
            output_dir,
            "05_weight_under.png"
        ),
        w_under
    )

    save_map(
        os.path.join(
            output_dir,
            "06_weight_normal.png"
        ),
        w_normal
    )

    save_map(
        os.path.join(
            output_dir,
            "07_weight_over.png"
        ),
        w_over
    )

    # ==============================================
    # SAVE FINAL IMAGE
    # ==============================================

    fused_bgr = cv2.cvtColor(
        np.uint8(
            fused * 255.0
        ),
        cv2.COLOR_RGB2BGR
    )

    output_path = os.path.join(
        output_dir,
        "FINAL_ADAPTIVE_FUSION_V3_8.jpg"
    )

    cv2.imwrite(
        output_path,
        fused_bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            100
        ]
    )

    # ==============================================
    # PRINT RESULTS
    # ==============================================

    print()
    print("====================================")
    print("ADAPTIVE HDR FUSION V3.8 COMPLETED")
    print("V3.3 BRIGHTNESS + DEPTH PRESERVATION")
    print("====================================")
    print()

    print(
        "Output:",
        output_path
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
        f"Over  : {np.mean(w_over):.4f}"
    )


# ==================================================
# ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
