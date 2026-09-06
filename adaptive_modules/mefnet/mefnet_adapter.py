#!/usr/bin/env python3

"""
MEF-NET CANDIDATE 3
===================

Direct inference adapter for the cloned MEF-Net repository.

IMPORTANT:
    We use the actual MEF-Net neural network from:

        adaptive_modules/mefnet/MEFNet_py3/e2emef.py

    and the actual pretrained checkpoint:

        adaptive_modules/mefnet/MEFNet_py3/checkpoint/
        MEFNet_release.pt

    We DO NOT use the old TrainModel.py / ImageDataset.py
    inference wrapper because it depends on obsolete
    Python/PyTorch/Pandas behavior.

    This adapter reproduces the actual inference logic:

        RGB exposures
             ↓
        RGB -> YCbCr
             ↓
        high-res Y
        low-res Y
             ↓
        E2EMEF
             ↓
        predicted high-res weights
             ↓
        luminance fusion
             ↓
        chroma fusion
             ↓
        YCbCr -> RGB
             ↓
        final image

Original cloned repository remains untouched.
"""

import os
import sys
import argparse

import cv2
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

MEFNET_DIR = os.path.join(
    PROJECT_ROOT,
    "adaptive_modules",
    "mefnet",
    "MEFNet_py3"
)

CHECKPOINT = os.path.join(
    MEFNET_DIR,
    "checkpoint",
    "MEFNet_release.pt"
)


# ============================================================
# SETTINGS
# ============================================================

# Match the official MEF-Net test behavior:
# smaller image edge is resized to 2048.
HIGH_SIZE = 2048

# Official low-resolution network input.
LOW_SIZE = 128

# Guided-filter parameters from E2EMEF.
GUIDED_RADIUS = 1
GUIDED_EPS = 1e-4


# ============================================================
# IMPORT MEF-NET
# ============================================================

def prepare_imports():

    if MEFNET_DIR not in sys.path:
        sys.path.insert(
            0,
            MEFNET_DIR
        )


# ============================================================
# CHECK FILE
# ============================================================

def check_file(path):

    if not os.path.isfile(path):

        raise FileNotFoundError(
            f"File not found:\n{path}"
        )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(path):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    # OpenCV BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    return image.astype(
        np.float32
    ) / 255.0


# ============================================================
# RESIZE KEEPING ASPECT RATIO
#
# This reproduces torchvision Resize(size)
# behavior reasonably closely:
# the smaller edge becomes target size.
# ============================================================

def resize_short_edge(
    image,
    target
):

    height, width = (
        image.shape[:2]
    )

    short_edge = min(
        height,
        width
    )

    if short_edge == target:

        return image.copy()

    scale = (
        float(target)
        /
        float(short_edge)
    )

    new_width = max(
        1,
        int(
            round(
                width * scale
            )
        )
    )

    new_height = max(
        1,
        int(
            round(
                height * scale
            )
        )
    )

    return cv2.resize(
        image,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# RESIZE EXPOSURE SEQUENCE
# ============================================================

def resize_sequence(
    images,
    target
):

    return [
        resize_short_edge(
            image,
            target
        )
        for image in images
    ]


# ============================================================
# RGB -> YCbCr
#
# This follows the formulas used by the official
# MEF-Net BatchRGBToYCbCr implementation.
# ============================================================

def rgb_to_ycbcr_tensor(
    rgb_tensor,
    torch
):

    # rgb_tensor:
    # [N, 3, H, W]

    r = rgb_tensor[:, 0:1, :, :]
    g = rgb_tensor[:, 1:2, :, :]
    b = rgb_tensor[:, 2:3, :, :]

    y = (
        0.299000 * r
        +
        0.587000 * g
        +
        0.114000 * b
    )

    cb = (
        128.0 / 256.0
        -
        0.168736 * r
        -
        0.331264 * g
        +
        0.500000 * b
    )

    cr = (
        128.0 / 256.0
        +
        0.500000 * r
        -
        0.418688 * g
        -
        0.081312 * b
    )

    return torch.cat(
        (
            y,
            cb,
            cr
        ),
        dim=1
    )


# ============================================================
# YCbCr -> RGB
#
# This follows the formulas used by the official
# MEF-Net YCbCrToRGB implementation.
# ============================================================

def ycbcr_to_rgb_tensor(
    ycbcr,
    torch
):

    y = ycbcr[:, 0:1, :, :]

    cb = ycbcr[:, 1:2, :, :]
    cr = ycbcr[:, 2:3, :, :]

    r = (
        y
        +
        (
            cr
            -
            128.0 / 256.0
        )
        *
        1.402
    )

    g = (
        y
        -
        (
            cb
            -
            128.0 / 256.0
        )
        *
        0.344136
        -
        (
            cr
            -
            128.0 / 256.0
        )
        *
        0.714136
    )

    b = (
        y
        +
        (
            cb
            -
            128.0 / 256.0
        )
        *
        1.772
    )

    return torch.cat(
        (
            r,
            g,
            b
        ),
        dim=1
    )


# ============================================================
# RGB NUMPY -> TORCH
# ============================================================

def numpy_rgb_to_tensor(
    images,
    torch
):

    # images:
    # list of [H, W, 3] RGB float arrays

    arrays = []

    for image in images:

        tensor = torch.from_numpy(
            image.transpose(
                2,
                0,
                1
            )
        ).float()

        arrays.append(
            tensor
        )

    return torch.stack(
        arrays,
        dim=0
    )


# ============================================================
# SAVE TENSOR RGB
# ============================================================

def save_tensor_rgb(
    tensor,
    path,
    torch
):

    tensor = tensor.detach().cpu()

    tensor = torch.clamp(
        tensor,
        0.0,
        1.0
    )

    # [1,3,H,W] -> [H,W,3]
    image = (
        tensor[0]
        .permute(
            1,
            2,
            0
        )
        .numpy()
    )

    image = np.clip(
        image,
        0.0,
        1.0
    )

    image = (
        image * 255.0
    ).astype(
        np.uint8
    )

    # RGB -> BGR for OpenCV
    image = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    output_dir = os.path.dirname(
        path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    success = cv2.imwrite(
        path,
        image,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            1
        ]
    )

    if not success:

        raise RuntimeError(
            f"Could not save:\n{path}"
        )


# ============================================================
# SAVE WEIGHT MAP
# ============================================================

def save_weight_map(
    weights,
    path,
    torch
):

    weights = (
        weights.detach()
        .cpu()
    )

    # weights:
    # [N,1,H,W]

    if weights.ndim == 4:

        weight = weights[0, 0]

    elif weights.ndim == 3:

        weight = weights[0]

    else:

        weight = weights

    weight = torch.clamp(
        weight,
        0.0,
        1.0
    )

    image = (
        weight.numpy()
        * 255.0
    ).astype(
        np.uint8
    )

    output_dir = os.path.dirname(
        path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    cv2.imwrite(
        path,
        image
    )


# ============================================================
# FUSE CHROMA
#
# This reproduces the logic from the official Trainer.eval():
#
# Wb = |Cb - 0.5|
# Wr = |Cr - 0.5|
# ============================================================

def fuse_chroma(
    cb,
    cr,
    torch
):

    weights_cb = (
        torch.abs(
            cb - 0.5
        )
        +
        1e-8
    )

    weights_cb = (
        weights_cb
        /
        torch.sum(
            weights_cb,
            dim=0,
            keepdim=True
        )
    )

    weights_cr = (
        torch.abs(
            cr - 0.5
        )
        +
        1e-8
    )

    weights_cr = (
        weights_cr
        /
        torch.sum(
            weights_cr,
            dim=0,
            keepdim=True
        )
    )

    cb_fused = torch.sum(
        weights_cb * cb,
        dim=0,
        keepdim=True
    ).clamp(
        0,
        1
    )

    cr_fused = torch.sum(
        weights_cr * cr,
        dim=0,
        keepdim=True
    ).clamp(
        0,
        1
    )

    return (
        cb_fused,
        cr_fused
    )


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(
    model,
    checkpoint_path,
    torch
):

    print()
    print(
        "[*] loading checkpoint:"
    )

    print(
        checkpoint_path
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    # Official MEF-Net checkpoint is a dictionary
    # containing state_dict.

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint[
                "state_dict"
            ]
        )

    else:

        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    print(
        "[*] checkpoint loaded successfully"
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "epoch" in checkpoint
    ):

        print(
            f"[*] checkpoint epoch: "
            f"{checkpoint['epoch']}"
        )


# ============================================================
# RUN MEF-NET
# ============================================================

def run_mefnet(
    under_path,
    normal_path,
    over_path,
    output_path,
    use_cuda=False
):

    print()
    print("=" * 60)
    print("CANDIDATE 3: MEF-NET DIRECT INFERENCE")
    print("=" * 60)

    print()

    print(
        "MEF-Net model:"
    )

    print(
        MEFNET_DIR
    )

    print()

    print(
        "Checkpoint:"
    )

    print(
        CHECKPOINT
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    check_file(
        CHECKPOINT
    )

    check_file(
        under_path
    )

    check_file(
        normal_path
    )

    check_file(
        over_path
    )

    # --------------------------------------------------------
    # Import actual MEF-Net architecture
    # --------------------------------------------------------

    prepare_imports()

    try:

        import torch

        from e2emef import E2EMEF

    except Exception as exc:

        raise RuntimeError(
            "Could not import the MEF-Net model from "
            f"{MEFNET_DIR}"
        ) from exc

    # --------------------------------------------------------
    # CUDA
    # --------------------------------------------------------

    cuda_available = (
        torch.cuda.is_available()
    )

    device = torch.device(
        "cuda"
        if (
            use_cuda
            and
            cuda_available
        )
        else
        "cpu"
    )

    print()

    print(
        f"CUDA available: "
        f"{cuda_available}"
    )

    print(
        f"Device: "
        f"{device}"
    )

    # --------------------------------------------------------
    # Load RGB exposures
    # --------------------------------------------------------

    print()
    print(
        "Loading exposures..."
    )

    under = load_image(
        under_path
    )

    normal = load_image(
        normal_path
    )

    over = load_image(
        over_path
    )

    # --------------------------------------------------------
    # Ensure matching original resolution
    # --------------------------------------------------------

    original_h, original_w = (
        normal.shape[:2]
    )

    if under.shape[:2] != (
        original_h,
        original_w
    ):

        under = cv2.resize(
            under,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_LINEAR
        )

    if over.shape[:2] != (
        original_h,
        original_w
    ):

        over = cv2.resize(
            over,
            (
                original_w,
                original_h
            ),
            interpolation=cv2.INTER_LINEAR
        )

    # --------------------------------------------------------
    # HIGH-RESOLUTION sequence
    #
    # Same idea as official BatchTestResolution(2048)
    # --------------------------------------------------------

    print(
        "Creating high-resolution sequence..."
    )

    high_images = resize_sequence(
        [
            under,
            normal,
            over
        ],
        HIGH_SIZE
    )

    high_tensor_rgb = (
        numpy_rgb_to_tensor(
            high_images,
            torch
        )
    )

    high_tensor_ycbcr = (
        rgb_to_ycbcr_tensor(
            high_tensor_rgb,
            torch
        )
    )

    # --------------------------------------------------------
    # LOW-RESOLUTION sequence
    #
    # Same idea as official low-resolution preprocessing.
    # --------------------------------------------------------

    print(
        "Creating low-resolution sequence..."
    )

    low_images = resize_sequence(
        [
            under,
            normal,
            over
        ],
        LOW_SIZE
    )

    low_tensor_rgb = (
        numpy_rgb_to_tensor(
            low_images,
            torch
        )
    )

    low_tensor_ycbcr = (
        rgb_to_ycbcr_tensor(
            low_tensor_rgb,
            torch
        )
    )

    # --------------------------------------------------------
    # Extract channels
    # --------------------------------------------------------

    Y_hr = (
        high_tensor_ycbcr[
            :,
            0:1,
            :,
            :
        ]
    )

    Cb_hr = (
        high_tensor_ycbcr[
            :,
            1:2,
            :,
            :
        ]
    )

    Cr_hr = (
        high_tensor_ycbcr[
            :,
            2:3,
            :,
            :
        ]
    )

    Y_lr = (
        low_tensor_ycbcr[
            :,
            0:1,
            :,
            :
        ]
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Creating E2EMEF model..."
    )

    model = E2EMEF(
        radius=GUIDED_RADIUS,
        eps=GUIDED_EPS,
        is_guided=True
    )

    # --------------------------------------------------------
    # Load official checkpoint
    # --------------------------------------------------------

    load_checkpoint(
        model,
        CHECKPOINT,
        torch
    )

    model = model.to(
        device
    )

    model.eval()

    # --------------------------------------------------------
    # Device transfer
    # --------------------------------------------------------

    Y_hr_device = (
        Y_hr
        .to(
            device
        )
    )

    Y_lr_device = (
        Y_lr
        .to(
            device
        )
    )

    Cb_hr_device = (
        Cb_hr
        .to(
            device
        )
    )

    Cr_hr_device = (
        Cr_hr
        .to(
            device
        )
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    print()
    print(
        "Running MEF-Net inference..."
    )

    with torch.no_grad():

        fused_y, weights = model(
            Y_lr_device,
            Y_hr_device
        )

        # Official repository chroma fusion.
        fused_cb, fused_cr = (
            fuse_chroma(
                Cb_hr_device,
                Cr_hr_device,
                torch
            )
        )

        fused_ycbcr = torch.cat(
            (
                fused_y,
                fused_cb,
                fused_cr
            ),
            dim=1
        )

        fused_rgb = (
            ycbcr_to_rgb_tensor(
                fused_ycbcr,
                torch
            )
        )

    # --------------------------------------------------------
    # Save final result at high-resolution processing size
    # --------------------------------------------------------

    output_temp = os.path.join(
        PROJECT_ROOT,
        "adaptive_outputs",
        "_mefnet_temp.png"
    )

    save_tensor_rgb(
        fused_rgb,
        output_temp,
        torch
    )

    # --------------------------------------------------------
    # Resize back to original resolution
    # --------------------------------------------------------

    result = cv2.imread(
        output_temp,
        cv2.IMREAD_COLOR
    )

    if result is None:

        raise RuntimeError(
            "MEF-Net temporary output could not be read."
        )

    result = cv2.resize(
        result,
        (
            original_w,
            original_h
        ),
        interpolation=cv2.INTER_CUBIC
    )

    output_dir = os.path.dirname(
        output_path
    )

    if output_dir:

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    success = cv2.imwrite(
        output_path,
        result,
        [
            cv2.IMWRITE_PNG_COMPRESSION,
            1
        ]
    )

    if not success:

        raise RuntimeError(
            f"Could not save MEF-Net output:\n"
            f"{output_path}"
        )

    # --------------------------------------------------------
    # Weight map
    # --------------------------------------------------------

    weight_map_path = (
        os.path.splitext(
            output_path
        )[0]
        +
        "_weight_map.png"
    )

    save_weight_map(
        weights,
        weight_map_path,
        torch
    )

    # --------------------------------------------------------
    # Cleanup temp output
    # --------------------------------------------------------

    try:

        os.remove(
            output_temp
        )

    except OSError:

        pass

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    weights_cpu = (
        weights.detach()
        .cpu()
    )

    print()
    print("=" * 60)
    print("MEF-NET CANDIDATE COMPLETED")
    print("=" * 60)

    print()

    print(
        f"Final output:\n"
        f"{output_path}"
    )

    print()

    print(
        f"Weight map:\n"
        f"{weight_map_path}"
    )

    print()

    print(
        "Mean MEF-Net exposure weights:"
    )

    for index, name in enumerate(
        [
            "Under",
            "Normal",
            "Over"
        ]
    ):

        mean_weight = float(
            weights_cpu[
                index
            ].mean()
        )

        print(
            f"{name:>6}: "
            f"{mean_weight:.4f}"
        )

    return output_path


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Direct MEF-Net inference adapter "
            "for adaptive FreeMEF."
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
        "--output",
        required=True
    )

    parser.add_argument(
        "--cuda",
        action="store_true",
        help="Use CUDA when available."
    )

    args = parser.parse_args()

    run_mefnet(
        under_path=args.under,
        normal_path=args.normal,
        over_path=args.over,
        output_path=args.output,
        use_cuda=args.cuda
    )


if __name__ == "__main__":

    main()
