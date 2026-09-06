#!/usr/bin/env python3

"""
CLIENT-PREFERENCE ADAPTIVE SELECTOR

Client-labeled examples:

    Scene002 -> Candidate A (V3.3 Client-Bright)
    Scene005 -> Candidate C (MEF-Net)

The selector uses scene characteristics rather than
hardcoded exposure strengths.

IMPORTANT:
This is a proof-of-concept with only two labeled scenes.
It is not yet a production-quality machine-learning model.
"""


import os
import argparse
import numpy as np
import cv2


# ============================================================
# FEATURE DEFINITIONS
# ============================================================

FEATURE_NAMES = [
    "mean_brightness",
    "median_brightness",
    "shadow_ratio",
    "deep_shadow_ratio",
    "highlight_ratio",
]


# ============================================================
# CLIENT-LABELED SCENES
# ============================================================

TRAINING_DATA = [

    # --------------------------------------------------------
    # SCENE002
    # Client preference:
    # Candidate A = V3.3 Client-Bright
    # --------------------------------------------------------

    {
        "scene": "scene002",

        "label": "A",

        "features": {
            "mean_brightness": 0.6008,
            "median_brightness": 0.6521,
            "shadow_ratio": 0.1589,
            "deep_shadow_ratio": 0.1026,
            "highlight_ratio": 0.1231,
        },
    },

    # --------------------------------------------------------
    # SCENE005
    # Client preference:
    # Candidate C = MEF-Net
    # --------------------------------------------------------

    {
        "scene": "scene005",

        "label": "C",

        "features": {
            "mean_brightness": 0.3634,
            "median_brightness": 0.3941,
            "shadow_ratio": 0.3621,
            "deep_shadow_ratio": 0.2186,
            "highlight_ratio": 0.0178,
        },
    },
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(
    normal_path
):

    image = cv2.imread(
        normal_path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise FileNotFoundError(
            f"Could not read:\n{normal_path}"
        )

    image = (
        image.astype(
            np.float32
        )
        / 255.0
    )

    lum = (
        0.0722 * image[:, :, 0]
        +
        0.7152 * image[:, :, 1]
        +
        0.2126 * image[:, :, 2]
    )

    return {

        "mean_brightness":
            float(
                np.mean(lum)
            ),

        "median_brightness":
            float(
                np.median(lum)
            ),

        "shadow_ratio":
            float(
                np.mean(
                    lum < 0.30
                )
            ),

        "deep_shadow_ratio":
            float(
                np.mean(
                    lum < 0.18
                )
            ),

        "highlight_ratio":
            float(
                np.mean(
                    lum > 0.85
                )
            ),
    }


# ============================================================
# VECTOR
# ============================================================

def to_vector(
    features
):

    return np.array(
        [
            features[name]
            for name in FEATURE_NAMES
        ],
        dtype=np.float32
    )


# ============================================================
# ADAPTIVE SELECTOR
# ============================================================

class AdaptiveSelector:

    def __init__(self):

        self.X = None

        self.labels = None

        self.mean = None

        self.std = None


    def fit(
        self,
        training_data
    ):

        X = []

        labels = []

        for row in training_data:

            X.append(
                to_vector(
                    row["features"]
                )
            )

            labels.append(
                row["label"]
            )

        self.X = np.stack(
            X
        )

        self.labels = labels

        self.mean = np.mean(
            self.X,
            axis=0
        )

        self.std = np.std(
            self.X,
            axis=0
        )

        # Two samples make some features have zero variance.
        # Prevent division by zero.

        self.std[
            self.std < 1e-8
        ] = 1.0


    def predict(
        self,
        features
    ):

        if self.X is None:

            raise RuntimeError(
                "Selector has not been fitted."
            )

        x = to_vector(
            features
        )

        # ----------------------------------------------------
        # Standardize
        # ----------------------------------------------------

        x_scaled = (
            x -
            self.mean
        ) / self.std

        X_scaled = (
            self.X -
            self.mean
        ) / self.std

        # ----------------------------------------------------
        # Euclidean distance
        # ----------------------------------------------------

        distances = np.linalg.norm(
            X_scaled -
            x_scaled[None, :],
            axis=1
        )

        # ----------------------------------------------------
        # Nearest client-labeled scene
        # ----------------------------------------------------

        nearest_index = int(
            np.argmin(
                distances
            )
        )

        predicted_label = (
            self.labels[
                nearest_index
            ]
        )

        nearest_distance = float(
            distances[
                nearest_index
            ]
        )

        # ----------------------------------------------------
        # Simple confidence
        # ----------------------------------------------------

        confidence = (
            1.0 /
            (
                1.0
                +
                nearest_distance
            )
        )

        return (
            predicted_label,
            confidence,
            distances
        )


# ============================================================
# LABEL DESCRIPTION
# ============================================================

def candidate_name(
    label
):

    mapping = {

        "A":
            "V3.3 Client-Bright",

        "B":
            "Mertens V2",

        "C":
            "MEF-Net",
    }

    return mapping.get(
        label,
        "Unknown"
    )


# ============================================================
# PRINT TRAINING DATA
# ============================================================

def print_training_data():

    print()
    print(
        "=" * 65
    )

    print(
        "CLIENT PREFERENCE DATA"
    )

    print(
        "=" * 65
    )

    for row in TRAINING_DATA:

        print()

        print(
            f"{row['scene']}: "
            f"Candidate {row['label']} "
            f"({candidate_name(row['label'])})"
        )

        for feature in FEATURE_NAMES:

            print(
                f"  {feature:<22} "
                f"{row['features'][feature]:.4f}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Client-preference adaptive "
            "fusion selector"
        )
    )

    parser.add_argument(
        "--normal",
        required=True,
        help="Normal exposure of the scene"
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Extract new scene features
    # --------------------------------------------------------

    features = extract_features(
        args.normal
    )

    # --------------------------------------------------------
    # Train selector
    # --------------------------------------------------------

    selector = (
        AdaptiveSelector()
    )

    selector.fit(
        TRAINING_DATA
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_label, confidence, distances = (
        selector.predict(
            features
        )
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print_training_data()

    print()
    print(
        "=" * 65
    )

    print(
        "CURRENT SCENE"
    )

    print(
        "=" * 65
    )

    for feature in FEATURE_NAMES:

        print(
            f"{feature:<22}: "
            f"{features[feature]:.4f}"
        )

    print()
    print(
        "DISTANCE TO CLIENT-LABELED SCENES"
    )

    print(
        "=" * 65
    )

    for index, row in enumerate(
        TRAINING_DATA
    ):

        print(
            f"{row['scene']:<15} "
            f"Candidate {row['label']} "
            f"({candidate_name(row['label'])}) "
            f"distance = "
            f"{distances[index]:.4f}"
        )

    print()
    print(
        "=" * 65
    )

    print(
        f"PREDICTED CANDIDATE: "
        f"{predicted_label}"
    )

    print(
        f"METHOD: "
        f"{candidate_name(predicted_label)}"
    )

    print(
        f"CONFIDENCE: "
        f"{confidence:.4f}"
    )

    print(
        "=" * 65
    )


if __name__ == "__main__":

    main()
