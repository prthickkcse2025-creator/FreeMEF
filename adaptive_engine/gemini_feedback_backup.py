#!/usr/bin/env python3

"""
GEMINI FEEDBACK TRANSLATOR
==========================

Customer text
    ↓
Gemini 3.6 Flash
    ↓
structured correction values
    ↓
correction_engine.py
    ↓
revised image

Gemini does NOT select A/B/C.
Gemini does NOT edit the image.
Gemini ONLY converts feedback into numeric controls.
"""

import os
import argparse

from google import genai
from pydantic import BaseModel, Field


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = os.environ.get(
    "GEMINI_FEEDBACK_MODEL",
    "gemini-3.6-flash"
)


# ============================================================
# STRUCTURED OUTPUT
# ============================================================

class FeedbackValues(BaseModel):

    brightness: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "-1 darker, +1 brighter"
        )
    )

    shadow: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "-1 reduce lifted shadows, "
            "+1 recover shadow detail"
        )
    )

    highlight: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "-1 reduce highlights, "
            "+1 recover highlight detail"
        )
    )

    depth: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "-1 less depth, +1 more natural depth"
        )
    )

    dehaze: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "-1 more haze, +1 less haze"
        )
    )

    contrast: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "-1 less contrast, +1 more contrast"
        )
    )

    saturation: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "-1 less saturation, +1 more saturation"
        )
    )

    color: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "-1 reduce color correction, "
            "+1 improve natural color"
        )
    )

    summary: str = Field(
        default="",
        description=(
            "Only the changes explicitly requested "
            "by the customer"
        )
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0
    )


# ============================================================
# PROMPT
# ============================================================

SYSTEM_INSTRUCTION = """
You are a strict professional image-feedback translator.

Your ONLY job is to translate customer feedback into
numeric image-correction parameters.

DO NOT edit images.
DO NOT choose Candidate A/B/C.
DO NOT invent corrections.
DO NOT change parameters that the customer says to preserve.

All numeric values must be between -1.0 and +1.0.

PARAMETERS
----------

brightness:
    -1 darker
    +1 brighter

shadow:
    -1 reduce lifted/open shadows
    +1 recover/open shadow detail

highlight:
    -1 reduce bright/lifted highlights
    +1 recover highlight detail

depth:
    -1 reduce depth
    +1 add natural midtone depth

dehaze:
    -1 increase haze
    +1 reduce haze

contrast:
    -1 reduce contrast
    +1 increase contrast

saturation:
    -1 reduce saturation
    +1 increase saturation

color:
    -1 reduce color correction
    +1 improve/correct unnatural color

IMPORTANT PRESERVATION RULE
---------------------------

Words such as:
    preserve
    keep
    protect
    maintain
    don't change
    leave unchanged

are NOT correction requests.

Examples:

"preserve highlight detail"
    highlight = 0

"keep the colors natural"
    color = 0

"protect shadow detail"
    shadow = 0

"keep brightness"
    brightness = 0

EXPLICIT CORRECTIONS
--------------------

"make it darker"
"darken the image"
"make the image dark"
"reduce brightness"

    brightness < 0

"make it brighter"
"brighten the image"
"increase brightness"

    brightness > 0

"reduce lifted shadows"
"shadows are too lifted"
"too much shadow lifting"
"darken shadows"

    shadow < 0

"recover shadow detail"
"open shadows"
"more shadow detail"

    shadow > 0

"reduce highlights"
"highlights are too bright"
"highlights are lifted"
"darken highlights"

    highlight < 0

"recover highlight detail"
"restore highlights"
"more highlight detail"

    highlight > 0

"more depth"
"add depth"
"less flat"
"more natural 3D"

    depth > 0

"hazy"
"foggy"
"remove haze"
"reduce haze"
"less fog"

    dehaze > 0

"more contrast"

    contrast > 0

"less contrast"

    contrast < 0

"more saturation"

    saturation > 0

"less saturation"
"too saturated"

    saturation < 0

"colors are unnatural"
"fix the colors"
"white balance is wrong"

    color > 0

IMPORTANT
---------

If the customer says:

"Make it darker but preserve highlight detail."

then:

    brightness < 0
    highlight = 0

If the customer says:

"Reduce lifted shadows but preserve shadow detail."

then:

    shadow < 0

The explicit correction takes priority over the
preservation wording for the same parameter.

If the customer says:

"Add depth but keep natural colors."

then:

    depth > 0
    color = 0

If the customer says:

"Reduce haze without changing brightness."

then:

    dehaze > 0
    brightness = 0

Only requested changes should be non-zero.

INTENSITY
---------

slightly:
    0.20 to 0.35

moderately:
    0.35 to 0.60

strongly:
    0.60 to 0.85

very strongly:
    0.85 to 1.00

For ambiguous feedback, choose a conservative value.

Return only the structured fields.
"""


# ============================================================
# CLIENT
# ============================================================

def get_client():

    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    return genai.Client(
        api_key=api_key
    )


# ============================================================
# SANITIZE
# ============================================================

def sanitize(values):

    keys = [
        "brightness",
        "shadow",
        "highlight",
        "depth",
        "dehaze",
        "contrast",
        "saturation",
        "color"
    ]

    for key in keys:

        try:
            value = float(
                values.get(
                    key,
                    0.0
                )
            )
        except Exception:
            value = 0.0

        values[key] = float(
            max(
                -1.0,
                min(
                    1.0,
                    value
                )
            )
        )

    try:
        confidence = float(
            values.get(
                "confidence",
                0.0
            )
        )
    except Exception:
        confidence = 0.0

    values["confidence"] = float(
        max(
            0.0,
            min(
                1.0,
                confidence
            )
        )
    )

    return values


# ============================================================
# TRANSLATE
# ============================================================

def translate_feedback(
    feedback
):

    feedback = feedback.strip()

    if not feedback:
        raise ValueError(
            "Customer feedback is empty."
        )

    client = get_client()

    prompt = (
        SYSTEM_INSTRUCTION
        +
        "\n\nCUSTOMER FEEDBACK:\n"
        +
        feedback
    )

    # Use the current Gemini SDK structured-output path.
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": FeedbackValues
        }
    )

    if not getattr(
        response,
        "text",
        None
    ):
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    try:

        values = FeedbackValues.model_validate_json(
            response.text
        ).model_dump()

    except Exception as exc:

        raise RuntimeError(
            "Could not parse Gemini structured output.\n\n"
            f"Raw response:\n{response.text}"
        ) from exc

    return sanitize(
        values
    )


# ============================================================
# PRINT
# ============================================================

def print_values(
    values
):

    print()
    print("=" * 70)
    print("GEMINI FEEDBACK TRANSLATION")
    print("=" * 70)

    print()

    print(
        "Summary:"
    )

    print(
        f"  {values['summary']}"
    )

    print()

    print(
        f"Confidence: "
        f"{values['confidence']:.3f}"
    )

    print()

    print(
        "Correction values:"
    )

    for key in [
        "brightness",
        "shadow",
        "highlight",
        "depth",
        "dehaze",
        "contrast",
        "saturation",
        "color"
    ]:

        print(
            f"  {key:<12}: "
            f"{values[key]:+.3f}"
        )

    print()
    print("=" * 70)


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "feedback",
        nargs="+"
    )

    args = parser.parse_args()

    feedback = " ".join(
        args.feedback
    )

    values = translate_feedback(
        feedback
    )

    print_values(
        values
    )


if __name__ == "__main__":
    main()
