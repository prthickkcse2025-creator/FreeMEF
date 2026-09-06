#!/usr/bin/env python3

"""
FreeMEF CONTROL SAFETY LAYER

Purpose:
    Validate and constrain correction controls before they reach V5.

Design goals:
    - Preserve the existing image appearance by default.
    - Prevent unnecessarily aggressive global corrections.
    - Keep color changes especially conservative.
    - Keep blending conservative.
    - Preserve all 9 supported controls.
"""

from typing import Dict


CONTROL_KEYS = [
    "brightness",
    "shadow",
    "highlight",
    "depth",
    "dehaze",
    "contrast",
    "saturation",
    "color",
    "blending",
]


# Conservative production limits.
# These are intentionally lower than the raw Gemini range.
SAFE_LIMITS = {
    "brightness": 0.45,
    "shadow": 0.45,
    "highlight": 0.45,
    "depth": 0.35,
    "dehaze": 0.30,
    "contrast": 0.30,
    "saturation": 0.25,
    "color": 0.20,
    "blending": 0.35,
}


def clamp(
    value: float,
    low: float,
    high: float,
) -> float:
    return max(
        low,
        min(high, float(value)),
    )


def sanitize_controls(
    values: Dict
) -> Dict[str, float]:
    """
    Convert arbitrary control input into a safe,
    complete 9-control dictionary.
    """

    if not isinstance(values, dict):
        values = {}

    safe = {}

    for key in CONTROL_KEYS:

        try:
            value = float(
                values.get(
                    key,
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            value = 0.0

        value = clamp(
            value,
            -1.0,
            1.0,
        )

        limit = SAFE_LIMITS[key]

        value = clamp(
            value,
            -limit,
            limit,
        )

        safe[key] = value

    return safe


def validate_feedback_controls(
    values: Dict
) -> Dict:
    """
    Return a structured safety result.

    The original controls are preserved for diagnostics,
    while the safe controls are what should be sent to V5.
    """

    original = sanitize_raw(values)
    safe = sanitize_controls(values)

    changed = {}

    for key in CONTROL_KEYS:

        original_value = original[key]
        safe_value = safe[key]

        if abs(
            original_value
            -
            safe_value
        ) > 1e-6:

            changed[key] = {
                "requested": original_value,
                "applied": safe_value,
            }

    return {
        "controls": safe,
        "changed": changed,
        "was_limited": bool(changed),
    }


def sanitize_raw(
    values: Dict
) -> Dict[str, float]:
    """
    Keep the Gemini numeric range intact for diagnostics.
    """

    if not isinstance(values, dict):
        values = {}

    result = {}

    for key in CONTROL_KEYS:

        try:
            value = float(
                values.get(
                    key,
                    0.0,
                )
            )
        except (
            TypeError,
            ValueError,
        ):
            value = 0.0

        result[key] = clamp(
            value,
            -1.0,
            1.0,
        )

    return result


def has_changes(
    values: Dict
) -> bool:
    safe = sanitize_controls(values)

    return any(
        abs(value) > 1e-6
        for value in safe.values()
    )


def describe_limits() -> Dict[str, float]:
    return dict(
        SAFE_LIMITS
    )
