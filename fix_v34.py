from pathlib import Path

path = Path("production/adaptive_fusion_v3_4.py")
text = path.read_text()

changes = [
    (
'''    shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.18,
            0.55
        )
    )''',

'''    shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.10,
            0.35
        )
    )'''
    ),

    (
'''    deep_shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.05,
            0.32
        )
    )''',

'''    deep_shadow_mask = (
        1.0 -
        smoothstep(
            lum_normal,
            0.03,
            0.20
        )
    )'''
    ),

    (
'''    w_normal = (
        2.5 +
        1.5 * q_normal -
        0.8 * deep_shadow_mask
    )''',

'''    w_normal = (
        3.5 +
        1.8 * q_normal -
        0.35 * deep_shadow_mask
    )'''
    ),

    (
'''    w_over = (
        0.05 +
        2.0 * shadow_recovery_mask +
        0.7 * q_over
    )''',

'''    w_over = (
        0.03 +
        1.0 * shadow_recovery_mask +
        0.45 * q_over
    )'''
    )
]

success = True

for old, new in changes:
    if old in text:
        text = text.replace(old, new, 1)
        print("✓ Block replaced")
    else:
        print("✗ Block NOT found")
        success = False

if success:
    path.write_text(text)
    print("\nSUCCESS: V3.4 created correctly.")
else:
    print("\nFAILED: V3.4 was NOT written.")
    print("Original file remains unchanged.")
