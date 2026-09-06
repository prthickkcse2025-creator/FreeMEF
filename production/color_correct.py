import cv2
import numpy as np
import os

# FreeMEF output = base image
base = cv2.imread("production/output/order_test/scene001.png")

# Over-exposed input
over = cv2.imread("my_test_context/scene001/03_over.JPG")

# Safety check
if base is None:
    raise FileNotFoundError("FreeMEF output not found")

if over is None:
    raise FileNotFoundError("Over exposure image not found")

# Convert to float
base_f = base.astype(np.float32) / 255.0
over_f = over.astype(np.float32) / 255.0

# Calculate mean color of both images
base_mean = np.mean(base_f.reshape(-1, 3), axis=0)
over_mean = np.mean(over_f.reshape(-1, 3), axis=0)

# Match OVER exposure color to FreeMEF output
scale = base_mean / (over_mean + 1e-6)

over_corrected = np.clip(over_f * scale, 0, 1)

# Convert back to uint8
over_corrected = (over_corrected * 255).astype(np.uint8)

# Save
os.makedirs("production/output/color_corrected", exist_ok=True)

output_path = "production/output/color_corrected/scene001.png"

cv2.imwrite(output_path, over_corrected)

print("Color correction complete")
print("Output:", output_path)
print("Base mean:", base_mean)
print("Over mean:", over_mean)
print("Scale:", scale)
