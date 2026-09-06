import os
import sys
import subprocess
import gradio as gr
from PIL import Image


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

ENGINE_PATH = os.path.join(
    PROJECT_DIR,
    "adaptive_engine",
    "correction_engine_final.py"
)

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    "adaptive_outputs"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# APPLY CORRECTIONS
# ============================================================

def apply_corrections(
    selected,
    normal,
    under,
    over,
    brightness,
    shadow,
    highlight,
    depth,
    dehaze,
    contrast,
    saturation,
    color
):

    if selected is None:
        raise gr.Error(
            "Please upload a fused image."
        )

    # --------------------------------------------------------
    # If reference images are missing, use selected image
    # temporarily.
    # --------------------------------------------------------

    if normal is None:
        normal = selected

    if under is None:
        under = selected

    if over is None:
        over = selected

    output_path = os.path.join(
        OUTPUT_DIR,
        "final_output.jpg"
    )

    # --------------------------------------------------------
    # Run correction engine
    # --------------------------------------------------------

    command = [

        sys.executable,

        ENGINE_PATH,

        "--selected",
        selected,

        "--normal",
        normal,

        "--under",
        under,

        "--over",
        over,

        "--brightness",
        str(brightness),

        "--shadow",
        str(shadow),

        "--highlight",
        str(highlight),

        "--depth",
        str(depth),

        "--dehaze",
        str(dehaze),

        "--contrast",
        str(contrast),

        "--saturation",
        str(saturation),

        "--color",
        str(color),

        "--output",
        output_path
    ]

    try:

        result = subprocess.run(

            command,

            capture_output=True,

            text=True,

            check=True

        )

        print(
            result.stdout
        )

    except subprocess.CalledProcessError as e:

        print(
            e.stdout
        )

        print(
            e.stderr
        )

        raise gr.Error(
            "Correction engine failed. "
            "Check the terminal."
        )

    if not os.path.exists(
        output_path
    ):

        raise gr.Error(
            "Output image was not created."
        )

    return output_path


# ============================================================
# RESET SETTINGS
# ============================================================

def reset_settings():

    return (

        0.0,   # brightness
        0.0,   # shadow
        0.0,   # highlight
        0.0,   # depth
        0.0,   # dehaze
        0.0,   # contrast
        0.0,   # saturation
        0.0    # color

    )


# ============================================================
# USER INTERFACE
# ============================================================

with gr.Blocks(
    title="FreeMEF - Interactive Image Enhancement"
) as demo:

    gr.Markdown(
        """
        # 📸 FreeMEF Image Enhancement

        Upload your fused image and optionally provide the
        Normal, Underexposed, and Overexposed reference images.

        Adjust the enhancement sliders and generate your final image.
        """
    )

    # --------------------------------------------------------
    # IMAGE INPUTS
    # --------------------------------------------------------

    with gr.Row():

        selected_input = gr.Image(
            label="Fused / Selected Image",
            type="filepath"
        )

        normal_input = gr.Image(
            label="Normal Reference",
            type="filepath"
        )

    with gr.Row():

        under_input = gr.Image(
            label="Underexposed Reference",
            type="filepath"
        )

        over_input = gr.Image(
            label="Overexposed Reference",
            type="filepath"
        )

    # --------------------------------------------------------
    # CORRECTION CONTROLS
    # --------------------------------------------------------

    gr.Markdown(
        "## 🎛️ Enhancement Controls"
    )

    with gr.Row():

        brightness_slider = gr.Slider(

            minimum=-1.0,
            maximum=1.0,
            value=0.0,
            step=0.05,

            label="Brightness"

        )

        shadow_slider = gr.Slider(

            minimum=-1.0,
            maximum=1.0,
            value=0.0,
            step=0.05,

            label="Shadows"

        )

    with gr.Row():

        highlight_slider = gr.Slider(

            minimum=-1.0,
            maximum=1.0,
            value=0.0,
            step=0.05,

            label="Highlights"

        )

        depth_slider = gr.Slider(

            minimum=-1.0,
            maximum=1.0,
            value=0.0,
            step=0.05,

            label="Depth"

        )

    with gr.Row():

        dehaze_slider = gr.Slider(

            minimum=-1.0,
            maximum=1.0,
            value=0.0,
            step=0.05,

            label="Dehaze"

        )

        contrast_slider = gr.Slider(

            minimum=-1.0,
            maximum=1.0,
            value=0.0,
            step=0.05,

            label="Contrast"

        )

    with gr.Row():

        saturation_slider = gr.Slider(

            minimum=-1.0,
            maximum=1.0,
            value=0.0,
            step=0.05,

            label="Saturation"

        )

        color_slider = gr.Slider(

            minimum=-1.0,
            maximum=1.0,
            value=0.0,
            step=0.05,

            label="Color"

        )

    # --------------------------------------------------------
    # BUTTONS
    # --------------------------------------------------------

    with gr.Row():

        apply_button = gr.Button(
            "✨ Apply Enhancement",
            variant="primary"
        )

        reset_button = gr.Button(
            "↩ Reset"
        )

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_image = gr.Image(
        label="Final Enhanced Image",
        type="filepath"
    )

    download_file = gr.File(
        label="Download Final Image"
    )

    # --------------------------------------------------------
    # APPLY EVENT
    # --------------------------------------------------------

    apply_button.click(

        fn=apply_corrections,

        inputs=[

            selected_input,
            normal_input,
            under_input,
            over_input,

            brightness_slider,
            shadow_slider,
            highlight_slider,
            depth_slider,
            dehaze_slider,
            contrast_slider,
            saturation_slider,
            color_slider

        ],

        outputs=[

            output_image

        ]

    )

    # --------------------------------------------------------
    # DOWNLOAD EVENT
    # --------------------------------------------------------

    apply_button.click(

        fn=lambda: os.path.join(
            OUTPUT_DIR,
            "final_output.jpg"
        ),

        outputs=download_file

    )

    # --------------------------------------------------------
    # RESET EVENT
    # --------------------------------------------------------

    reset_button.click(

        fn=reset_settings,

        outputs=[

            brightness_slider,
            shadow_slider,
            highlight_slider,
            depth_slider,
            dehaze_slider,
            contrast_slider,
            saturation_slider,
            color_slider

        ]

    )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False
    )
