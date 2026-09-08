#!/usr/bin/env python3

import os
from pathlib import Path
import sys
import json
import subprocess
from datetime import datetime

import cv2
import numpy as np
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


V33_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "production",
    "adaptive_fusion_v3_3_client_bright.py"
)

MERTENS_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "adaptive_modules",
    "mertens",
    "mertens_candidate_v3.py"
)

MEFNET_SCRIPT = os.path.join(
    PROJECT_ROOT,
    "adaptive_modules",
    "mefnet",
    "mefnet_adapter.py"
)

FEEDBACK_DIR = os.path.join(
    PROJECT_ROOT,
    "adaptive_outputs",
    "feedback"
)

FEEDBACK_FILE = os.path.join(
    FEEDBACK_DIR,
    "client_sessions.json"
)

SESSION_ROOT = os.path.join(
    PROJECT_ROOT,
    "adaptive_outputs",
    "streamlit_sessions"
)


# ============================================================
# NEW FEEDBACK SYSTEM
# ============================================================

from adaptive_engine.gemini_feedback import (
    translate_feedback
)

from adaptive_engine.correction_engine_final import (
    refine_with_values
)

from adaptive_engine.input_validation import (
    validate_and_align_exposures
)


# ============================================================
# STREAMLIT PAGE
# ============================================================

st.set_page_config(
    page_title="FreeMEF Client Fusion",
    page_icon="🖼️",
    layout="wide"
)


# ============================================================
# UI STYLE
# ============================================================

st.markdown(
    """
    <style>

    .title {
        text-align: center;
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 25px;
    }

    .candidate-title {
        text-align: center;
        font-size: 24px;
        font-weight: 700;
    }

    .candidate-method {
        text-align: center;
        color: #777;
        margin-bottom: 8px;
    }

    .selected-box {
        padding: 15px;
        border-radius: 10px;
        background: #eef6ff;
        border: 1px solid #9ec5ff;
    }

    .approved-box {
        padding: 18px;
        border-radius: 10px;
        background: #eef9ee;
        border: 1px solid #8bc48b;
    }

    .feedback-box {
        padding: 12px;
        border-radius: 10px;
        background: #f7f7f7;
        border: 1px solid #dddddd;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SESSION STATE
# ============================================================


# ============================================================
# IMAGE PREVIEW / ZOOM HELPER
# ============================================================

@st.dialog(
    "🔍 Image Viewer",
    width="large"
)
def show_zoom_dialog(
    image_path,
    caption="Image"
):
    """Show the original image only when Zoom is clicked."""

    if not image_path or not os.path.isfile(image_path):

        st.error(
            "The image could not be found."
        )
        return

    st.image(
        image_path,
        caption=caption,
        width="stretch"
    )


def show_image_preview(
    image_path,
    caption=None,
    key_prefix="image",
    preview_width=None,
    center=True
):
    """
    Display a clean responsive preview.

    The source file is never resized, cropped, or overwritten.
    Only the browser presentation changes.
    """

    if not image_path or not os.path.isfile(image_path):

        st.warning(
            "Image unavailable."
        )
        return

    if center:

        preview_col = st.columns(
            [1, 2, 1]
        )[1]

    else:

        preview_col = st.container()

    with preview_col:

        if preview_width is None:

            st.image(
                image_path,
                caption=caption,
                width="stretch"
            )

        else:

            st.image(
                image_path,
                caption=caption,
                width=preview_width
            )

        st.button(
            "🔍 Zoom",
            key=f"{key_prefix}_zoom",
            on_click=show_zoom_dialog,
            args=(
                image_path,
                caption or "Image"
            )
        )



def initialize_state():

    defaults = {

        "session_id": None,

        "session_dir": None,

        "under_path": None,

        "normal_path": None,

        "over_path": None,

        "candidate_a": None,

        "candidate_b": None,

        "candidate_c": None,

        "candidate_logs": {},

        "generation_done": False,

        "selected_candidate": None,

        # Clean candidate selected by the customer.
        # Revisions are rendered from this baseline.
        "baseline_image": None,

        # Latest displayed revision.
        "current_image": None,

        "current_label": None,

        "revision_number": 0,

        # Persistent customer correction state.
        "control_state": {
            "brightness": 0.0,
            "shadow": 0.0,
            "highlight": 0.0,
            "depth": 0.0,
            "dehaze": 0.0,
            "contrast": 0.0,
            "saturation": 0.0,
            "color": 0.0,
            "blending": 0.0,
        },

        "approved": False,

        "history": []
    }

    for key, value in defaults.items():

        if key not in st.session_state:

            st.session_state[key] = value


initialize_state()


# ============================================================
# SESSION CREATION
# ============================================================

def create_session():

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    session_dir = os.path.join(
        SESSION_ROOT,
        timestamp
    )

    os.makedirs(
        session_dir,
        exist_ok=True
    )

    st.session_state.session_id = timestamp

    st.session_state.session_dir = session_dir

    return session_dir


# ============================================================
# SESSION PERSISTENCE
# ============================================================

SESSION_STATE_FILE = "session.json"


def session_manifest_path(session_dir=None):
    """
    Return the persistent metadata file for the current session.
    """

    if session_dir is None:
        session_dir = st.session_state.session_dir

    if not session_dir:
        return None

    return os.path.join(
        session_dir,
        SESSION_STATE_FILE
    )


def serializable_session_state():
    """
    Build a JSON-safe snapshot of the workflow state.

    Only workflow metadata is stored here. Image files themselves
    remain as files in the session directory.
    """

    return {
        "session_id":
            st.session_state.session_id,

        "under_path":
            st.session_state.under_path,

        "normal_path":
            st.session_state.normal_path,

        "over_path":
            st.session_state.over_path,

        "candidate_a":
            st.session_state.candidate_a,

        "candidate_b":
            st.session_state.candidate_b,

        "candidate_c":
            st.session_state.candidate_c,

        "selected_candidate":
            st.session_state.selected_candidate,

        "baseline_image":
            st.session_state.baseline_image,

        "current_image":
            st.session_state.current_image,

        "current_label":
            st.session_state.current_label,

        "revision_number":
            st.session_state.revision_number,

        "control_state":
            dict(
                st.session_state.control_state
            ),

        "approved":
            st.session_state.approved,

        "history":
            list(
                st.session_state.history
            )
    }


def save_session_state():
    """
    Persist the current workflow state to session.json.

    Writes atomically so an interrupted write is less likely to
    leave a corrupted manifest.
    """

    session_dir = (
        st.session_state.session_dir
    )

    if not session_dir:
        return

    os.makedirs(
        session_dir,
        exist_ok=True
    )

    manifest_path = session_manifest_path(
        session_dir
    )

    temporary_path = (
        manifest_path + ".tmp"
    )

    data = (
        serializable_session_state()
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False
        )

        f.flush()
        os.fsync(
            f.fileno()
        )

    os.replace(
        temporary_path,
        manifest_path
    )


def load_session_state(
    session_dir
):
    """
    Load workflow metadata from session.json.

    Returns True when a valid session manifest was loaded.
    Returns False when the manifest is unavailable or invalid.
    """

    manifest_path = (
        session_manifest_path(
            session_dir
        )
    )

    if (
        not manifest_path
        or not os.path.isfile(
            manifest_path
        )
    ):
        return False

    try:

        with open(
            manifest_path,
            "r",
            encoding="utf-8"
        ) as f:

            data = json.load(
                f
            )

        if not isinstance(
            data,
            dict
        ):
            return False

        # Validate the session identity.
        session_id = data.get(
            "session_id"
        )

        if not session_id:
            return False

        # Validate the control structure.
        control_keys = [
            "brightness",
            "shadow",
            "highlight",
            "depth",
            "dehaze",
            "contrast",
            "saturation",
            "color",
            "blending"
        ]

        saved_controls = data.get(
            "control_state",
            {}
        )

        if not isinstance(
            saved_controls,
            dict
        ):
            return False

        restored_controls = {}

        for key in control_keys:

            try:

                value = float(
                    saved_controls.get(
                        key,
                        0.0
                    )
                )

            except (
                TypeError,
                ValueError
            ):

                return False

            if not (
                np.isfinite(
                    value
                )
            ):
                return False

            if value < -1.0 or value > 1.0:
                return False

            restored_controls[key] = value

        # Validate saved exposure paths.
        saved_paths = {}

        for key in [
            "under_path",
            "normal_path",
            "over_path"
        ]:

            saved_path = data.get(key)

            if not saved_path:
                return False

            try:
                resolved_path = (
                    Path(saved_path)
                    .resolve()
                )

                resolved_session = (
                    Path(session_dir)
                    .resolve()
                )

                resolved_path.relative_to(
                    resolved_session
                )

            except (
                OSError,
                ValueError,
                TypeError
            ):

                return False

            if not resolved_path.is_file():
                return False

            saved_paths[key] = str(
                resolved_path
            )

        # Restore only after all validation succeeds.
        st.session_state.session_id = (
            session_id
        )

        st.session_state.under_path = (
            saved_paths["under_path"]
        )

        st.session_state.normal_path = (
            saved_paths["normal_path"]
        )

        st.session_state.over_path = (
            saved_paths["over_path"]
        )

        st.session_state.session_dir = (
            session_dir
        )

        # Validate and restore generated candidate paths.
        candidate_paths = {}

        for key in [
            "candidate_a",
            "candidate_b",
            "candidate_c"
        ]:

            saved_path = data.get(key)

            if not saved_path:
                return False

            try:
                resolved_path = (
                    Path(saved_path)
                    .resolve()
                )

                resolved_session = (
                    Path(session_dir)
                    .resolve()
                )

                resolved_path.relative_to(
                    resolved_session
                )

            except (
                OSError,
                ValueError,
                TypeError
            ):

                return False

            if not resolved_path.is_file():
                return False

            candidate_paths[key] = str(
                resolved_path
            )

        st.session_state.candidate_a = (
            candidate_paths["candidate_a"]
        )

        st.session_state.candidate_b = (
            candidate_paths["candidate_b"]
        )

        st.session_state.candidate_c = (
            candidate_paths["candidate_c"]
        )

        st.session_state.generation_done = True

        st.session_state.selected_candidate = (
            data.get(
                "selected_candidate"
            )
        )

        st.session_state.baseline_image = (
            data.get(
                "baseline_image"
            )
        )

        st.session_state.current_image = (
            data.get(
                "current_image"
            )
        )

        st.session_state.current_label = (
            data.get(
                "current_label"
            )
        )

        try:

            revision_number = int(
                data.get(
                    "revision_number",
                    0
                )
            )

        except (
            TypeError,
            ValueError
        ):

            return False

        if revision_number < 0:
            return False

        st.session_state.revision_number = (
            revision_number
        )

        st.session_state.control_state = (
            restored_controls
        )

        st.session_state.approved = bool(
            data.get(
                "approved",
                False
            )
        )

        history = data.get(
            "history",
            []
        )

        if not isinstance(
            history,
            list
        ):
            return False

        st.session_state.history = (
            history
        )

        return True

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError
    ):

        return False


# ============================================================
# SAVE UPLOAD
# ============================================================

def save_upload(
    uploaded_file,
    filename
):

    path = os.path.join(
        st.session_state.session_dir,
        filename
    )

    with open(
        path,
        "wb"
    ) as f:

        f.write(
            uploaded_file.getbuffer()
        )

    return path


# ============================================================
# VERIFY IMAGE
# ============================================================

def verify_image(
    path
):

    image = cv2.imread(
        path,
        cv2.IMREAD_COLOR
    )

    if image is None:

        raise RuntimeError(
            f"Could not read image:\n{path}"
        )

    return image.shape


# ============================================================
# RUN EXTERNAL PROGRAM
# ============================================================

def run_command(
    command
):

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            result.stdout
        )

    return result.stdout


# ============================================================
# CANDIDATE A
# ============================================================

def generate_candidate_a():

    output = os.path.join(
        st.session_state.session_dir,
        "candidate_a_v33.jpg"
    )

    command = [
        sys.executable,
        V33_SCRIPT,

        "--under",
        st.session_state.under_path,

        "--normal",
        st.session_state.normal_path,

        "--over",
        st.session_state.over_path,

        "--output",
        output
    ]

    log = run_command(
        command
    )

    return output, log


# ============================================================
# CANDIDATE B
# ============================================================

def generate_candidate_b():

    output = os.path.join(
        st.session_state.session_dir,
        "candidate_b_mertens.png"
    )

    command = [
        sys.executable,
        MERTENS_SCRIPT,

        "--under",
        st.session_state.under_path,

        "--normal",
        st.session_state.normal_path,

        "--over",
        st.session_state.over_path,

        "--output",
        output
    ]

    log = run_command(
        command
    )

    return output, log


# ============================================================
# CANDIDATE C
# ============================================================

def generate_candidate_c():

    output = os.path.join(
        st.session_state.session_dir,
        "candidate_c_mefnet.png"
    )

    command = [
        sys.executable,
        MEFNET_SCRIPT,

        "--under",
        st.session_state.under_path,

        "--normal",
        st.session_state.normal_path,

        "--over",
        st.session_state.over_path,

        "--output",
        output
    ]

    log = run_command(
        command
    )

    return output, log


# ============================================================
# SAVE FEEDBACK RECORD
# ============================================================

def save_feedback_record(
    record
):

    os.makedirs(
        FEEDBACK_DIR,
        exist_ok=True
    )

    records = []

    if os.path.isfile(
        FEEDBACK_FILE
    ):

        try:

            with open(
                FEEDBACK_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                records = json.load(
                    f
                )

            if not isinstance(
                records,
                list
            ):

                records = []

        except Exception:

            records = []

    records.append(
        record
    )

    with open(
        FEEDBACK_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            indent=2
        )


# ============================================================
# RESET WORKFLOW
# ============================================================

def reset_workflow():

    keys = [

        "session_id",
        "session_dir",

        "under_path",
        "normal_path",
        "over_path",

        "candidate_a",
        "candidate_b",
        "candidate_c",

        "candidate_logs",

        "generation_done",

        "selected_candidate",
        "baseline_image",
        "current_image",
        "current_label",

        "revision_number",

        "control_state",

        "approved",

        "history"
    ]

    for key in keys:

        if key in st.session_state:

            del st.session_state[key]

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">FreeMEF Client-Guided Fusion</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Generate three candidates, let the customer choose, '
    'then refine exactly what they request.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# RECOVER PREVIOUS SESSION
# ============================================================

with st.expander(
    "Recover Previous Session"
):

    recoverable_sessions = []

    if os.path.isdir(
        SESSION_ROOT
    ):

        for name in sorted(
            os.listdir(SESSION_ROOT),
            reverse=True
        ):

            session_dir = os.path.join(
                SESSION_ROOT,
                name
            )

            manifest = os.path.join(
                session_dir,
                SESSION_STATE_FILE
            )

            if (
                os.path.isdir(session_dir)
                and
                os.path.isfile(manifest)
            ):

                recoverable_sessions.append(
                    name
                )

    if not recoverable_sessions:

        st.info(
            "No recoverable sessions found."
        )

    else:

        selected_session = st.selectbox(
            "Select a saved session",
            recoverable_sessions,
            key="recovery_session"
        )

        if st.button(
            "Recover Selected Session",
            use_container_width=True
        ):

            recovery_dir = os.path.join(
                SESSION_ROOT,
                selected_session
            )

            if load_session_state(
                recovery_dir
            ):

                for widget_key in [
                    "under_file",
                    "normal_file",
                    "over_file",
                    "customer_feedback"
                ]:

                    st.session_state.pop(
                        widget_key,
                        None
                    )

                st.success(
                    f"Session {selected_session} recovered."
                )

                st.rerun()

            else:

                st.error(
                    "This saved session is invalid or incomplete "
                    "and could not be recovered."
                )


# ============================================================
# STEP 1 — UPLOAD
# ============================================================

st.header(
    "1. Upload the three exposures"
)

col1, col2, col3 = st.columns(3)


with col1:

    under_file = st.file_uploader(
        "Under exposure",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="under_file"
    )


with col2:

    normal_file = st.file_uploader(
        "Normal exposure",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="normal_file"
    )


with col3:

    over_file = st.file_uploader(
        "Over exposure",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="over_file"
    )


# ============================================================
# GENERATE A/B/C
# ============================================================

ready = (
    under_file is not None
    and
    normal_file is not None
    and
    over_file is not None
)

if ready:

    if st.button(
        "Generate Candidate A / B / C",
        type="primary",
        use_container_width=True
    ):

        try:

            create_session()

            st.session_state.under_path = (
                save_upload(
                    under_file,
                    "01_under.jpg"
                )
            )

            st.session_state.normal_path = (
                save_upload(
                    normal_file,
                    "02_normal.jpg"
                )
            )

            st.session_state.over_path = (
                save_upload(
                    over_file,
                    "03_over.jpg"
                )
            )

            verify_image(
                st.session_state.under_path
            )

            verify_image(
                st.session_state.normal_path
            )

            verify_image(
                st.session_state.over_path
            )

            # ------------------------------------------------
            # INPUT VALIDATION + ALIGNMENT
            # ------------------------------------------------

            with st.spinner(
                "Validating exposure alignment..."
            ):

                input_check = validate_and_align_exposures(
                    st.session_state.under_path,
                    st.session_state.normal_path,
                    st.session_state.over_path,
                )

            if not input_check["overall_ok"]:
                raise RuntimeError(
                    "The three exposure images failed "
                    "alignment validation. Please upload "
                    "a well-aligned Under, Normal, and Over set."
                )

            if not input_check["same_dimensions"]:
                st.warning(
                    "The exposure images have different dimensions. "
                    "They passed alignment validation, but matching "
                    "source dimensions are recommended."
                )

            st.success(
                "Exposure validation and alignment check passed."
            )

            # ------------------------------------------------
            # Candidate A
            # ------------------------------------------------

            with st.spinner(
                "Generating Candidate A — V3.3..."
            ):

                a_path, a_log = (
                    generate_candidate_a()
                )

            # ------------------------------------------------
            # Candidate B
            # ------------------------------------------------

            with st.spinner(
                "Generating Candidate B — Mertens..."
            ):

                b_path, b_log = (
                    generate_candidate_b()
                )

            # ------------------------------------------------
            # Candidate C
            # ------------------------------------------------

            with st.spinner(
                "Generating Candidate C — MEF-Net..."
            ):

                c_path, c_log = (
                    generate_candidate_c()
                )

            st.session_state.candidate_a = a_path

            st.session_state.candidate_b = b_path

            st.session_state.candidate_c = c_path

            st.session_state.candidate_logs = {

                "A": a_log,

                "B": b_log,

                "C": c_log
            }

            st.session_state.generation_done = True

            st.session_state.selected_candidate = None

            st.session_state.current_image = None

            st.session_state.current_label = None

            st.session_state.revision_number = 0

            st.session_state.approved = False

            st.session_state.history = []

            save_session_state()

            st.success(
                "All three candidates generated."
            )

        except Exception as exc:

            st.error(
                f"Generation failed:\n{exc}"
            )


# ============================================================
# STEP 2 — DISPLAY CANDIDATES
# ============================================================

if st.session_state.generation_done:

    st.header(
        "2. Choose the preferred image"
    )

    st.info(
        "The customer makes the final choice. "
        "There is no automatic winner."
    )

    ca, cb, cc = st.columns(3)


    # --------------------------------------------------------
    # A
    # --------------------------------------------------------

    with ca:

        st.markdown(
            '<div class="candidate-title">'
            'Candidate A'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="candidate-method">'
            'V3.3 Client-Bright'
            '</div>',
            unsafe_allow_html=True
        )

        show_image_preview(
            st.session_state.candidate_a,
            caption="Candidate A",
            key_prefix="candidate_a",
            preview_width=None,
            center=False
        )

        if st.button(
            "Choose A",
            key="select_a",
            use_container_width=True
        ):

            st.session_state.selected_candidate = "A"

            st.session_state.baseline_image = (
                st.session_state.candidate_a
            )

            st.session_state.current_image = (
                st.session_state.candidate_a
            )

            st.session_state.current_label = (
                "Candidate A — V3.3"
            )

            st.session_state.revision_number = 0

            st.session_state.control_state = {
                "brightness": 0.0,
                "shadow": 0.0,
                "highlight": 0.0,
                "depth": 0.0,
                "dehaze": 0.0,
                "contrast": 0.0,
                "saturation": 0.0,
                "color": 0.0,
                "blending": 0.0,
            }

            st.session_state.history = []

            st.session_state.approved = False

            save_session_state()

            st.rerun()


    # --------------------------------------------------------
    # B
    # --------------------------------------------------------

    with cb:

        st.markdown(
            '<div class="candidate-title">'
            'Candidate B'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="candidate-method">'
            'Mertens V3'
            '</div>',
            unsafe_allow_html=True
        )

        show_image_preview(
            st.session_state.candidate_b,
            caption="Candidate B",
            key_prefix="candidate_b",
            preview_width=None,
            center=False
        )

        if st.button(
            "Choose B",
            key="select_b",
            use_container_width=True
        ):

            st.session_state.selected_candidate = "B"

            st.session_state.baseline_image = (
                st.session_state.candidate_b
            )

            st.session_state.current_image = (
                st.session_state.candidate_b
            )

            st.session_state.current_label = (
                "Candidate B — Mertens V3"
            )

            st.session_state.revision_number = 0

            st.session_state.control_state = {
                "brightness": 0.0,
                "shadow": 0.0,
                "highlight": 0.0,
                "depth": 0.0,
                "dehaze": 0.0,
                "contrast": 0.0,
                "saturation": 0.0,
                "color": 0.0,
                "blending": 0.0,
            }

            st.session_state.history = []

            st.session_state.approved = False

            save_session_state()

            st.rerun()


    # --------------------------------------------------------
    # C
    # --------------------------------------------------------

    with cc:

        st.markdown(
            '<div class="candidate-title">'
            'Candidate C'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="candidate-method">'
            'MEF-Net'
            '</div>',
            unsafe_allow_html=True
        )

        show_image_preview(
            st.session_state.candidate_c,
            caption="Candidate C",
            key_prefix="candidate_c",
            preview_width=None,
            center=False
        )

        if st.button(
            "Choose C",
            key="select_c",
            use_container_width=True
        ):

            st.session_state.selected_candidate = "C"

            st.session_state.baseline_image = (
                st.session_state.candidate_c
            )

            st.session_state.current_image = (
                st.session_state.candidate_c
            )

            st.session_state.current_label = (
                "Candidate C — MEF-Net"
            )

            st.session_state.revision_number = 0

            st.session_state.control_state = {
                "brightness": 0.0,
                "shadow": 0.0,
                "highlight": 0.0,
                "depth": 0.0,
                "dehaze": 0.0,
                "contrast": 0.0,
                "saturation": 0.0,
                "color": 0.0,
                "blending": 0.0,
            }

            st.session_state.history = []

            st.session_state.approved = False

            save_session_state()

            st.rerun()


# ============================================================
# STEP 3 — GEMINI FEEDBACK REFINEMENT
# ============================================================

if st.session_state.selected_candidate:

    st.header(
        "3. Refine the selected image"
    )

    st.markdown(
        f"""
        <div class="selected-box">
        <b>Selected:</b>
        {st.session_state.current_label}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div style=\"height:20px;\"></div>",
        unsafe_allow_html=True
    )

    show_image_preview(
        st.session_state.current_image,
        caption="Current result",
        key_prefix="current_result",
        preview_width=620,
        center=True
    )

    feedback = st.text_area(
        "What would you like to change?",
        placeholder=(
            "Example:\n"
            "Make it slightly darker but preserve "
            "the highlight detail and keep the colors natural."
        ),
        height=140,
        key="customer_feedback"
    )


    # --------------------------------------------------------
    # APPLY FEEDBACK
    # --------------------------------------------------------

    if st.button(
        "Apply Feedback",
        type="primary",
        use_container_width=True
    ):

        if not feedback.strip():

            st.warning(
                "Please describe the change you want."
            )

        else:

            try:

                # --------------------------------------------
                # GEMINI TRANSLATION
                # --------------------------------------------

                with st.spinner(
                    "Understanding your feedback..."
                ):

                    values = translate_feedback(
                        feedback
                    )


                # --------------------------------------------
                # SHOW GEMINI TRANSLATION
                # --------------------------------------------

                st.subheader(
                    "Detected feedback"
                )

                st.write(
                    values.get(
                        "summary",
                        ""
                    )
                )

                st.write(
                    f"Confidence: "
                    f"{float(values.get('confidence', 0.0)):.3f}"
                )


                correction_keys = [

                    "brightness",
                    "shadow",
                    "highlight",
                    "depth",
                    "dehaze",
                    "contrast",
                    "saturation",
                    "color",
                    "blending"
                ]

                non_zero = []

                for key in correction_keys:

                    value = float(
                        values.get(
                            key,
                            0.0
                        )
                    )

                    if abs(value) > 1e-6:

                        non_zero.append(
                            (
                                key,
                                value
                            )
                        )


                if not non_zero:

                    st.warning(
                        "No supported correction was detected "
                        "from the feedback."
                    )

                    st.stop()


                st.write(
                    "**Correction values:**"
                )

                for key, value in non_zero:

                    st.write(
                        f"✓ {key}: {value:+.3f}"
                    )


                # --------------------------------------------
                # UPDATE PERSISTENT CORRECTION STATE
                # --------------------------------------------

                for key in correction_keys:

                    try:
                        delta = float(
                            values.get(
                                key,
                                0.0
                            )
                        )
                    except (TypeError, ValueError):
                        delta = 0.0

                    current_value = float(
                        st.session_state.control_state.get(
                            key,
                            0.0
                        )
                    )

                    st.session_state.control_state[key] = max(
                        -1.0,
                        min(
                            1.0,
                            current_value + delta
                        )
                    )

                # --------------------------------------------
                # REVISION NUMBER
                # --------------------------------------------

                revision = (
                    st.session_state.revision_number
                    + 1
                )

                output_path = os.path.join(
                    st.session_state.session_dir,
                    f"revision_{revision}.jpg"
                )


                # --------------------------------------------
                # APPLY EXACT GEMINI VALUES
                # --------------------------------------------

                with st.spinner(
                    "Applying your requested correction..."
                ):

                    diagnostics = refine_with_values(

                        selected_path=
                            st.session_state.baseline_image,

                        normal_path=
                            st.session_state.normal_path,

                        under_path=
                            st.session_state.under_path,

                        over_path=
                            st.session_state.over_path,

                        values=st.session_state.control_state,

                        output_path=
                            output_path
                    )


                result_path = diagnostics[
                    "output"
                ]


                # --------------------------------------------
                # UPDATE CURRENT IMAGE
                # --------------------------------------------

                st.session_state.current_image = (
                    result_path
                )

                st.session_state.current_label = (
                    f"{st.session_state.current_label} "
                    f"→ Revision {revision}"
                )

                st.session_state.revision_number = (
                    revision
                )

                st.session_state.approved = False


                # --------------------------------------------
                # HISTORY RECORD
                # --------------------------------------------

                record = {

                    "timestamp":
                        datetime.now().isoformat(),

                    "session_id":
                        st.session_state.session_id,

                    "selected_candidate":
                        st.session_state.selected_candidate,

                    "revision":
                        revision,

                    "feedback":
                        feedback,

                    "gemini_values":
                        values,

                    "output":
                        result_path,

                    "approved":
                        False
                }


                st.session_state.history.append(
                    record
                )

                save_feedback_record(
                    record
                )

                save_session_state()


                # --------------------------------------------
                # RESULT
                # --------------------------------------------

                st.success(
                    f"Revision {revision} generated."
                )

                show_image_preview(
                    result_path,
                    caption=f"Revision {revision}",
                    key_prefix=f"generated_revision_{revision}",
                    preview_width=620,
                    center=True
                )


                # --------------------------------------------
                # DIAGNOSTICS
                # --------------------------------------------

                if "before_mean_brightness" in diagnostics:

                    st.write(
                        f"Brightness before: "
                        f"{diagnostics['before_mean_brightness']:.4f}"
                    )

                if "after_mean_brightness" in diagnostics:

                    st.write(
                        f"Brightness after: "
                        f"{diagnostics['after_mean_brightness']:.4f}"
                    )

                if "brightness_difference" in diagnostics:

                    st.write(
                        f"Brightness change: "
                        f"{diagnostics['brightness_difference']:+.4f}"
                    )


            except Exception as exc:

                # Keep technical details in the server terminal.
                print(
                    "\n[REFINEMENT ERROR]",
                    repr(exc)
                )

                # Show only a customer-safe message in the UI.
                st.error(
                    "We couldn't apply that correction. "
                    "Please try again with a simpler or more specific request."
                )


    # --------------------------------------------------------
    # APPROVE
    # --------------------------------------------------------

    if st.button(
        "Approve This Result",
        use_container_width=True
    ):

        st.session_state.approved = True

        record = {

            "timestamp":
                datetime.now().isoformat(),

            "session_id":
                st.session_state.session_id,

            "selected_candidate":
                st.session_state.selected_candidate,

            "revision":
                st.session_state.revision_number,

            "feedback":
                "",

            "gemini_values":
                {},

            "output":
                st.session_state.current_image,

            "approved":
                True
        }

        st.session_state.history.append(
            record
        )

        save_feedback_record(
            record
        )

        save_session_state()

        st.rerun()


# ============================================================
# APPROVED RESULT
# ============================================================

if st.session_state.approved:

    st.markdown(
        """
        <div class="approved-box">
        <h2>✅ Approved Final Image</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    show_image_preview(
        st.session_state.current_image,
        caption="Approved Final Image",
        key_prefix="approved_result",
        preview_width=620,
        center=True
    )

    st.write(
        "Approved image:"
    )

    st.code(
        st.session_state.current_image
    )


# ============================================================
# REVISION HISTORY
# ============================================================

if st.session_state.history:

    st.header(
        "4. Revision history"
    )

    for history_index, item in enumerate(
        st.session_state.history
    ):

        revision = item.get(
            "revision",
            0
        )

        feedback = item.get(
            "feedback",
            ""
        )

        approved = item.get(
            "approved",
            False
        )

        output_path = item.get(
            "output",
            ""
        )

        if approved:

            st.write(
                f"Revision {revision} — APPROVED"
            )

        else:

            st.write(
                f"Revision {revision} — {feedback}"
            )

        # ----------------------------------------------------
        # HISTORICAL REVISION IMAGE
        # ----------------------------------------------------

        if output_path and os.path.isfile(output_path):

            show_image_preview(
                output_path,
                caption=f"Revision {revision}",
                key_prefix=f"revision_{revision}_{history_index}",
                preview_width=620,
                center=True
            )

        else:

            st.warning(
                f"Revision {revision} image is no longer available."
            )

        # ----------------------------------------------------
        # GEMINI CONTROL SUMMARY
        # ----------------------------------------------------

        gemini_values = item.get(
            "gemini_values",
            {}
        )

        if gemini_values:

            changes = []

            for key in [
                "brightness",
                "shadow",
                "highlight",
                "depth",
                "dehaze",
                "contrast",
                "saturation",
                "color",
                "blending"
            ]:

                try:

                    value = float(
                        gemini_values.get(
                            key,
                            0.0
                        )
                    )

                except (TypeError, ValueError):

                    value = 0.0

                if abs(value) > 1e-6:

                    changes.append(
                        f"{key}={value:+.3f}"
                    )

            if changes:

                st.caption(
                    "Gemini controls: "
                    +
                    ", ".join(changes)
                )

        # ----------------------------------------------------
        # RESTORE REVISION
        # ----------------------------------------------------

        if (
            not approved
            and output_path
            and os.path.isfile(output_path)
        ):

            if st.button(
                f"Restore Revision {revision}",
                key=f"restore_revision_{revision}"
            ):

                correction_keys = [
                    "brightness",
                    "shadow",
                    "highlight",
                    "depth",
                    "dehaze",
                    "contrast",
                    "saturation",
                    "color",
                    "blending"
                ]

                restored_state = {
                    key: 0.0
                    for key in correction_keys
                }

                # Rebuild the cumulative control state
                # represented by this revision.
                for previous_item in st.session_state.history:

                    previous_revision = previous_item.get(
                        "revision",
                        0
                    )

                    if (
                        previous_revision > revision
                        or previous_item.get(
                            "approved",
                            False
                        )
                    ):
                        continue

                    previous_values = previous_item.get(
                        "gemini_values",
                        {}
                    )

                    for key in correction_keys:

                        try:

                            delta = float(
                                previous_values.get(
                                    key,
                                    0.0
                                )
                            )

                        except (TypeError, ValueError):

                            delta = 0.0

                        restored_state[key] = max(
                            -1.0,
                            min(
                                1.0,
                                restored_state[key] + delta
                            )
                        )

                st.session_state.control_state = (
                    restored_state
                )

                st.session_state.current_image = (
                    output_path
                )

                st.session_state.revision_number = (
                    revision
                )

                st.session_state.current_label = (
                    f"Candidate "
                    f"{st.session_state.selected_candidate} "
                    f"→ Revision {revision}"
                )

                st.session_state.approved = False

                st.success(
                    f"Revision {revision} restored."
                )

                save_session_state()

                st.rerun()


# ============================================================
# NEW SESSION
# ============================================================

if st.session_state.generation_done:

    st.divider()

    if st.button(
        "Start New Scene"
    ):

        reset_workflow()
# ============================================================
# APPROVED FINAL RESULT
# ============================================================

if st.session_state.approved:

    st.markdown(
        """
        <div class="approved-box">
        <h2>✅ Approved Final Image</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

    show_image_preview(
        st.session_state.current_image,
        caption="Approved Final Image",
        key_prefix="approved_final",
        preview_width=620,
        center=True
    )

    final_path = st.session_state.current_image

    if final_path and os.path.isfile(final_path):

        with open(
            final_path,
            "rb"
        ) as f:

            final_bytes = f.read()

        st.download_button(
            label="⬇️ Download Final JPG",
            data=final_bytes,
            file_name="freemef_final.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

    else:

        st.error(
            "Approved final image file could not be found."
        )
