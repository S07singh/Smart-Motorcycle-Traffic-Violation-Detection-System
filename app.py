import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

from utils.detector import load_model, load_coco_model, run_detection, detect_motorcycles, detect_persons
from utils.ocr_engine import init_ocr_reader, extract_plate_text
from utils.violation_checker import check_violations, check_triple_riding
from utils.visualizer import draw_detections

st.set_page_config(
    page_title="Smart Motorcycle Violation Detector",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --primary: #FF4B4B;
        --success: #28a745;
        --warning: #ffc107;
        --info: #17a2b8;
    }

    .main .block-container {
        padding-top: 2rem;
        max-width: 1200px;
    }

    .hero-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #888;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    .violation-card {
        background: linear-gradient(135deg, #ff4b4b22 0%, #ff8e5322 100%);
        border-left: 4px solid #FF4B4B;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
        font-size: 1.05rem;
    }
    .safe-card {
        background: linear-gradient(135deg, #28a74522 0%, #20c99722 100%);
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.75rem;
        font-weight: 600;
        font-size: 1.05rem;
        color: #28a745;
    }

    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        border: 1px solid #e9ecef;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #333;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .plate-text {
        background: #1a1a2e;
        color: #e0e0e0;
        font-family: 'Courier New', monospace;
        font-size: 1.5rem;
        font-weight: 700;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        text-align: center;
        letter-spacing: 3px;
        border: 2px solid #16213e;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e0e0 !important;
    }

    .styled-divider {
        border: none;
        border-top: 2px solid #e9ecef;
        margin: 1.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def get_model():
    model_path = os.path.join("model", "best.pt")
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at `{model_path}`. Please place your trained model there.")
        st.stop()
    return load_model(model_path)


@st.cache_resource
def get_coco_model():
    model_path = os.path.join("model", "yolov8n.pt")
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at `{model_path}`. Please place your trained model there.")
        st.stop()
    return load_coco_model(model_path)


@st.cache_resource
def get_ocr_reader():
    return init_ocr_reader()


with st.sidebar:
    st.markdown("## ⚙️ Settings")

    confidence_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.10,
        max_value=0.95,
        value=0.25,
        step=0.05,
        help="Only detections above this score will be shown. "
             "Lower → more detections (but more false positives). "
             "Higher → fewer detections (but more precise).",
    )

    st.markdown("---")
    st.markdown("## 📋 About")
    st.markdown(
        """
        **Smart Motorcycle Traffic Violation Detection System**

        This application uses **dual YOLOv8 models**:
        - 🏍️ **COCO model** — motorcycle detection
        - 🎯 **Custom model** — helmet, no_helmet, person, license plate

        **Violations detected:**
        - 🪖 No Helmet (per rider)
        - 👥 Triple Riding (per motorcycle, center-based association)
        - 🔢 License Plate OCR

        **Tech Stack:**
        - Ultralytics YOLOv8
        - EasyOCR
        - OpenCV
        - Streamlit
        """
    )
    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#666; font-size:0.8rem;'>"
        "Built with ❤️ using Streamlit</p>",
        unsafe_allow_html=True,
    )


st.markdown(
    '<p class="hero-title">🏍️ Smart Motorcycle Violation Detector</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-subtitle">'
    "AI-powered traffic violation detection using YOLOv8 & EasyOCR — "
    "detect helmet violations, triple riding, and extract license plate numbers."
    "</p>",
    unsafe_allow_html=True,
)

st.markdown("### 📤 Upload Image or Video")

uploaded_file = st.file_uploader(
    "Choose an image or video file",
    type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"],
    help="Supported formats: JPG, PNG, BMP (image) · MP4, AVI, MOV, MKV (video)",
)

is_video = False
if uploaded_file is not None:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    is_video = file_ext in ["mp4", "avi", "mov", "mkv"]


def process_image(image_bgr: np.ndarray, conf: float):
    model = get_model()
    coco_model = get_coco_model()
    reader = get_ocr_reader()

    # Run custom model for helmet/no_helmet/person/license_plate
    detections = run_detection(model, image_bgr, confidence=conf)

    # Run COCO model for motorcycle detection
    motorcycles = detect_motorcycles(coco_model, image_bgr, confidence=conf)

    # Extract person detections for association
    persons = detect_persons(detections)

    # Per-motorcycle triple riding check
    triple_riding_results = check_triple_riding(motorcycles, persons)

    # Full violation report with per-motorcycle data
    violation_report = check_violations(detections, triple_riding_results)

    plate_results = []
    for det in detections:
        if det["class_name"] == "license_plate":
            ocr_result = extract_plate_text(reader, image_bgr, det["bbox"])
            plate_results.append(ocr_result)

    annotated = draw_detections(image_bgr, detections, violation_report, motorcycles)

    return annotated, detections, violation_report, plate_results, motorcycles


def display_results(annotated_rgb, detections, violation_report, plate_results):
    st.markdown("### 🖼️ Detection Results")
    st.image(annotated_rgb, caption="Annotated Image", width="stretch")

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    st.markdown("### 📊 Detection Summary")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'{violation_report["person_count"]}</div>'
            f'<div class="metric-label">Persons</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'{violation_report["motorcycle_count"]}</div>'
            f'<div class="metric-label">Motorcycles</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'{violation_report["helmet_count"]}</div>'
            f'<div class="metric-label">Helmets</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'{violation_report["no_helmet_count"]}</div>'
            f'<div class="metric-label">No Helmet</div></div>',
            unsafe_allow_html=True,
        )
    with col5:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">'
            f'{violation_report["license_plate_count"]}</div>'
            f'<div class="metric-label">Plates</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    st.markdown("### 🚨 Violation Report")
    if violation_report["violations"]:
        for v in violation_report["violations"]:
            st.markdown(f'<div class="violation-card">{v}</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            '<div class="safe-card">✅ No Violations Detected — All Clear!</div>',
            unsafe_allow_html=True,
        )

    if violation_report["violation_details"]:
        with st.expander("📝 Violation Details", expanded=False):
            for i, detail in enumerate(violation_report["violation_details"], 1):
                extra = ""
                if detail.get("persons_count"):
                    extra = f" — Persons on bike: **{detail['persons_count']}**"
                st.markdown(
                    f"**{i}.** {detail['violation_type']} — "
                    f"`{detail['class_name']}` — "
                    f"Confidence: **{detail['confidence']:.1%}** — "
                    f"BBox: `{detail['bbox']}`{extra}"
                )

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    st.markdown("### 🔢 License Plate Recognition")
    if plate_results:
        for i, pr in enumerate(plate_results):
            col_crop, col_text = st.columns([1, 2])
            with col_crop:
                if pr["plate_crop"] is not None:
                    crop_rgb = cv2.cvtColor(pr["plate_crop"], cv2.COLOR_BGR2RGB)
                    st.image(crop_rgb, caption=f"Plate Crop #{i+1}", width=200)
            with col_text:
                if pr["cleaned_text"]:
                    st.markdown(
                        f'<div class="plate-text">{pr["cleaned_text"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"OCR Confidence: {pr['confidence']:.1%}")
                    with st.expander("📋 OCR Details", expanded=False):
                        st.markdown(f"**Raw OCR Text:** `{pr['raw_text']}`")
                        st.markdown(f"**Cleaned Plate:** `{pr['cleaned_text']}`")
                        st.markdown(f"**Confidence:** {pr['confidence']:.1%}")
                else:
                    st.warning("Could not read plate text. The plate may be blurry or at an angle.")
                    if pr["raw_text"]:
                        st.caption(f"Raw OCR attempt: `{pr['raw_text']}`")
    else:
        st.info("No license plates detected in this image.")

    with st.expander("🔍 All Detections", expanded=False):
        if detections:
            for i, det in enumerate(detections, 1):
                st.markdown(
                    f"**{i}.** `{det['class_name']}` — "
                    f"Confidence: **{det['confidence']:.1%}** — "
                    f"BBox: `{det['bbox']}`"
                )
        else:
            st.write("No objects detected.")


if uploaded_file is not None:

    run_button = st.button("🚀 Run Detection", type="primary", use_container_width=True)

    if run_button:

        if not is_video:
            with st.spinner("🔍 Running dual-model detection & OCR..."):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

                if image_bgr is None:
                    st.error("❌ Failed to read the uploaded image. Please try a different file.")
                    st.stop()

                annotated, detections, violation_report, plate_results, motorcycles = (
                    process_image(image_bgr, confidence_threshold)
                )

                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            display_results(annotated_rgb, detections, violation_report, plate_results)

        else:
            st.markdown("### 🎬 Video Processing")

            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_ext}")
            tfile.write(uploaded_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)

            if not cap.isOpened():
                st.error("❌ Failed to open the uploaded video. Please try a different file.")
                os.unlink(tfile.name)
                st.stop()

            fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            out_path = tempfile.NamedTemporaryFile(
                delete=False, suffix=".mp4"
            ).name
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

            progress_bar = st.progress(0, text="Processing video frames...")
            status_text = st.empty()

            all_violations = set()
            all_plate_texts = []
            total_detections = 0
            frame_count = 0
            max_persons_in_frame = 0
            total_no_helmets = 0

            model = get_model()
            coco_model = get_coco_model()
            reader = get_ocr_reader()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                detections = run_detection(model, frame, confidence=confidence_threshold)
                motorcycles = detect_motorcycles(coco_model, frame, confidence=confidence_threshold)
                persons = detect_persons(detections)
                triple_riding_results = check_triple_riding(motorcycles, persons)
                violation_report = check_violations(detections, triple_riding_results)
                annotated_frame = draw_detections(frame, detections, violation_report, motorcycles)

                out.write(annotated_frame)

                total_detections += len(detections)
                for v in violation_report["violations"]:
                    all_violations.add(v)
                max_persons_in_frame = max(
                    max_persons_in_frame, violation_report["person_count"]
                )
                total_no_helmets += violation_report["no_helmet_count"]

                if frame_count % 10 == 0:
                    for det in detections:
                        if det["class_name"] == "license_plate":
                            ocr_result = extract_plate_text(
                                reader, frame, det["bbox"]
                            )
                            if ocr_result["cleaned_text"]:
                                all_plate_texts.append(
                                    {"text": ocr_result["cleaned_text"], "confidence": ocr_result["confidence"]}
                                )

                progress = frame_count / max(total_frames, 1)
                progress_bar.progress(
                    min(progress, 1.0),
                    text=f"Processing frame {frame_count}/{total_frames}...",
                )

            cap.release()
            out.release()
            progress_bar.empty()
            status_text.empty()

            st.video(out_path)

            try:
                os.unlink(tfile.name)
            except OSError:
                pass

            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

            st.markdown("### 📊 Video Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Frames", frame_count)
            with col2:
                st.metric("Total Detections", total_detections)
            with col3:
                st.metric("Max Persons/Frame", max_persons_in_frame)
            with col4:
                st.metric("No-Helmet Instances", total_no_helmets)

            st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

            st.markdown("### 🚨 Violations Found in Video")
            if all_violations:
                for v in all_violations:
                    st.markdown(
                        f'<div class="violation-card">{v}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown(
                    '<div class="safe-card">✅ No Violations Detected — All Clear!</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("### 🔢 License Plates Detected in Video")
            if all_plate_texts:
                seen = set()
                unique_plates = []
                for pt in all_plate_texts:
                    if pt["text"] not in seen:
                        seen.add(pt["text"])
                        unique_plates.append(pt)
                for pt in unique_plates:
                    st.markdown(
                        f'<div class="plate-text">{pt["text"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.caption(f"OCR Confidence: {pt['confidence']:.1%}")
            else:
                st.info("No license plates detected or read in this video.")
else:
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; padding: 3rem 1rem; color: #888;">
            <p style="font-size: 3rem; margin-bottom: 0.5rem;">📷</p>
            <p style="font-size: 1.2rem; font-weight: 600;">
                Upload an image or video to get started
            </p>
            <p style="font-size: 0.9rem;">
                Supported: JPG, PNG, BMP &nbsp;|&nbsp; MP4, AVI, MOV, MKV
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
