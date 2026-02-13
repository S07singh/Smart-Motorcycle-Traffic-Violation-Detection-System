# 🏍️ Smart Motorcycle Traffic Violation Detection System

AI-powered traffic violation detection using **dual YOLOv8 models** and **EasyOCR**. This system detects helmet violations, per-motorcycle triple riding, and extracts license plate numbers from images and videos through a professional Streamlit web interface.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?logo=streamlit&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?logo=yolo&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🧠 Model Training

The custom YOLOv8 model was trained on Google Colab using a curated dataset of motorcycle traffic images.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/S07singh/Smart-Motorcycle-Traffic-Violation-Detection-System/blob/main/Smart_Motorcycle_Traffic_Violation_Detection_System.ipynb)

---

## ✨ Features

- **Dual YOLOv8 Detection** — Custom model for helmet/person/plate + pretrained COCO model for motorcycle detection.
- **Per-Motorcycle Triple Riding** — Center-based person-motorcycle spatial association instead of naive global person count.
- **No Helmet Detection** — Flags riders without helmets with per-detection confidence scores.
- **Enhanced License Plate OCR** — Multi-stage preprocessing (upscale → adaptive threshold → morphological cleanup) + Indian plate regex validation.
- **Image & Video Support** — Upload JPG/PNG images or MP4/AVI/MOV/MKV videos for analysis.
- **Interactive Confidence Tuning** — Adjust the detection threshold via a sidebar slider.
- **Structured Violation Reports** — Summary metrics, violation cards, detailed logs, and plate OCR details.
- **Production-Ready UI** — Modern Streamlit interface with custom CSS styling.

---

## 📁 Project Structure

```
Smart Motorcycle Traffic Violation Detection System/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── LICENSE                    # MIT License
├── .gitignore                 # Git ignore rules
├── Smart_Motorcycle_Traffic_Violation_Detection_System.ipynb  # Training notebook
├── model/
│   ├── best.pt                # Custom trained YOLOv8 weights
│   └── yolov8n.pt             # Pretrained COCO YOLOv8n weights
├── utils/
│   ├── __init__.py            # Package init
│   ├── detector.py            # Dual YOLO detection (custom + COCO motorcycle)
│   ├── ocr_engine.py          # Enhanced OCR pipeline with preprocessing
│   ├── violation_checker.py   # Per-motorcycle violation logic
│   └── visualizer.py          # Selective bounding box annotation
├── data/
│   ├── classes.txt            # Class names
│   ├── traffic.yaml           # YOLO training config
│   ├── train/                 # Training images & labels
│   └── val/                   # Validation images & labels
└── test/                      # Test images
```

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/S07singh/Smart-Motorcycle-Traffic-Violation-Detection-System.git
cd Smart-Motorcycle-Traffic-Violation-Detection-System
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv env
# Windows
env\Scripts\activate
# Linux/macOS
source env/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place the model weights

- **Custom model:** Place your trained YOLOv8 weights at `model/best.pt`.
- **COCO model:** Place `yolov8n.pt` at `model/yolov8n.pt` (download from [Ultralytics](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt)).

### 5. Run the application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 🎯 How It Works

### Detection Pipeline

```
Input Frame
    │
    ├──► Custom YOLOv8 ──► helmet, no_helmet, person, license_plate
    │
    └──► COCO YOLOv8n  ──► motorcycle
              │
              ▼
     Per-Motorcycle Association (center-based)
              │
              ▼
     Violation Report + Annotated Image + OCR Results
```

1. **Upload** — User uploads an image or video through the Streamlit interface.
2. **Dual-Model Inference** — Two YOLOv8 models run in parallel:
   - **Custom model** → detects helmets, no-helmet riders, persons, and license plates.
   - **COCO model** → detects motorcycles (class 3).
3. **Triple Riding Check** — For each motorcycle, persons are associated using center-point-in-bbox matching. If a motorcycle has >2 associated persons, it is flagged.
4. **No Helmet Check** — Any `no_helmet` detection triggers a helmet violation.
5. **OCR Extraction** — Detected plates are preprocessed (2-3x upscale → adaptive threshold → morphological cleanup) and read using EasyOCR, then cleaned with Indian plate regex `[A-Z]{2}\d{2}[A-Z]{1,2}\d{4}`.
6. **Visualisation** — Color-coded bounding boxes:
   - 🟢 Green — helmet
   - 🔴 Red — no helmet / violating motorcycle / violating persons
   - 🟠 Orange — person (non-violating)
   - 🟡 Cyan — motorcycle (safe) / license plate
7. **Report** — Structured violation report with metrics, violation cards, plate OCR details, and detection logs.

### Classes Detected

| Model | Class ID | Class Name | Description |
|-------|----------|------------|-------------|
| Custom | 0 | `helmet` | Rider wearing a helmet |
| Custom | 1 | `no_helmet` | Rider without a helmet |
| Custom | 2 | `person` | Person on/near motorcycle |
| Custom | 3 | `license_plate` | Vehicle license plate |
| COCO | 3 | `motorcycle` | Motorcycle vehicle |

### OCR Preprocessing Pipeline

| Step | Technique | Purpose |
|------|-----------|---------|
| 1 | Bicubic upscale (2-3x) | Small plates need ~32px char height for OCR |
| 2 | Grayscale conversion | Reduces color noise |
| 3 | Adaptive Gaussian threshold | Handles uneven lighting → clean B&W text |
| 4 | Morphological close + open | Fills char gaps, removes noise dots |
| 5 | Indian plate regex | Extracts valid `XX00XX0000` pattern from noisy OCR |

---

## ⚙️ Configuration

| Setting | Location | Default | Description |
|---------|----------|---------|-------------|
| Confidence Threshold | Sidebar slider | 0.25 | Min confidence for YOLO detections |
| OCR GPU | `utils/ocr_engine.py` | `False` | Set to `True` if CUDA GPU available |
| Custom Model Path | `app.py` | `model/best.pt` | Path to custom YOLOv8 weights |
| COCO Model Path | `app.py` | `model/yolov8n.pt` | Path to COCO YOLOv8n weights |

---

## ☁️ Deploy on Streamlit Cloud

1. Push your repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repo and select `app.py` as the main file.
4. Ensure `model/best.pt` and `model/yolov8n.pt` are in the repo (use Git LFS for large files).
5. Deploy!

> **Note:** Streamlit Cloud provides CPU-only instances. The app is configured to run without GPU by default.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| **Python 3.9+** | Core programming language |
| **Streamlit** | Web application framework |
| **Ultralytics YOLOv8** | Object detection (custom + COCO) |
| **OpenCV** | Image/video processing & plate preprocessing |
| **EasyOCR** | Optical character recognition |
| **NumPy** | Array operations |
| **Pillow** | Image format handling |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📬 Contact

For questions or suggestions, please open an issue on the GitHub repository.
