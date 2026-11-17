# 🧠 NeuroVision AI — Uncertainty-Guided Brain Tumour Segmentation & Quality Assessment

[![FastAPI](https://img.shields.io/badge/FastAPI-0.143+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-EE4C2C.svg?logo=pytorch)](https://pytorch.org)
[![React](https://img.shields.io/badge/React-19-61DAFB.svg?logo=react)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-8.3+-646CFF.svg?logo=vite)](https://vitejs.dev)
[![Three.js](https://img.shields.io/badge/Three.js-0.186+-000000.svg?logo=three.js)](https://threejs.org)
[![TailwindCSS](https://img.shields.io/badge/TailwindCSS-v4-38B2AC.svg?logo=tailwind-css)](https://tailwindcss.com)

**NeuroVision AI** is an advanced medical AI research demonstration platform designed for technical evaluation of multi-task deep learning architectures, Monte Carlo dropout boundary uncertainty estimation, 3D anatomical polygon reconstruction, and automated imaging quality assurance.

---

## 🌟 Key Highlights & System Architecture

1. **4-Class Tumour Classification**:
   - Classifies scans into **Glioma**, **Meningioma**, **Pituitary Tumour**, and **No Tumour (Healthy)**.
   - Outputs confidence scores, comparative probability distributions via Recharts, and suggested clinical steps.
2. **Precision nnU-Net Segmentation**:
   - Trained on the **BRISC 2025** dataset with InstanceNorm, strided convolutions, and paired Z-score intensity normalization.
   - Generates translucent crimson lesion core overlays with high-contrast gold border contours.
3. **Monte Carlo Dropout Uncertainty Mapping**:
   - Executes $T$ stochastic forward passes with active dropout layers.
   - Computes pixel-wise variance and entropy maps (Jet Colormap) with automated boundary risk gating (35% threshold).
4. **Interactive 3D Marching Cubes Mesh Reconstruction**:
   - Extracts watertight 3D polygonal surface meshes using Marching Cubes rendered via **Three.js**.
   - Features anatomical reference cerebral cage, camera presets (Isometric, Axial, Sagittal, Coronal), and direct **STL & OBJ** export.
5. **Automated 8-Point Quality Assessment (QA)**:
   - Evaluates input readability, spatial resolution, motion blur (Laplacian variance), tissue contrast dynamic range, cranial domain fit, boundary uncertainty, and lesion component plausibility.
6. **Edge Lightweight Model Benchmarking**:
   - Features **FastBrain-UNet** (depthwise separable convolutions), ~0.19M parameters (**38.7x** fewer than the nnU-Net). Its weights are **not shipped**: train them with `training/train_segmentation.py --model fast --teacher models/best_nnunet_model.pt`. Until then the API returns a clear `503` for `fast-brain-unet` instead of serving a randomly-initialised network. Latency / Dice numbers shown in the benchmark modal are measured or read from `models/segmentation_metrics.json`, never hardcoded.
7. **Complete Data & Report Export**:
   - Download results as **JSON data packages**, **CSV metric sheets**, **Segmentation PNG overlays**, **3D STL/OBJ meshes**, or printable clinical research PDF reports.

---

## 📁 Project Structure

```
c:\Srihaas\ALTHON\
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI application entrypoint & middleware
│   │   ├── api/
│   │   │   └── routes.py               # REST API endpoints (analyze, benchmark, export, demo)
│   │   ├── models/
│   │   │   ├── classifier.py           # 4-Class CNN classifier & clinical details
│   │   │   ├── segmenter.py            # Trained nnU-Net architecture with MC Dropout
│   │   │   └── lightweight_segmenter.py# FastBrain-UNet depthwise separable edge model
│   │   ├── services/
│   │   │   ├── preprocessing.py        # NIfTI & standard 2D image decoding & overlay generation
│   │   │   ├── uncertainty.py          # Monte Carlo Dropout predictive variance engine
│   │   │   ├── quality_assessment.py   # 8-Point automated rule-based QA auditor
│   │   │   ├── measurements.py         # Calibrated spatial & pixel geometry calculator
│   │   │   └── reconstruction.py       # Marching Cubes 3D surface mesh & STL/OBJ generator
│   │   └── schemas/
│   │       └── responses.py            # Pydantic data schemas
│   └── tests/
│       └── test_api.py                 # Automated unit tests for all backend routes
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── Navbar.tsx              # Navigation & live backend health indicator
│   │   │   ├── Footer.tsx              # Research disclaimers & tech specs
│   │   │   ├── Brain3DViewer.tsx       # Interactive Three.js 3D WebGL viewport
│   │   │   ├── SegmentationViewer.tsx  # MRI overlay viewer with opacity slider & zoom
│   │   │   ├── UncertaintyViewer.tsx   # Jet heatmap & boundary variance threshold
│   │   │   ├── ClassificationPanel.tsx # Recharts probability distribution bar chart
│   │   │   ├── QualityAssessmentPanel.tsx # 8-point automated validation card
│   │   │   ├── HumanReviewPanel.tsx    # Safety triage advisory & clinical recommendations
│   │   │   ├── ModelBenchmarkModal.tsx # nnU-Net vs FastBrain-UNet latency & memory
│   │   │   └── ExportModal.tsx         # JSON, CSV, PNG, STL & PDF export dialog
│   │   ├── pages/
│   │   │   ├── LandingPage.tsx         # Modern homepage with workflow cards & features
│   │   │   ├── UploadPage.tsx          # Drag-and-drop, 1-click demos, and NIfTI slider
│   │   │   └── DashboardPage.tsx       # Complete structured diagnostic dashboard
│   │   ├── services/api.ts             # Typed REST API client
│   │   ├── types/analysis.ts           # TypeScript interfaces
│   │   ├── App.tsx                     # Main view router
│   │   └── index.css                   # Tailwind v4 medical styling & dark theme
│   ├── package.json
│   └── vite.config.ts
├── models/
│   ├── best_nnunet_model.pt            # Trained nnU-Net weights (BRISC 2025, 29.5MB)
│   ├── best_model.pt                   # (you create it) calibrated classifier: training/train.py
│   ├── fast_brain_unet.pt              # (you create it) distilled edge model: training/train_segmentation.py
│   ├── segmentation_metrics.json       # (generated) real Dice / IoU / HD95 for the benchmark modal
│   ├── model_metadata.json             # Model training metadata
│   └── training_curves.png             # Historical training metrics plot
├── brain_tumour.ipynb                  # Colab training notebook (nnU-Net pipeline)
├── .env.example
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- Python 3.10+
- Node.js v18+ and npm

### 2. Backend Setup & Startup
```powershell
# In project root: c:\Srihaas\ALTHON
pip install fastapi uvicorn pydantic python-multipart torch torchvision numpy opencv-python pillow scipy scikit-image nibabel

# Start FastAPI backend (port 8000)
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```
- Interactive Swagger API Documentation: [http://127.0.0.1:8000/api/docs](http://127.0.0.1:8000/api/docs)
- Alternative ReDoc: [http://127.0.0.1:8000/api/redoc](http://127.0.0.1:8000/api/redoc)

### 3. Frontend Setup & Startup
```powershell
# In frontend directory: c:\Srihaas\ALTHON\frontend
cd frontend
npm install
npm run dev
```
- Open your browser to: [http://localhost:5173/](http://localhost:5173/)

### 4. Running Automated Tests
```powershell
python -m unittest backend/tests/test_api.py
```

---

## 🎯 Training & Evaluation (accuracy-focused pipeline)

```bash
pip install -r requirements.txt

# 1) Classifier - pretrained EfficientNet-B0, patient-aware validation split, label smoothing,
#    early stopping on VALIDATION, temperature-calibrated confidence. Test set touched once.
python training/train.py --backbone efficientnet_b0 --epochs 30 --class-weights
python training/evaluate.py            # accuracy + 95% CI, macro-F1, AUC, ECE, confusion matrix (with/without TTA)

# 2) Segmentation - measure your existing weights (writes models/segmentation_metrics.json)
python training/train_segmentation.py --model nnunet --eval-only
#    ...or retrain with augmentation + MC-dropout layers + validation-based selection
python training/train_segmentation.py --model nnunet --epochs 40
#    ...and train the lightweight edge model by distillation
python training/train_segmentation.py --model fast --teacher models/best_nnunet_model.pt
```

**Patient-level splitting.** If your dataset has a `manifest.csv` with a patient column, it is used automatically. Otherwise pass `--group-regex "<regex with one capture group>"`; without either, the split is slice-level and the scripts warn that validation numbers may be optimistic.

**Honest behaviour.** With no `models/best_model.pt` the API reports *"Classification unavailable"* (no tumour type is invented). Low calibrated confidence (< `ABSTAIN_THRESHOLD`) makes the classifier *abstain*. A classifier/segmentation disagreement is flagged for human review instead of one model silently overriding the other. See `CHANGES.md` for the full list of fixes.

---

## 🧪 1-Click Demonstration Scans

The application comes pre-loaded with clinical demonstration samples for instant evaluation without requiring manual file searching:
- **Glioma (High Grade)**: Infiltrative intra-axial lesion exhibiting surrounding vasogenic edema.
- **Meningioma**: Well-circumscribed extra-axial lesion with prominent dural margin.
- **Pituitary Tumour**: Sellar mass with suprasellar extension.
- **No Tumour**: Healthy brain parenchyma negative scan.

---

## ⚖️ Responsible Use & Research Disclaimer

> **Important Medical Notice:**  
> NeuroVision AI is a research demonstration prototype developed for academic evaluation, AI hackathons, and technical benchmarking. It does **not** provide definitive diagnoses, treatment plans, or clinical recommendations. All imaging outputs, uncertainty maps, and 3D surface reconstructions require independent evaluation by a qualified medical professional.
