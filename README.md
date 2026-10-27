# Edge-Powered Multimodal Context-Aware Emergency Triage & Automated Dispatch System

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Vision Transformer](https://img.shields.io/badge/Vision-Google%20ViT--Base-purple.svg)](https://huggingface.co/google/vit-base-patch16-224)
[![NLP](https://img.shields.io/badge/NLP-DistilBERT-yellow.svg)](https://huggingface.co/distilbert-base-uncased)
[![Ensemble](https://img.shields.io/badge/Classifier-Balanced%20Random%20Forest%20(1568--D)-emerald.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

Traditional emergency dispatch centers (PSAP / 911 / 112) suffer from acute perceptual bottlenecks, relying predominantly on fragmented voice transcripts and subjective verbal reporting during mass casualty incidents and catastrophic disasters. Such unimodal reporting often introduces dangerous triage delays, misprioritizes walking wounded over critically injured victims, or fails completely under network congestion.

This project introduces an **Edge-Powered Multimodal Context-Aware Emergency Triage System**. By performing **Early Feature Concatenation** across:
1. **Surveillance/Drone Computer Vision** ($768$-D ViT [CLS] representations)
2. **Acoustic/Transcript NLP Urgency** ($768$-D DistilBERT contextual embeddings)
3. **IoT & Spatio-Temporal Sensor Telemetry** ($32$-D environmental telemetry including thermal gradients, smoke PPM, and GIS proximity)

The system synthesizes a **$1,568$-dimensional latent representation vector** evaluated by a **Balanced Random Forest Classifier** to assign standardized emergency priority ranks (**P1 Critical**, **P2 High**, **P3 Moderate**, **P4 Low**) in **$22.4\text{ ms}$** edge latency, while reducing cloud uplink communication bandwidth by **$98.2\%$**.

---

## 🏛️ System Architecture

```
                                 [ MULTIMODAL INGESTION LAYER ]
                                                │
       ┌────────────────────────────────────────┼────────────────────────────────────────┐
       │                                        │                                        │
  [Visual Feed]                          [Voice / Text]                          [IoT Telemetry]
  CCTV / Drone / Citizen Photos          911 Call / SMS Transcript               Temperature, Smoke PPM, GIS
       │                                        │                                        │
       ▼                                        ▼                                        ▼
 Google ViT-Base                        DistilBERT NLP                           Feature Normalizer
 224x224 RGB -> 16x16 Patches           768-D Contextual Vector                  32-D Telemetric Vector
 (768-D [CLS] Embedding)                (768-D Semantic Vector)                  (Thermal / Gas / Vibration)
       │                                        │                                        │
       └────────────────────────────────────────┼────────────────────────────────────────┘
                                                │
                                                ▼
                             [ EARLY MULTIMODAL FUSION LAYER ]
                                      1,568-D Vector
                                                │
                                                ▼
                             [ ENSEMBLE CLASSIFICATION LAYER ]
                           Balanced Random Forest (100 Trees)
                           With Clinical Guardrail Overrides
                                                │
                        ┌───────────────────────┴───────────────────────┐
                        ▼                                               ▼
         [ 4-Class Triage Priority ]                     [ Automated Situational Dispatch ]
         P1: Critical (Immediate Red)                    - Units: Air Medevac / Hazmat / Fire
         P2: High (Urgent Yellow)                        - Facility: Level 1 Trauma ICU Routing
         P3: Moderate (Ambulatory Green)                 - ETA: First-responder arrival bounds
         P4: Low (Non-Emergency Blue)                    - Live Edge Console Audit Trail
```

---

### Summary Ablation Table

| Config ID | System Configuration | Accuracy | Macro F1 | Specificity | Edge Latency |
| :---: | :--- | :---: | :---: | :---: | :---: |
| **C1** | **Proposed Full Multimodal (ViT + DistilBERT + IoT + GIS)** | **95.32%** | **94.58%** | **96.82%** | **22.4 ms** |
| **C2** | w/o Social-Temporal Synthesis (ViT + DistilBERT + IoT) | 91.21% | 90.11% | 93.15% | 21.8 ms |
| **C3** | w/o GPS Proximity (ViT + DistilBERT + Sensors) | 90.34% | 88.95% | 92.40% | 21.0 ms |
| **C4** | w/o Textual NLP (ViT Vision + Sensors + GIS) | 88.36% | 87.09% | 90.80% | 18.2 ms |
| **C5** | w/o Video Extraction (ViT Image + DistilBERT + GIS) | 87.45% | 86.18% | 89.92% | 19.5 ms |
| **C6** | Unimodal Baseline: Image Only (Google ViT-Base) | 78.62% | 76.89% | 82.10% | 15.6 ms |
| **C7** | Traditional Baseline: Non-Deep Learning RF (Tabular) | 71.34% | 69.22% | 74.50% | 4.2 ms |

---

## ⚡ Interactive Web Command Center & Live Edge Terminal

The project features a **Mission Control Web Dashboard** located in `triage_app/`:
- **Panel 01 — Ingestion**: Select presets (Fire, Flash Flood, Expressway Pileup, Collapse), upload custom images, type ultra-short voice transcripts, and toggle baseline ambient telemetry ("Simulate Zero Rich IoT Data").
- **Panel 02 — Latent Space & Terminal**: Inspect the $1,568$-D feature breakdown, view real-time modality contribution bars, and watch the **Live Edge Inference Terminal** stream stdout logs sequentially.
- **Panel 03 — Prioritization & Dispatch**: Instant priority hero badge, softmax probability distribution, tactical unit recommendations, and acoustic siren alert.

---

## 🚀 Quickstart & Setup Guide

### 1. Clone Repository & Setup Environment
```bash
git clone https://github.com/vinuraghav/MMAI-Project-Team10.git
cd MMAI-Project-Team10

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Edge Triage Command Center
```bash
python triage_app/server.py 8080
```
Open **`http://localhost:8080`** in any modern web browser.

## Structure

```text
MMAI-Project-Team10/
├── SOS-Team-10.pptx                # Final Academic Presentation Slide Deck
├── generate_presentation_charts.py # Script generating 5 presentation figures
├── presentation_results/           # 300 DPI high-res charts and summary tables
│   ├── figure1_confusion_matrix.png
│   ├── figure2_ablation_study.png
│   ├── figure3_multiclass_roc.png
│   ├── figure4_edge_vs_cloud_benchmark.png
│   ├── figure5_radar_comparison.png
│   └── table_performance_metrics.md
├── requirements.txt                # Core Python package dependencies
├── triage_pipeline.py              # Foundational training & evaluation pipeline
├── triage_simulation.py            # Monte-Carlo incident simulation pipeline
└── triage_app/                     # Interactive Command Center Application
    ├── server.py                   # Lightweight HTTP Edge Server (Port 8080)
    ├── triage_engine.py            # Multimodal Inference Engine & Clinical Guardrails
    ├── train_export_model.py       # RF Model Training & Checkpoint Export
    ├── presets.py                  # Standardized Disaster Incident Presets
    ├── models/
    │   └── trained_triage_rf.joblib# Pre-trained 1,568-D Random Forest Classifier
    └── web/
        ├── index.html              # Modern 3-Panel Mission Control UI
        ├── style.css               # Neon Glassmorphic Theme & Terminal Styling
        └── app.js                  # Asynchronous Client Controller & Live Terminal Stream
```
