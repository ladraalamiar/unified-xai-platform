# Unified XAI Platform - Multi-Modal Explainable AI System

A comprehensive explainable AI platform for medical image analysis (chest X-rays) and audio deepfake detection, featuring 7 state-of-the-art XAI methods across both modalities.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [XAI Methods](#xai-methods)
- [Models](#models)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)


---

## Features

### Multi-Modal Support
- **Medical Imaging**: Chest X-ray pathology detection (18 pathologies)
- **Audio Analysis**: Deepfake audio detection
- **Cross-Modal Comparison**: Compare XAI methods across modalities

### XAI Methods (7 Total)

#### Images (3 methods)
- **LIME** - Local Interpretable Model-agnostic Explanations
- **Grad-CAM** - Gradient-weighted Class Activation Mapping
- **SHAP** - SHapley Additive exPlanations

#### Audio (4 methods)
- **Gradient Visualization** - Attention heatmaps on spectrograms
- **LIME** - Time-frequency region importance
- **SHAP** - Audio feature importance analysis
- **Comprehensive Analysis** - Full statistical breakdown

### Pre-trained Models

#### Image Models (7)
- 6x DenseNet121 variants (different training datasets)
- 1x ResNet50 (alternative architecture)
- Trained on: CheXpert, MIMIC-CXR, NIH ChestX-ray14, PadChest

#### Audio Models (2)
- wav2vec2 (HuggingFace transformer)
- TensorFlow CNN (spectrogram-based)

### Interactive Features
- Side-by-side XAI method comparison
- Real-time visualization
- Exportable results
- Comprehensive interpretation guides
- Progress tracking and error handling

---

## Architecture

```
┌─────────────────────────────────────────────────┐
│           Unified XAI Platform                  │
├─────────────────────────────────────────────────┤
│                                                 │
│  ┌──────────────┐          ┌──────────────┐     │
│  │   Image      │          │    Audio     │     │
│  │ Pipeline     │          │  Pipeline    │     │
│  └──────┬───────┘          └──────┬───────┘     │
│         │                         │             │
│  ┌──────▼─────────────────────────▼─────────┐   │
│  │         XAI Methods Layer                │   │
│  │  LIME | Grad-CAM | SHAP | Gradient       │   │
│  └──────────────────────────────────────────┘   │
│         │                          │            │
│  ┌──────▼───────┐          ┌───────▼───────┐    │
│  │   7 Image    │          │   2 Audio     │    │
│  │   Models     │          │   Models      │    │
│  └──────────────┘          └───────────────┘    │
│                                                 │
│  ┌────────────────────────────────────────┐     │
│  │     Streamlit Web Interface            │     │
│  │  Dashboard | Comparison | Exports      │     │
│  └────────────────────────────────────────┘     │
└─────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, for faster inference)

### Step 1: Clone Repository

```bash
git clone https://github.com/ladraalamiar/unified-xai-platform.git
cd unified-xai-platform
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt**:
```

streamlit>=1.24
numpy
pillow
torch
torchvision
torchaudio
librosa
scikit-image
lime
shap
torchxrayvision
matplotlib
opencv-python-headless
pandas
protobuf>=3.20,<5
tensorflow-cpu>=2.13
transformers
altair
llvmlite
pyarrow
```

### Step 4: Download Pre-trained Models

```bash
# Image models (torchxrayvision - automatic download)
python -c "import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-all')"

# Audio models (place in models/audio/)
# Download from [your model source]
```

---

## Quick Start

### Launch Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

### Basic Workflow

1. **Select Data Type**: Choose Image or Audio from sidebar
2. **Upload File**: 
   - Images: .jpg, .png, .jpeg (chest X-rays)
   - Audio: .wav files (2-5 seconds optimal)
3. **Choose Model**: Select from available pre-trained models
4. **Classify**: Click "Classify" to get prediction
5. **Explain**: Select XAI method and generate explanation
6. **Compare** (optional): Use comparison page for multi-method analysis

---

## XAI Methods

### Image XAI

#### 1. LIME (Local Interpretable Model-agnostic Explanations)

**Type**: Model-agnostic  
**Time**: ~30-60 seconds  
**Output**: Superpixel segmentation with importance highlighting

**When to use**:
- Need model-agnostic explanations
- Want to understand regional importance
- Require interpretable segments

**Example**:
```python
from lime import lime_image
explainer = lime_image.LimeImageExplainer()
explanation = explainer.explain_instance(image, predict_fn, top_labels=1)
```

#### 2. Grad-CAM (Gradient-weighted Class Activation Mapping)

**Type**: Gradient-based  
**Time**: ~1-2 seconds  
**Output**: Pixel-level attention heatmap

**When to use**:
- Need fast explanations
- Working with CNN models
- Want direct model attention visualization

**Example**:
```python
gradients = model.backward(target_class)
cam = gradient_weighted_average(gradients, activations)
```

#### 3. SHAP (SHapley Additive exPlanations)

**Type**: Game theory-based  
**Time**: ~1-2 minutes  
**Output**: Pixel-level contribution values

**When to use**:
- Need theoretically rigorous explanations
- Want feature attribution
- Require consistent explanations

**Example**:
```python
from XAI_methods.shap_image import explain_image_with_shap
result = explain_image_with_shap(model, image, target_idx, num_samples=50)
```

### Audio XAI

#### 1. Gradient Visualization

**Time**: ~5-10 seconds  
**Output**: Attention heatmap over time-frequency representation

#### 2. LIME (Time-Frequency)

**Time**: ~30-60 seconds  
**Output**: Important time-frequency regions

#### 3. SHAP (Feature Importance)

**Time**: ~10-20 seconds  
**Output**: Importance of audio features (spectral, temporal, cepstral)

#### 4. Comprehensive Analysis

**Time**: ~5 seconds  
**Output**: Complete statistical breakdown (waveform, spectrogram, MFCCs, etc.)

---

## Models

### Image Models (Chest X-Ray)

| Model | Datasets | Pathologies | Best For |
|-------|----------|-------------|----------|
| DenseNet121 (All) | CheXpert, MIMIC, NIH, PadChest | 18 | General screening |
| DenseNet121 (MIMIC+CheXpert) | Hospital data | 14 | Clinical diagnosis |
| DenseNet121 (MIMIC) | MIMIC-CXR | 14 | ICU patients |
| DenseNet121 (NIH) | NIH ChestX-ray14 | 14 | Research standard |
| DenseNet121 (PadChest) | PadChest | 18 | European population |
| DenseNet121 (CheXpert) | Stanford CheXpert | 14 | US population |
| ResNet50 (All) | All datasets | 18 | Alternative architecture |

**Detected Pathologies**:
Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax, and more.

### Audio Models (Deepfake Detection)

| Model | Architecture | Training Data | Accuracy |
|-------|--------------|---------------|----------|
| wav2vec2 | Transformer | Large-scale deepfake corpus | High |
| TensorFlow CNN | CNN on spectrograms | Custom dataset | Fast |

---

## Project Structure

```
unified-xai-platform/
├── Dockerfile
├── README.md
├── XAI_methods
│   ├── __init__.py
│   ├── __pycache__
│   ├── audio_xai_advanced.py
│   ├── gradcam_image.py
│   ├── lime_image.py
│   ├── shap_audio.py
│   └── shap_image.py
├── app.py
├── comparison.py
├── docker-compose.yml
├── models
│   ├── audio
|   |     ├── .pb
|   |     |    ├── keras_metadat.pb
|   |     |    └── saved_model.pb
|   |     └── model_loader.py
│   └── images
|         └── model_loader.py
├── pages
│   ├── 1_Audio_Classification.py
│   ├── 2_Image_Classification.py
│   └── 3_XAI_Comparison.py
├── requirements.txt
├── test_audio_models.py
├── test_shap.py
└── utils
    ├── compatibility_checker.py
    └── data_loader.py
```

---

## Usage Examples

### Example 1: Image Classification with SHAP

```python
import streamlit as st
from images.model_loader import load_image_model, predict_image
from XAI_methods.shap_image import explain_image_with_shap, visualize_shap_image

# Load model
model = load_image_model('densenet121-res224-all')

# Predict
result = predict_image(model, image)
top_pathology = result['top_pathology']
top_idx = list(model.pathologies).index(top_pathology)

# Generate SHAP explanation
shap_result = explain_image_with_shap(model, image, top_idx, num_samples=50)

# Visualize
fig = visualize_shap_image(shap_result, top_pathology, original_image=image)
st.pyplot(fig)
```

### Example 2: Audio Analysis with Multiple Methods

```python
from audio.model_loader import load_audio_model, predict_audio
from XAI_methods.audio_xai_advanced import (
    gradient_audio_visualization,
    audio_shap_explanation
)

# Load and predict
model = load_audio_model('huggingface')
prediction = predict_audio(model, audio_file)

# Generate multiple explanations
gradient_fig = gradient_audio_visualization(model, audio_file, prediction)
shap_fig = audio_shap_explanation(model, audio_file, prediction)

# Compare results
st.pyplot(gradient_fig)
st.pyplot(shap_fig)
```

### Example 3: Side-by-Side Comparison

```python
# On the comparison page, select multiple methods
methods = ["LIME", "Grad-CAM", "SHAP"]

for method in methods:
    with st.spinner(f"Generating {method}..."):
        explanation = generate_explanation(method, model, image)
        display_explanation(explanation, method)
```

---

## Troubleshooting

### Common Issues

**Issue**: SHAP takes too long  
**Solution**: Reduce `num_samples` parameter (e.g., from 50 to 20)

**Issue**: "Module not found" errors  
**Solution**: Ensure all dependencies installed: `pip install -r requirements.txt`

**Issue**: LIME fails on images  
**Solution**: Check image format (RGB/Grayscale) and size

**Issue**: Audio model not loading  
**Solution**: Verify model files in `models/audio/` directory

See `docs/DEBUGGING_GUIDE.md` for comprehensive troubleshooting.

---
