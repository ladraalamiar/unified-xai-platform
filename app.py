import streamlit as st

st.set_page_config(
    page_title="Unified XAI Platform",
    #page_icon="🔍",
    layout="wide"
)

# Header avec logo
st.title(" Unified Explainable AI Platform")
st.caption("Multi-Modal Classification & Explainability System")

# Bannière info
st.info("""
**Mission**: Rendre l'intelligence artificielle transparente et compréhensible à travers des méthodes d'explainability avancées pour les images médicales et l'audio.
""")

# Métriques principales
st.markdown("## Platform Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Models",
        value="9",
        delta="7 Image + 2 Audio",
        help="Pre-trained state-of-the-art models"
    )

with col2:
    st.metric(
        label="XAI Methods",
        value="7",
        delta="3 Image + 4 Audio",
        help="Explainability techniques"
    )

with col3:
    st.metric(
        label="Input Types",
        value="2",
        delta="Image + Audio",
        help="Supported data modalities"
    )

with col4:
    st.metric(
        label="Pathologies",
        value="18",
        delta="Max detectable",
        help="For chest X-ray analysis"
    )

st.divider()

# Section principale avec colonnes
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("### Image Classification")
    st.markdown("""
    **Task**: Chest X-ray Pathology Detection
    
    **Models Available** (7):
    - DenseNet121 (All Datasets) - *Recommended*
    - DenseNet121 (MIMIC + CheXpert)
    - DenseNet121 (MIMIC only)
    - DenseNet121 (NIH ChestX-ray14)
    - DenseNet121 (PadChest)
    - DenseNet121 (CheXpert)
    - ResNet50 (All Datasets)
    
    **XAI Methods** (3):
    - 🟢 **LIME** - Superpixel segmentation (~30-60s)
    - 🔴 **Grad-CAM** - Attention heatmap (~1-2s)
    - 🔵 **SHAP** - Pixel importance (~1-2min)
    
    **Detects**:
    Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Infiltration, Mass, Nodule, Pleural Thickening, Pneumonia, Pneumothorax, and more.
    """)
    
    if st.button("Go to Image Classification", use_container_width=True, type="primary"):
        st.switch_page("pages/1_Image_Classification.py")

with col_right:
    st.markdown("### Audio Classification")
    st.markdown("""
    **Task**: Deepfake Audio Detection
    
    **Models Available** (2):
    - wav2vec2 (HuggingFace) - *Transformer-based*
    - TensorFlow CNN - *Fast inference*
    
    **XAI Methods** (4):
    - 🎨 **Gradient Visualization** - Attention heatmap (~5-10s)
    - 🟢 **LIME** - Time-frequency regions (~30-60s)
    - 📊 **SHAP** - Feature importance (~10-20s)
    - 📈 **Comprehensive Analysis** - All features (~5s)
    
    **Features Analyzed**:
    Spectral (Centroid, Rolloff, Flatness), Temporal (ZCR, Energy), Cepstral (MFCCs), Harmonic (Chroma), Rhythmic (Tempo)
    """)
    
    if st.button("🎵 Go to Audio Classification", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Audio_Classification.py")

st.divider()

# Section comparaison
st.markdown("### XAI Methods Comparison")

col_comp1, col_comp2 = st.columns([2, 1])

with col_comp1:
    st.markdown("""
    Compare multiple explainability methods **side-by-side** on the same input.
    
    **Features**:
    - Visual comparison (2-4 methods simultaneously)
    - Works for both images and audio
    - Detailed comparison tables
    - Interpretation guides included
    
    **Use Cases**:
    - Research: Cross-validate explanations
    - Clinical: Understand model decisions
    - Education: Learn XAI differences
    - Production: Choose best method
    """)

with col_comp2:
    
    if st.button("Go to XAI Comparison", use_container_width=True, type="secondary"):
        st.switch_page("pages/3_XAI_Comparison.py")

st.divider()

# Guide de démarrage rapide
with st.expander("Quick Start Guide", expanded=False):
    st.markdown("""
    ### Getting Started in 5 Steps
    
    1. **Select Data Type** 
       - Click "Image Classification" for chest X-rays
       - Click "Audio Classification" for deepfake detection
    
    2. **Upload Your File**
       - Images: .jpg, .png, .jpeg (chest X-rays)
       - Audio: .wav files (2-5 seconds optimal)
    
    3. **Choose a Model**
       - For images: Try DenseNet121 (All Datasets) first
       - For audio: Try wav2vec2 (HuggingFace) first
    
    4. **Get Prediction**
       - Click "Classify" button
       - View results with confidence scores
    
    5. **Generate Explanation**
       - Select XAI method (LIME, Grad-CAM, SHAP, etc.)
       - Click "Generate Explanation"
       - Understand why the model made its decision
    
    ### Pro Tips
    
    - **Fast preview**: Use Grad-CAM (images) or Gradient (audio)
    - **Detailed analysis**: Use LIME or SHAP
    - **Cross-validation**: Use Comparison page for all methods
    - **Research**: Try all methods and compare results
    """)

# Caractéristiques détaillées
st.markdown("## Key Features")

feat_col1, feat_col2, feat_col3 = st.columns(3)

with feat_col1:
    st.markdown("""
    ### Multi-Modal
    
    - **Images**: Medical X-rays
    - **Audio**: Speech/voice
    - **Unified**: Same interface
    - **Extensible**: Easy to add more
    """)

with feat_col2:
    st.markdown("""
    ### Advanced XAI
    
    - **7 Methods**: LIME, SHAP, Grad-CAM, etc.
    - **Validated**: Academic standards
    - **Interpretable**: Clear visualizations
    - **Comparative**: Side-by-side analysis
    """)

with feat_col3:
    st.markdown("""
    ### User-Friendly
    
    - **No Code**: Web interface
    - **Real-time**: Instant results
    - **Guides**: Built-in explanations
    - **Robust**: Error handling
    """)

st.divider()

# Méthodes XAI détaillées
st.markdown("## XAI Methods Overview")

xai_tab1, xai_tab2, xai_tab3, xai_tab4 = st.tabs([
    "LIME", "Grad-CAM / Gradient", "SHAP", "Comprehensive (Audio)"
])

with xai_tab1:
    st.markdown("""
    ### 🟢 LIME - Local Interpretable Model-agnostic Explanations
    
    **Type**: Model-agnostic  
    **Speed**: Medium (~30-60 seconds)
    
    **How it works**:
    - Perturbs the input (adds noise, removes segments)
    - Observes how predictions change
    - Fits a simple linear model locally
    - Identifies important regions/features
    
    **Pros**:
    - ✅ Works with any model (black box)
    - ✅ Intuitive segmentation (images) or regions (audio)
    - ✅ Widely validated in literature
    
    **Cons**:
    - ⚠️ Slower than gradient-based methods
    - ⚠️ Approximation (not exact)
    - ⚠️ Sensitive to hyperparameters
    
    **Best for**: Understanding which regions/segments are important
    """)

with xai_tab2:
    st.markdown("""
    ### 🔴 Grad-CAM / 🎨 Gradient Visualization
    
    **Type**: Gradient-based  
    **Speed**: Fast (~1-10 seconds)
    
    **How it works**:
    - Uses gradients from the neural network
    - Computes importance weights for activations
    - Creates attention heatmap
    - Shows where the model "looks" or "listens"
    
    **Pros**:
    - ✅ Very fast (1-2 seconds for images, 5-10s for audio)
    - ✅ Pixel/frequency-level precision
    - ✅ Direct view of model attention
    
    **Cons**:
    - ⚠️ Requires gradient access (CNN/transformer models)
    - ⚠️ Not model-agnostic
    
    **Best for**: Quick visual insight into model focus
    """)

with xai_tab3:
    st.markdown("""
    ### 🔵 SHAP - SHapley Additive exPlanations
    
    **Type**: Game theory-based  
    **Speed**: Slow (~1-2 minutes for images, ~10-20s for audio)
    
    **How it works**:
    - Based on Shapley values from cooperative game theory
    - Fairly distributes prediction across features
    - Guarantees consistency and local accuracy
    - Computes contribution of each pixel/feature
    
    **Pros**:
    - ✅ Theoretically rigorous (only solution with Shapley properties)
    - ✅ Consistent explanations
    - ✅ Feature attribution at fine granularity
    
    **Cons**:
    - ⚠️ Computationally expensive (slowest method)
    - ⚠️ High memory usage
    - ⚠️ May take 1-2 minutes for images
    
    **Best for**: When you need mathematically grounded explanations
    """)

with xai_tab4:
    st.markdown("""
    ### 📈 Comprehensive Analysis (Audio Only)
    
    **Type**: Statistical analysis  
    **Speed**: Fast (~5 seconds)
    
    **What it shows**:
    - **Waveform**: Time-domain signal
    - **Mel Spectrogram**: Frequency content
    - **MFCCs**: Speech characteristics (13 coefficients)
    - **Chromagram**: Pitch class distribution
    - **Spectral Features**: Centroid, Rolloff, ZCR
    - **Summary Statistics**: All features quantified
    
    **Pros**:
    - ✅ Complete technical breakdown
    - ✅ Fast generation
    - ✅ 7 visualizations in one
    
    **Best for**: Technical deep-dive and feature engineering
    """)

st.divider()

# Datasets et modèles
with st.expander("📚 Datasets & Training Details", expanded=False):
    st.markdown("""
    ### Image Models Training
    
    | Dataset | Images | Pathologies | Quality |
    |---------|--------|-------------|---------|
    | **CheXpert** | 224,316 | 14 | ⭐⭐⭐⭐⭐ |
    | **MIMIC-CXR** | 377,110 | 14 | ⭐⭐⭐⭐⭐ |
    | **NIH ChestX-ray14** | 112,120 | 14 | ⭐⭐⭐⭐ |
    | **PadChest** | 160,000 | 18 | ⭐⭐⭐⭐ |
    
    ### Audio Models Training
    
    - **wav2vec2**: Large-scale transformer model, pre-trained on deepfake corpus
    - **TensorFlow CNN**: Custom CNN on mel spectrograms, optimized for speed
    
    ### Model Architectures
    
    **Images**:
    - **DenseNet121**: 121 layers, dense connections, 7M parameters
    - **ResNet50**: 50 layers, residual connections, 25M parameters
    
    **Audio**:
    - **wav2vec2**: Transformer encoder, self-supervised learning
    - **CNN**: 5 conv layers + 2 FC layers on spectrograms
    """)

# Footer
st.divider()

st.markdown("""
---

**Built with ❤️ for transparent and explainable AI**

*Version 1.0 • January 2026*
""")

# Sidebar avec navigation rapide
with st.sidebar:
    
    
    st.markdown("### ℹ️ About")
    st.info("""
    **Unified XAI Platform** is a comprehensive system for explaining AI decisions in medical imaging and audio analysis.
    
    **Features:**
    - 9 pre-trained models
    - 7 XAI methods
    - Real-time explanations
    - Multi-modal support
    """)
    
    st.divider()
    
    st.markdown("### 📊 Platform Stats")
    st.metric("Total XAI Methods", "7")
    st.metric("Models Available", "9")
    st.metric("Data Types", "2")
    
    



# import streamlit as st

# st.set_page_config(
#     page_title="Unified XAI Platform",
#     page_icon="🔍",
#     layout="wide"
# )

# st.title("🔍 Unified Explainable AI Platform")

# st.markdown("""
# ## Multi-Modal Classification & Explainability

# ### 🎵 Audio Classification
# - **Models**: 2 pre-trained models (VGG16, MobileNet or ResNet)
# - **Task**: Deepfake detection
# - **XAI**: LIME, Grad-CAM

# ### 🫁 Image Classification  
# - **Model**: DenseNet121 (torchxrayvision)
# - **Task**: Lung abnormality detection
# - **XAI**: LIME, Grad-CAM

# ---

# 👈 **Select a page from the sidebar:**
# - 📊 **Audio Classification** - Upload .wav files
# - 🖼️ **Image Classification** - Upload chest X-rays
# - 🔄 **XAI Comparison** - Compare methods side-by-side

# ---

# ### ℹ️ Quick Start
# 1. Navigate to Audio or Image page
# 2. Upload your file
# 3. Select a model
# 4. Choose XAI method (auto-filtered based on input type)
# 5. View predictions & explanations

# **Note**: Only compatible XAI methods will be shown for each data type.
# """)

# # Stats
# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric("Models", "3")
# with col2:
#     st.metric("XAI Methods", "2")
# with col3:
#     st.metric("Input Types", "2")

# st.info("💡 **Tip**: Start with the Audio or Image classification page to test the system!")
