import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import os

# Ajouter chemin modèles
sys.path.insert(0, os.path.join(os.getcwd(), 'models'))

st.title("🫁 Chest X-Ray Classification")

# Cache le modèle
@st.cache_resource
def load_model(model_key):
    """Charge et met en cache le modèle"""
    try:
        from models.images.model_loader import load_image_model
        model = load_image_model(model_key)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ============================================
# SECTION UPLOAD / TEST FILES (CORRIGÉE)
# ============================================

st.subheader("📸 Select Input Image")

tab_upload, tab_test = st.tabs(["📤 Upload Image", "🧪 Test Images"])

image = None
image_source = None  # Pour tracker la source

# TAB 1: Upload
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload chest X-ray image",
        type=['jpg', 'png', 'jpeg'],
        key='img_upload',
        help="Supported formats: JPG, PNG, JPEG"
    )
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        image_source = "uploaded"
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Uploaded X-ray", use_column_width=True)
        with col2:
            st.write("**Image Information**")
            st.metric("Filename", uploaded_file.name)
            st.metric("Size", f"{image.size[0]} x {image.size[1]} pixels")
            st.metric("Mode", image.mode)

# TAB 2: Test Images
with tab_test:
    st.write("**Available test images:**")
    
    
    test_images = {
        'Sample 1': 'models/images/test_lung.jpg',
        'Sample 2': 'models/images/radio_lung.jpg',
        'Sample 3': 'models/images/test_radio.jpeg'
    }
    
    # Filtrer ceux qui existent
    available = {n: p for n, p in test_images.items() if os.path.exists(p)}
    
    if not available:
        st.warning("""
        ⚠️ **No test images found**
        
        Add test images to one of these locations:
        - `models/images/`
        - `test_data/images/`
        
        Then refresh the page.
        """)
    else:
        selected = st.selectbox(
            "Choose a test image:",
            ["-- Select --"] + list(available.keys()),
            key="test_image_selector"
        )
        
        if selected != "-- Select --":
            test_image_path = available[selected]
            
            try:
                image = Image.open(test_image_path).convert('RGB')
                image_source = "test"
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image(image, caption=selected, use_column_width=True)
                
                with col2:
                    st.success(f"✅ **{selected}** loaded successfully")
                    st.info(f"""
                    **Path**: `{test_image_path}`
                    
                    **Dimensions**: {image.size[0]} x {image.size[1]} px
                    
                    **Mode**: {image.mode}
                    
                    *This is a pre-loaded test image for demonstration.*
                    """)
                    
            except Exception as e:
                st.error(f"❌ Error loading test image: {str(e)}")
                image = None

# Indicateur de source
if image:
    if image_source == "uploaded":
        st.caption("🔹 Using uploaded image")
    elif image_source == "test":
        st.caption("🔹 Using test image")

# Arrêter si pas d'image
if image is None:
    st.info("👆 **Please upload an image or select a test image to begin**")
    
    # Guide d'utilisation
    with st.expander("ℹ️ How to use this page", expanded=True):
        st.markdown("""
        ### 🎯 Step-by-Step Guide
        
        1. **Upload or Select Image**:
           - Tab "📤 Upload": Upload your own chest X-ray
           - Tab "🧪 Test Images": Select from pre-loaded samples
        
        2. **Choose Model**: Select from 7 pre-trained DenseNet/ResNet models
        
        3. **Classify**: Click "Classify X-Ray" to get predictions
        
        4. **View Results**: See top 5 predicted pathologies with confidence scores
        
        5. **Explore XAI**: Use LIME, Grad-CAM, or SHAP to understand the decision
        
        ### 🤖 Available Models
        
        All models are pre-trained on chest X-ray datasets:
        - **Model 1**: All Datasets (most comprehensive)
        - **Model 2-6**: Specialized on specific datasets
        - **Model 7**: ResNet50 alternative architecture
        
        ### 💡 Tips
        
        - **Not sure which model?** → Start with Model 1 (All Datasets)
        - **For research?** → Use Model 4 (NIH ChestX-ray14)
        - **Compare models** to see consistency
        - **Use XAI** to understand predictions
        """)
    
    st.stop()

# ============================================
# RESTE DU CODE (FONCTIONNE AVEC LES DEUX SOURCES)
# ============================================

st.divider()

# Sélection modèle avec conseils
st.subheader("🤖 Model Selection")

# Info sur les modèles
with st.expander("ℹ️ Model Guide - Click to see recommendations", expanded=False):
    st.markdown("""
    ### 🎯 Quick Recommendations
    
    **🏆 Best Overall Performance** → Model 1 (All Datasets)
    - Most comprehensive training
    - Best for general use
    - 18 pathologies detected
    
    **⚡ Fast & Reliable** → Model 2 (MIMIC + CheXpert)
    - Hospital-grade datasets
    - Clinically validated
    - Good balance speed/accuracy
    
    **🎓 Research Standard** → Model 4 (NIH ChestX-ray14)
    - Most cited in literature
    - 14 common pathologies
    - Standardized benchmark
    
    ---
    
    ### 📊 All Models Comparison
    
    | Model | Datasets | Pathologies | Best For |
    |-------|----------|-------------|----------|
    | **1. All Datasets** | CheXpert, MIMIC, NIH, PadChest | 18 | General screening |
    | **2. MIMIC + CheXpert** | Hospital data | 14 | Clinical diagnosis |
    | **3. MIMIC only** | MIMIC-CXR | 14 | ICU patients |
    | **4. NIH ChestX-ray14** | NIH public | 14 | Research/benchmarking |
    | **5. PadChest** | Spanish dataset | 18 | European population |
    | **6. CheXpert** | Stanford | 14 | US population |
    | **7. ResNet50** | All datasets | 18 | Alternative architecture |
    
    ### 💡 Tips
    - **Not sure?** → Use Model 1 (All Datasets)
    - **Research paper?** → Use Model 4 (NIH)
    - **Speed matters?** → Use Model 6 (CheXpert)
    - **Want diversity?** → Compare Model 1, 2, and 4
    """)

# Sélecteur de modèle
model_options = {
    "🏆 Model 1: DenseNet121 (All Datasets) - RECOMMENDED": 'densenet121-res224-all',
    "⚡ Model 2: DenseNet121 (MIMIC + CheXpert)": 'densenet121-res224-mimic_ch',
    "🏥 Model 3: DenseNet121 (MIMIC only)": 'densenet121-res224-mimic_nb',
    "🎓 Model 4: DenseNet121 (NIH ChestX-ray14)": 'densenet121-res224-nih',
    "🌍 Model 5: DenseNet121 (PadChest)": 'densenet121-res224-pc',
    "🇺🇸 Model 6: DenseNet121 (CheXpert)": 'densenet121-res224-chex',
    "🔄 Model 7: ResNet50 (All Datasets)": 'resnet50-res512-all'
}

selected_display = st.selectbox(
    "Select Classification Model",
    list(model_options.keys()),
    help="See guide above for recommendations on which model to use"
)

model_key = model_options[selected_display]

# Afficher info du modèle sélectionné
model_descriptions = {
    'densenet121-res224-all': "🏆 Most comprehensive model, trained on 4 major datasets. Best overall performance.",
    'densenet121-res224-mimic_ch': "⚡ Hospital-grade model combining MIMIC-CXR and CheXpert. Clinically validated.",
    'densenet121-res224-mimic_nb': "🏥 Specialized on ICU patients from MIMIC-CXR dataset.",
    'densenet121-res224-nih': "🎓 Research standard, trained on NIH ChestX-ray14 (112,120 images).",
    'densenet121-res224-pc': "🌍 Trained on PadChest dataset (160,000 Spanish images).",
    'densenet121-res224-chex': "🇺🇸 Stanford's CheXpert dataset (224,316 images).",
    'resnet50-res512-all': "🔄 Alternative ResNet50 architecture for comparison."
}

st.info(model_descriptions[model_key])

# Bouton classification
if st.button("🔍 Classify X-Ray", type="primary", use_container_width=True):
    with st.spinner(f"Loading model..."):
        model = load_model(model_key)
    
    if model is None or model.model is None:
        st.error("❌ Failed to load model")
    else:
        st.success(f"✅ Model loaded: {model.name}")
        
        with st.spinner("Analyzing X-ray..."):
            try:
                # Prédiction avec l'image PIL
                from models.images.model_loader import predict_image
                result = predict_image(model, image)
                
                # Afficher résultats
                st.success("✅ Classification complete!")
                
                st.subheader("📊 Top 5 Pathology Predictions")
                
                # Top 5 avec barres
                for i, pred in enumerate(result['top_predictions'], 1):
                    pathology = pred['pathology']
                    probability = pred['probability']
                    
                    # Couleur selon probabilité
                    if probability > 0.7:
                        color = "🔴"
                        bar_color = "#e74c3c"
                    elif probability > 0.4:
                        color = "🟠"
                        bar_color = "#f39c12"
                    else:
                        color = "🟢"
                        bar_color = "#2ecc71"
                    
                    col1, col2, col3 = st.columns([3, 2, 1])
                    
                    with col1:
                        st.write(f"**{i}. {color} {pathology}**")
                    
                    with col2:
                        st.progress(float(probability))
                    
                    with col3:
                        st.metric("", f"{probability:.1%}")
                
                # Détails
                with st.expander("📋 View All Pathology Scores"):
                    all_probs = result['all_probabilities']
                    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                    
                    for pathology, prob in sorted_probs:
                        st.write(f"**{pathology}**: {prob:.2%}")
                
                # Info source
                st.caption(f"🔹 Image source: {image_source}")
                
                # Sauvegarder dans session
                st.session_state['image_result'] = result
                st.session_state['image_model'] = model
                st.session_state['image_file'] = image
                st.session_state['image_source'] = image_source
                
            except Exception as e:
                st.error(f"❌ Classification error: {str(e)}")
                with st.expander("Show error details"):
                    import traceback
                    st.code(traceback.format_exc())

# XAI Section
if 'image_result' in st.session_state:
    st.divider()
    st.subheader("🔍 Explainability Analysis (XAI)")
    
    result = st.session_state['image_result']
    top_path = result['top_pathology']
    top_prob = result['top_probability']
    
    st.info(f"📊 Understanding why the model predicted **{top_path}** ({top_prob:.1%})")
    
    # Guide XAI
    with st.expander("ℹ️ XAI Methods for Images", expanded=False):
        st.markdown("""
        ### Available Methods
        
        **🟢 LIME** (~30-60s)
        - Superpixel segmentation
        - Shows important regions
        - Model-agnostic
        
        **🔴 Grad-CAM** (~1-2s)
        - Attention heatmap
        - Shows where model looks
        - Fast and intuitive
        
        **🔵 SHAP** (~1-2min)
        - Pixel-level importance
        - Game theory-based
        - Rigorous attribution
        
        ### Comparison
        - **LIME**: Best for regional understanding
        - **Grad-CAM**: Fastest, good for quick insight
        - **SHAP**: Most rigorous, slower
        """)
    
    # Sélection méthode XAI
    xai_col1, xai_col2 = st.columns([3, 2])
    
    with xai_col1:
        xai_method = st.selectbox(
            "Select XAI Method",
            [
                "🟢 LIME (Superpixel Segmentation)",
                "🔴 Grad-CAM (Attention Heatmap)",
                "🔵 SHAP (Pixel Importance)"
            ],
            help="LIME: Shows important regions | Grad-CAM: Shows model attention | SHAP: Pixel-level importance"
        )
    
    with xai_col2:
        st.write("")
        st.write("")
        generate_xai = st.button(
            "✨ Generate Explanation",
            type="secondary",
            use_container_width=True
        )
    
    if generate_xai:
        with st.spinner(f"Generating {xai_method.split()[1]}..."):
            try:
                model = st.session_state['image_model']
                image = st.session_state['image_file']
                
                if xai_method.startswith("🟢 LIME"):
                    st.write("**🔧 LIME: Local Interpretable Model-agnostic Explanations**")
                    
                    from lime import lime_image
                    from skimage.segmentation import mark_boundaries
                    import torchxrayvision as xrv
                    
                    # Fonction de prédiction pour LIME
                    def predict_fn(images):
                        batch = []
                        for img in images:
                            if len(img.shape) == 3:
                                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                            img = xrv.datasets.normalize(img, 255)
                            img = cv2.resize(img, (224, 224))
                            batch.append(img)
                        
                        batch_tensor = torch.from_numpy(np.array(batch)).unsqueeze(1).float()
                        
                        with torch.no_grad():
                            outputs = model.model(batch_tensor)
                            return torch.sigmoid(outputs).cpu().numpy()
                    
                    # LIME explainer
                    explainer = lime_image.LimeImageExplainer()
                    
                    img_array = np.array(image)
                    
                    with st.spinner("LIME is analyzing... (may take 30-60s)"):
                        explanation = explainer.explain_instance(
                            img_array,
                            predict_fn,
                            top_labels=3,
                            hide_color=0,
                            num_samples=100
                        )
                    
                    # Trouver l'index du top pathology
                    top_idx = list(model.pathologies).index(top_path)
                    
                    # Visualiser
                    temp, mask = explanation.get_image_and_mask(
                        top_idx,
                        positive_only=True,
                        num_features=5,
                        hide_rest=False
                    )
                    
                    # Créer figure
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Original
                    axes[0].imshow(img_array)
                    axes[0].set_title("Original X-Ray", fontsize=14, fontweight='bold')
                    axes[0].axis('off')
                    
                    # LIME avec boundaries
                    axes[1].imshow(mark_boundaries(temp / 255.0, mask))
                    axes[1].set_title(f"LIME - {top_path}", fontsize=14, fontweight='bold')
                    axes[1].axis('off')
                    
                    # Mask seul
                    axes[2].imshow(mask, cmap='Greens', alpha=0.8)
                    axes[2].set_title("Important Regions", fontsize=14, fontweight='bold')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.success("✅ LIME explanation generated!")
                    
                    st.caption("""
                    **How to interpret**: Green boundaries highlight regions that most influenced the prediction.
                    Darker green = more important for this pathology.
                    """)
                
                elif xai_method.startswith("🔴 Grad-CAM"):
                    st.write("**🔥 Grad-CAM: Gradient-weighted Class Activation Mapping**")
                    
                    import torchxrayvision as xrv
                    
                    img_array = np.array(image)
                    if len(img_array.shape) == 3:
                        img_array_gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                    else:
                        img_array_gray = img_array
                    
                    img_norm = xrv.datasets.normalize(img_array_gray, 255)
                    img_resized = cv2.resize(img_norm, (224, 224))
                    img_tensor = torch.from_numpy(img_resized).unsqueeze(0).unsqueeze(0).float()
                    
                    # Forward avec gradients
                    img_tensor.requires_grad = True
                    output = model.model(img_tensor)
                    
                    # Backward
                    model.model.zero_grad()
                    top_idx = list(model.pathologies).index(top_path)
                    output[0, top_idx].backward()
                    
                    # Grad-CAM
                    gradients = img_tensor.grad.data
                    weights = gradients.mean(dim=(2, 3), keepdim=True)
                    cam = (weights * img_tensor.data).sum(dim=1).squeeze()
                    cam = torch.relu(cam).cpu().numpy()
                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    
                    # Resize à taille originale
                    cam_resized = cv2.resize(cam, (image.size[0], image.size[1]))
                    
                    # Créer heatmap
                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    # Superposer
                    superimposed = heatmap * 0.4 + img_array * 0.6
                    
                    # Visualiser
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Original
                    axes[0].imshow(img_array, cmap='gray' if len(img_array.shape) == 2 else None)
                    axes[0].set_title("Original X-Ray", fontsize=14, fontweight='bold')
                    axes[0].axis('off')
                    
                    # Heatmap
                    axes[1].imshow(heatmap)
                    axes[1].set_title(f"Attention Heatmap", fontsize=14, fontweight='bold')
                    axes[1].axis('off')
                    
                    # Overlay
                    axes[2].imshow(superimposed.astype(np.uint8))
                    axes[2].set_title(f"Grad-CAM - {top_path}", fontsize=14, fontweight='bold')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.success("✅ Grad-CAM explanation generated!")
                    
                    st.caption("""
                    **How to interpret**: Red/yellow areas show where the model paid most attention.
                    Warmer colors = higher importance for prediction.
                    """)
                
                elif xai_method.startswith("🔵 SHAP"):
                    st.write("**📊 SHAP: SHapley Additive exPlanations**")
                    
                    from XAI_methods.shap_image import explain_image_with_shap
                    
                    top_idx = list(model.pathologies).index(top_path)
                    
                    with st.spinner("SHAP is analyzing... (may take 1-2 minutes)"):
                        shap_result = explain_image_with_shap(
                            model,
                            image,
                            target_pathology_idx=top_idx,
                            num_samples=20
                        )
                    
                    # Extraire valeurs
                    shap_vals = shap_result['shap_values']
                    
                    # Normaliser dimensions
                    while len(shap_vals.shape) > 2:
                        shap_vals = shap_vals.squeeze()
                    
                    # Vérifier non vide
                    if shap_vals.size == 0:
                        raise ValueError("Empty SHAP values")
                    
                    # Normaliser
                    shap_norm = (shap_vals - shap_vals.min()) / (shap_vals.max() - shap_vals.min() + 1e-8)
                    
                    # Resize avec dimensions explicites
                    h, w = image.size[1], image.size[0]
                    shap_resized = cv2.resize(shap_norm, (w, h))
                    
                    # Heatmap
                    heatmap = cv2.applyColorMap(np.uint8(255 * shap_resized), cv2.COLORMAP_JET)
                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                    
                    # Image en RGB
                    img_arr = np.array(image)
                    if len(img_arr.shape) == 2:
                        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
                    
                    # Overlay
                    overlay = (heatmap * 0.4 + img_arr * 0.6).astype(np.uint8)
                    
                    # Visualiser
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    
                    # Original
                    axes[0].imshow(img_arr)
                    axes[0].set_title("Original X-Ray", fontsize=14, fontweight='bold')
                    axes[0].axis('off')
                    
                    # SHAP heatmap
                    axes[1].imshow(heatmap)
                    axes[1].set_title("SHAP Values", fontsize=14, fontweight='bold')
                    axes[1].axis('off')
                    
                    # Overlay
                    axes[2].imshow(overlay)
                    axes[2].set_title(f"SHAP - {top_path}", fontsize=14, fontweight='bold')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
                    
                    st.success("✅ SHAP explanation generated!")
                    
                    st.caption("""
                    **How to interpret**: Shows pixel-level contribution to prediction.
                    Red = increases probability, Blue = decreases probability.
                    Based on game theory (Shapley values).
                    """)
                
                # Guide d'interprétation
                with st.expander("💡 Understanding This Explanation"):
                    if "LIME" in xai_method:
                        st.markdown(f"""
                        ### LIME Interpretation for {top_path}
                        
                        **What LIME shows**:
                        - Segments (superpixels) that most influenced the decision
                        - Green boundaries = important regions
                        
                        **For {top_path}**:
                        - Look for anatomical consistency
                        - Check if highlighted areas match known pathology locations
                        - More concentrated regions = clearer evidence
                        
                        **Confidence**: {top_prob:.1%}
                        """)
                    
                    elif "Grad-CAM" in xai_method:
                        st.markdown(f"""
                        ### Grad-CAM Interpretation for {top_path}
                        
                        **What Grad-CAM shows**:
                        - Model's attention at pixel level
                        - Warm colors (red/yellow) = high attention
                        - Cool colors (blue/purple) = low attention
                        
                        **For {top_path}**:
                        - Attention should align with pathology location
                        - Diffuse attention may indicate uncertainty
                        - Check if focus is anatomically correct
                        
                        **Confidence**: {top_prob:.1%}
                        """)
                    
                    elif "SHAP" in xai_method:
                        st.markdown(f"""
                        ### SHAP Interpretation for {top_path}
                        
                        **What SHAP shows**:
                        - Contribution of each pixel to prediction
                        - Based on game theory (Shapley values)
                        - Red = increases {top_path} probability
                        - Blue = decreases {top_path} probability
                        
                        **For {top_path}**:
                        - Red areas are evidence FOR this pathology
                        - Blue areas are evidence AGAINST
                        - Intensity shows strength of contribution
                        
                        **Confidence**: {top_prob:.1%}
                        **Method**: {shap_result.get('method', 'Unknown')}
                        """)
                
            except Exception as e:
                st.error(f"❌ XAI error: {str(e)}")
                with st.expander("Show error details"):
                    import traceback
                    st.code(traceback.format_exc())

# import streamlit as st
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2
# import sys
# import os

# # Ajouter chemin modèles
# sys.path.insert(0, os.path.join(os.getcwd(), 'models'))

# st.title("🫁 Chest X-Ray Classification")

# # Cache le modèle
# @st.cache_resource
# def load_model(model_key):
#     """Charge et met en cache le modèle"""
#     try:
#         from images.model_loader import load_image_model
#         model = load_image_model(model_key)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Upload
# #uploaded_file = st.file_uploader("Upload a chest X-ray image", type=['jpg', 'png', 'jpeg'])

# st.subheader("📸 Select Input Image")

# tab_upload, tab_test = st.tabs(["📤 Upload", "🧪 Test Images"])

# image = None

# with tab_upload:
#     uploaded_file = st.file_uploader("Upload X-ray", type=['jpg', 'png', 'jpeg'])
#     if uploaded_file:
#         image = Image.open(uploaded_file).convert('RGB')
#         st.image(image, width=400)

# with tab_test:
#     test_images = {
#         'Sample 1': 'models/images/test_lung.jpg',
#         'Sample 2': 'models/images/radio_lung.jpg',
#         'Sample 3': 'models/images/test_radio.jpeg'
#     }
    
#     available = {n: p for n, p in test_images.items() if os.path.exists(p)}
    
#     if available:
#         selected = st.selectbox("Choose:", ["-- Select --"] + list(available.keys()))
#         if selected != "-- Select --":
#             image = Image.open(available[selected]).convert('RGB')
#             st.image(image, caption=selected, width=400)
#             st.success(f"✅ {selected}")

# if image is None:
#     st.info("Please select an image")
#     st.stop()

# # Votre code de classification continue ici...

# if uploaded_file:
#     # Charger et afficher image
#     image = Image.open(uploaded_file).convert('RGB')
    
#     col1, col2 = st.columns([1, 1])
#     with col1:
#         st.image(image, caption="Uploaded X-ray", use_column_width=True)
    
#     with col2:
#         st.write("**Image Information**")
#         st.write(f"Size: {image.size[0]} x {image.size[1]} pixels")
#         st.write(f"Mode: {image.mode}")
#         st.write(f"Format: {uploaded_file.type}")
    
#     st.divider()
    
#     # Sélection modèle avec conseils
#     st.subheader("🤖 Model Selection")
    
#     # Info sur les modèles
#     with st.expander("ℹ️ Model Guide - Click to see recommendations", expanded=False):
#         st.markdown("""
#         ### 🎯 Quick Recommendations
        
#         **🏆 Best Overall Performance** → Model 1 (All Datasets)
#         - Most comprehensive training
#         - Best for general use
#         - 18 pathologies detected
        
#         **⚡ Fast & Reliable** → Model 2 (MIMIC + CheXpert)
#         - Hospital-grade datasets
#         - Clinically validated
#         - Good balance speed/accuracy
        
#         **🎓 Research Standard** → Model 4 (NIH ChestX-ray14)
#         - Most cited in literature
#         - 14 common pathologies
#         - Standardized benchmark
        
#         ---
        
#         ### 📊 All Models Comparison
        
#         | Model | Datasets | Pathologies | Best For |
#         |-------|----------|-------------|----------|
#         | **1. All Datasets** | CheXpert, MIMIC, NIH, PadChest | 18 | General screening |
#         | **2. MIMIC + CheXpert** | Hospital data | 14 | Clinical diagnosis |
#         | **3. MIMIC only** | MIMIC-CXR | 14 | ICU patients |
#         | **4. NIH ChestX-ray14** | NIH public | 14 | Research/benchmarking |
#         | **5. PadChest** | Spanish dataset | 18 | European population |
#         | **6. CheXpert** | Stanford | 14 | US population |
#         | **7. ResNet50** | All datasets | 18 | Alternative architecture |
        
#         ### 💡 Tips
#         - **Not sure?** → Use Model 1 (All Datasets)
#         - **Research paper?** → Use Model 4 (NIH)
#         - **Speed matters?** → Use Model 6 (CheXpert)
#         - **Want diversity?** → Compare Model 1, 2, and 4
#         """)
    
#     # Sélecteur de modèle
#     model_options = {
#         "🏆 Model 1: DenseNet121 (All Datasets) - RECOMMENDED": 'densenet121-res224-all',
#         "⚡ Model 2: DenseNet121 (MIMIC + CheXpert)": 'densenet121-res224-mimic_ch',
#         "🏥 Model 3: DenseNet121 (MIMIC only)": 'densenet121-res224-mimic_nb',
#         "🎓 Model 4: DenseNet121 (NIH ChestX-ray14)": 'densenet121-res224-nih',
#         "🌍 Model 5: DenseNet121 (PadChest)": 'densenet121-res224-pc',
#         "🇺🇸 Model 6: DenseNet121 (CheXpert)": 'densenet121-res224-chex',
#         "🔄 Model 7: ResNet50 (All Datasets)": 'resnet50-res512-all'
#     }
    
#     selected_display = st.selectbox(
#         "Select Classification Model",
#         list(model_options.keys()),
#         help="See guide above for recommendations on which model to use"
#     )
    
#     model_key = model_options[selected_display]
    
#     # Afficher info du modèle sélectionné
#     model_descriptions = {
#         'densenet121-res224-all': "🏆 Most comprehensive model, trained on 4 major datasets. Best overall performance.",
#         'densenet121-res224-mimic_ch': "⚡ Hospital-grade model combining MIMIC-CXR and CheXpert. Clinically validated.",
#         'densenet121-res224-mimic_nb': "🏥 Specialized on ICU patients from MIMIC-CXR dataset.",
#         'densenet121-res224-nih': "🎓 Research standard, trained on NIH ChestX-ray14 (112,120 images).",
#         'densenet121-res224-pc': "🌍 Trained on PadChest dataset (160,000 Spanish images).",
#         'densenet121-res224-chex': "🇺🇸 Stanford's CheXpert dataset (224,316 images).",
#         'resnet50-res512-all': "🔄 Alternative ResNet50 architecture for comparison."
#     }
    
#     st.info(model_descriptions[model_key])
    
#     # Bouton classification
#     if st.button("🔍 Classify X-Ray", type="primary", use_container_width=True):
#         with st.spinner(f"Loading model..."):
#             model = load_model(model_key)
        
#         if model is None or model.model is None:
#             st.error("❌ Failed to load model")
#         else:
#             st.success(f"✅ Model loaded: {model.name}")
            
#             with st.spinner("Analyzing X-ray..."):
#                 try:
#                     # Prédiction avec l'image PIL (pas uploaded_file)
#                     from images.model_loader import predict_image
#                     result = predict_image(model, image)
                    
#                     # Afficher résultats
#                     st.success("✅ Classification complete!")
                    
#                     st.subheader("📊 Top 5 Pathology Predictions")
                    
#                     # Top 5 avec barres
#                     for i, pred in enumerate(result['top_predictions'], 1):
#                         pathology = pred['pathology']
#                         probability = pred['probability']
                        
#                         # Couleur selon probabilité
#                         if probability > 0.7:
#                             color = "🔴"
#                             bar_color = "#e74c3c"
#                         elif probability > 0.4:
#                             color = "🟠"
#                             bar_color = "#f39c12"
#                         else:
#                             color = "🟢"
#                             bar_color = "#2ecc71"
                        
#                         col1, col2, col3 = st.columns([3, 2, 1])
                        
#                         with col1:
#                             st.write(f"**{i}. {color} {pathology}**")
                        
#                         with col2:
#                             st.progress(float(probability))
                        
#                         with col3:
#                             st.metric("", f"{probability:.1%}")
                    
#                     # Détails
#                     with st.expander("📋 View All Pathology Scores"):
#                         all_probs = result['all_probabilities']
#                         sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
                        
#                         for pathology, prob in sorted_probs:
#                             st.write(f"**{pathology}**: {prob:.2%}")
                    
#                     # Sauvegarder dans session
#                     st.session_state['image_result'] = result
#                     st.session_state['image_model'] = model
#                     st.session_state['image_file'] = image
                    
#                 except Exception as e:
#                     st.error(f"❌ Classification error: {str(e)}")
#                     with st.expander("Show error details"):
#                         import traceback
#                         st.code(traceback.format_exc())
    
#     # XAI Section
#     if 'image_result' in st.session_state:
#         st.divider()
#         st.subheader("🔍 Explainability Analysis (XAI)")
        
#         result = st.session_state['image_result']
#         top_path = result['top_pathology']
#         top_prob = result['top_probability']
        
#         st.info(f"📊 Understanding why the model predicted **{top_path}** ({top_prob:.1%})")
        
#         # Sélection méthode XAI
#         xai_col1, xai_col2 = st.columns([3, 2])
        
#         with xai_col1:
#             xai_method = st.selectbox(
#                 "Select XAI Method",
#                 ["LIME (Superpixel Segmentation)", "Grad-CAM (Attention Heatmap)", "SHAP (Pixel Importance)"],
#                 help="LIME: Shows important regions | Grad-CAM: Shows model attention | SHAP: Pixel-level importance"
#             )
        
#         with xai_col2:
#             st.write("")
#             st.write("")
#             generate_xai = st.button(
#                 "Generate Explanation",
#                 type="secondary",
#                 use_container_width=True
#             )
        
#         if generate_xai:
#             with st.spinner(f"Generating {xai_method.split()[0]}..."):
#                 try:
#                     model = st.session_state['image_model']
#                     image = st.session_state['image_file']
                    
#                     if xai_method.startswith("LIME"):
#                         st.write("**🔧 LIME: Local Interpretable Model-agnostic Explanations**")
                        
#                         from lime import lime_image
#                         from skimage.segmentation import mark_boundaries
#                         import torchxrayvision as xrv
                        
#                         # Fonction de prédiction pour LIME
#                         def predict_fn(images):
#                             batch = []
#                             for img in images:
#                                 if len(img.shape) == 3:
#                                     img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#                                 img = xrv.datasets.normalize(img, 255)
#                                 img = cv2.resize(img, (224, 224))
#                                 batch.append(img)
                            
#                             batch_tensor = torch.from_numpy(np.array(batch)).unsqueeze(1).float()
                            
#                             with torch.no_grad():
#                                 outputs = model.model(batch_tensor)
#                                 return torch.sigmoid(outputs).cpu().numpy()
                        
#                         # LIME explainer
#                         explainer = lime_image.LimeImageExplainer()
                        
#                         img_array = np.array(image)
                        
#                         with st.spinner("LIME is analyzing... (may take 30-60s)"):
#                             explanation = explainer.explain_instance(
#                                 img_array,
#                                 predict_fn,
#                                 top_labels=3,
#                                 hide_color=0,
#                                 num_samples=100  # Réduit pour rapidité
#                             )
                        
#                         # Trouver l'index du top pathology
#                         top_idx = list(model.pathologies).index(top_path)
                        
#                         # Visualiser
#                         temp, mask = explanation.get_image_and_mask(
#                             top_idx,
#                             positive_only=True,
#                             num_features=5,
#                             hide_rest=False
#                         )
                        
#                         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                        
#                         ax1.imshow(image)
#                         ax1.set_title("Original X-ray", fontsize=14, fontweight='bold')
#                         ax1.axis('off')
                        
#                         ax2.imshow(mark_boundaries(temp / 255.0, mask))
#                         ax2.set_title(f"LIME Explanation\nHighlighted: {top_path}", 
#                                      fontsize=14, fontweight='bold')
#                         ax2.axis('off')
                        
#                         st.pyplot(fig)
#                         plt.close()
                        
#                         st.caption("🟢 **Green boundaries** highlight regions that support the prediction of " + top_path)
                        
#                     elif xai_method.startswith("Grad-CAM"):
#                         st.write("**🔧 Grad-CAM: Gradient-weighted Class Activation Mapping**")
                        
#                         import torchxrayvision as xrv
                        
#                         # Préparer image
#                         img_array = np.array(image)
#                         if len(img_array.shape) == 3:
#                             img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                        
#                         img_array = xrv.datasets.normalize(img_array, 255)
#                         img_array_resized = cv2.resize(img_array, (224, 224))
#                         img_tensor = torch.from_numpy(img_array_resized).unsqueeze(0).unsqueeze(0).float()
                        
#                         # Grad-CAM
#                         img_tensor.requires_grad = True
#                         output = model.model(img_tensor)
                        
#                         # Backward sur top pathology
#                         top_idx = list(model.pathologies).index(top_path)
#                         model.model.zero_grad()
#                         output[0, top_idx].backward()
                        
#                         # Gradients
#                         gradients = img_tensor.grad.data
#                         weights = gradients.mean(dim=(2, 3), keepdim=True)
#                         cam = (weights * img_tensor.data).sum(dim=1).squeeze()
#                         cam = torch.relu(cam).cpu().numpy()
#                         cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                        
#                         # Resize à taille originale
#                         cam_resized = cv2.resize(cam, (image.size[0], image.size[1]))
                        
#                         # Heatmap
#                         heatmap = cv2.applyColorMap(
#                             np.uint8(255 * cam_resized),
#                             cv2.COLORMAP_JET
#                         )
#                         heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                        
#                         # Overlay
#                         img_array_vis = np.array(image)
#                         superimposed = heatmap * 0.4 + img_array_vis * 0.6
                        
#                         # Visualiser
#                         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
                        
#                         ax1.imshow(image, cmap='gray')
#                         ax1.set_title("Original X-ray", fontsize=14, fontweight='bold')
#                         ax1.axis('off')
                        
#                         ax2.imshow(cam_resized, cmap='jet')
#                         ax2.set_title("Grad-CAM Heatmap", fontsize=14, fontweight='bold')
#                         ax2.axis('off')
                        
#                         ax3.imshow(superimposed.astype(np.uint8))
#                         ax3.set_title(f"Overlay - Focus on {top_path}", fontsize=14, fontweight='bold')
#                         ax3.axis('off')
                        
#                         st.pyplot(fig)
#                         plt.close()
                        
#                         st.caption("🔴 **Red areas** = High attention | 🔵 **Blue areas** = Low attention")
                    
                    
#                     elif xai_method.startswith("SHAP"):
#                         st.write("**🔧 SHAP: SHapley Additive exPlanations**")
                        
#                         try:
#                             from XAI_methods.shap_image import explain_image_with_shap, visualize_shap_image
                            
#                             model = st.session_state['image_model']
#                             image = st.session_state['image_file']
#                             top_idx = list(model.pathologies).index(top_path)
                            
#                             st.info(f"🎯 Target: {top_path} (index: {top_idx})")
                            
#                             # Barre de progression
#                             progress_bar = st.progress(0)
#                             status_text = st.empty()
                            
#                             status_text.text("Generating SHAP explanation (1-2 min)...")
#                             progress_bar.progress(20)
                            
#                             # IMPORTANT: Réduire num_samples pour éviter timeout
#                             shap_result = explain_image_with_shap(
#                                 model, 
#                                 image, 
#                                 target_pathology_idx=top_idx,
#                                 num_samples=20  # Réduit pour éviter timeout
#                             )
                            
#                             progress_bar.progress(70)
#                             status_text.text("Creating visualization...")
                            
#                             # Visualiser
#                             fig = visualize_shap_image(shap_result, top_path, original_image=image)
                            
#                             progress_bar.progress(100)
#                             status_text.empty()
#                             progress_bar.empty()
                            
#                             st.pyplot(fig)
#                             plt.close()
                            
#                             st.caption("🔴 **Red pixels** = Increase prediction | 🔵 **Blue pixels** = Decrease prediction")
                            
#                             # Statistiques
#                             shap_vals = shap_result['shap_values']
#                             if len(shap_vals.shape) == 4:
#                                 shap_vals = shap_vals[0, 0, :, :]
#                             elif len(shap_vals.shape) == 3:
#                                 shap_vals = shap_vals[0, :, :]
                            
#                             st.write("**📊 SHAP Statistics**")
#                             col1, col2, col3 = st.columns(3)
#                             with col1:
#                                 st.metric("Max SHAP", f"{shap_vals.max():.4f}")
#                             with col2:
#                                 st.metric("Min SHAP", f"{shap_vals.min():.4f}")
#                             with col3:
#                                 st.metric("Mean |SHAP|", f"{np.abs(shap_vals).mean():.4f}")
                            
#                             st.success("✅ SHAP complete!")
                            
#                         except Exception as e:
#                             st.error(f"❌ SHAP error: {str(e)}")
#                             with st.expander("Show error"):
#                                 import traceback
#                                 st.code(traceback.format_exc())
#                             st.info("💡 Try Grad-CAM (faster) or reduce image complexity")


#                     st.success(f"✅ {xai_method.split()[0]} explanation generated!")
                    
#                     # Conseils d'interprétation
#                     with st.expander("💡 How to interpret this explanation"):
#                         if xai_method.startswith("LIME"):
#                             st.markdown("""
#                             ### Understanding LIME
                            
#                             - **Green boundaries**: Regions that strongly support the prediction
#                             - **Superpixels**: Image divided into meaningful segments
#                             - **Interpretation**: Focus on highlighted anatomical structures
                            
#                             ⚠️ **Note**: LIME is model-agnostic but computationally intensive
#                             """)
#                         elif xai_method.startswith("Grad-CAM"):
#                             st.markdown("""
#                             ### Understanding Grad-CAM
                            
#                             - **Red/Yellow areas**: Where the model focuses most
#                             - **Blue/Purple areas**: Less important for decision
#                             - **Heatmap**: Direct visualization of neural network attention
                            
#                             ✅ **Advantage**: Fast and shows exact model focus
#                             """)
                
#                 except Exception as e:
#                     st.error(f"❌ XAI error: {str(e)}")
#                     with st.expander("Show error details"):
#                         import traceback
#                         st.code(traceback.format_exc())

# else:
#     # Page d'accueil
#     st.info("👆 **Upload a chest X-ray image to begin analysis**")
    
#     with st.expander("ℹ️ How to use this page", expanded=True):
#         st.markdown("""
#         ### 🎯 Quick Start Guide
        
#         1. **Upload X-ray**: Click "Browse files" and select an image (JPG, PNG)
#         2. **Choose Model**: Select from 7 pre-trained models
#            - 🏆 **Recommended**: Model 1 (All Datasets) for general use
#            - Click "Model Guide" for detailed recommendations
#         3. **Classify**: Click "Classify X-Ray" to detect pathologies
#         4. **Review Results**: See top 5 predictions with confidence scores
#         5. **Explain**: Generate LIME or Grad-CAM visualizations
        
#         ### 🤖 Available Models (All Pre-trained)
        
#         We provide **7 state-of-the-art models**:
#         - 6 x DenseNet121 variants (different training datasets)
#         - 1 x ResNet50 (alternative architecture)
        
#         All models detect multiple pathologies including:
#         - Atelectasis, Cardiomegaly, Consolidation, Edema
#         - Effusion, Emphysema, Pneumonia, Pneumothorax
#         - And more (up to 18 pathologies depending on model)
        
#         ### 🔍 XAI Methods
        
#         **LIME** (Local Interpretable Model-agnostic Explanations)
#         - Shows which image regions are important
#         - Model-agnostic (works with any model)
#         - Takes 30-60 seconds to generate
        
#         **Grad-CAM** (Gradient-weighted Class Activation Mapping)
#         - Visualizes model's attention as heatmap
#         - Fast generation (~1-2 seconds)
#         - Shows exact neural network focus
                    
#         **SHAP** (SHapley Additive exPlanations)
#         - Pixel-level importance scores
#         - More detailed but slower (1-2 minutes)
#         - Useful for in-depth analysis
        
#         ### 💡 Tips
        
#         - **High quality images** work best (clear, well-exposed X-rays)
#         - **Try multiple models** to compare predictions
#         - **Use XAI** to validate clinical reasoning
#         - **Check "View All Scores"** for complete pathology analysis
#         """)
    
#     # Exemples de fichiers
#     st.write("### 🧪 Test Files")
#     test_paths = [
#         'models/images/test_lung.jpg',
#         'models/images/radio_lung.jpg',
#         'models/images/test_radio.jpeg'
#     ]
    
#     for path in test_paths:
#         if os.path.exists(path):
#             st.write(f"✅ {os.path.basename(path)} available")
#         else:
#             st.write(f"⚠️ {os.path.basename(path)} (add to test)")