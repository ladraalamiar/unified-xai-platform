import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import sys
import os
import pandas as pd

# Ajouter chemin modèles
sys.path.insert(0, os.path.join(os.getcwd(), 'models'))

st.title("🔄 XAI Methods Comparison")

st.markdown("""
Compare multiple explainability techniques side-by-side on the same input.
**Analyze the same prediction using different XAI methods to gain comprehensive insights.**
""")

# Sélection type de données
data_type = st.radio("Select Data Type", ["🫁 Image (X-Ray)", "🎵 Audio (Deepfake)"], horizontal=True)

# ========================================
# IMAGE COMPARISON
# ========================================
if "Image" in data_type:
    st.subheader("📸 Image XAI Comparison")
    
    with st.expander("ℹ️ Image XAI Methods Guide", expanded=False):
        st.markdown("""
        ### Available Methods
        
        | Method | Speed | Type | Shows |
        |--------|-------|------|-------|
        | **LIME** | ~30-60s | Model-agnostic | Important regions (superpixels) |
        | **Grad-CAM** | ~1-2s | Gradient-based | Model attention (heatmap) |
        | **SHAP** | ~1-2min | Game theory | Pixel importance (contribution) |
        
        ### Comparison Strategy
        - **LIME + Grad-CAM**: Quick overview (regions + attention)
        - **All three**: Complete analysis (recommended for research)
        - **LIME + SHAP**: Model-agnostic deep dive
        """)
    
    uploaded_file = st.file_uploader("Upload chest X-ray image", type=['jpg', 'png', 'jpeg'], key='img_upload')
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(image, caption="Input X-Ray", use_column_width=True)
        
        with col2:
            st.write("**Select XAI methods to compare:**")
            
            method_col1, method_col2, method_col3 = st.columns(3)
            with method_col1:
                use_lime = st.checkbox("🟢 LIME", value=True, help="Superpixel regions")
            with method_col2:
                use_gradcam = st.checkbox("🔴 Grad-CAM", value=True, help="Attention heatmap")
            with method_col3:
                use_shap = st.checkbox("🔵 SHAP", value=False, help="Pixel importance (slow)")
            
            if not (use_lime or use_gradcam or use_shap):
                st.warning("⚠️ Please select at least one XAI method")
            
            # Sélection modèle
            model_choice = st.selectbox(
                "Select Model",
                ["DenseNet121 (All Datasets)", "DenseNet121 (NIH)", "ResNet50 (All Datasets)"],
                help="Model to analyze"
            )
        
        if st.button("🔍 Generate & Compare All Methods", type="primary", use_container_width=True):
            # Map modèle
            model_map = {
                "DenseNet121 (All Datasets)": "densenet121-res224-all",
                "DenseNet121 (NIH)": "densenet121-res224-nih",
                "ResNet50 (All Datasets)": "resnet50-res512-all"
            }
            model_key = model_map[model_choice]
            
            # Charger modèle
            with st.spinner("Loading model..."):
                from images.model_loader import load_image_model, predict_image
                model = load_image_model(model_key)
            
            if model is None:
                st.error("❌ Failed to load model")
            else:
                st.success(f"✅ Model loaded: {model.name}")
                
                # Prédiction
                with st.spinner("Analyzing image..."):
                    result = predict_image(model, image)
                    top_path = result['top_pathology']
                    top_prob = result['top_probability']
                    top_idx = list(model.pathologies).index(top_path)
                
                # Afficher prédiction
                st.info(f"**🎯 Prediction**: {top_path} ({top_prob:.1%} confidence)")
                
                st.divider()
                st.subheader("📊 XAI Methods Side-by-Side Comparison")
                
                # Créer colonnes pour comparaison
                methods_selected = []
                if use_lime:
                    methods_selected.append("LIME")
                if use_gradcam:
                    methods_selected.append("Grad-CAM")
                if use_shap:
                    methods_selected.append("SHAP")
                
                cols = st.columns(len(methods_selected))
                
                for idx, method in enumerate(methods_selected):
                    with cols[idx]:
                        st.markdown(f"### {method}")
                        
                        if method == "LIME":
                            time_est = "~30-60 seconds"
                        elif method == "Grad-CAM":
                            time_est = "~1-2 seconds"
                        else:  # SHAP
                            time_est = "~1-2 minutes"
                        
                        st.caption(f"⏱️ {time_est}")
                        
                        with st.spinner(f"Generating {method}..."):
                            try:
                                if method == "LIME":
                                    # LIME
                                    from lime import lime_image
                                    from skimage.segmentation import mark_boundaries
                                    import torchxrayvision as xrv
                                    
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
                                    
                                    explainer = lime_image.LimeImageExplainer()
                                    explanation = explainer.explain_instance(
                                        np.array(image),
                                        predict_fn,
                                        top_labels=1,
                                        hide_color=0,
                                        num_samples=100
                                    )
                                    
                                    temp, mask = explanation.get_image_and_mask(
                                        top_idx,
                                        positive_only=True,
                                        num_features=5,
                                        hide_rest=False
                                    )
                                    
                                    fig, ax = plt.subplots(figsize=(7, 7))
                                    ax.imshow(mark_boundaries(temp / 255.0, mask))
                                    ax.set_title(f"LIME\n{top_path}", fontsize=12, fontweight='bold')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                    st.caption("🟢 Green = Important regions")
                                    
                                elif method == "Grad-CAM":
                                    # Grad-CAM
                                    import torchxrayvision as xrv
                                    
                                    img_array = np.array(image)
                                    if len(img_array.shape) == 3:
                                        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                                    
                                    img_array = xrv.datasets.normalize(img_array, 255)
                                    img_array_resized = cv2.resize(img_array, (224, 224))
                                    img_tensor = torch.from_numpy(img_array_resized).unsqueeze(0).unsqueeze(0).float()
                                    
                                    img_tensor.requires_grad = True
                                    output = model.model(img_tensor)
                                    model.model.zero_grad()
                                    output[0, top_idx].backward()
                                    
                                    gradients = img_tensor.grad.data
                                    weights = gradients.mean(dim=(2, 3), keepdim=True)
                                    cam = (weights * img_tensor.data).sum(dim=1).squeeze()
                                    cam = torch.relu(cam).cpu().numpy()
                                    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                                    
                                    cam_resized = cv2.resize(cam, (image.size[0], image.size[1]))
                                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
                                    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                    
                                    img_array_vis = np.array(image)
                                    superimposed = heatmap * 0.4 + img_array_vis * 0.6
                                    
                                    fig, ax = plt.subplots(figsize=(7, 7))
                                    ax.imshow(superimposed.astype(np.uint8))
                                    ax.set_title(f"Grad-CAM\n{top_path}", fontsize=12, fontweight='bold')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                    st.caption("🔴 Red = High attention")
                                
                                # elif method == "SHAP":
                                #     # SHAP
                                #     from XAI_methods.shap_image import explain_image_with_shap, visualize_shap_image
                                    
                                #     shap_result = explain_image_with_shap(
                                #         model,
                                #         image,
                                #         target_pathology_idx=top_idx,
                                #         num_samples=20  # Réduit pour comparaison
                                #     )
                                    
                                #     # Juste le overlay (pas toute la figure)
                                #     shap_vals = shap_result['shap_values']
                                #     if len(shap_vals.shape) == 4:
                                #         shap_vals = shap_vals[0, 0, :, :]
                                #     elif len(shap_vals.shape) == 3:
                                #         shap_vals = shap_vals[0, :, :]
                                    
                                #     # Normaliser
                                #     shap_normalized = (shap_vals - shap_vals.min()) / (shap_vals.max() - shap_vals.min() + 1e-8)
                                #     shap_resized = cv2.resize(shap_normalized, (image.size[0], image.size[1]))
                                    
                                #     # Overlay
                                #     heatmap = cv2.applyColorMap(np.uint8(255 * shap_resized), cv2.COLORMAP_JET)
                                #     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                    
                                #     img_display = np.array(image)
                                #     if len(img_display.shape) == 3:
                                #         img_display_gray = cv2.cvtColor(img_display, cv2.COLOR_RGB2GRAY)
                                #     else:
                                #         img_display_gray = img_display
                                    
                                #     img_display_rgb = cv2.cvtColor(
                                #         ((img_display_gray - img_display_gray.min()) / 
                                #          (img_display_gray.max() - img_display_gray.min() + 1e-8) * 255).astype(np.uint8),
                                #         cv2.COLOR_GRAY2RGB
                                #     )
                                    
                                #     overlay = heatmap * 0.4 + img_display_rgb * 0.6
                                    
                                #     fig, ax = plt.subplots(figsize=(7, 7))
                                #     ax.imshow(overlay.astype(np.uint8))
                                #     ax.set_title(f"SHAP\n{top_path}", fontsize=12, fontweight='bold')
                                #     ax.axis('off')
                                #     st.pyplot(fig)
                                #     plt.close()
                                    
                                #     st.caption("🔵 Blue/Red = Pixel contribution")
                                
                                elif method == "SHAP":
                                    from XAI_methods.shap_image import explain_image_with_shap
                                    
                                    shap_result = explain_image_with_shap(
                                        model, image, 
                                        target_pathology_idx=top_idx,
                                        num_samples=20
                                    )
                                    
                                    # Extraire valeurs
                                    shap_vals = shap_result['shap_values']
                                    
                                    # Squeeze robuste
                                    while len(shap_vals.shape) > 2:
                                        shap_vals = shap_vals.squeeze()
                                    
                                    # Vérifier non vide
                                    if shap_vals.size == 0:
                                        raise ValueError("Empty SHAP values")
                                    
                                    # Normaliser
                                    shap_norm = (shap_vals - shap_vals.min()) / (shap_vals.max() - shap_vals.min() + 1e-8)
                                    
                                    # Resize - CORRECTION: dimensions explicites
                                    h, w = image.size[1], image.size[0]  # PIL: (width, height)
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
                                    
                                    # Plot
                                    fig, ax = plt.subplots(figsize=(7, 7))
                                    ax.imshow(overlay)
                                    ax.set_title(f"SHAP\n{top_path}", fontsize=12, fontweight='bold')
                                    ax.axis('off')
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                    st.caption("🔵 Blue/Red = Pixel contribution")



                                st.success(f"✅ {method}")
                                
                            except Exception as e:
                                st.error(f"❌ {method} failed")
                                with st.expander("Error details"):
                                    st.code(str(e))
                
                # Tableau comparatif détaillé
                st.divider()
                st.subheader("📋 Methods Comparison Table")
                
                import pandas as pd
                
                comparison_data = {
                    "Method": ["LIME", "Grad-CAM", "SHAP"],
                    "Type": ["Model-agnostic", "Gradient-based", "Game theory"],
                    "Speed": ["⚠️ Slow (30-60s)", "✅ Fast (1-2s)", "⚠️ Very Slow (1-2min)"],
                    "Granularity": ["Superpixel regions", "Pixel-level", "Pixel-level"],
                    "Accuracy": ["Approximation", "Direct", "Theoretically rigorous"],
                    "Best For": ["Region importance", "Quick attention", "Feature attribution"]
                }
                
                df = pd.DataFrame(comparison_data)
                
                # Filter based on selected methods
                selected_methods = []
                if use_lime:
                    selected_methods.append("LIME")
                if use_gradcam:
                    selected_methods.append("Grad-CAM")
                if use_shap:
                    selected_methods.append("SHAP")
                
                df_filtered = df[df['Method'].isin(selected_methods)]
                st.dataframe(df_filtered, use_container_width=True, hide_index=True)
                
                # Insights
                with st.expander("💡 Interpretation Guide"):
                    st.markdown(f"""
                    ### How to Interpret for {top_path}
                    
                    **LIME (Green boundaries)**:
                    - Shows which **anatomical regions** the model focuses on
                    - Segments represent coherent image areas
                    - More segments = more distributed attention
                    
                    **Grad-CAM (Red heatmap)**:
                    - Shows **exact pixel attention** from model's gradients
                    - Red/Yellow = High importance
                    - Blue/Purple = Low importance
                    - Direct view of neural network focus
                    
                    **SHAP (Blue/Red overlay)**:
                    - Shows **pixel contributions** to prediction
                    - Red = Increases {top_path} probability
                    - Blue = Decreases {top_path} probability
                    - Based on Shapley values (game theory)
                    
                    ### Agreement Analysis
                    - **High agreement** (all methods highlight same areas) → Confident prediction
                    - **Partial agreement** → Some features more important than others
                    - **Low agreement** → Model uncertainty or complex decision
                    """)
    
    else:
        st.info("👆 Upload a chest X-ray image to start comparison")

# ========================================
# AUDIO COMPARISON
# ========================================
else:  # Audio
    st.subheader("🎵 Audio XAI Comparison")
    
    with st.expander("ℹ️ Audio XAI Methods Guide", expanded=False):
        st.markdown("""
        ### Available Methods
        
        | Method | Speed | Shows | Best For |
        |--------|-------|-------|----------|
        | **Gradient Visualization** | ~5-10s | Attention heatmap | Quick insight |
        | **LIME** | ~30-60s | Time-frequency regions | Region importance |
        | **SHAP** | ~10-20s | Feature importance | Which features matter |
        | **Comprehensive** | ~5s | All features | Technical analysis |
        
        ### Comparison Strategy
        - **Gradient + SHAP**: Quick + detailed (recommended)
        - **All methods**: Complete analysis
        - **LIME + SHAP**: Model-agnostic deep dive
        """)
    
    uploaded_file = st.file_uploader("Upload audio file (.wav)", type=['wav'], key='audio_upload')
    
    if uploaded_file:
        st.audio(uploaded_file, format='audio/wav')
        
        # Charger audio pour info
        try:
            import soundfile as sf
            uploaded_file.seek(0)
            y, sr = sf.read(uploaded_file)
            if len(y.shape) > 1:
                y = y[:, 0]
            
            duration = len(y) / sr
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Sample Rate", f"{sr} Hz")
            with col2:
                st.metric("Duration", f"{duration:.2f}s")
            with col3:
                st.metric("Samples", f"{len(y):,}")
        except:
            pass
        
        st.write("**Select XAI methods to compare:**")
        
        method_col1, method_col2, method_col3, method_col4 = st.columns(4)
        with method_col1:
            use_gradient = st.checkbox("🎨 Gradient", value=True, help="Attention visualization")
        with method_col2:
            use_lime_audio = st.checkbox("🟢 LIME", value=False, help="Time-frequency regions")
        with method_col3:
            use_shap_audio = st.checkbox("📊 SHAP", value=True, help="Feature importance")
        with method_col4:
            use_comprehensive = st.checkbox("📈 Comprehensive", value=False, help="All features")
        
        if not (use_gradient or use_lime_audio or use_shap_audio or use_comprehensive):
            st.warning("⚠️ Please select at least one XAI method")
        
        # Sélection modèle
        model_audio_choice = st.selectbox(
            "Select Audio Model",
            ["wav2vec2 (HuggingFace)", "TensorFlow CNN"],
            help="Model to analyze"
        )
        
        if st.button("🔍 Generate & Compare Audio Methods", type="primary", use_container_width=True):
            # Map modèle
            model_audio_map = {
                "wav2vec2 (HuggingFace)": "huggingface",
                "TensorFlow CNN": "tensorflow"
            }
            model_key_audio = model_audio_map[model_audio_choice]
            
            # Charger modèle
            with st.spinner("Loading audio model..."):
                from audio.model_loader import load_audio_model, predict_audio
                audio_model = load_audio_model(model_key_audio)
            
            if audio_model is None:
                st.error("❌ Failed to load audio model")
            else:
                st.success(f"✅ Model loaded: {audio_model.name}")
                
                # Prédiction
                with st.spinner("Classifying audio..."):
                    uploaded_file.seek(0)
                    prediction = predict_audio(audio_model, uploaded_file)
                    pred_class = prediction['class']
                    confidence = prediction['confidence']
                
                # Afficher prédiction
                color_emoji = "🔴" if pred_class == "FAKE" else "🟢"
                st.info(f"**🎯 Prediction**: {color_emoji} {pred_class} ({confidence:.1%} confidence)")
                
                st.divider()
                st.subheader("📊 Audio XAI Methods Side-by-Side")
                
                # Recharger audio
                uploaded_file.seek(0)
                y, sr = sf.read(uploaded_file)
                if len(y.shape) > 1:
                    y = y[:, 0]
                
                # Créer colonnes
                methods_audio_selected = []
                if use_gradient:
                    methods_audio_selected.append("Gradient")
                if use_lime_audio:
                    methods_audio_selected.append("LIME")
                if use_shap_audio:
                    methods_audio_selected.append("SHAP")
                if use_comprehensive:
                    methods_audio_selected.append("Comprehensive")
                
                # Afficher en grille 2x2 si 4 méthodes, sinon en ligne
                if len(methods_audio_selected) <= 2:
                    cols = st.columns(len(methods_audio_selected))
                    rows = [cols]
                else:
                    # Grille 2x2
                    cols_row1 = st.columns(2)
                    cols_row2 = st.columns(2)
                    rows = [cols_row1, cols_row2]
                
                for idx, method in enumerate(methods_audio_selected):
                    if len(methods_audio_selected) <= 2:
                        container = rows[0][idx]
                    else:
                        row_idx = idx // 2
                        col_idx = idx % 2
                        container = rows[row_idx][col_idx]
                    
                    with container:
                        st.markdown(f"### {method}")
                        
                        with st.spinner(f"Generating {method}..."):
                            try:
                                from XAI_methods.audio_xai_advanced import (
                                    gradient_audio_visualization,
                                    audio_lime_explanation,
                                    audio_shap_explanation,
                                    comprehensive_audio_analysis
                                )
                                
                                uploaded_file.seek(0)
                                
                                if method == "Gradient":
                                    fig = gradient_audio_visualization(audio_model, uploaded_file, prediction, sr=sr)
                                    st.pyplot(fig)
                                    plt.close()
                                    st.caption("🔴 Red = High attention")
                                
                                elif method == "LIME":
                                    fig = audio_lime_explanation(audio_model, uploaded_file, prediction, sr=sr, n_segments=15)
                                    st.pyplot(fig)
                                    plt.close()
                                    st.caption("🟢 Green = Important regions")
                                
                                elif method == "SHAP":
                                    fig = audio_shap_explanation(audio_model, uploaded_file, prediction, sr=sr)
                                    st.pyplot(fig)
                                    plt.close()
                                    st.caption("🔴🔵 Red/Blue = Feature impact")
                                
                                elif method == "Comprehensive":
                                    fig = comprehensive_audio_analysis(audio_model, uploaded_file, prediction, sr=sr)
                                    st.pyplot(fig)
                                    plt.close()
                                    st.caption("📊 Complete analysis")
                                
                                st.success(f"✅ {method}")
                                
                            except Exception as e:
                                st.error(f"❌ {method} failed")
                                with st.expander("Error details"):
                                    st.code(str(e))
                
                # Tableau comparatif
                st.divider()
                st.subheader("📋 Audio Methods Comparison")
                
                comparison_audio = {
                    "Method": ["Gradient", "LIME", "SHAP", "Comprehensive"],
                    "Speed": ["✅ Fast (5-10s)", "⚠️ Slow (30-60s)", "⚙️ Medium (10-20s)", "✅ Fast (5s)"],
                    "Shows": ["Attention heatmap", "Time-frequency regions", "Feature importance", "All features"],
                    "Type": ["Gradient-based", "Model-agnostic", "Game theory", "Statistical"],
                    "Best For": ["Quick insight", "Region analysis", "Feature analysis", "Technical report"]
                }
                
                df_audio = pd.DataFrame(comparison_audio)
                selected_audio = []
                if use_gradient:
                    selected_audio.append("Gradient")
                if use_lime_audio:
                    selected_audio.append("LIME")
                if use_shap_audio:
                    selected_audio.append("SHAP")
                if use_comprehensive:
                    selected_audio.append("Comprehensive")
                
                df_audio_filtered = df_audio[df_audio['Method'].isin(selected_audio)]
                st.dataframe(df_audio_filtered, use_container_width=True, hide_index=True)
                
                # Insights audio
                with st.expander("💡 Audio Interpretation Guide"):
                    st.markdown(f"""
                    ### How to Interpret for {pred_class} Audio
                    
                    **Gradient Visualization**:
                    - Heatmap shows where model **"listens"** most
                    - Red/Yellow = High attention areas
                    - For FAKE: Often focuses on high frequencies (artifacts)
                    - For REAL: More balanced across frequencies
                    
                    **LIME (Time-Frequency)**:
                    - Green boundaries = Important time-frequency regions
                    - Shows **when and at what frequency** decision is made
                    - Concentrated regions = specific artifacts
                    - Distributed = general characteristics
                    
                    **SHAP (Features)**:
                    - Bar chart shows **which audio features** matter
                    - Red = Increases FAKE probability
                    - Blue = Increases REAL probability
                    - Key features: Spectral centroid, MFCCs, ZCR, etc.
                    
                    **Comprehensive Analysis**:
                    - Complete breakdown of all audio properties
                    - Waveform, spectrogram, MFCCs, chroma, etc.
                    - Best for technical deep-dive
                    
                    ### Agreement Analysis
                    - **Methods agree** → Confident prediction
                    - **Methods differ** → Complex audio characteristics
                    - Compare Gradient + SHAP to see if attention matches important features
                    """)
    
    else:
        st.info("👆 Upload a .wav file to start audio comparison")

# Section générale
st.divider()
with st.expander("ℹ️ Why Compare Multiple XAI Methods?"):
    st.markdown("""
    ### Benefits of Multi-Method XAI
    
    **1. Validation**
    - If multiple methods agree → High confidence in explanation
    - If methods disagree → Model uncertainty or complex decision
    
    **2. Complementary Insights**
    - LIME: Shows **where** (regions)
    - Grad-CAM/Gradient: Shows **attention** (what model looks at)
    - SHAP: Shows **contribution** (what influences decision)
    
    **3. Robustness**
    - Model-agnostic (LIME) vs Model-specific (Grad-CAM)
    - Different mathematical foundations
    - Cross-validation of explanations
    
    **4. Different Use Cases**
    - **Quick demo**: Grad-CAM/Gradient (fastest)
    - **Research**: LIME + SHAP (comprehensive)
    - **Clinical validation**: All methods (thorough)
    - **Production**: Grad-CAM + SHAP (balance speed/detail)
    
    ### Interpretation Tips
    
    ✅ **Good Sign**: All methods highlight similar areas
    → Model has learned relevant features
    
    ⚠️ **Warning Sign**: Methods completely disagree
    → Possible overfitting or spurious correlations
    
    💡 **Mixed Results**: Some agreement, some difference
    → Normal - each method measures different aspects
    """)


# import streamlit as st
# import torch
# import torchxrayvision as xrv
# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import cv2
# from lime import lime_image
# from skimage.segmentation import mark_boundaries

# st.title("🔄 XAI Methods Comparison")

# st.markdown("""
# Compare multiple explainability techniques side-by-side on the same input.
# **Note**: Only compatible methods for the selected data type will be shown.
# """)

# # Sélection type de données
# data_type = st.radio("Select Data Type", ["Image", "Audio"], horizontal=True)

# if data_type == "Image":
#     st.subheader("📸 Image Comparison")
    
#     uploaded_file = st.file_uploader("Upload chest X-ray", type=['jpg', 'png', 'jpeg'])
    
#     if uploaded_file:
#         image = Image.open(uploaded_file).convert('RGB')
#         st.image(image, caption="Input Image", width=400)
        
#         # Sélection des méthodes
#         st.write("**Select XAI methods to compare:**")
#         col1, col2 = st.columns(2)
        
#         with col1:
#             use_lime = st.checkbox("LIME", value=True)
#         with col2:
#             use_gradcam = st.checkbox("Grad-CAM", value=True)
        
#         if not (use_lime or use_gradcam):
#             st.warning("⚠️ Please select at least one XAI method")
        
#         if st.button("🔍 Generate Comparisons"):
#             # Charger modèle
#             with st.spinner("Loading model..."):
#                 model = xrv.models.DenseNet(weights="densenet121-res224-all")
#                 model.eval()
                
#                 # Préparer image
#                 img_array = np.array(image)
#                 if len(img_array.shape) == 3:
#                     img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
#                 img_array = xrv.datasets.normalize(img_array, 255)
#                 img_array = cv2.resize(img_array, (224, 224))
#                 img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()
                
#                 # Prédiction
#                 with torch.no_grad():
#                     outputs = model(img_tensor)
#                     predictions = torch.sigmoid(outputs).cpu().numpy()[0]
                
#                 top_idx = predictions.argmax()
#                 top_class = model.pathologies[top_idx]
#                 top_prob = predictions[top_idx]
            
#             # Afficher prédiction
#             st.success(f"**Prediction**: {top_class} ({top_prob:.1%})")
            
#             st.divider()
#             st.subheader("📊 Side-by-Side Comparison")
            
#             # Créer colonnes pour comparaison
#             methods_selected = []
#             if use_lime:
#                 methods_selected.append("LIME")
#             if use_gradcam:
#                 methods_selected.append("Grad-CAM")
            
#             cols = st.columns(len(methods_selected))
            
#             for idx, method in enumerate(methods_selected):
#                 with cols[idx]:
#                     st.markdown(f"### {method}")
                    
#                     with st.spinner(f"Generating {method}..."):
#                         try:
#                             if method == "LIME":
#                                 # LIME
#                                 def predict_fn(images):
#                                     batch = []
#                                     for img in images:
#                                         if len(img.shape) == 3:
#                                             img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#                                         img = xrv.datasets.normalize(img, 255)
#                                         img = cv2.resize(img, (224, 224))
#                                         batch.append(img)
                                    
#                                     batch_tensor = torch.from_numpy(np.array(batch)).unsqueeze(1).float()
#                                     with torch.no_grad():
#                                         outputs = model(batch_tensor)
#                                         return torch.sigmoid(outputs).cpu().numpy()
                                
#                                 explainer = lime_image.LimeImageExplainer()
#                                 explanation = explainer.explain_instance(
#                                     np.array(image),
#                                     predict_fn,
#                                     top_labels=1,
#                                     hide_color=0,
#                                     num_samples=50
#                                 )
                                
#                                 temp, mask = explanation.get_image_and_mask(
#                                     top_idx,
#                                     positive_only=True,
#                                     num_features=5,
#                                     hide_rest=False
#                                 )
                                
#                                 fig, ax = plt.subplots(figsize=(6, 6))
#                                 ax.imshow(mark_boundaries(temp / 255.0, mask))
#                                 ax.set_title(f"{method}\nHighlighted Regions")
#                                 ax.axis('off')
#                                 st.pyplot(fig)
                                
#                             elif method == "Grad-CAM":
#                                 # Grad-CAM
#                                 img_tensor.requires_grad = True
#                                 output = model(img_tensor)
#                                 model.zero_grad()
#                                 output[0, top_idx].backward()
                                
#                                 gradients = img_tensor.grad.data
#                                 weights = gradients.mean(dim=(2, 3), keepdim=True)
#                                 cam = (weights * img_tensor).sum(dim=1).squeeze().detach().numpy()
#                                 cam = np.maximum(cam, 0)
#                                 cam = cam / cam.max()
                                
#                                 cam_resized = cv2.resize(cam, (image.size[0], image.size[1]))
#                                 heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
#                                 heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                                
#                                 img_array_vis = np.array(image)
#                                 superimposed = heatmap * 0.4 + img_array_vis * 0.6
                                
#                                 fig, ax = plt.subplots(figsize=(6, 6))
#                                 ax.imshow(superimposed.astype(np.uint8))
#                                 ax.set_title(f"{method}\nAttention Heatmap")
#                                 ax.axis('off')
#                                 st.pyplot(fig)
                            
#                             st.caption(f"✅ {method} completed")
                            
#                         except Exception as e:
#                             st.error(f"Error with {method}: {str(e)}")
            
#             # Tableau comparatif
#             st.divider()
#             st.subheader("📋 Method Characteristics")
            
#             comparison_data = {
#                 "Method": methods_selected,
#                 "Type": ["Model-agnostic", "Model-specific"] if len(methods_selected) == 2 else (["Model-agnostic"] if "LIME" in methods_selected else ["Model-specific"]),
#                 "Speed": ["⚠️ Slower", "✅ Fast"] if len(methods_selected) == 2 else (["⚠️ Slower"] if "LIME" in methods_selected else ["✅ Fast"]),
#                 "Granularity": ["🔹 Regions", "🔸 Pixels"] if len(methods_selected) == 2 else (["🔹 Regions"] if "LIME" in methods_selected else ["🔸 Pixels"])
#             }
            
#             import pandas as pd
#             df = pd.DataFrame(comparison_data)
#             st.dataframe(df, use_container_width=True, hide_index=True)
    
#     else:
#         st.info("👆 Upload an image to start comparison")

# else:  # Audio
#     st.subheader("🎵 Audio Comparison")
    
#     uploaded_file = st.file_uploader("Upload audio file (.wav)", type=['wav'])
    
#     if uploaded_file:
#         st.audio(uploaded_file)
        
#         st.info("""
#         🚧 **Audio XAI Comparison**
        
#         For audio comparison:
#         1. Upload your .wav file
#         2. Select methods (LIME, Grad-CAM on spectrogram)
#         3. Compare explanations side-by-side
        
#         **Note**: This requires your pre-trained audio models.
#         Implement similar to image comparison above.
#         """)
        
#         # Placeholder structure
#         col1, col2 = st.columns(2)
#         with col1:
#             use_lime = st.checkbox("LIME (Audio)", value=True)
#         with col2:
#             use_gradcam = st.checkbox("Grad-CAM (Spectrogram)", value=True)
        
#         if st.button("Generate Audio Comparisons"):
#             st.warning("⚠️ Implement with your pre-trained audio models")
#     else:
#         st.info("👆 Upload a .wav file to start comparison")

# # Section informative
# with st.expander("ℹ️ Understanding XAI Methods"):
#     st.markdown("""
#     ### LIME (Local Interpretable Model-agnostic Explanations)
#     - **Pros**: Works with any model, intuitive segmentation
#     - **Cons**: Slower, approximation-based
#     - **Best for**: Understanding which regions matter
    
#     ### Grad-CAM (Gradient-weighted Class Activation Mapping)
#     - **Pros**: Fast, pixel-level precision, uses model's gradients
#     - **Cons**: Requires gradient-based models
#     - **Best for**: Visual attention heatmaps
    
#     ### When to use each?
#     - **LIME**: When you need model-agnostic explanations
#     - **Grad-CAM**: When you have CNNs and want gradient-based insights
#     - **Both**: For comprehensive analysis
#     """)