import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import sys
import os

# Ajouter le chemin des modèles
sys.path.insert(0, os.path.join(os.getcwd(), 'models'))

st.title("🎵 Audio Classification - Deepfake Detection")

# Charger modèles avec cache
@st.cache_resource
def load_model(model_name):
    """Charge et met en cache le modèle"""
    try:
        from audio.model_loader import load_audio_model
        model = load_audio_model(model_name)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Upload
#uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=['wav'])
tab_upload, tab_test = st.tabs(["📤 Upload", "🧪 Test Audio"])

audio_file = None

with tab_upload:
    uploaded_file = st.file_uploader("Upload .wav", type=['wav'])
    if uploaded_file:
        audio_file = uploaded_file
        st.audio(uploaded_file)

with tab_test:
    test_audios = {
        'Real': 'models/audio/.pb/wav_de_test/test_audio.wav',      # ← AJUSTEZ
        'Fake': 'models/audio/.pb/wav_de_test/test_audio.wav',
    }
    
    available = {n: p for n, p in test_audios.items() if os.path.exists(p)}
    
    if available:
        selected = st.selectbox("Choose:", ["-- Select --"] + list(available.keys()))
        if selected != "-- Select --":
            with open(available[selected], 'rb') as f:
                st.audio(f.read())
            audio_file = open(available[selected], 'rb')

if audio_file is None:
    st.stop()

if uploaded_file:
    # Player audio
    st.audio(uploaded_file, format='audio/wav')
    
    # Charger pour visualisation
    try:
        import soundfile as sf
        uploaded_file.seek(0)
        y, sr = sf.read(uploaded_file)
        
        # Si stéréo, prendre premier canal
        if len(y.shape) > 1:
            y = y[:, 0]
        
        # Infos
        duration = len(y) / sr
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Rate", f"{sr} Hz")
        with col2:
            st.metric("Duration", f"{duration:.2f}s")
        with col3:
            st.metric("Samples", f"{len(y):,}")
        
        # Visualisations rapides
        with st.expander("📊 Quick Audio Visualizations", expanded=False):
            tab1, tab2 = st.tabs(["Waveform", "Mel Spectrogram"])
            
            with tab1:
                fig, ax = plt.subplots(figsize=(12, 4))
                time_axis = np.arange(len(y)) / sr
                ax.plot(time_axis, y, linewidth=0.5, color='steelblue')
                ax.set_xlabel("Time (s)", fontsize=12)
                ax.set_ylabel("Amplitude", fontsize=12)
                ax.set_title("Audio Waveform", fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
            
            with tab2:
                # Spectrogramme
                D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
                D_db = librosa.power_to_db(D, ref=np.max)
                
                fig, ax = plt.subplots(figsize=(12, 5))
                img = librosa.display.specshow(
                    D_db,
                    sr=sr,
                    x_axis='time',
                    y_axis='mel',
                    ax=ax,
                    cmap='viridis'
                )
                ax.set_title("Mel Spectrogram", fontsize=14, fontweight='bold')
                fig.colorbar(img, ax=ax, format='%+2.0f dB')
                st.pyplot(fig)
                plt.close()
        
    except Exception as e:
        st.error(f"Error loading audio: {e}")
    
    st.divider()
    
    # Sélection modèle
    st.subheader("🤖 Model Selection & Classification")
    
    with st.expander("ℹ️ Model Guide", expanded=False):
        st.markdown("""
        ### Available Models
        
        **Model 1 - wav2vec2 (HuggingFace)**
        - Pre-trained transformer model
        - Trained on large-scale deepfake datasets
        - High accuracy on various audio types
        - Processing time: ~2-3 seconds
        
        **Model 2 - TensorFlow CNN**
        - Custom spectrogram-based CNN
        - Optimized for audio characteristics
        - Fast inference: ~1 second
        - Good for real-time detection
        
        ### 💡 Tips
        - Try both models for cross-validation
        - Clear audio (minimal noise) works best
        - 2-5 seconds duration optimal
        """)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        model_choice = st.selectbox(
            "Select Pre-trained Model",
            [
                "🏆 Model 1: wav2vec2 (HuggingFace) - RECOMMENDED",
                "⚡ Model 2: TensorFlow CNN (Fast)"
            ],
            help="Choose between wav2vec2 or TensorFlow model"
        )
        
        # Map vers nom interne
        model_map = {
            "🏆 Model 1: wav2vec2 (HuggingFace) - RECOMMENDED": "huggingface",
            "⚡ Model 2: TensorFlow CNN (Fast)": "tensorflow"
        }
        model_key = model_map[model_choice]
    
    with col2:
        st.write("")
        st.write("")
        classify_btn = st.button(
            "🔍 Classify Audio",
            type="primary",
            use_container_width=True
        )
    
    if classify_btn:
        with st.spinner(f"Loading model..."):
            model = load_model(model_key)
        
        if model is None:
            st.error("❌ Failed to load model")
        else:
            st.success(f"✅ Model loaded: {model.name}")
            
            with st.spinner("Classifying audio..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Prédiction
                    from audio.model_loader import predict_audio
                    prediction = predict_audio(model, uploaded_file)
                    
                    # Afficher résultat
                    st.success("✅ Classification complete!")
                    
                    # Résultat visuel
                    result_col1, result_col2, result_col3 = st.columns([2, 2, 3])
                    
                    with result_col1:
                        pred_class = prediction['class']
                        color = "🔴" if pred_class == "FAKE" else "🟢"
                        st.markdown(f"### {color} {pred_class}")
                    
                    with result_col2:
                        confidence = prediction['confidence']
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with result_col3:
                        # Jauge visuelle
                        if pred_class == "FAKE":
                            st.error(f"⚠️ Likely deepfake ({confidence:.1%})")
                        else:
                            st.success(f"✓ Likely authentic ({confidence:.1%})")
                    
                    # Probabilités détaillées
                    st.write("**Detailed Probabilities:**")
                    prob_col1, prob_col2 = st.columns(2)
                    
                    with prob_col1:
                        real_prob = prediction['probabilities']['REAL']
                        st.metric("🟢 REAL", f"{real_prob:.2%}")
                        st.progress(float(real_prob))
                    
                    with prob_col2:
                        fake_prob = prediction['probabilities']['FAKE']
                        st.metric("🔴 FAKE", f"{fake_prob:.2%}")
                        st.progress(float(fake_prob))
                    
                    # Sauvegarder dans session state
                    st.session_state['audio_prediction'] = prediction
                    st.session_state['audio_model'] = model
                    st.session_state['audio_file'] = uploaded_file
                    st.session_state['audio_data'] = (y, sr)
                    
                except Exception as e:
                    st.error(f"❌ Classification error: {str(e)}")
                    with st.expander("Show error details"):
                        import traceback
                        st.code(traceback.format_exc())
    
    # XAI Section - AMÉLIORÉE
    if 'audio_prediction' in st.session_state:
        st.divider()
        st.subheader("🔍 Explainability Analysis (XAI)")
        
        prediction = st.session_state['audio_prediction']
        
        st.info(f"📊 Understanding why the model classified this audio as **{prediction['class']}** ({prediction['confidence']:.1%})")
        
        # Guide XAI
        with st.expander("ℹ️ XAI Methods Explained", expanded=False):
            st.markdown("""
            ### 🎯 Available XAI Methods
            
            **LIME (Local Interpretable Model-agnostic Explanations)**
            - Shows which **time-frequency regions** influenced the decision
            - Highlights important segments in the spectrogram
            - Processing time: ~30-60 seconds
            - Best for: Understanding temporal patterns
            
            **SHAP (SHapley Additive exPlanations)**
            - Shows **feature importance** (spectral, temporal, MFCC)
            - Based on game theory (Shapley values)
            - Processing time: ~10-20 seconds
            - Best for: Understanding which audio characteristics matter
            
            **Gradient Visualization**
            - Shows where the model **"listens"** most
            - Attention heatmap over spectrogram
            - Processing time: ~5-10 seconds
            - Best for: Quick visual understanding
            
            **Comprehensive Analysis**
            - Complete breakdown of all audio features
            - Spectral, temporal, and frequency analysis
            - Processing time: ~5 seconds
            - Best for: Detailed technical analysis
            
            ### 💡 Which to Choose?
            
            - **Quick insight** → Gradient Visualization or Comprehensive Analysis
            - **Detailed understanding** → LIME (shows exact regions)
            - **Feature analysis** → SHAP (shows what features matter)
            - **Complete picture** → Try all methods!
            """)
        
        # Sélection méthode XAI
        xai_col1, xai_col2 = st.columns([3, 2])
        
        with xai_col1:
            xai_method = st.selectbox(
                "Select XAI Method",
                [
                    "🎨 Gradient Visualization (Fast - Recommended)",
                    "📊 Comprehensive Analysis (All Features)",
                    "🔍 LIME (Time-Frequency Regions)",
                    "📈 SHAP (Feature Importance)"
                ],
                help="Choose explainability method based on your needs"
            )
        
        with xai_col2:
            st.write("")
            st.write("")
            xai_btn = st.button(
                "Generate Explanation",
                type="secondary",
                use_container_width=True
            )
        
        if xai_btn:
            method_name = xai_method.split('(')[0].strip()
            
            with st.spinner(f"Generating {method_name}..."):
                try:
                    y, sr = st.session_state['audio_data']
                    audio_file = st.session_state['audio_file']
                    model = st.session_state['audio_model']
                    
                    # Import XAI functions
                    from XAI_methods.audio_xai_advanced import (
                        gradient_audio_visualization,
                        comprehensive_audio_analysis,
                        audio_lime_explanation,
                        audio_shap_explanation
                    )
                    
                    if "Gradient" in xai_method:
                        st.write("**🎨 Gradient-based Attention Visualization**")
                        st.info("Showing where the model focuses its attention in the audio...")
                        
                        fig = gradient_audio_visualization(model, audio_file, prediction, sr=sr)
                        st.pyplot(fig)
                        plt.close()
                        
                        st.caption("🔴 **Red/Yellow areas** = High attention | 🔵 **Blue areas** = Low attention")
                        
                        with st.expander("💡 How to interpret"):
                            st.markdown("""
                            ### Understanding Gradient Visualization
                            
                            - **Attention Map**: Shows which time-frequency regions the model focuses on
                            - **Red areas**: High importance for the prediction
                            - **Blue areas**: Low importance
                            - **Temporal Profile**: Shows attention over time
                            
                            **For FAKE audio**: Often shows attention on high frequencies (artifacts)
                            **For REAL audio**: More balanced attention across all frequencies
                            """)
                    
                    elif "Comprehensive" in xai_method:
                        st.write("**📊 Comprehensive Audio Analysis**")
                        st.info("Complete breakdown of all audio characteristics...")
                        
                        fig = comprehensive_audio_analysis(model, audio_file, prediction, sr=sr)
                        st.pyplot(fig)
                        plt.close()
                        
                        st.caption("Complete technical analysis of waveform, spectrogram, MFCCs, chroma, and spectral features")
                        
                        with st.expander("💡 How to interpret"):
                            st.markdown("""
                            ### Understanding Comprehensive Analysis
                            
                            - **Waveform**: Time-domain signal representation
                            - **Mel Spectrogram**: Frequency content over time (human perception scale)
                            - **MFCCs**: Key features for speech/audio classification
                            - **Chromagram**: Pitch class distribution
                            - **Spectral Features**: Centroid (brightness), Rolloff (bandwidth)
                            - **Zero Crossing Rate**: Noisiness indicator
                            
                            **Deepfake indicators**: Unusual spectral patterns, unnatural MFCCs, irregular ZCR
                            """)
                    
                    elif "LIME" in xai_method:
                        st.write("**🔍 LIME: Local Interpretable Model-agnostic Explanations**")
                        st.info("Analyzing time-frequency regions... (30-60 seconds)")
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Generating LIME explanation...")
                        progress_bar.progress(20)
                        
                        fig = audio_lime_explanation(model, audio_file, prediction, sr=sr, n_segments=20)
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        st.caption("🟢 **Green boundaries** highlight time-frequency regions that influenced the prediction")
                        
                        with st.expander("💡 How to interpret"):
                            st.markdown(
                                f"""
                        ### Understanding LIME for Audio

                        - **Left**: Original mel spectrogram
                        - **Middle**: LIME explanation with highlighted regions
                        - **Right**: Overlay showing important areas

                        **Green boundaries** = Regions that support the **{prediction['class']}** prediction

                        **What to look for:**
                        - Concentrated regions → specific artifacts or patterns
                        - Distributed regions → general audio characteristics
                        - High frequency focus → often indicates processing artifacts (deepfake)
                        - Low frequency focus → fundamental speech characteristics (often real)
                        """
                            )

                    elif "SHAP" in xai_method:
                        st.write("**📈 SHAP: Feature Importance Analysis**")
                        st.info("Calculating feature importance... (10-20 seconds)")
                        
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        status_text.text("Extracting audio features...")
                        progress_bar.progress(30)
                        
                        fig = audio_shap_explanation(model, audio_file, prediction, sr=sr)
                        
                        progress_bar.progress(100)
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.pyplot(fig)
                        plt.close()
                        
                        st.caption("🔴 **Red bars** = Increase FAKE probability | 🔵 **Blue bars** = Increase REAL probability")
                        
                        # Statistiques SHAP
                        st.write("")
                        st.write("**Key Feature Insights:**")
                        
                        insights_col1, insights_col2 = st.columns(2)
                        
                        with insights_col1:
                            st.markdown("""
                            **Spectral Features**
                            - Centroid: Brightness/timbre
                            - Bandwidth: Frequency spread
                            - Rolloff: Energy concentration
                            - Flatness: Noisiness
                            - Contrast: Frequency peaks
                            """)
                        
                        with insights_col2:
                            st.markdown("""
                            **Temporal Features**
                            - Zero Crossing Rate: Signal changes
                            - RMS Energy: Overall loudness
                            - MFCCs: Speech characteristics
                            - Chroma: Pitch content
                            - Tempo: Rhythmic patterns
                            """)
                        
                        with st.expander("💡 How to interpret"):
                            st.markdown("""
                            ### Understanding SHAP Values
                            
                            SHAP values show **how much each feature contributed** to the prediction.
                            
                            **Positive values (Red)**: Increase likelihood of FAKE
                            **Negative values (Blue)**: Increase likelihood of REAL
                            
                            **Common patterns:**
                            
                            **FAKE indicators:**
                            - High spectral flatness (artificial noise)
                            - Unusual MFCC patterns (synthesized speech)
                            - Low spectral contrast (processing artifacts)
                            - Irregular zero crossing rate
                            
                            **REAL indicators:**
                            - Natural spectral centroid variation
                            - Coherent chroma features
                            - Consistent temporal patterns
                            - Balanced energy distribution
                            """)
                    
                    st.success(f"✅ {method_name} explanation generated!")
                    
                except Exception as e:
                    st.error(f"❌ XAI error: {str(e)}")
                    with st.expander("Show error details"):
                        import traceback
                        st.code(traceback.format_exc())

else:
    # Page d'accueil
    st.info("👆 **Upload a .wav audio file to begin deepfake detection**")
    
    # Guide d'utilisation
    with st.expander("ℹ️ How to use this page", expanded=True):
        st.markdown("""
        ### 🎯 Step-by-Step Guide
        
        1. **Upload Audio**: Click "Browse files" and select a `.wav` file
        2. **View Visualizations**: Examine waveform and mel spectrogram (optional)
        3. **Select Model**:
           - **Model 1 (wav2vec2)**: Best accuracy, HuggingFace transformer
           - **Model 2 (TensorFlow)**: Fastest, CNN-based
        4. **Classify**: Click "Classify Audio" to detect deepfake
        5. **View Results**: See prediction with confidence scores
        6. **Explore XAI**: Generate detailed explanations (4 methods available!)
        
        ### 🔍 XAI Methods Overview
        
        | Method | Processing Time | Best For |
        |--------|----------------|----------|
        | **Gradient Visualization** | ~5-10s | Quick visual insight |
        | **Comprehensive Analysis** | ~5s | Technical deep-dive |
        | **LIME** | ~30-60s | Time-frequency regions |
        | **SHAP** | ~10-20s | Feature importance |
        
        ### 💡 Pro Tips
        
        - **Audio Quality**: Clear audio works best (minimal background noise)
        - **Duration**: 2-5 seconds optimal
        - **Format**: .wav format, 16kHz sample rate recommended
        - **Compare**: Try both models for validation
        - **XAI**: Start with Gradient Visualization, then try LIME/SHAP for details
        """)
    
    # Test files
    st.write("### 🧪 Test Files")
    test_files = [
        "models/audio/.pb/wav_de_test/test_audio.wav",
        "models/audio/.pb/wav_de_test/test_audio2222.wav"
    ]
    
    for test_file in test_files:
        if os.path.exists(test_file):
            st.write(f"✅ {os.path.basename(test_file)}")
        else:
            st.write(f"⚠️ {os.path.basename(test_file)} (not found)")

# import streamlit as st
# import numpy as np
# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
# import sys
# import os

# # Ajouter le chemin des modèles
# sys.path.insert(0, os.path.join(os.getcwd(), 'models'))

# st.title("🎵 Audio Classification - Deepfake Detection")

# # Charger modèles avec cache
# @st.cache_resource
# def load_model(model_name):
#     """Charge et met en cache le modèle"""
#     try:
#         from audio.model_loader import load_audio_model
#         model = load_audio_model(model_name)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {e}")
#         return None

# # Upload
# uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=['wav'])

# if uploaded_file:
#     # Player audio
#     st.audio(uploaded_file, format='audio/wav')
    
#     # Charger pour visualisation
#     try:
#         import soundfile as sf
#         uploaded_file.seek(0)
#         y, sr = sf.read(uploaded_file)
        
#         # Si stéréo, prendre premier canal
#         if len(y.shape) > 1:
#             y = y[:, 0]
        
#         # Infos
#         duration = len(y) / sr
        
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Sample Rate", f"{sr} Hz")
#         with col2:
#             st.metric("Duration", f"{duration:.2f}s")
#         with col3:
#             st.metric("Samples", f"{len(y):,}")
        
#         # Visualisations
#         st.subheader("📊 Audio Visualizations")
        
#         tab1, tab2 = st.tabs(["Waveform", "Mel Spectrogram"])
        
#         with tab1:
#             fig, ax = plt.subplots(figsize=(12, 4))
#             time_axis = np.arange(len(y)) / sr
#             ax.plot(time_axis, y, linewidth=0.5, color='steelblue')
#             ax.set_xlabel("Time (s)", fontsize=12)
#             ax.set_ylabel("Amplitude", fontsize=12)
#             ax.set_title("Audio Waveform", fontsize=14, fontweight='bold')
#             ax.grid(True, alpha=0.3)
#             st.pyplot(fig)
#             plt.close()
        
#         with tab2:
#             # Spectrogramme
#             D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#             D_db = librosa.power_to_db(D, ref=np.max)
            
#             fig, ax = plt.subplots(figsize=(12, 5))
#             img = librosa.display.specshow(
#                 D_db,
#                 sr=sr,
#                 x_axis='time',
#                 y_axis='mel',
#                 ax=ax,
#                 cmap='viridis'
#             )
#             ax.set_title("Mel Spectrogram", fontsize=14, fontweight='bold')
#             fig.colorbar(img, ax=ax, format='%+2.0f dB')
#             st.pyplot(fig)
#             plt.close()
        
#     except Exception as e:
#         st.error(f"Error loading audio: {e}")
    
#     st.divider()
    
#     # Sélection modèle
#     st.subheader("🤖 Model Selection & Classification")
    
#     col1, col2 = st.columns([3, 2])
    
#     with col1:
#         model_choice = st.selectbox(
#             "Select Pre-trained Model",
#             [
#                 "Model 1 (wav2vec2 HuggingFace)",
#                 "Model 2 (TensorFlow Local)"
#             ],
#             help="Choose between wav2vec2 or local TensorFlow model"
#         )
        
#         # Map vers nom interne
#         model_map = {
#             "Model 1 (wav2vec2 HuggingFace)": "huggingface",
#             "Model 2 (TensorFlow Local)": "tensorflow"
#         }
#         model_key = model_map[model_choice]
    
#     with col2:
#         st.write("")
#         st.write("")
#         classify_btn = st.button(
#             "🔍 Classify Audio",
#             type="primary",
#             use_container_width=True
#         )
    
#     if classify_btn:
#         with st.spinner(f"Loading {model_choice}..."):
#             model = load_model(model_key)
        
#         if model is None:
#             st.error("❌ Failed to load model. Check console for errors.")
#         else:
#             st.success(f"✅ Model loaded: {model.name}")
            
#             with st.spinner("Classifying audio..."):
#                 try:
#                     # Reset file pointer
#                     uploaded_file.seek(0)
                    
#                     # Prédiction
#                     from audio.model_loader import predict_audio
#                     prediction = predict_audio(model, uploaded_file)
                    
#                     # Afficher résultat
#                     st.success("✅ Classification complete!")
                    
#                     # Résultat visuel
#                     result_col1, result_col2, result_col3 = st.columns([2, 2, 3])
                    
#                     with result_col1:
#                         pred_class = prediction['class']
#                         color = "🔴" if pred_class == "FAKE" else "🟢"
#                         st.markdown(f"### {color} {pred_class}")
                    
#                     with result_col2:
#                         confidence = prediction['confidence']
#                         st.metric("Confidence", f"{confidence:.1%}")
                    
#                     with result_col3:
#                         # Jauge visuelle
#                         if pred_class == "FAKE":
#                             st.error(f"⚠️ Likely deepfake ({confidence:.1%})")
#                         else:
#                             st.success(f"✓ Likely authentic ({confidence:.1%})")
                    
#                     # Probabilités détaillées
#                     st.write("**Detailed Probabilities:**")
#                     prob_col1, prob_col2 = st.columns(2)
                    
#                     with prob_col1:
#                         real_prob = prediction['probabilities']['REAL']
#                         st.metric("🟢 REAL", f"{real_prob:.2%}")
#                         st.progress(float(real_prob))
                    
#                     with prob_col2:
#                         fake_prob = prediction['probabilities']['FAKE']
#                         st.metric("🔴 FAKE", f"{fake_prob:.2%}")
#                         st.progress(float(fake_prob))
                    
#                     # Sauvegarder dans session state
#                     st.session_state['audio_prediction'] = prediction
#                     st.session_state['audio_model'] = model
#                     st.session_state['audio_file'] = uploaded_file
#                     st.session_state['audio_data'] = (y, sr)
                    
#                 except Exception as e:
#                     st.error(f"❌ Classification error: {str(e)}")
#                     with st.expander("Show error details"):
#                         import traceback
#                         st.code(traceback.format_exc())
    
#     # XAI Section
#     if 'audio_prediction' in st.session_state:
#         st.divider()
#         st.subheader("🔍 Explainability Analysis (XAI)")
        
#         st.info("📊 Understand **why** the model classified this audio as " + 
#                 st.session_state['audio_prediction']['class'])
        
#         xai_col1, xai_col2 = st.columns([3, 2])
        
#         with xai_col1:
#             xai_method = st.selectbox(
#                 "Select XAI Method",
#                 ["Spectrogram Visualization", "Feature Importance (Basic)"],
#                 help="Advanced XAI methods for audio"
#             )
        
#         with xai_col2:
#             st.write("")
#             st.write("")
#             xai_btn = st.button(
#                 "Generate Explanation",
#                 type="secondary",
#                 use_container_width=True
#             )
        
#         if xai_btn:
#             with st.spinner(f"Generating {xai_method}..."):
#                 try:
#                     y, sr = st.session_state['audio_data']
#                     prediction = st.session_state['audio_prediction']
                    
#                     if xai_method == "Spectrogram Visualization":
#                         st.write("**Mel Spectrogram with Prediction Overlay**")
                        
#                         # Créer spectrogramme
#                         D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#                         D_db = librosa.power_to_db(D, ref=np.max)
                        
#                         fig, ax = plt.subplots(figsize=(14, 6))
#                         img = librosa.display.specshow(
#                             D_db,
#                             sr=sr,
#                             x_axis='time',
#                             y_axis='mel',
#                             ax=ax,
#                             cmap='coolwarm'
#                         )
                        
#                         pred_class = prediction['class']
#                         conf = prediction['confidence']
                        
#                         ax.set_title(
#                             f"Spectrogram - Predicted: {pred_class} ({conf:.1%})",
#                             fontsize=16,
#                             fontweight='bold'
#                         )
                        
#                         fig.colorbar(img, ax=ax, format='%+2.0f dB')
                        
#                         # Ajouter annotation
#                         ax.text(
#                             0.02, 0.98,
#                             f"Model: {st.session_state['audio_model'].name}",
#                             transform=ax.transAxes,
#                             fontsize=10,
#                             verticalalignment='top',
#                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
#                         )
                        
#                         st.pyplot(fig)
#                         plt.close()
                        
#                         st.caption("The spectrogram shows time-frequency representation. " +
#                                  "Deepfakes may show artifacts in certain frequency bands.")
                    
#                     elif xai_method == "Feature Importance (Basic)":
#                         st.write("**Basic Audio Features Analysis**")
                        
#                         # Extraire features basiques
#                         spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
#                         spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
#                         zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
#                         mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                        
#                         # Statistiques
#                         features = {
#                             'Spectral Centroid (mean)': np.mean(spectral_centroids),
#                             'Spectral Rolloff (mean)': np.mean(spectral_rolloff),
#                             'Zero Crossing Rate (mean)': np.mean(zero_crossing_rate),
#                             'MFCC 1 (mean)': np.mean(mfccs[0]),
#                             'MFCC 2 (mean)': np.mean(mfccs[1]),
#                             'MFCC 3 (mean)': np.mean(mfccs[2])
#                         }
                        
#                         # Visualiser
#                         fig, ax = plt.subplots(figsize=(12, 6))
                        
#                         names = list(features.keys())
#                         values = list(features.values())
                        
#                         # Normaliser pour visualisation
#                         values_norm = (values - np.min(values)) / (np.max(values) - np.min(values) + 1e-8)
                        
#                         colors = ['green' if v > 0.5 else 'orange' for v in values_norm]
                        
#                         ax.barh(names, values_norm, color=colors, alpha=0.7)
#                         ax.set_xlabel("Normalized Feature Value", fontsize=12)
#                         ax.set_title(
#                             f"Basic Audio Features - {pred_class} Audio",
#                             fontsize=14,
#                             fontweight='bold'
#                         )
#                         ax.grid(axis='x', alpha=0.3)
                        
#                         st.pyplot(fig)
#                         plt.close()
                        
#                         st.caption("These basic features can indicate audio quality and processing artifacts.")
                    
#                     st.success(f"✅ {xai_method} generated!")
                    
#                 except Exception as e:
#                     st.error(f"❌ XAI error: {str(e)}")
#                     with st.expander("Show error details"):
#                         import traceback
#                         st.code(traceback.format_exc())

# else:
#     # Page d'accueil
#     st.info("👆 **Upload a .wav audio file to begin deepfake detection**")
    
#     # Guide d'utilisation
#     with st.expander("ℹ️ How to use this page", expanded=True):
#         st.markdown("""
#         ### 🎯 Step-by-Step Guide
        
#         1. **Upload Audio**: Click "Browse files" and select a `.wav` file
#         2. **View Visualizations**: Examine waveform and mel spectrogram
#         3. **Select Model**:
#            - **Model 1 (wav2vec2)**: HuggingFace transformer model
#            - **Model 2 (TensorFlow)**: Local custom-trained model
#         4. **Classify**: Click "Classify Audio" to detect deepfake
#         5. **View Results**: See prediction with confidence scores
#         6. **Explore XAI**: Generate explanations to understand the decision
        
#         ### 🤖 Available Models
        
#         **Model 1 - wav2vec2 (HuggingFace)**
#         - Pre-trained on large-scale deepfake dataset
#         - Transformer-based architecture
#         - High accuracy on various audio types
        
#         **Model 2 - TensorFlow Local**
#         - Custom-trained model
#         - Spectrogram-based CNN
#         - Optimized for specific audio characteristics
        
#         ### 🔍 XAI Methods
        
#         - **Spectrogram Visualization**: See time-frequency representation
#         - **Feature Importance**: Understand which audio features matter
        
#         ### 💡 Tips
        
#         - **Audio Quality**: Clear audio works best (minimal background noise)
#         - **Duration**: 2-5 seconds optimal
#         - **Format**: .wav format required (16kHz recommended)
#         - **Compare Models**: Try both models for cross-validation
#         """)
    
#     # Test files
#     st.write("### 🧪 Test Files Available")
#     test_files = [
#         "models/audio/.pb/wav_de_test/test_audio.wav",
#         "models/audio/.pb/wav_de_test/test_audio2222.wav"
#     ]
    
#     for test_file in test_files:
#         if os.path.exists(test_file):
#             st.write(f"✅ {os.path.basename(test_file)}")
#         else:
#             st.write(f"⚠️ {os.path.basename(test_file)} (not found)")
    
#     st.caption("Use these test files to verify the system works correctly.")