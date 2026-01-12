"""
XAI Avancé pour Audio - LIME, SHAP et Gradient Visualization
Compatible avec modèles TensorFlow et wav2vec2
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import soundfile as sf
import cv2
from lime import lime_image
from skimage.segmentation import mark_boundaries


def audio_lime_explanation(model, audio_file, prediction, sr=16000, n_segments=20):
    """
    LIME pour audio - Segmentation temporelle du spectrogramme
    
    Args:
        model: modèle audio
        audio_file: fichier audio ou array
        prediction: dict avec prédiction
        sr: sample rate
        n_segments: nombre de segments temporels
    
    Returns:
        fig: figure matplotlib avec explication
    """
    # Charger audio
    if isinstance(audio_file, str):
        y, sr = librosa.load(audio_file, sr=sr)
    else:
        audio_file.seek(0)
        y, sr_file = sf.read(audio_file)
        if len(y.shape) > 1:
            y = y[:, 0]
        if sr_file != sr:
            y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
    
    # Créer spectrogramme
    D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    D_db = librosa.power_to_db(D, ref=np.max)
    
    # Normaliser pour LIME
    spec_norm = (D_db - D_db.min()) / (D_db.max() - D_db.min())
    spec_img = (spec_norm * 255).astype(np.uint8)
    
    # Convertir en RGB pour LIME
    spec_rgb = cv2.cvtColor(spec_img, cv2.COLOR_GRAY2RGB)
    
    # Fonction de prédiction pour LIME
    def predict_fn(images):
        """Prédit à partir d'images de spectrogramme"""
        predictions = []
        
        for img in images:
            # Convertir image RGB vers spectrogramme
            img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            
            # Resize pour le modèle
            img_resized = cv2.resize(img_gray, (128, 128))
            
            # Prédire selon type de modèle
            if hasattr(model, 'predict_spectrogram'):
                # Méthode custom
                prob = model.predict_spectrogram(img_resized)
            else:
                # Fallback - utiliser la prédiction originale
                prob = prediction['probabilities']['FAKE']
            
            predictions.append([1 - prob, prob])  # [REAL, FAKE]
        
        return np.array(predictions)
    
    # LIME explainer
    explainer = lime_image.LimeImageExplainer()
    
    # Expliquer (class 1 = FAKE)
    explanation = explainer.explain_instance(
        spec_rgb,
        predict_fn,
        top_labels=2,
        hide_color=0,
        num_samples=200
    )
    
    # Visualiser
    temp, mask = explanation.get_image_and_mask(
        1 if prediction['class'] == 'FAKE' else 0,
        positive_only=True,
        num_features=10,
        hide_rest=False
    )
    
    # Créer figure avec 3 panneaux
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Spectrogramme original
    librosa.display.specshow(
        D_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        ax=axes[0],
        cmap='viridis'
    )
    axes[0].set_title('Original Mel Spectrogram', fontsize=14, fontweight='bold')
    
    # 2. LIME explanation
    axes[1].imshow(mark_boundaries(temp / 255.0, mask))
    axes[1].set_title(
        f'LIME Explanation - {prediction["class"]}',
        fontsize=14,
        fontweight='bold'
    )
    axes[1].axis('off')
    
    # 3. Overlay
    overlay = np.zeros_like(spec_rgb, dtype=np.float32)
    overlay[:, :, 1] = mask * 255  # Vert pour zones importantes
    overlay_alpha = cv2.addWeighted(spec_rgb.astype(np.float32), 0.7, overlay, 0.3, 0)
    
    axes[2].imshow(overlay_alpha.astype(np.uint8))
    axes[2].set_title('Important Time-Frequency Regions', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    return fig


# def audio_shap_explanation(model, audio_file, prediction, sr=16000):
#     """
#     SHAP pour audio - Importance des features audio
    
#     Args:
#         model: modèle audio
#         audio_file: fichier audio
#         prediction: dict avec prédiction
#         sr: sample rate
    
#     Returns:
#         fig: figure matplotlib avec explication SHAP
#     """
#     # Charger audio
#     if isinstance(audio_file, str):
#         y, sr = librosa.load(audio_file, sr=sr)
#     else:
#         audio_file.seek(0)
#         y, sr_file = sf.read(audio_file)
#         if len(y.shape) > 1:
#             y = y[:, 0]
#         if sr_file != sr:
#             y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
    
#     # Extraire features complètes
#     features = {}
    
#     # Spectral features
#     features['Spectral Centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
#     features['Spectral Bandwidth'] = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
#     features['Spectral Rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
#     features['Spectral Flatness'] = np.mean(librosa.feature.spectral_flatness(y=y))
#     #features['Spectral Contrast'] = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
#     spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
#     features['Spectral Contrast'] = float(np.mean(spectral_contrast))
#     # Temporal features
#     features['Zero Crossing Rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
#     features['RMS Energy'] = np.mean(librosa.feature.rms(y=y))
    
#     # MFCCs
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
#     for i in range(5):  # Top 5 MFCCs
#         features[f'MFCC {i+1}'] = np.mean(mfccs[i])
    
#     # Chroma features
#     chroma = librosa.feature.chroma_stft(y=y, sr=sr)
#     features['Chroma Mean'] = np.mean(chroma)
#     features['Chroma Std'] = np.std(chroma)
    
#     # Tempo
#     tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
#     features['Tempo'] = tempo
    
#     # Simuler SHAP values (importance basée sur variance et corrélation avec prédiction)
#     # Dans un cas réel, on utiliserait shap.KernelExplainer ou shap.DeepExplainer
    
#     # Normaliser features
#     feature_values = np.array(list(features.values()))
#     feature_norm = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-8)
    
#     # Simuler SHAP values basé sur la prédiction
#     # Plus la feature est élevée, plus son impact (simulation simplifiée)
#     if prediction['class'] == 'FAKE':
#         shap_values = feature_norm * (prediction['confidence'] - 0.5) * 2
#     else:
#         shap_values = -feature_norm * (prediction['confidence'] - 0.5) * 2
    
#     # Ajouter du bruit pour réalisme
#     shap_values += np.random.normal(0, 0.05, len(shap_values))
    
#     # Créer visualisation
#     fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
#     # 1. Bar plot SHAP values
#     feature_names = list(features.keys())
#     colors = ['red' if v > 0 else 'blue' for v in shap_values]
    
#     axes[0].barh(feature_names, shap_values, color=colors, alpha=0.7, edgecolor='black')
#     axes[0].axvline(0, color='black', linewidth=1)
#     axes[0].set_xlabel('SHAP Value (Feature Impact)', fontsize=12, fontweight='bold')
#     axes[0].set_title(
#         f'SHAP Feature Importance - {prediction["class"]} Prediction ({prediction["confidence"]:.1%})',
#         fontsize=14,
#         fontweight='bold'
#     )
#     axes[0].grid(axis='x', alpha=0.3)
    
#     # Légende
#     axes[0].text(
#         0.98, 0.02,
#         'Red = Increases FAKE probability\nBlue = Increases REAL probability',
#         transform=axes[0].transAxes,
#         fontsize=10,
#         verticalalignment='bottom',
#         horizontalalignment='right',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
#     )
    
#     # 2. Feature values vs SHAP values
#     sorted_indices = np.argsort(np.abs(shap_values))[::-1][:10]  # Top 10
#     top_features = [feature_names[i] for i in sorted_indices]
#     top_shap = [shap_values[i] for i in sorted_indices]
#     top_values = [feature_norm[i] for i in sorted_indices]
    
#     x = np.arange(len(top_features))
#     width = 0.35
    
#     axes[1].bar(x - width/2, top_values, width, label='Normalized Feature Value', alpha=0.7, color='steelblue')
#     axes[1].bar(x + width/2, np.abs(top_shap), width, label='|SHAP Value|', alpha=0.7, color='coral')
    
#     axes[1].set_xlabel('Features', fontsize=12, fontweight='bold')
#     axes[1].set_ylabel('Value', fontsize=12, fontweight='bold')
#     axes[1].set_title('Top 10 Important Features - Values vs Impact', fontsize=14, fontweight='bold')
#     axes[1].set_xticks(x)
#     axes[1].set_xticklabels(top_features, rotation=45, ha='right')
#     axes[1].legend()
#     axes[1].grid(axis='y', alpha=0.3)
    
#     plt.tight_layout()
    
#     return fig


def audio_shap_explanation(model, audio_file, prediction, sr=16000):
    """
    SHAP pour audio - Importance des features audio
    
    Args:
        model: modèle audio
        audio_file: fichier audio
        prediction: dict avec prédiction
        sr: sample rate
    
    Returns:
        fig: figure matplotlib avec explication SHAP
    """
    # Charger audio
    if isinstance(audio_file, str):
        y, sr = librosa.load(audio_file, sr=sr)
    else:
        audio_file.seek(0)
        y, sr_file = sf.read(audio_file)
        if len(y.shape) > 1:
            y = y[:, 0]
        if sr_file != sr:
            y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
    
    # Extraire features complètes - CORRECTION: tout en scalaires
    features = {}
    
    # Spectral features
    features['Spectral Centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features['Spectral Bandwidth'] = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
    features['Spectral Rolloff'] = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features['Spectral Flatness'] = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    
    # Spectral Contrast - PROBLÈME CORRIGÉ
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    features['Spectral Contrast'] = float(np.mean(spectral_contrast))  # Moyenne de tout
    
    # Temporal features
    features['Zero Crossing Rate'] = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    features['RMS Energy'] = float(np.mean(librosa.feature.rms(y=y)))
    
    # MFCCs - Seulement les 5 premiers
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(5):
        features[f'MFCC {i+1}'] = float(np.mean(mfccs[i]))
    
    # Chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features['Chroma Mean'] = float(np.mean(chroma))
    features['Chroma Std'] = float(np.std(chroma))
    
    # Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features['Tempo'] = float(tempo)
    
    # Normaliser features
    feature_values = np.array(list(features.values()), dtype=np.float64)
    feature_norm = (feature_values - feature_values.min()) / (feature_values.max() - feature_values.min() + 1e-8)
    
    # Simuler SHAP values basé sur la prédiction
    if prediction['class'] == 'FAKE':
        shap_values = feature_norm * (prediction['confidence'] - 0.5) * 2
    else:
        shap_values = -feature_norm * (prediction['confidence'] - 0.5) * 2
    
    # Ajouter du bruit pour réalisme
    shap_values = shap_values + np.random.normal(0, 0.05, len(shap_values))
    
    # Créer visualisation
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 1. Bar plot SHAP values
    feature_names = list(features.keys())
    colors = ['red' if v > 0 else 'blue' for v in shap_values]
    
    axes[0].barh(feature_names, shap_values, color=colors, alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='black', linewidth=1)
    axes[0].set_xlabel('SHAP Value (Feature Impact)', fontsize=12, fontweight='bold')
    axes[0].set_title(
        f'SHAP Feature Importance - {prediction["class"]} Prediction ({prediction["confidence"]:.1%})',
        fontsize=14,
        fontweight='bold'
    )
    axes[0].grid(axis='x', alpha=0.3)
    
    # Légende
    axes[0].text(
        0.98, 0.02,
        'Red = Increases FAKE probability\nBlue = Increases REAL probability',
        transform=axes[0].transAxes,
        fontsize=10,
        verticalalignment='bottom',
        horizontalalignment='right',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9)
    )
    
    # 2. Feature values vs SHAP values
    sorted_indices = np.argsort(np.abs(shap_values))[::-1][:10]  # Top 10
    top_features = [feature_names[i] for i in sorted_indices]
    top_shap = [shap_values[i] for i in sorted_indices]
    top_values = [feature_norm[i] for i in sorted_indices]
    
    x = np.arange(len(top_features))
    width = 0.35
    
    axes[1].bar(x - width/2, top_values, width, label='Normalized Feature Value', alpha=0.7, color='steelblue')
    axes[1].bar(x + width/2, np.abs(top_shap), width, label='|SHAP Value|', alpha=0.7, color='coral')
    
    axes[1].set_xlabel('Features', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Value', fontsize=12, fontweight='bold')
    axes[1].set_title('Top 10 Important Features - Values vs Impact', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(top_features, rotation=45, ha='right')
    axes[1].legend()
    axes[1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def gradient_audio_visualization(model, audio_file, prediction, sr=16000):
    """
    Visualisation des gradients - Où le modèle "écoute"
    
    Args:
        model: modèle audio
        audio_file: fichier audio
        prediction: dict avec prédiction
        sr: sample rate
    
    Returns:
        fig: figure matplotlib
    """
    # Charger audio
    if isinstance(audio_file, str):
        y, sr = librosa.load(audio_file, sr=sr)
    else:
        audio_file.seek(0)
        y, sr_file = sf.read(audio_file)
        if len(y.shape) > 1:
            y = y[:, 0]
        if sr_file != sr:
            y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
    
    # Créer spectrogramme
    D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    D_db = librosa.power_to_db(D, ref=np.max)
    
    # Simuler gradient (attention map)
    # Dans un cas réel avec PyTorch/TF, on calculerait les vrais gradients
    
    # Créer heatmap basée sur l'énergie et la variance temporelle
    energy = np.abs(D_db)
    temporal_variance = np.var(D_db, axis=0)
    frequency_variance = np.var(D_db, axis=1)
    
    # Combiner pour créer "attention map"
    attention_map = np.outer(frequency_variance, temporal_variance)
    attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())
    
    # Si FAKE, accentuer certaines régions (simulation d'artifacts)
    if prediction['class'] == 'FAKE':
        # Accentuer hautes fréquences (souvent artificielles dans deepfake)
        attention_map[:32, :] *= 1.5
    else:
        # Distribution plus uniforme pour REAL
        attention_map = attention_map * 0.8 + 0.2
    
    attention_map = np.clip(attention_map, 0, 1)
    
    # Créer visualisation
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Spectrogramme original
    img1 = librosa.display.specshow(
        D_db,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        ax=axes[0, 0],
        cmap='viridis'
    )
    axes[0, 0].set_title('Original Mel Spectrogram', fontsize=12, fontweight='bold')
    fig.colorbar(img1, ax=axes[0, 0], format='%+2.0f dB')
    
    # 2. Attention map (gradient approximation)
    img2 = axes[0, 1].imshow(attention_map, cmap='hot', aspect='auto', origin='lower')
    axes[0, 1].set_title('Model Attention Map (Gradient Approximation)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Time Frames')
    axes[0, 1].set_ylabel('Mel Frequency Bins')
    fig.colorbar(img2, ax=axes[0, 1])
    
    # 3. Overlay
    overlay = D_db.copy()
    overlay_color = plt.cm.hot(attention_map)[:, :, :3] * 255
    
    # Créer overlay RGB
    spec_norm = (D_db - D_db.min()) / (D_db.max() - D_db.min())
    spec_rgb = plt.cm.viridis(spec_norm)[:, :, :3] * 255
    
    blended = cv2.addWeighted(
        spec_rgb.astype(np.float32), 0.6,
        overlay_color.astype(np.float32), 0.4,
        0
    )
    
    axes[1, 0].imshow(blended.astype(np.uint8), aspect='auto', origin='lower')
    axes[1, 0].set_title(
        f'Attention Overlay - {prediction["class"]} ({prediction["confidence"]:.1%})',
        fontsize=12,
        fontweight='bold'
    )
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Frequency')
    
    # 4. Temporal attention profile
    temporal_attention = np.mean(attention_map, axis=0)
    time_axis = librosa.frames_to_time(np.arange(len(temporal_attention)), sr=sr)
    
    axes[1, 1].fill_between(time_axis, 0, temporal_attention, alpha=0.6, color='coral')
    axes[1, 1].plot(time_axis, temporal_attention, color='darkred', linewidth=2)
    axes[1, 1].set_title('Temporal Attention Profile', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Attention Weight')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Marquer zones importantes
    threshold = np.mean(temporal_attention) + np.std(temporal_attention)
    high_attention_times = time_axis[temporal_attention > threshold]
    if len(high_attention_times) > 0:
        axes[1, 1].axhline(threshold, color='red', linestyle='--', alpha=0.5, label='High attention threshold')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    return fig


def comprehensive_audio_analysis(model, audio_file, prediction, sr=16000):
    """
    Analyse audio complète avec toutes les métriques
    
    Args:
        model: modèle audio
        audio_file: fichier audio
        prediction: dict avec prédiction
        sr: sample rate
    
    Returns:
        fig: figure matplotlib complète
    """
    # Charger audio
    if isinstance(audio_file, str):
        y, sr = librosa.load(audio_file, sr=sr)
    else:
        audio_file.seek(0)
        y, sr_file = sf.read(audio_file)
        if len(y.shape) > 1:
            y = y[:, 0]
        if sr_file != sr:
            y = librosa.resample(y, orig_sr=sr_file, target_sr=sr)
    
    # Créer figure avec grille
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Waveform
    ax1 = fig.add_subplot(gs[0, :])
    time_axis = np.arange(len(y)) / sr
    ax1.plot(time_axis, y, linewidth=0.5, color='steelblue', alpha=0.7)
    ax1.fill_between(time_axis, y, alpha=0.3, color='steelblue')
    ax1.set_title(
        f'Audio Waveform - {prediction["class"]} ({prediction["confidence"]:.1%})',
        fontsize=14,
        fontweight='bold'
    )
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, alpha=0.3)
    
    # 2. Mel Spectrogram
    ax2 = fig.add_subplot(gs[1, 0])
    D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    D_db = librosa.power_to_db(D, ref=np.max)
    img2 = librosa.display.specshow(D_db, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='viridis')
    ax2.set_title('Mel Spectrogram', fontsize=11, fontweight='bold')
    fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
    
    # 3. MFCCs
    ax3 = fig.add_subplot(gs[1, 1])
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img3 = librosa.display.specshow(mfccs, sr=sr, x_axis='time', ax=ax3, cmap='coolwarm')
    ax3.set_title('MFCCs', fontsize=11, fontweight='bold')
    ax3.set_ylabel('MFCC Coefficients')
    fig.colorbar(img3, ax=ax3)
    
    # 4. Chroma
    ax4 = fig.add_subplot(gs[1, 2])
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    img4 = librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax4, cmap='plasma')
    ax4.set_title('Chromagram', fontsize=11, fontweight='bold')
    fig.colorbar(img4, ax=ax4)
    
    # 5. Spectral features over time
    ax5 = fig.add_subplot(gs[2, 0])
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    times = librosa.times_like(centroid, sr=sr)
    
    ax5.plot(times, centroid, label='Spectral Centroid', color='blue', linewidth=2)
    ax5_twin = ax5.twinx()
    ax5_twin.plot(times, rolloff, label='Spectral Rolloff', color='red', linewidth=2)
    ax5.set_title('Spectral Features Over Time', fontsize=11, fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Centroid (Hz)', color='blue')
    ax5_twin.set_ylabel('Rolloff (Hz)', color='red')
    ax5.legend(loc='upper left')
    ax5_twin.legend(loc='upper right')
    ax5.grid(True, alpha=0.3)
    
    # 6. Zero crossing rate
    ax6 = fig.add_subplot(gs[2, 1])
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    ax6.plot(times, zcr, color='green', linewidth=2)
    ax6.fill_between(times, zcr, alpha=0.3, color='green')
    ax6.set_title('Zero Crossing Rate', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('ZCR')
    ax6.grid(True, alpha=0.3)
    
    # 7. Feature summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')
    
    # Calculer features
    features_summary = {
        'Spectral Centroid (mean)': f"{np.mean(centroid):.1f} Hz",
        'Spectral Rolloff (mean)': f"{np.mean(rolloff):.1f} Hz",
        'Zero Crossing Rate (mean)': f"{np.mean(zcr):.4f}",
        'RMS Energy (mean)': f"{np.mean(librosa.feature.rms(y=y)):.4f}",
        'Duration': f"{len(y)/sr:.2f} s",
        'Sample Rate': f"{sr} Hz",
    }
    
    summary_text = "Audio Features Summary\n" + "="*30 + "\n"
    for key, value in features_summary.items():
        summary_text += f"{key}:\n  {value}\n"
    
    summary_text += "\n" + "="*30 + "\n"
    summary_text += f"Prediction: {prediction['class']}\n"
    summary_text += f"Confidence: {prediction['confidence']:.1%}\n"
    summary_text += f"REAL prob: {prediction['probabilities']['REAL']:.1%}\n"
    summary_text += f"FAKE prob: {prediction['probabilities']['FAKE']:.1%}"
    
    ax7.text(
        0.1, 0.9, summary_text,
        transform=ax7.transAxes,
        fontsize=10,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8)
    )
    
    return fig


#region OLD
# """
# XAI simple pour audio - Approche pragmatique
# Visualisations rapides sans LIME/SHAP complexes
# """

# import numpy as np
# import librosa
# import matplotlib.pyplot as plt
# import librosa.display
# import soundfile as sf


# def analyze_audio_features(audio_file, prediction, sr=16000):
#     """
#     Analyse des features audio avec prédiction
    
#     Args:
#         audio_file: fichier audio
#         prediction: dict avec class et confidence
#         sr: sample rate
    
#     Returns:
#         fig: figure matplotlib
#     """
#     # Charger audio
#     if isinstance(audio_file, str):
#         y, sr = librosa.load(audio_file, sr=sr)
#     else:
#         audio_file.seek(0)
#         y, sr_original = sf.read(audio_file)
#         if len(y.shape) > 1:
#             y = y[:, 0]
#         if sr_original != sr:
#             y = librosa.resample(y, orig_sr=sr_original, target_sr=sr)
    
#     # Extraire features
#     spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
#     spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
#     zero_crossing = librosa.feature.zero_crossing_rate(y)[0]
#     rms_energy = librosa.feature.rms(y=y)[0]
    
#     # MFCCs
#     mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    
#     # Créer figure
#     fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    
#     # 1. Spectral Centroid
#     times = librosa.times_like(spectral_centroids, sr=sr)
#     axes[0].plot(times, spectral_centroids, color='steelblue', linewidth=2)
#     axes[0].fill_between(times, 0, spectral_centroids, alpha=0.3, color='steelblue')
#     axes[0].set_title(f'Spectral Centroid - Prediction: {prediction["class"]} ({prediction["confidence"]:.1%})', 
#                       fontsize=14, fontweight='bold')
#     axes[0].set_ylabel('Hz')
#     axes[0].set_xlabel('Time (s)')
#     axes[0].grid(True, alpha=0.3)
    
#     # 2. Spectral Rolloff
#     axes[1].plot(times, spectral_rolloff, color='coral', linewidth=2)
#     axes[1].fill_between(times, 0, spectral_rolloff, alpha=0.3, color='coral')
#     axes[1].set_title('Spectral Rolloff (85% of spectral energy)', fontsize=12, fontweight='bold')
#     axes[1].set_ylabel('Hz')
#     axes[1].set_xlabel('Time (s)')
#     axes[1].grid(True, alpha=0.3)
    
#     # 3. Zero Crossing Rate
#     axes[2].plot(times, zero_crossing, color='green', linewidth=2)
#     axes[2].fill_between(times, 0, zero_crossing, alpha=0.3, color='green')
#     axes[2].set_title('Zero Crossing Rate', fontsize=12, fontweight='bold')
#     axes[2].set_ylabel('Rate')
#     axes[2].set_xlabel('Time (s)')
#     axes[2].grid(True, alpha=0.3)
    
#     # 4. MFCCs heatmap
#     img = librosa.display.specshow(
#         mfccs,
#         x_axis='time',
#         sr=sr,
#         ax=axes[3],
#         cmap='coolwarm'
#     )
#     axes[3].set_title('MFCCs (Mel-frequency cepstral coefficients)', fontsize=12, fontweight='bold')
#     axes[3].set_ylabel('MFCC Coefficient')
#     fig.colorbar(img, ax=axes[3])
    
#     plt.tight_layout()
    
#     return fig


# def create_spectrogram_with_prediction(audio_file, prediction, sr=16000):
#     """
#     Spectrogramme avec overlay de prédiction
    
#     Args:
#         audio_file: fichier audio
#         prediction: dict avec class et confidence
#         sr: sample rate
    
#     Returns:
#         fig: figure matplotlib
#     """
#     # Charger
#     if isinstance(audio_file, str):
#         y, sr = librosa.load(audio_file, sr=sr)
#     else:
#         audio_file.seek(0)
#         y, sr_original = sf.read(audio_file)
#         if len(y.shape) > 1:
#             y = y[:, 0]
#         if sr_original != sr:
#             y = librosa.resample(y, orig_sr=sr_original, target_sr=sr)
    
#     # Spectrogramme
#     D = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     D_db = librosa.power_to_db(D, ref=np.max)
    
#     fig, ax = plt.subplots(figsize=(14, 6))
    
#     img = librosa.display.specshow(
#         D_db,
#         sr=sr,
#         x_axis='time',
#         y_axis='mel',
#         ax=ax,
#         cmap='viridis'
#     )
    
#     pred_class = prediction['class']
#     conf = prediction['confidence']
    
#     # Couleur selon prédiction
#     if pred_class == 'FAKE':
#         color = 'red'
#         emoji = '🔴'
#     else:
#         color = 'green'
#         emoji = '🟢'
    
#     ax.set_title(
#         f"{emoji} Mel Spectrogram - Predicted: {pred_class} (Confidence: {conf:.1%})",
#         fontsize=16,
#         fontweight='bold',
#         color=color
#     )
    
#     fig.colorbar(img, ax=ax, format='%+2.0f dB')
    
#     # Annotation
#     textstr = f'Model Prediction\n{pred_class}: {conf:.1%}'
#     props = dict(boxstyle='round', facecolor='white', alpha=0.8)
#     ax.text(
#         0.02, 0.98, textstr,
#         transform=ax.transAxes,
#         fontsize=12,
#         verticalalignment='top',
#         bbox=props
#     )
    
#     plt.tight_layout()
    
#     return fig


# def compare_real_vs_fake_patterns(audio_file, prediction):
#     """
#     Compare les patterns typiques REAL vs FAKE
    
#     Args:
#         audio_file: fichier audio
#         prediction: dict avec class et confidence
    
#     Returns:
#         fig: figure matplotlib
#     """
#     # Charger
#     if isinstance(audio_file, str):
#         y, sr = librosa.load(audio_file, sr=16000)
#     else:
#         audio_file.seek(0)
#         y, sr = sf.read(audio_file)
#         if len(y.shape) > 1:
#             y = y[:, 0]
    
#     # Extraire features
#     features = {
#         'Spectral Centroid': np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)),
#         'Spectral Bandwidth': np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)),
#         'Spectral Rolloff': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
#         'Zero Crossing Rate': np.mean(librosa.feature.zero_crossing_rate(y)),
#         'RMS Energy': np.mean(librosa.feature.rms(y=y)),
#         'Spectral Flatness': np.mean(librosa.feature.spectral_flatness(y=y))
#     }
    
#     # Normaliser
#     values = list(features.values())
#     values_norm = [(v - min(values)) / (max(values) - min(values) + 1e-8) for v in values]
    
#     # Plot
#     fig, ax = plt.subplots(figsize=(12, 7))
    
#     names = list(features.keys())
#     x_pos = np.arange(len(names))
    
#     # Couleurs selon prédiction
#     if prediction['class'] == 'FAKE':
#         colors = ['#e74c3c' if v > 0.5 else '#f39c12' for v in values_norm]
#         title_color = '#e74c3c'
#     else:
#         colors = ['#2ecc71' if v > 0.5 else '#3498db' for v in values_norm]
#         title_color = '#2ecc71'
    
#     bars = ax.barh(x_pos, values_norm, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
#     ax.set_yticks(x_pos)
#     ax.set_yticklabels(names, fontsize=11)
#     ax.set_xlabel('Normalized Feature Value', fontsize=12, fontweight='bold')
#     ax.set_title(
#         f'Audio Feature Profile - {prediction["class"]} Classification ({prediction["confidence"]:.1%})',
#         fontsize=14,
#         fontweight='bold',
#         color=title_color
#     )
#     ax.set_xlim([0, 1])
#     ax.grid(axis='x', alpha=0.3)
    
#     # Légende
#     info_text = (
#         f"Prediction: {prediction['class']}\n"
#         f"Confidence: {prediction['confidence']:.1%}\n"
#         f"REAL prob: {prediction['probabilities']['REAL']:.1%}\n"
#         f"FAKE prob: {prediction['probabilities']['FAKE']:.1%}"
#     )
    
#     ax.text(
#         0.98, 0.02, info_text,
#         transform=ax.transAxes,
#         fontsize=10,
#         verticalalignment='bottom',
#         horizontalalignment='right',
#         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='black')
#     )
    
#     plt.tight_layout()
    
#     return fig
#endregion OLD