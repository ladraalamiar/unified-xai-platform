"""
Chargeur unifié pour modèles audio - VERSION CORRIGÉE
- Modèle 1: HuggingFace wav2vec2
- Modèle 2: TensorFlow SavedModel (avec fix Keras 3)
"""

import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# ==================== MODÈLE 1: HUGGINGFACE ====================

class HuggingFaceAudioModel:
    """Wrapper pour modèle HuggingFace wav2vec2"""
    
    def __init__(self):
        self.name = "wav2vec2 (HuggingFace)"
        self.pipeline = None
        self.load_model()
    
    def load_model(self):
        """Charge le modèle depuis HuggingFace"""
        try:
            print("Loading HuggingFace model...")
            self.pipeline = pipeline(
                "audio-classification",
                model="Gustking/wav2vec2-large-xlsr-deepfake-audio-classification"
            )
            print("✅ HuggingFace model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading HuggingFace model: {e}")
            print("💡 Try: pip install tf-keras")
            self.pipeline = None
    
    def preprocess(self, audio_file, target_sr=16000):
        """Prétraite audio pour wav2vec2"""
        import tempfile
        
        if isinstance(audio_file, str):
            return audio_file
        else:
            audio_file.seek(0)
            y, sr = sf.read(audio_file)
            
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                sf.write(tmp.name, y, target_sr)
                return tmp.name
    
    def predict(self, audio_file):
        """Prédiction avec wav2vec2"""
        if self.pipeline is None:
            raise RuntimeError("Model not loaded! Install tf-keras first.")
        
        audio_path = self.preprocess(audio_file)
        results = self.pipeline(audio_path)
        
        # Parser résultats HuggingFace
        probs = {item['label'].upper(): item['score'] for item in results}
        
        prob_fake = probs.get('FAKE', probs.get('SPOOF', 0.0))
        prob_real = probs.get('REAL', probs.get('BONA-FIDE', 1.0 - prob_fake))
        
        predicted_class = 'FAKE' if prob_fake > prob_real else 'REAL'
        
        return {
            'class': predicted_class,
            'confidence': max(prob_real, prob_fake),
            'probabilities': {
                'REAL': float(prob_real),
                'FAKE': float(prob_fake)
            },
            'raw_output': results
        }


# ==================== MODÈLE 2: TENSORFLOW (CORRIGÉ) ====================

class TensorFlowLocalModel:
    """Wrapper pour TensorFlow SavedModel - COMPATIBLE KERAS 3"""
    
    def __init__(self, model_path='models/audio/.pb'):
        self.name = "TensorFlow Local"
        self.model_path = model_path
        self.model = None
        self.infer_fn = None
        self.load_model()
    
    def load_model(self):
        """Charge SavedModel avec méthode compatible Keras 3"""
        try:
            print(f"Loading TensorFlow model from {self.model_path}...")
            
            # Charger avec tf.saved_model.load (marche avec Keras 3)
            loaded = tf.saved_model.load(self.model_path)
            
            # Extraire la fonction d'inférence
            self.infer_fn = loaded.signatures["serving_default"]
            
            print("✅ TensorFlow model loaded successfully!")
            print(f"   Input spec: {self.infer_fn.structured_input_signature[1]}")
            print(f"   Output spec: {list(self.infer_fn.structured_outputs.keys())}")
            
            self.model = loaded  # Garder référence
            
        except Exception as e:
            print(f"❌ Error loading TensorFlow model: {e}")
            self.model = None
            self.infer_fn = None
    
    def preprocess(self, audio_file, target_sr=16000, duration=3.0):
        """
        Prétraite audio pour modèle TensorFlow
        Output shape doit être (batch, height, width, 3) selon le diagnostic
        """
        # Charger audio
        if isinstance(audio_file, str):
            y, sr = librosa.load(audio_file, sr=target_sr)
        else:
            audio_file.seek(0)
            y, sr = sf.read(audio_file)
            if sr != target_sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
        # Mono
        if len(y.shape) > 1:
            y = librosa.to_mono(y)
        
        # Tronquer/padder
        target_length = int(target_sr * duration)
        if len(y) > target_length:
            y = y[:target_length]
        else:
            y = np.pad(y, (0, target_length - len(y)))
        
        # Spectrogramme Mel
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=target_sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512
        )
        
        # dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normaliser (0-1)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Le modèle attend (batch, height, width, 3) = RGB
        # mel_spec_norm shape: (mels, time) = (128, ~94)
        
        # Transposer d'abord : (time, mels)
        mel_spec_t = mel_spec_norm.T  # (time, mels) ex: (94, 128)
        
        # Dupliquer en 3 canaux RGB : (time, mels, 3)
        mel_spec_rgb = np.stack([mel_spec_t, mel_spec_t, mel_spec_t], axis=-1)
        
        # Ajouter batch dimension : (1, time, mels, 3)
        mel_spec_rgb = np.expand_dims(mel_spec_rgb, axis=0)  # (1, 94, 128, 3)
        
        # Convertir en tensor TensorFlow
        return tf.constant(mel_spec_rgb, dtype=tf.float32)
    
    def predict(self, audio_file):
        """Prédiction avec TensorFlow"""
        if self.infer_fn is None:
            raise RuntimeError("Model not loaded!")
        
        # Prétraiter
        input_tensor = self.preprocess(audio_file)
        
        # Prédiction avec la bonne clé d'input
        try:
            predictions = self.infer_fn(input_1=input_tensor)
            
            # Extraire output (clé: 'dense_1' selon diagnostic)
            output = predictions['dense_1'].numpy()
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            raise RuntimeError(f"Could not run inference: {e}")
        
        # Parser output (2 classes)
        if output.shape[-1] == 2:
            prob_real = float(output[0][0])
            prob_fake = float(output[0][1])
        else:
            raise ValueError(f"Unexpected output shape: {output.shape}")
        
        predicted_class = 'FAKE' if prob_fake > prob_real else 'REAL'
        
        return {
            'class': predicted_class,
            'confidence': max(prob_real, prob_fake),
            'probabilities': {
                'REAL': prob_real,
                'FAKE': prob_fake
            },
            'raw_output': output
        }


# ==================== INTERFACE UNIFIÉE ====================

def load_audio_model(model_name):
    """
    Charge un modèle audio
    
    Args:
        model_name: 'huggingface', 'wav2vec2', 'model1' OU 
                    'tensorflow', 'local', 'model2'
    
    Returns:
        model wrapper
    """
    model_name_lower = model_name.lower()
    
    if model_name_lower in ['huggingface', 'wav2vec2', 'model1', 'hf']:
        return HuggingFaceAudioModel()
    
    elif model_name_lower in ['tensorflow', 'local', 'model2', 'tf']:
        return TensorFlowLocalModel()
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def predict_audio(model, audio_file):
    """Interface unifiée pour prédiction"""
    return model.predict(audio_file)


def get_available_models():
    """Retourne modèles disponibles"""
    return {
        'Model 1 (wav2vec2 HuggingFace)': 'huggingface',
        'Model 2 (TensorFlow Local)': 'tensorflow'
    }


# ==================== TEST ====================

if __name__ == "__main__":
    print("="*70)
    print("🔍 TESTING AUDIO MODELS")
    print("="*70)
    
    # Test Model 1
    print("\n[1/2] Testing HuggingFace...")
    try:
        model1 = load_audio_model('huggingface')
        if model1.pipeline:
            print(f"✅ Model 1: {model1.name}")
        else:
            print("⚠️ Model 1 loaded but needs tf-keras")
    except Exception as e:
        print(f"❌ Model 1: {e}")
    
    # Test Model 2
    print("\n[2/2] Testing TensorFlow...")
    try:
        model2 = load_audio_model('tensorflow')
        if model2.infer_fn:
            print(f"✅ Model 2: {model2.name}")
        else:
            print("❌ Model 2: Failed to load")
    except Exception as e:
        print(f"❌ Model 2: {e}")
    
    # Test prédiction
    test_audio = 'models/audio/.pb/wav_de_test/test_audio.wav'
    if os.path.exists(test_audio):
        print(f"\n[BONUS] Testing with {test_audio}...")
        
        if 'model2' in locals() and model2.infer_fn:
            try:
                result = model2.predict(test_audio)
                print(f"Model 2: {result['class']} ({result['confidence']:.2%})")
            except Exception as e:
                print(f"Prediction failed: {e}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print("="*70)