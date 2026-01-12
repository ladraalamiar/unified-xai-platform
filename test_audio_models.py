
import sys
import os
sys.path.insert(0, 'models')

print("="*70)
print(" TESTING AUDIO MODELS")
print("="*70)

# Test imports
print("\n [1/4] Testing imports...")
try:
    import tensorflow as tf
    print("TensorFlow:", tf.__version__)
except Exception as e:
    print(f" TensorFlow: {e}")

try:
    import transformers
    print(" Transformers:", transformers.__version__)
except Exception as e:
    print(f" Transformers: {e}")

try:
    import librosa
    print(" Librosa:", librosa.__version__)
except Exception as e:
    print(f" Librosa: {e}")

# Test model loader
print("\n[2/4] Testing model loader...")
try:
    from audio.model_loader import load_audio_model, get_available_models
    print(" Model loader imported successfully")
    
    available = get_available_models()
    print(f"   Available models: {list(available.keys())}")
except Exception as e:
    print(f" Model loader: {e}")
    import traceback
    traceback.print_exc()

# Test Model 1 (HuggingFace)
print("\n[3/4] Testing Model 1 (HuggingFace wav2vec2)...")
try:
    model1 = load_audio_model('huggingface')
    if model1 and model1.pipeline:
        print(f" Model 1 loaded: {model1.name}")
        print(f"   Pipeline ready: {model1.pipeline is not None}")
    else:
        print(" Model 1 loaded but pipeline is None")
except Exception as e:
    print(f" Model 1 failed: {e}")
    import traceback
    traceback.print_exc()

# Test Model 2 (TensorFlow)
print("\n[4/4] Testing Model 2 (TensorFlow Local)...")
try:
    model2 = load_audio_model('tensorflow')
    if model2 and model2.model:
        print(f"Model 2 loaded: {model2.name}")
        print(f"   Model ready: {model2.model is not None}")
    else:
        print(" Model 2 loaded but model is None")
except Exception as e:
    print(f"Model 2 failed: {e}")
    import traceback
    traceback.print_exc()

# Test avec fichier audio
test_audio_files = [
    'models/audio/.pb/wav_de_test/test_audio.wav',
    'models/audio/.pb/wav_de_test/test_audio2222.wav'
]

found_test = None
for test_file in test_audio_files:
    if os.path.exists(test_file):
        found_test = test_file
        break

if found_test:
    print(f"\n[BONUS] Testing predictions with: {found_test}")
    print("-"*70)
    
    # Test Model 1
    if 'model1' in locals() and model1.pipeline:
        try:
            print("\nModel 1 (HuggingFace) prediction:")
            result1 = model1.predict(found_test)
            print(f"  Prediction: {result1['class']}")
            print(f"  Confidence: {result1['confidence']:.2%}")
            print(f"  REAL: {result1['probabilities']['REAL']:.2%}")
            print(f"  FAKE: {result1['probabilities']['FAKE']:.2%}")
        except Exception as e:
            print(f"  Prediction failed: {e}")
    
    # Test Model 2
    if 'model2' in locals() and model2.model:
        try:
            print("\nModel 2 (TensorFlow) prediction:")
            result2 = model2.predict(found_test)
            print(f"  Prediction: {result2['class']}")
            print(f"  Confidence: {result2['confidence']:.2%}")
            print(f"  REAL: {result2['probabilities']['REAL']:.2%}")
            print(f"  FAKE: {result2['probabilities']['FAKE']:.2%}")
        except Exception as e:
            print(f"  Prediction failed: {e}")
            import traceback
            traceback.print_exc()
else:
    print("\n No test audio files found")
    print("   Add .wav files to: models/audio/.pb/wav_de_test/")

# Summary
print("\n" + "="*70)
print("TEST SUMMARY")
print("="*70)

checks = {
    'TensorFlow installed': 'tf' in dir(),
    'Transformers installed': 'transformers' in dir(),
    'Model loader works': 'load_audio_model' in dir(),
    'Model 1 (HF) loaded': 'model1' in locals() and model1 is not None,
    'Model 2 (TF) loaded': 'model2' in locals() and model2 is not None,
    'Test predictions': found_test is not None
}

for check, status in checks.items():
    icon = "YES" if status else "NO"
    print(f"{icon} {check}")

print("\n🚀 Next Steps:")
if all(checks.values()):
    print("All checks passed! Ready to run:")
    print("   streamlit run app.py")
else:
    print("Some checks failed. Review errors above.")
    print("   Fix issues then run: python test_audio_models.py")

print("="*70)