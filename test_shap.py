"""
Test de la version corrigée de shap_image.py
"""

import sys
import torch
import numpy as np
from PIL import Image

print("=" * 60)
print("TEST: Version Corrigée de SHAP")
print("=" * 60)

# Créer une image de test
test_image = Image.fromarray(np.random.randint(0, 255, (224, 224), dtype=np.uint8))
print("✅ Image de test créée: 224x224")

# Créer un modèle factice qui simule torchxrayvision
class DummyXRayModel:
    def __init__(self):
        # Modèle qui attend (batch, 1, 224, 224)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(16, 18)
        )
        self.pathologies = ['Pathology_' + str(i) for i in range(18)]
    
dummy_model = DummyXRayModel()
print("✅ Modèle factice créé (simule torchxrayvision)")

# Tester la fonction
print("\n" + "=" * 60)
print("Test 1: explain_image_with_shap avec GradientExplainer")
print("=" * 60)

try:
    # Importer la version CORRIGÉE
    import XAI_methods.shap_image as shap_img
    
    result = shap_img.explain_image_with_shap(
        dummy_model, 
        test_image, 
        target_pathology_idx=0,
        num_samples=10
    )
    
    print("✅ explain_image_with_shap: SUCCESS!")
    print(f"   Keys: {result.keys()}")
    print(f"   SHAP values shape: {result['shap_values'].shape}")
    print(f"   Image array shape: {result['image_array'].shape}")
    print(f"   Target idx: {result['target_idx']}")
    
except Exception as e:
    print(f"❌ explain_image_with_shap: FAILED")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

# Test visualisation
print("\n" + "=" * 60)
print("Test 2: visualize_shap_image")
print("=" * 60)

try:
    import matplotlib
    matplotlib.use('Agg')  # Backend non-interactif
    import XAI_methods.shap_image as shap_img

    fig = shap_img.visualize_shap_image(result, "Test Pathology", original_image=test_image)
    print("✅ visualize_shap_image: SUCCESS!")
    print(f"   Figure created with {len(fig.axes)} axes")
    
except Exception as e:
    print(f"❌ visualize_shap_image: FAILED")
    print(f"   Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("RÉSUMÉ")
print("=" * 60)
print("✅ Si les tests passent, remplacez XAI_methods/shap_image.py")
print("   par le contenu de shap_image_fixed.py")
print("=" * 60)