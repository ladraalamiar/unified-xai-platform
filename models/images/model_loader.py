"""
Chargeur pour PLUSIEURS modèles de classification d'images X-ray
Tous pré-entraînés sur datasets médicaux (CheXpert, MIMIC-CXR, NIH, etc.)
"""

import torch
import torchxrayvision as xrv
import numpy as np
import cv2
from PIL import Image


class ChestXRayModel:
    """Classe de base pour modèles X-ray"""
    
    def __init__(self, model_name, weights):
        self.model_name = model_name
        self.weights = weights
        self.name = f"{model_name} ({weights})"
        self.model = None
        self.pathologies = None
        self.load_model()
    
    def load_model(self):
        """Charge le modèle spécifié"""
        try:
            print(f"Loading {self.model_name} with weights: {self.weights}...")
            
            # Charger selon le type de modèle
            if 'densenet' in self.model_name.lower():
                self.model = xrv.models.DenseNet(weights=self.weights)
            elif 'resnet' in self.model_name.lower():
                self.model = xrv.models.ResNet(weights=self.weights)
            else:
                raise ValueError(f"Unknown model type: {self.model_name}")
            
            self.model.eval()
            self.pathologies = self.model.pathologies
            
            print(f"✅ {self.name} loaded successfully!")
            print(f"   Pathologies: {len(self.pathologies)} classes")
            
        except Exception as e:
            print(f"❌ Error loading {self.name}: {e}")
            self.model = None
    
    def preprocess(self, image):
        """Prétraite une image pour le modèle"""
        # Convertir en array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Grayscale
        if len(img_array.shape) == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Normaliser
        img_array = xrv.datasets.normalize(img_array, 255)
        
        # Resize à 224x224
        img_array = cv2.resize(img_array, (224, 224))
        
        # Tensor
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()
        
        return img_tensor
    
    def predict(self, image):
        """Prédiction"""
        if self.model is None:
            raise RuntimeError(f"Model {self.name} not loaded!")
        
        # Charger si path
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        
        # Prétraiter
        img_tensor = self.preprocess(image)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
        
        # Top 5
        top_indices = probabilities.argsort()[-5:][::-1]
        
        return {
            'model_name': self.name,
            'top_predictions': [
                {
                    'pathology': self.pathologies[idx],
                    'probability': float(probabilities[idx])
                }
                for idx in top_indices
            ],
            'all_probabilities': {
                pathology: float(prob)
                for pathology, prob in zip(self.pathologies, probabilities)
            },
            'top_pathology': self.pathologies[top_indices[0]],
            'top_probability': float(probabilities[top_indices[0]]),
            'raw_output': probabilities
        }


# ==================== MODÈLES DISPONIBLES ====================

AVAILABLE_MODELS = {
    # DenseNet models
    'densenet121-res224-all': {
        'class': 'DenseNet121',
        'weights': 'densenet121-res224-all',
        'description': 'DenseNet121 trained on all datasets (CheXpert, MIMIC, NIH, PadChest)',
        'datasets': ['CheXpert', 'MIMIC-CXR', 'NIH', 'PadChest']
    },
    'densenet121-res224-mimic_ch': {
        'class': 'DenseNet121',
        'weights': 'densenet121-res224-mimic_ch',
        'description': 'DenseNet121 trained on MIMIC-CXR and CheXpert',
        'datasets': ['MIMIC-CXR', 'CheXpert']
    },
    'densenet121-res224-mimic_nb': {
        'class': 'DenseNet121',
        'weights': 'densenet121-res224-mimic_nb',
        'description': 'DenseNet121 trained on MIMIC-CXR (NIH-Biobank)',
        'datasets': ['MIMIC-CXR']
    },
    'densenet121-res224-nih': {
        'class': 'DenseNet121',
        'weights': 'densenet121-res224-nih',
        'description': 'DenseNet121 trained on NIH ChestX-ray14',
        'datasets': ['NIH ChestX-ray14']
    },
    'densenet121-res224-pc': {
        'class': 'DenseNet121',
        'weights': 'densenet121-res224-pc',
        'description': 'DenseNet121 trained on PadChest',
        'datasets': ['PadChest']
    },
    'densenet121-res224-chex': {
        'class': 'DenseNet121',
        'weights': 'densenet121-res224-chex',
        'description': 'DenseNet121 trained on CheXpert',
        'datasets': ['CheXpert']
    },
    
    # ResNet models
    'resnet50-res512-all': {
        'class': 'ResNet50',
        'weights': 'resnet50-res512-all',
        'description': 'ResNet50 trained on all datasets',
        'datasets': ['CheXpert', 'MIMIC-CXR', 'NIH', 'PadChest']
    }
}


# ==================== INTERFACE UNIFIÉE ====================

def load_image_model(model_key):
    """
    Charge un modèle de classification X-ray
    
    Args:
        model_key: clé du modèle dans AVAILABLE_MODELS
    
    Returns:
        ChestXRayModel instance
    """
    if model_key not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_key}. Available: {list(AVAILABLE_MODELS.keys())}")
    
    model_info = AVAILABLE_MODELS[model_key]
    
    return ChestXRayModel(
        model_name=model_info['class'],
        weights=model_info['weights']
    )


def predict_image(model, image):
    """Interface unifiée pour prédiction"""
    return model.predict(image)


def get_available_models():
    """
    Retourne les modèles disponibles avec descriptions
    
    Returns:
        dict: {display_name: model_key}
    """
    return {
        f"{info['class']} - {', '.join(info['datasets'][:2])}": key
        for key, info in AVAILABLE_MODELS.items()
    }


def get_model_info(model_key):
    """Retourne les infos détaillées d'un modèle"""
    if model_key in AVAILABLE_MODELS:
        return AVAILABLE_MODELS[model_key]
    return None


# ==================== MODÈLES RECOMMANDÉS ====================

RECOMMENDED_MODELS = {
    'best_overall': 'densenet121-res224-all',
    'fastest': 'densenet121-res224-chex',
    'most_accurate': 'densenet121-res224-all'
}


def get_recommended_models():
    """Retourne les modèles recommandés pour l'interface"""
    return {
        'Model 1: DenseNet121 (All Datasets)': 'densenet121-res224-all',
        'Model 2: DenseNet121 (CheXpert + MIMIC)': 'densenet121-res224-mimic_ch',
        'Model 3: DenseNet121 (NIH ChestX-ray14)': 'densenet121-res224-nih'
    }


# ==================== TEST ====================

if __name__ == "__main__":
    print("="*70)
    print("🔍 TESTING MULTIPLE IMAGE MODELS")
    print("="*70)
    
    print(f"\n📋 Available models: {len(AVAILABLE_MODELS)}")
    for key, info in AVAILABLE_MODELS.items():
        print(f"  - {info['class']}: {info['description']}")
    
    # Test quelques modèles
    test_models = [
        'densenet121-res224-all',
        'densenet121-res224-mimic_ch',
        'densenet121-res224-nih'
    ]
    
    loaded_models = []
    
    for model_key in test_models:
        print(f"\n[Testing] {model_key}")
        try:
            model = load_image_model(model_key)
            if model.model:
                print(f"✅ {model.name} loaded")
                print(f"   Pathologies: {model.pathologies[:5]}...")
                loaded_models.append((model_key, model))
            else:
                print(f"❌ {model_key} failed")
        except Exception as e:
            print(f"❌ Error with {model_key}: {e}")
    
    # Test prédiction si image disponible
    import os
    test_images = [
        'Lamia/image/test_lung.jpg',
        'Lamia/image/test_lung511.jpg',
        'models/images/test.jpg'
    ]
    
    found_test = None
    for test_path in test_images:
        if os.path.exists(test_path):
            found_test = test_path
            break
    
    if found_test and loaded_models:
        print(f"\n{'='*70}")
        print(f"🧪 TESTING PREDICTIONS with: {found_test}")
        print("="*70)
        
        for model_key, model in loaded_models:
            print(f"\n[{model.name}]")
            try:
                result = model.predict(found_test)
                
                print("Top 3 predictions:")
                for i, pred in enumerate(result['top_predictions'][:3], 1):
                    print(f"  {i}. {pred['pathology']}: {pred['probability']:.2%}")
                
            except Exception as e:
                print(f"❌ Prediction failed: {e}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE")
    print(f"Loaded {len(loaded_models)}/{len(test_models)} models successfully")
    print("="*70)