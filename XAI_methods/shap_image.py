"""
SHAP (SHapley Additive exPlanations) pour images X-ray
Version corrigée pour torchxrayvision
"""

import torch
import shap
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2


def explain_image_with_shap(model, image, target_pathology_idx=None, num_samples=50):
    """
    Génère une explication SHAP pour une image X-ray
    
    Args:
        model: modèle PyTorch (torchxrayvision)
        image: PIL Image ou numpy array
        target_pathology_idx: index de la pathologie cible (None = top prediction)
        num_samples: nombre d'échantillons pour SHAP (50-100 recommandé)
    
    Returns:
        dict: {'shap_values': array, 'image_array': array, 'explanation': shap.Explanation}
    """
    
    # Préparer l'image
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image
    
    # Grayscale
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # Normaliser et resize
    import torchxrayvision as xrv
    img_array = xrv.datasets.normalize(img_array, 255)
    img_array = cv2.resize(img_array, (224, 224))
    
    # Tensor avec la bonne shape pour torchxrayvision: (1, 1, 224, 224)
    img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()
    
    # Créer background (moyenne d'images avec bruit)
    # Pour simplifier, on utilise plusieurs versions bruitées de l'image
    background_samples = []
    for _ in range(min(5, num_samples)):
        noisy = img_tensor + torch.randn_like(img_tensor) * 0.1
        noisy = torch.clamp(noisy, img_tensor.min(), img_tensor.max())
        background_samples.append(noisy)
    
    background = torch.cat(background_samples, dim=0)
    
    # Si target_pathology_idx non spécifié, utiliser le top
    if target_pathology_idx is None:
        with torch.no_grad():
            outputs = model.model(img_tensor)
            probs = torch.sigmoid(outputs)
            target_pathology_idx = probs.argmax().item()
    
    try:
        # Méthode 1: GradientExplainer
        print("Trying GradientExplainer...")
        
        # Créer un wrapper pour le modèle qui retourne seulement la sortie cible
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model, target_idx):
                super().__init__()
                self.model = model
                self.target_idx = target_idx
            
            def forward(self, x):
                # S'assurer que x a la bonne shape (batch, 1, 224, 224)
                if len(x.shape) == 3:
                    x = x.unsqueeze(1)
                
                outputs = self.model(x)
                # Retourner seulement la sortie pour la pathologie cible
                return outputs[:, self.target_idx:self.target_idx+1]
        
        wrapped_model = ModelWrapper(model.model, target_pathology_idx)
        
        explainer = shap.GradientExplainer(
            wrapped_model,
            background
        )
        
        # Calculer SHAP values
        shap_values = explainer.shap_values(img_tensor, nsamples=num_samples)
        
        # shap_values devrait être de shape (1, 1, 224, 224)
        if isinstance(shap_values, list):
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values
        
        return {
            'shap_values': shap_vals,
            'image_array': img_tensor.squeeze().cpu().numpy(),
            'target_idx': target_pathology_idx,
            'explainer': explainer
        }
    
    except Exception as e:
        print(f"Error with GradientExplainer: {e}")
        
        try:
            # Méthode 2: Utiliser Kernel SHAP (plus lent mais plus robuste)
            print("Trying KernelExplainer as fallback...")
            
            # Fonction de prédiction pour KernelSHAP
            def predict_fn(x):
                """
                x: array de shape (n_samples, n_features) 
                   où n_features = 1 * 224 * 224 = 50176
                """
                batch = []
                for sample in x:
                    # Reshape de (50176,) vers (1, 1, 224, 224)
                    img_reshaped = sample.reshape(1, 1, 224, 224)
                    batch.append(img_reshaped)
                
                batch_tensor = torch.from_numpy(np.concatenate(batch, axis=0)).float()
                
                with torch.no_grad():
                    outputs = model.model(batch_tensor)
                    probs = torch.sigmoid(outputs)
                    # Retourner seulement la probabilité pour la pathologie cible
                    return probs[:, target_pathology_idx].cpu().numpy()
            
            # Aplatir l'image pour KernelSHAP
            img_flat = img_tensor.squeeze().numpy().flatten().reshape(1, -1)
            background_flat = background.squeeze().numpy().reshape(background.shape[0], -1)
            
            # KernelExplainer
            explainer = shap.KernelExplainer(
                predict_fn, 
                background_flat,
                link="identity"
            )
            
            # Calculer SHAP values avec moins d'échantillons pour KernelSHAP
            shap_values = explainer.shap_values(
                img_flat, 
                nsamples=min(100, num_samples*2)
            )
            
            # Reshape de (50176,) vers (224, 224)
            shap_vals = shap_values.reshape(1, 1, 224, 224)
            
            return {
                'shap_values': shap_vals,
                'image_array': img_tensor.squeeze().cpu().numpy(),
                'target_idx': target_pathology_idx,
                'explainer': explainer
            }
        
        except Exception as e2:
            print(f"Error with KernelExplainer: {e2}")
            
            # Méthode 3: Gradient simple (fallback final)
            print("Using simple gradient as final fallback...")
            
            try:
                img_tensor_grad = img_tensor.clone().requires_grad_(True)
                
                output = model.model(img_tensor_grad)
                output[0, target_pathology_idx].backward()
                
                gradients = img_tensor_grad.grad.data
                
                # Utiliser le gradient absolu comme approximation de SHAP
                shap_vals = gradients.abs()
                
                return {
                    'shap_values': shap_vals.cpu().numpy(),
                    'image_array': img_tensor.squeeze().cpu().numpy(),
                    'target_idx': target_pathology_idx,
                    'explainer': None
                }
            
            except Exception as e3:
                print(f"Error with gradient fallback: {e3}")
                raise RuntimeError(f"All SHAP methods failed. Last error: {e3}")


def visualize_shap_image(shap_result, pathology_name, original_image=None):
    """
    Visualise l'explication SHAP
    
    Args:
        shap_result: résultat de explain_image_with_shap()
        pathology_name: nom de la pathologie
        original_image: image PIL originale (optionnel)
    
    Returns:
        fig: figure matplotlib
    """
    shap_vals = shap_result['shap_values']
    img_array = shap_result['image_array']
    
    # Squeeze pour avoir (H, W)
    if len(shap_vals.shape) == 4:
        shap_vals = shap_vals[0, 0, :, :]
    elif len(shap_vals.shape) == 3:
        shap_vals = shap_vals[0, :, :]
    elif len(shap_vals.shape) == 2:
        pass  # Déjà la bonne shape
    else:
        shap_vals = shap_vals.squeeze()
    
    # Créer figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. Image originale
    if original_image:
        axes[0].imshow(original_image, cmap='gray')
    else:
        axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title("Original X-ray", fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # 2. SHAP values (heatmap)
    im = axes[1].imshow(shap_vals, cmap='seismic', vmin=-np.abs(shap_vals).max(), 
                        vmax=np.abs(shap_vals).max())
    axes[1].set_title("SHAP Feature Importance", fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], label='SHAP value', fraction=0.046)
    
    # 3. Overlay
    # Normaliser SHAP values pour overlay
    shap_normalized = (shap_vals - shap_vals.min()) / (shap_vals.max() - shap_vals.min() + 1e-8)
    
    # Resize si nécessaire
    if original_image:
        shap_resized = cv2.resize(shap_normalized, (original_image.size[0], original_image.size[1]))
        img_display = np.array(original_image)
        if len(img_display.shape) == 3:
            img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2GRAY)
    else:
        shap_resized = shap_normalized
        img_display = img_array
    
    # Créer overlay coloré
    heatmap = cv2.applyColorMap(np.uint8(255 * shap_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Normaliser image pour overlay
    img_display_norm = ((img_display - img_display.min()) / 
                        (img_display.max() - img_display.min() + 1e-8) * 255).astype(np.uint8)
    img_display_rgb = cv2.cvtColor(img_display_norm, cv2.COLOR_GRAY2RGB)
    
    # Overlay
    overlay = heatmap * 0.4 + img_display_rgb * 0.6
    
    axes[2].imshow(overlay.astype(np.uint8))
    axes[2].set_title(f"SHAP Overlay - {pathology_name}", fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    return fig


def compare_shap_predictions(model, image, top_k=3):
    """
    Compare SHAP explanations pour les top K prédictions
    
    Args:
        model: modèle
        image: PIL Image
        top_k: nombre de top prédictions à expliquer
    
    Returns:
        dict: {pathology: shap_result}
    """
    # Prédiction
    from images.model_loader import predict_image
    result = predict_image(model, image)
    top_predictions = result['top_predictions'][:top_k]
    
    explanations = {}
    
    for pred in top_predictions:
        pathology = pred['pathology']
        idx = list(model.pathologies).index(pathology)
        
        try:
            shap_result = explain_image_with_shap(model, image, target_pathology_idx=idx)
            explanations[pathology] = shap_result
        except Exception as e:
            print(f"Failed to explain {pathology}: {e}")
    
    return explanations


def create_shap_summary_plot(shap_results, pathology_names):
    """
    Crée un plot comparatif des SHAP values pour plusieurs pathologies
    
    Args:
        shap_results: liste de résultats SHAP
        pathology_names: liste des noms de pathologies
    
    Returns:
        fig: figure matplotlib
    """
    n = len(shap_results)
    fig, axes = plt.subplots(2, n, figsize=(6*n, 10))
    
    if n == 1:
        axes = axes.reshape(2, 1)
    
    for i, (shap_res, path_name) in enumerate(zip(shap_results, pathology_names)):
        shap_vals = shap_res['shap_values']
        
        if len(shap_vals.shape) == 4:
            shap_vals = shap_vals[0, 0, :, :]
        elif len(shap_vals.shape) == 3:
            shap_vals = shap_vals[0, :, :]
        
        # Heatmap
        im1 = axes[0, i].imshow(shap_vals, cmap='seismic', 
                               vmin=-np.abs(shap_vals).max(), 
                               vmax=np.abs(shap_vals).max())
        axes[0, i].set_title(f"{path_name}\nSHAP Heatmap", fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        plt.colorbar(im1, ax=axes[0, i])
        
        # Histogramme des valeurs
        axes[1, i].hist(shap_vals.flatten(), bins=50, alpha=0.7, color='steelblue')
        axes[1, i].set_title(f"{path_name}\nDistribution", fontsize=12, fontweight='bold')
        axes[1, i].set_xlabel("SHAP Value")
        axes[1, i].set_ylabel("Frequency")
        axes[1, i].grid(alpha=0.3)
    
    plt.tight_layout()
    
    return fig