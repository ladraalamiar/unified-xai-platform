from lime import lime_image
from skimage.segmentation import mark_boundaries
import numpy as np

def explain_with_lime_image(model, image, num_samples=1000):
    """
    Génère une explication LIME pour une image
    """
    explainer = lime_image.LimeImageExplainer()
    
    # Convertir l'image en array numpy
    img_array = np.array(image)
    
    explanation = explainer.explain_instance(
        img_array,
        model.predict,
        top_labels=3,
        hide_color=0,
        num_samples=num_samples
    )
    
    # Visualisation
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )
    
    img_boundry = mark_boundaries(temp / 255.0, mask)
    
    return img_boundry