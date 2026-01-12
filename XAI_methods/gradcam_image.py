# import torch
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import numpy as np

# def explain_with_gradcam_image(model, image, target_layer=None):
#     """
#     Génère une heatmap Grad-CAM
#     """
#     if target_layer is None:
#         target_layer = model.layer4[-1]  # Dernière couche conv
    
#     cam = GradCAM(model=model, target_layers=[target_layer])
    
#     # Préparer l'image
#     input_tensor = preprocess_image(image)
    
#     # Générer CAM
#     grayscale_cam = cam(input_tensor=input_tensor)
#     grayscale_cam = grayscale_cam[0, :]
    
#     # Superposer sur l'image originale
#     rgb_img = np.array(image) / 255.0
#     visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
#     return visualization