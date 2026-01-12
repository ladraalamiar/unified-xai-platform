"""
XAI Methods Module
Explainable AI methods for medical image analysis
"""

# Make imports easier
try:
    from .shap_image import (
        explain_image_with_shap,
        visualize_shap_image,
        compare_shap_predictions,
        create_shap_summary_plot
    )
except ImportError as e:
    print(f"Warning: Could not import SHAP methods: {e}")

__all__ = [
    'explain_image_with_shap',
    'visualize_shap_image', 
    'compare_shap_predictions',
    'create_shap_summary_plot'
]