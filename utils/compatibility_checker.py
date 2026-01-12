"""
Filtre automatique des méthodes XAI selon le type de données
"""

def get_available_xai_methods(data_type):
    """
    Retourne les méthodes XAI compatibles avec le type de données
    
    Args:
        data_type: 'audio' ou 'image'
    
    Returns:
        list: méthodes XAI disponibles
    """
    if data_type == "audio":
        return {
            "LIME": True,
            "SHAP": True,
            "Grad-CAM": True,  # Fonctionne sur spectrogrammes
        }
    elif data_type == "image":
        return {
            "LIME": True,
            "SHAP": True,
            "Grad-CAM": True,
        }
    else:
        return {}

def filter_xai_methods(data_type, selected_methods):
    """
    Filtre les méthodes sélectionnées selon compatibilité
    """
    available = get_available_xai_methods(data_type)
    return [m for m in selected_methods if available.get(m, False)]