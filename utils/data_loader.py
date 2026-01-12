"""
Chargement et gestion des modèles audio pré-entraînés
"""
import torch
import torch.nn as nn
import torchaudio
import librosa
import numpy as np

# ========== DÉFINIR VOS ARCHITECTURES ==========

class VGG16Audio(nn.Module):
    """Architecture VGG16 pour audio - ADAPTER SELON VOTRE MODÈLE"""
    def __init__(self, num_classes=2):
        super(VGG16Audio, self).__init__()
        # TODO: Copier l'architecture exacte de votre modèle
        self.features = nn.Sequential(
            # Exemple - REMPLACER par votre architecture
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # ... reste des couches
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ResNetAudio(nn.Module):
    """Architecture ResNet pour audio - ADAPTER SELON VOTRE MODÈLE"""
    def __init__(self, num_classes=2):
        super(ResNetAudio, self).__init__()
        # TODO: Copier l'architecture exacte de votre modèle
        pass
    
    def forward(self, x):
        # TODO: Implémenter forward pass
        pass


# ========== FONCTIONS DE CHARGEMENT ==========

def load_audio_model(model_name, model_path=None):
    """
    Charge un modèle audio pré-entraîné
    
    Args:
        model_name: 'vgg16' ou 'resnet'
        model_path: chemin vers le fichier .pth (optionnel)
    
    Returns:
        model: modèle chargé en mode eval
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if model_name.lower() == 'vgg16':
        model = VGG16Audio(num_classes=2)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            # Chercher dans dossier par défaut
            model.load_state_dict(torch.load('Lamia/audio/vgg16_audio.pth', map_location=device))
    
    elif model_name.lower() == 'resnet':
        model = ResNetAudio(num_classes=2)
        if model_path:
            model.load_state_dict(torch.load(model_path, map_location=device))
        else:
            model.load_state_dict(torch.load('Lamia/audio/resnet_audio.pth', map_location=device))
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    model.to(device)
    model.eval()
    return model


# ========== PRÉTRAITEMENT AUDIO ==========

def preprocess_audio(audio_file, target_sr=16000, duration=3.0):
    """
    Prétraite un fichier audio pour le modèle
    
    Args:
        audio_file: fichier audio uploadé
        target_sr: sample rate cible
        duration: durée en secondes
    
    Returns:
        tensor: audio prétraité (1, 1, H, W) pour spectrogramme
    """
    # Charger audio
    waveform, sr = torchaudio.load(audio_file)
    
    # Resample si nécessaire
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Prendre mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Tronquer ou padder à durée fixe
    target_length = int(target_sr * duration)
    if waveform.shape[1] > target_length:
        waveform = waveform[:, :target_length]
    else:
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    
    # Créer spectrogramme (Mel)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=target_sr,
        n_fft=2048,
        hop_length=512,
        n_mels=128
    )
    
    spectrogram = mel_spectrogram(waveform)
    
    # Convertir en dB
    spectrogram = torchaudio.transforms.AmplitudeToDB()(spectrogram)
    
    # Normaliser
    spectrogram = (spectrogram - spectrogram.mean()) / spectrogram.std()
    
    # Ajouter dimension batch
    spectrogram = spectrogram.unsqueeze(0)  # (1, 1, n_mels, time)
    
    return spectrogram


# ========== PRÉDICTION ==========

def predict_audio(model, audio_file):
    """
    Prédit si audio est REAL ou FAKE
    
    Args:
        model: modèle chargé
        audio_file: fichier audio uploadé
    
    Returns:
        dict: {'class': 'REAL' ou 'FAKE', 'confidence': float, 'probabilities': [real_prob, fake_prob]}
    """
    device = next(model.parameters()).device
    
    # Prétraiter
    spectrogram = preprocess_audio(audio_file)
    spectrogram = spectrogram.to(device)
    
    # Prédiction
    with torch.no_grad():
        outputs = model(spectrogram)
        probabilities = torch.softmax(outputs, dim=1)[0].cpu().numpy()
    
    # Interpréter (supposant que classe 0 = REAL, 1 = FAKE)
    fake_prob = probabilities[1]
    
    prediction = {
        'class': 'FAKE' if fake_prob > 0.5 else 'REAL',
        'confidence': max(probabilities),
        'probabilities': {
            'REAL': float(probabilities[0]),
            'FAKE': float(probabilities[1])
        }
    }
    
    return prediction


# ========== ALTERNATIVE : SI VOUS AVEZ UN MODÈLE COMPLET ==========

def load_complete_model(model_path):
    """
    Si votre modèle est sauvegardé avec torch.save(model, 'model.pth')
    au lieu de torch.save(model.state_dict(), 'model.pth')
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model