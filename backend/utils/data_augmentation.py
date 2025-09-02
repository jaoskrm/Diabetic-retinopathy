import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image, ImageEnhance, ImageFilter
import cv2

class MedicalImageAugmentation:
    """
    Specialized augmentation techniques for medical/retinal images
    Implements domain-specific augmentations that preserve medical relevance
    """

    def __init__(self, severity_level='moderate'):
        self.severity_level = severity_level
        self.setup_augmentation_params()

    def setup_augmentation_params(self):
        """Set augmentation parameters based on severity level"""
        if self.severity_level == 'light':
            self.rotation_limit = 10
            self.brightness_limit = 0.1
            self.contrast_limit = 0.1
            self.noise_limit = 20
        elif self.severity_level == 'moderate':
            self.rotation_limit = 20
            self.brightness_limit = 0.2
            self.contrast_limit = 0.2
            self.noise_limit = 40
        else:  # aggressive
            self.rotation_limit = 30
            self.brightness_limit = 0.3
            self.contrast_limit = 0.3
            self.noise_limit = 60

def apply_test_time_augmentation(model, image_tensor, num_augmentations=5):
    """
    Apply test-time augmentation for more robust predictions
    """
    model.eval()
    predictions = []

    # Original prediction
    with torch.no_grad():
        pred = model(image_tensor.unsqueeze(0))
        predictions.append(torch.softmax(pred, dim=1))

    # Augmented predictions (simplified for compatibility)
    for i in range(num_augmentations-1):
        # Simple augmentations that work with tensors
        augmented = image_tensor.clone()

        # Random horizontal flip
        if random.random() > 0.5:
            augmented = torch.flip(augmented, [2])

        # Random vertical flip
        if random.random() > 0.5:
            augmented = torch.flip(augmented, [1])

        with torch.no_grad():
            pred = model(augmented.unsqueeze(0))
            predictions.append(torch.softmax(pred, dim=1))

    # Average predictions
    final_prediction = torch.mean(torch.stack(predictions), dim=0)

    return final_prediction

# Configuration for different augmentation strategies
AUGMENTATION_CONFIGS = {
    'light': {
        'severity': 'light',
        'tta_enabled': False
    },
    'moderate': {
        'severity': 'moderate', 
        'tta_enabled': True
    },
    'aggressive': {
        'severity': 'aggressive',
        'tta_enabled': True
    }
}
