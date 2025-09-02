import torch
import torch.nn as nn
import os
import logging
from pathlib import Path
import json
import numpy as np

logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping utility to prevent overfitting"""

    def __init__(self, patience=10, min_delta=1e-4, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_metric):
        if self.mode == 'min':
            score = -val_metric
        else:
            score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        return self.early_stop

def load_model_for_inference(model_path, device='cpu'):
    """Load model for inference"""

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # This would need proper model class mapping
        model_class = checkpoint.get('model_class', 'EfficientNetDRClassifier')
        config = checkpoint.get('config', {})

        # Initialize model (simplified - would need proper factory)
        if model_class == 'EfficientNetDRClassifier':
            from models.efficientnet_model import EfficientNetDRClassifier
            model = EfficientNetDRClassifier(num_classes=config.get('num_classes', 5))
        elif model_class == 'ViTDRClassifier':
            from models.vision_transformer import ViTDRClassifier
            model = ViTDRClassifier(num_classes=config.get('num_classes', 5))
        elif model_class == 'HybridDRClassifier':
            from models.hybrid_model import HybridDRClassifier
            model = HybridDRClassifier(num_classes=config.get('num_classes', 5))
        else:
            # Default fallback
            from models.efficientnet_model import EfficientNetDRClassifier
            model = EfficientNetDRClassifier(num_classes=5)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        return model

    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        # Return a default model
        from models.efficientnet_model import EfficientNetDRClassifier
        model = EfficientNetDRClassifier(num_classes=5)
        model.to(device)
        model.eval()
        return model

def save_model_for_inference(model, model_path, model_config=None):
    """Save model in format suitable for inference"""

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_class': model.__class__.__name__,
        'config': model_config,
        'class_names': getattr(model, 'class_names', None)
    }

    torch.save(checkpoint, model_path)
    logger.info(f"Model saved for inference: {model_path}")
