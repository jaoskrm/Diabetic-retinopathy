import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
import logging

logger = logging.getLogger(__name__)

class DRClassificationEvaluator:
    """
    Comprehensive evaluation metrics for Diabetic Retinopathy classification
    Implements medical AI evaluation best practices
    """

    def __init__(self, class_names=None):
        self.class_names = class_names or [
            'No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR'
        ]
        self.num_classes = len(self.class_names)

    def evaluate_model(self, model, dataloader, device='cpu', return_predictions=False):
        """Comprehensive model evaluation"""
        model.eval()
        all_predictions = []
        all_probabilities = []
        all_labels = []
        total_loss = 0.0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                probabilities = F.softmax(outputs, dim=1)
                predictions = torch.argmax(probabilities, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(dataloader)

        # Calculate metrics
        metrics = self.calculate_metrics(all_labels, all_predictions, all_probabilities)
        metrics['loss'] = avg_loss

        if return_predictions:
            return metrics, all_labels, all_predictions, all_probabilities

        return metrics

    def calculate_metrics(self, y_true, y_pred, y_prob=None):
        """Calculate comprehensive classification metrics"""

        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)

        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class,
        }

        return metrics
