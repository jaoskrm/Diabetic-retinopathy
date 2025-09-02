import torch
import torch.nn as nn
import timm
from torchvision import transforms
import torch.nn.functional as F

class ViTDRClassifier(nn.Module):
    """
    Vision Transformer for Diabetic Retinopathy Classification
    Based on research showing ViT achieving 93%+ accuracy with proper preprocessing
    """

    def __init__(self, model_name='vit_base_patch16_224', num_classes=5, pretrained=True):
        super(ViTDRClassifier, self).__init__()

        # Load pre-trained Vision Transformer from timm
        try:
            self.vit = timm.create_model(
                model_name, 
                pretrained=pretrained,
                num_classes=0  # Remove the original classifier head
            )
        except:
            # Fallback to a simpler model if timm is not available
            from torchvision.models import vision_transformer
            self.vit = vision_transformer.vit_b_16(pretrained=pretrained)
            self.vit.heads = nn.Identity()  # Remove classifier head

        # Get the feature dimension from the model
        try:
            self.feature_dim = self.vit.num_features
        except:
            self.feature_dim = 768  # Default for ViT-Base

        # Custom classification head with dropout for regularization
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

        # Class names
        self.class_names = [
            'No_DR',
            'Mild', 
            'Moderate',
            'Severe',
            'Proliferate_DR'
        ]

    def forward(self, x):
        # Extract features using ViT backbone
        features = self.vit(x)
        # Pass through custom classifier
        logits = self.classifier(features)
        return logits

    def predict_proba(self, x):
        """Returns prediction probabilities"""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

    def predict(self, x):
        """Returns predicted class and confidence"""
        probabilities = self.predict_proba(x)
        predicted_class = torch.argmax(probabilities, dim=1)
        confidence = torch.max(probabilities, dim=1)[0]

        return {
            'predicted_class': predicted_class.cpu().numpy(),
            'class_name': [self.class_names[i] for i in predicted_class.cpu().numpy()],
            'confidence': confidence.cpu().numpy(),
            'probabilities': probabilities.cpu().numpy()
        }

# Model configuration for ViT
VIT_CONFIG = {
    'model_name': 'vit_base_patch16_224',
    'image_size': 224,
    'num_classes': 5,
    'learning_rate': 0.0001,  # Lower LR for ViT
    'batch_size': 16,         # Smaller batch size due to memory
    'num_epochs': 30,
    'warmup_epochs': 5,
    'weight_decay': 0.05
}
