import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import torch.nn.functional as F

class EfficientNetDRClassifier(nn.Module):
    """
    EfficientNet-based Diabetic Retinopathy Classifier
    Based on research showing EfficientNet achieving 97%+ accuracy on DR classification
    """

    def __init__(self, num_classes=5, dropout_rate=0.3, pretrained=True):
        super(EfficientNetDRClassifier, self).__init__()

        # Load pre-trained EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)

        # Get the number of features from the classifier
        num_features = self.backbone.classifier[1].in_features

        # Replace the classifier with our custom head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            nn.Linear(256, num_classes)
        )

        # Class names for DR severity levels
        self.class_names = [
            'No_DR',           # Grade 0
            'Mild',            # Grade 1
            'Moderate',        # Grade 2
            'Severe',          # Grade 3
            'Proliferate_DR'   # Grade 4
        ]

    def forward(self, x):
        return self.backbone(x)

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

# Model configuration
MODEL_CONFIG = {
    'image_size': 224,
    'num_classes': 5,
    'dropout_rate': 0.3,
    'learning_rate': 0.001,
    'batch_size': 32,
    'num_epochs': 50,
    'early_stopping_patience': 10,
    'weight_decay': 1e-4
}
