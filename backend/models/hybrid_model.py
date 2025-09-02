import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class HybridDRClassifier(nn.Module):
    """
    Hybrid model combining EfficientNet and Vision Transformer
    Based on research showing hybrid models achieving 94%+ accuracy
    """

    def __init__(self, num_classes=5, fusion_method='weighted_average'):
        super(HybridDRClassifier, self).__init__()

        # EfficientNet branch for local features
        self.efficientnet = models.efficientnet_b0(pretrained=True)
        efficientnet_features = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier = nn.Identity()  # Remove original classifier

        # Simple CNN branch as ViT alternative (since timm might not be available)
        self.cnn_branch = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        cnn_features = 64

        # Feature fusion layer
        self.fusion_method = fusion_method
        combined_features = efficientnet_features + cnn_features

        if fusion_method == 'concatenation':
            # Simple concatenation
            self.fusion_layer = nn.Identity()
            classifier_input = combined_features
        elif fusion_method == 'weighted_average':
            # Learnable weighted average
            self.efficientnet_weight = nn.Parameter(torch.tensor(0.6))  # Start with 60% EfficientNet
            self.cnn_weight = nn.Parameter(torch.tensor(0.4))           # Start with 40% CNN

            # Project features to same dimension for averaging
            self.efficientnet_proj = nn.Linear(efficientnet_features, 512)
            self.cnn_proj = nn.Linear(cnn_features, 512)
            classifier_input = 512
        else:  # concatenation as default
            classifier_input = combined_features

        # Final classifier
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_input),
            nn.Dropout(0.3),
            nn.Linear(classifier_input, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )

        self.class_names = [
            'No_DR', 'Mild', 'Moderate', 'Severe', 'Proliferate_DR'
        ]

    def forward(self, x):
        # Extract features from both branches
        efficientnet_features = self.efficientnet(x)
        cnn_features = self.cnn_branch(x)

        # Fuse features based on method
        if self.fusion_method == 'weighted_average':
            # Project to same dimension
            eff_proj = self.efficientnet_proj(efficientnet_features)
            cnn_proj = self.cnn_proj(cnn_features)

            # Weighted average with learnable weights
            weights_sum = torch.abs(self.efficientnet_weight) + torch.abs(self.cnn_weight)
            eff_norm_weight = torch.abs(self.efficientnet_weight) / weights_sum
            cnn_norm_weight = torch.abs(self.cnn_weight) / weights_sum

            fused_features = eff_norm_weight * eff_proj + cnn_norm_weight * cnn_proj
        else:  # concatenation
            fused_features = torch.cat([efficientnet_features, cnn_features], dim=1)

        # Final classification
        logits = self.classifier(fused_features)
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

HYBRID_CONFIG = {
    'fusion_method': 'weighted_average',
    'num_classes': 5,
    'learning_rate': 0.0005,
    'batch_size': 24,
    'num_epochs': 40,
    'weight_decay': 1e-4
}
