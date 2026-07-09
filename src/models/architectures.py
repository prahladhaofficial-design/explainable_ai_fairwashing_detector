"""
Model architectures for fairwashing experiments.
Includes tabular classifiers and image CNNs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict


class TabularClassifier(nn.Module):
    """Simple MLP for tabular data classification."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32], 
                 num_classes: int = 2, dropout: float = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class BiasedClassifier(nn.Module):
    """
    Biased classifier for fairwashing experiments.
    Explicitly uses sensitive attribute for predictions.
    """

    def __init__(self, input_dim: int, sensitive_weight: float = 0.9,
                 other_weight: float = 0.1, num_classes: int = 2):
        super().__init__()
        # For logistic regression style: weight on sensitive, small weight on others
        self.sensitive_weight = sensitive_weight
        self.other_weight = other_weight
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

    def set_biased_weights(self, sensitive_idx: int):
        """Set weights to heavily rely on sensitive attribute."""
        with torch.no_grad():
            weight = torch.zeros_like(self.fc.weight)
            weight[:, sensitive_idx] = self.sensitive_weight
            # Small random weights for other features
            weight[:, [i for i in range(weight.shape[1]) if i != sensitive_idx]] =                 torch.randn(weight.shape[0], weight.shape[1] - 1) * self.other_weight
            self.fc.weight.copy_(weight)


class Conv4(nn.Module):
    """
    4-layer CNN for MNIST/Fashion-MNIST.
    Architecture from Anders et al. (Off-Manifold Detergent).
    """

    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        # After 2 poolings: 7x7 for 28x28 input
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class VGG16(nn.Module):
    """VGG16 for CIFAR-10."""

    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 4
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 5
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ScaffoldingClassifier(nn.Module):
    """
    Scaffolding classifier for LIME/SHAP scaffolding scenario (Fooling LIME/SHAP).
    Wraps a biased classifier with decoy features.
    """

    def __init__(self, base_classifier: nn.Module, n_decoy_features: int = 2,
                 decoy_std: float = 0.1):
        super().__init__()
        self.base_classifier = base_classifier
        self.n_decoy_features = n_decoy_features
        self.decoy_std = decoy_std

        # Freeze base classifier
        for param in self.base_classifier.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x contains original features + decoy features
        # Split into original and decoy
        orig_features = x[:, :-self.n_decoy_features]

        # Pass original features through base classifier
        output = self.base_classifier(orig_features)
        return output

    def generate_decoy_features(self, x_orig: torch.Tensor) -> torch.Tensor:
        """Generate uncorrelated decoy features."""
        batch_size = x_orig.size(0)
        decoy = torch.randn(batch_size, self.n_decoy_features, 
                           device=x_orig.device) * self.decoy_std
        return torch.cat([x_orig, decoy], dim=1)


def get_model(model_name: str, **kwargs) -> nn.Module:
    """Factory function to get models by name."""
    models = {
        "tabular": TabularClassifier,
        "biased": BiasedClassifier,
        "conv4": Conv4,
        "vgg16": VGG16,
        "scaffolding": ScaffoldingClassifier,
    }

    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}")

    return models[model_name](**kwargs)
