from torch import nn
import torch.nn.functional as F
from .feature_extractors.base import FeatureExtractor


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.net(x)
        return out


class Classifier(nn.Module):
    def __init__(self, feature_extractor: FeatureExtractor, head: nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, x):
        out = self.feature_extractor(x)
        return self.head(out)
