from torch import nn
from .feature_extractors.base import FeatureExtractor


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.net(x)
        return out
