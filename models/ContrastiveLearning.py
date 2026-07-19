import torch
from torch import nn
from torch.nn import functional as F
import open_clip

from models.SemanticPositioning import SemanticPositioning


gpu_ids = [1]
device = torch.device(f"cuda:{gpu_ids[0]}" if torch.cuda.is_available() else "cpu")


class TextEncoder(nn.Module):
    def __init__(self, model_name="ViT-B-32", pretrained="openai"):
        super().__init__()
        self.model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained)

    def forward(self, tokens):
        tokens = open_clip.tokenize(tokens).to(device)
        text_features = self.model.encode_text(tokens)
        return text_features


class ContrastiveLearning(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLearning, self).__init__()
        self.text_encoder = TextEncoder()
        self.semantic_positioning = SemanticPositioning()
        self.temperature = temperature
        self.log_softmax = nn.LogSoftmax(dim=-1)

    def forward(self, text, feature_t1, feature_t2):
        text_features = self.text_encoder(text)
        image_features = self.semantic_positioning(feature_t2) - self.semantic_positioning(feature_t1)

        text_features = F.normalize(text_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)

        similarity_matrix = image_features @ text_features.T
        logits_per_image = similarity_matrix / self.temperature
        logits_per_text = similarity_matrix.T / self.temperature

        labels = torch.arange(len(logits_per_image)).to(logits_per_image.device)
        loss_img = F.cross_entropy(logits_per_image, labels)
        loss_txt = F.cross_entropy(logits_per_text, labels)

        return (loss_img + loss_txt) / 2
