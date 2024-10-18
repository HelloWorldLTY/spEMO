# Source: https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from hest_matching.model.nn_utils import MLPWrapper


class CLIPModel(nn.Module):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.image_projection = MLPWrapper(
            config.image_d_input, config.d_model, config.d_model, config.activation, norm_layer=nn.LayerNorm, drop=config.drop, drop_last=False
        )
        self.gene_projection = MLPWrapper(
            config.gene_d_input, config.d_model, config.d_model, config.activation, norm_layer=nn.LayerNorm, drop=config.drop, drop_last=False
        )
        self.temperature = config.temperature

    def inference_embedding(self, image_features, gene_features):
        image_embeddings = self.image_projection(image_features)
        gene_embeddings = self.gene_projection(gene_features)
        return image_embeddings, gene_embeddings

    def forward(self, image_features, gene_features):
        # Getting Image and Text Embeddings (with same dimension)
        image_embeddings, gene_embeddings = self.inference_embedding(image_features, gene_features)

        # Calculating the Loss
        logits = (gene_embeddings @ image_embeddings.T) / self.temperature
        images_similarity = image_embeddings @ image_embeddings.T
        genes_similarity = gene_embeddings @ gene_embeddings.T
        targets = F.softmax(
            (images_similarity + genes_similarity) / 2 * self.temperature, dim=-1
        )
        genes_loss = cross_entropy(logits, targets, reduction='none')
        images_loss = cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (images_loss + genes_loss) / 2.0 # shape: (batch_size)
        return loss.mean()


def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()
