import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

from hest_matching.model.clip import CLIPModel


class TensorPairDataset(Dataset):
    def __init__(self, image_tensors, gene_tensors):
        self.image_tensors = image_tensors
        self.gene_tensors = gene_tensors

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        tensor1 = self.image_tensors[idx]
        tensor2 = self.gene_tensors[idx]
        return tensor1, tensor2


def test_metrics(similarity_mat, k=[1, 10, 20]):
    # entropy loss
    loss = torch.nn.CrossEntropyLoss()(similarity_mat, torch.eye(*similarity_mat.shape)).item()

    # precision@K
    precision = []
    pos_scores = torch.diag(similarity_mat)
    topk_values = torch.topk(similarity_mat, k=max(k)).values
    for i in k:
        precision.append((pos_scores >= topk_values[:, i - 1]).float().mean().item())

    return loss, precision


def train_clip_model(args, train_pair, test_pair, device=0):
    train_loader = DataLoader(TensorPairDataset(*train_pair), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(TensorPairDataset(*test_pair), batch_size=args.batch_size, shuffle=False)

    model = CLIPModel(args).to(device)

    criterion = torch.nn.CosineEmbeddingLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_metirc = {
        'loss': float('inf'),
        'accuracy': 0,
        'precision@1': 0,
        'precision@10': 0,
        'precision@20': 0,
    }

    for epoch in tqdm(range(args.epochs), ncols=100, desc='Training CLIP model'):
        model.train()
        for i, (image, gene) in enumerate(train_loader):
            image = image.to(device)
            gene = gene.float().to(device)
            optimizer.zero_grad()
            loss = model(image, gene)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            image_embeddings = []
            gene_embeddings = []
            for i, (image, gene) in enumerate(train_loader):
                image = image.to(device)
                gene = gene.float().to(device)
                image_embedding, gene_embedding = model.inference_embedding(image, gene)
                image_embeddings.append(image_embedding)
                gene_embeddings.append(gene_embedding)

            image_embeddings = torch.cat(image_embeddings, dim=0).cpu()
            gene_embeddings = torch.cat(gene_embeddings, dim=0).cpu()

            # Compute similarity matrix
            similarity_matrix = torch.matmul(gene_embeddings, image_embeddings.T)
            loss, precision = test_metrics(similarity_matrix)
            if loss < best_metirc['loss']:
                best_metirc['loss'] = loss
                best_metirc['precision@1'] = precision[0]
                best_metirc['precision@10'] = precision[1]
                best_metirc['precision@20'] = precision[2]

    return best_metirc
