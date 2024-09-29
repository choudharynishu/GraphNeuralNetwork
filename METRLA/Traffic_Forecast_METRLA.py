"""
Traffic Forecasting through Attention Temporal Graph Neural Network
Reference: https://arxiv.org/pdf/2006.11583
"""
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

from METRLADataset import METRLADataset
from torch_geometric_temporal.signal import temporal_signal_split

# Source code for A3TGCN:
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/nn/recurrent/attentiontemporalgcn.html

# ---------------------------------------------------Data Import -------------------------------------------------- #
loader = METRLADataset()
# loader = METRLADatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=12, num_timesteps_out=12)
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
print(f"Length of Training Dataset: {train_dataset.snapshot_count}")

# -------------------- Attention Temporal Graph Convolutional Network for Traffic Forecasting---------------------- #

class TemporalGNN(nn.Module):
    def __init__(self, node_features, periods):
        """
        :param node_features: int, number of input features
        :param periods: int, number of time periods for historical input
        """
        super(TemporalGNN, self).__init__()
        self.tgnn = A3TGCN(in_channels=node_features,
                           out_channels=32,
                           periods=periods)
        self.linear = nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        :param x: Node features for num_timesteps_in
        :param edge_index: Adjacency matrix (in COO format)
        :return: output of the forward pass
        """
        x = self.tgnn(x, edge_index)
        x = F.relu(x)
        x = self.linear(x)
        return x

model = TemporalGNN(node_features=2, periods=12)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_function = nn.MSELoss()
model.train()
num_epochs = 10
subset = 2000
for epoch in tqdm(range(num_epochs)):
    loss = 0
    step = 0
    for signal_snapshot in train_dataset:
        step+=1
        predictions = model(signal_snapshot.x, signal_snapshot.edge_index)

        loss += loss_function(predictions, signal_snapshot.y)
        if step>subset:
            break
    loss = loss / (subset)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Epoch ({epoch}): {loss.item()}")
# ----------------------------------------------------Evaluation------------------------------------------------------ #
model.eval()
test_subset = 288
test_step = 0
test_loss = 0
with torch.no_grad():
    for signal_snapshot in test_dataset:
        test_step += 1
        test_prediction = model(signal_snapshot.x, signal_snapshot.edge_index)
        test_loss += loss_function(test_prediction, signal_snapshot.y)
        if test_step >= test_subset:
            break
test_loss = test_loss/test_subset
print(f"Test MSE: {test_loss.item()}")



