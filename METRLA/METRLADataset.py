'''
Creating PyTorch Temporal Graph Dataset
Dataset source: https://graphmining.ai/temporal_datasets/
Basic Structure of the Data:
Traffic Data from 207 Loop Detectors aggregated at 5 minute intervals from March 2012 to June 2012
x = last 60 minutes' traffic data (12 data points)
y = next 60 minutes' traffic data (12 data points)
'''
# Import Required Packages
import os
import zipfile
import urllib

import numpy as np
import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj
from torch_geometric_temporal.nn.recurrent import GConvGRU


base_url = "https://graphmining.ai/temporal_datasets/"
metrla_file = "METR-LA.zip"
adjacency_file = "adj_mat.npy"
node_val_file = "node_values.npy"