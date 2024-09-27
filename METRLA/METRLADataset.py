'''
Creating PyTorch Temporal Graph Dataset
Dataset source: https://graphmining.ai/temporal_datasets/
Basic Structure of the Data:
Traffic Data from 207 Loop Detectors aggregated at 5 minute intervals from March 2012 to June 2012
x = last 60 minutes' traffic data along time-axis=12 points, along spatial axis = 207 points
y = next 60 minutes' traffic data (12 data points)
Node-Values file: Consists of every 5-minute's worth of average speed and occupancy rate values
Adj_mat file: Consists of adjacency matrix in shape: [num_nodes, num_nodes]
'''

# Import Required Packages
import os
import ssl
import zipfile
import urllib
import numpy as np

import torch
import torch_geometric
from torch_geometric.utils import to_dense_adj
from torch_geometric_temporal.signal import StaticGraphTemporalSignal

# Turning off ssl certification is also an option - not recommended
ssl._create_default_https_context = ssl._create_unverified_context

class METRLADataset(object):
    def __init__(self, root_dir='../data'):
        super(METRLADataset, self).__init__()
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir
        self._read_web_data()

    def _read_web_data(self):
        base_url = "https://graphmining.ai/temporal_datasets/"

        # --- Check if the zip file is present in the data directory of GNN folder

            # GraphNeuralNetwork/
            #   |
            #   |- data/
            #   |- METRLA/

        metrla_file = "METR-LA.zip"
        adjacency_file = "adj_mat.npy"
        node_val_file = "node_values.npy"

        metrla_zipfile_path = os.path.join(self.root_dir, metrla_file)

        if not os.path.isfile(metrla_zipfile_path):
            self._download_url(os.path.join(base_url, metrla_file), metrla_zipfile_path)

        # --- Check if the METRLA file has been unzipped and the two files are extracted
        if not os.path.isfile(os.path.join(self.root_dir, adjacency_file)) \
                or not os.path.isfile(os.path.join(self.root_dir, adjacency_file)):
            # Extract Zipfile
            with zipfile.ZipFile(metrla_zipfile_path, 'r') as zip_fph:
                zip_fph.extractall(self.root_dir)

        # --- Additional Checks if Adjacency and Node Value Numpy files exists
        assert os.path.isfile(os.path.join(self.root_dir, adjacency_file)), "Adjacency file does not exist"
        assert os.path.isfile(os.path.join(self.root_dir, node_val_file)), "Node Values file does not exist"


        self.A = None
        self.X = None


    def _download_url(self, url, directory):

        with urllib.request.urlopen(url) as metrla_zipfile:
            with open(directory, "wb") as file:
                file.write(metrla_zipfile.read())


    def _get_edges_and_weights(self):
        self.edge_index = None
        self.edge_weight = None

    def _generate_task(self):
        features = None
        targets = None
        self.features = features
        self.targets = targets

    def get_dataset(self):
        dataset = StaticGraphTemporalSignal(self.edge_index, self.edge_weight, self.features, self.targets)
        return dataset

check = METRLADataset(root_dir='../data')

