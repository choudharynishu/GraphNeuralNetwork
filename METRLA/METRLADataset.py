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
from torch_geometric.utils import dense_to_sparse
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
        adj_filepath = os.path.join(self.root_dir, adjacency_file)
        nodeval_filepath = os.path.join(self.root_dir, node_val_file)

        if not os.path.isfile(metrla_zipfile_path):
            self._download_url(os.path.join(base_url, metrla_file), metrla_zipfile_path)

        # --- Check if the METRLA file has been unzipped and the two files are extracted
        if not os.path.isfile(adj_filepath) or not os.path.isfile(nodeval_filepath):
            # Extract Zipfile
            with zipfile.ZipFile(metrla_zipfile_path, 'r') as zip_fph:
                zip_fph.extractall(self.root_dir)

        # --- Additional Checks if Adjacency and Node Value Numpy files exists
        assert os.path.isfile(adj_filepath), "Adjacency file does not exist"
        assert os.path.isfile(nodeval_filepath), "Node Values file does not exist"

        self.A = torch.from_numpy(np.load(adj_filepath))

        # --- Z-score Normalization Data Pre-processing before assigning as X
        X = np.load(nodeval_filepath)
        X = X.astype(np.float32)
        # X.shape = [34272, 207, 2] => [num_samples, num_nodes, num_features(avg_speed, occupancy_rate)]
        X = X.transpose(1, 2, 0)

        # X.shape = [207, 2, 34272] => [num_samples, num_nodes, num_features(avg_speed, occupancy_rate)]
        # --- Estimate mean value of speed and occupancy rate for all nodes (Assumed uniformity) 
        # --- and entire sample (Data Leak)

        means = np.mean(X, axis=(0, 2)).reshape(1, -1, 1)
        stds = np.std(X, axis=(0, 2)).reshape(1, -1, 1)

        X = (X-means)
        X = X / stds
        self.X = torch.from_numpy(X)


    def _download_url(self, url, directory):

        with urllib.request.urlopen(url) as metrla_zipfile:
            with open(directory, "wb") as file:
                file.write(metrla_zipfile.read())


    def _get_edges_and_weights(self):
        edge_index, edge_weights = dense_to_sparse(self.A)
        self.edge_index = edge_index.numpy()  # StaticGraphTemporalSignal Edge_Index = Union[np.ndarray, None]
        self.edge_weight = edge_weights.numpy() # StaticGraphTemporalSignal, Edge_Weight = Union[np.ndarray, None]

    def _generate_task(self, num_timesteps_in=12, num_timesteps_out=12):
        """
        :param num_timesteps_in: number of historical timesteps model utilizes for a single snapshot (default: 12)
        :param num_timesteps_out: number of future timesteps model predicts for a single snapshot (default: 12)
        :return: features: Sequence[Union[np.ndarray, None]], targets: Sequence[Union[np.ndarray, None]]
        """
        delta = (num_timesteps_in+num_timesteps_out)
        # Initialize features and targets as lists
        features, targets = [], []

        # X.shape = [207, 2, 34272] => [num_samples, num_nodes, num_features(avg_speed, occupancy_rate)]
        indices = [(i, (i+delta)) for i in range(self.X.shape[2]-(delta)+1)]
        for i, j in indices:

            features.append((self.X[:, :, i:i+num_timesteps_in]).numpy())
            # For Targets we only want speed
            targets.append((self.X[:, 0, i+num_timesteps_in:j]).numpy())
        self.features = features
        self.targets = targets


    def get_dataset(self, num_timesteps_in=12, num_timesteps_out=12):
        """Returns data iterator for METR-LA dataset as an instance of the Static Graph Temporal Signal class.
        :return: **dataset** *(StaticGraphTemporalSignal)* - The METR-LA traffic forecasting dataset.
        """
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(self.edge_index, self.edge_weight, self.features, self.targets)
        return dataset



