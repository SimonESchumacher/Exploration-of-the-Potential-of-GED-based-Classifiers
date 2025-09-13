# Imports
from multiprocessing import Pool, cpu_count
from functools import partial
from sklearn.model_selection import train_test_split
# Imports
import networkx as nx
# Import NetworkX for Graphs
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
from config_loader import get_conifg_param
from Dataset import Dataset
LOCAL_DATA_PATH = get_conifg_param('Dataset', 'local_data_path', type='str') # Path to the local data directory
# DOWNLOAD_SOURCE = 'TUD'
DATASET_NAME = 'NONE' # Placeholder, will be set later
DEBUG = get_conifg_param('Dataset','DEBUG') # Set to False to disable debug prints

# Constants
SHUFFLE = get_conifg_param('Dataset', 'shuffle_default', type='bool')# laod the default setting
RANDOM_STATE = get_conifg_param('Dataset', 'random_state', type='int') # Default random state for reproducibility
TEST_SIZE = get_conifg_param('Dataset', 'test_size', type='float') # Default test size for train-test split
DATASTE_LOG_FILE = get_conifg_param('Dataset', 'datasets_log', type='str') # Path to the dataset log file

USE_NODE_LABELS = get_conifg_param('Dataset', 'use_node_labels', type='bool') # Whether to use node labels
USE_EDGE_LABELS = get_conifg_param('Dataset', 'use_edge_labels', type='bool') # Whether to use edge labels
USE_NODE_ATTRIBUTES = get_conifg_param('Dataset', 'use_node_attributes', type='bool') # Whether to use node attributes
USE_EDGE_ATTRIBUTES = get_conifg_param('Dataset', 'use_edge_attributes', type='bool') # Whether to use edge attributes
from tqdm import tqdm  # Import tqdm for progress bar


class PreloadedDataset(Dataset):
    def __init__(self, name: str, source: str, domain: str, ged_calculator=None, use_node_labels=None, use_edge_labels=None, use_node_attributes=None, use_edge_attributes=None, load_now=True):
        super().__init__(name, source, domain, ged_calculator, use_node_labels, use_edge_labels, use_node_attributes, use_edge_attributes, load_now)
        self.preloaded_graphs = None  # Placeholder for preloaded graphs

    def load(self):
        """Load the dataset and preload graphs into memory."""
        super().load()  # Call the parent class load method