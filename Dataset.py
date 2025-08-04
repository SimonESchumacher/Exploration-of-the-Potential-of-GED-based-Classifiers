# Imports
from sklearn.model_selection import train_test_split
# Imports
import networkx as nx
# Import NetworkX for Graphs
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import KFold
from config_loader import get_conifg_param

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

def load_dataset_into_networkx(data_dir, dataset_name,use_node_labels="label", use_edge_labels="label", use_node_attributes:str=None, use_edge_attributes:str=None):
    """
    General function to load datasets into NetworkX.
    Handles cases where node labels or edge labels may be missing.
    """
    print(f"Loading {dataset_name} into NetworkX from {data_dir}...")
    adj_file = os.path.join(data_dir, f"{dataset_name}_A.txt")
    graph_indicator_file = os.path.join(data_dir, f"{dataset_name}_graph_indicator.txt")
    graph_labels_file = os.path.join(data_dir, f"{dataset_name}_graph_labels.txt")

    # Load edges 
    with open(adj_file, 'r') as f:
        edges_raw = [list(map(int, line.strip().split(','))) for line in f]
    edges = [(u - 1, v - 1) for u, v in edges_raw]  # Convert to 0-indexed
    print(f"Loaded {len(edges)} edges.")
    # Load node-to-graph mapping
    with open(graph_indicator_file, 'r') as f:
        node_to_graph_map = np.array([int(line.strip()) for line in f]) - 1
    print(f"Loaded {len(node_to_graph_map)} node-to-graph mappings.")
    # Load graph labels
    with open(graph_labels_file, 'r') as f:
        graph_labels_raw = [int(line.strip()) for line in f]
    y = np.array([(1 if label == 1 else 0) for label in graph_labels_raw])
    print(f"Loaded {len(y)} graph labels.")

    # Try to load node labels if available
    node_labels = None
    if USE_NODE_LABELS and use_node_labels is not None:
        # if use_node_labels is a boolean of value True, we assume labels are in the format "label"
        if use_node_labels is True:
            use_node_labels = "label"
        node_labels_file = os.path.join(data_dir, f"{dataset_name}_node_labels.txt")
        if os.path.exists(node_labels_file):
            with open(node_labels_file, 'r') as f:
                node_labels_raw = [int(line.strip()) for line in f]
            node_labels = {i: label for i, label in enumerate(node_labels_raw)}
            print(f"Loaded node labels for {len(node_labels)} nodes.")
    
    # Try to load edge labels if available
    edge_labels = None
    if USE_EDGE_LABELS and use_edge_labels is not None:
        if use_edge_labels is True:
            use_edge_labels = "label"
        edge_labels_file = os.path.join(data_dir, f"{dataset_name}_edge_labels.txt")
        if os.path.exists(edge_labels_file):
            with open(edge_labels_file, 'r') as f:
                edge_labels_raw = [int(line.strip()) for line in f]
            # edge labels should be mapped to their node pair
            edge_labels = {(u, v): label for (u, v), label in zip(edges, edge_labels_raw)}
            print(f"Loaded edge labels for {len(edge_labels)} edges.")
    
    # Try to load node attributes if availabl
    node_attributes = None
    if USE_NODE_ATTRIBUTES and use_node_attributes is not None:
        node_attributes_file = os.path.join(data_dir, f"{dataset_name}_node_attributes.txt")
        if os.path.exists(node_attributes_file):
            with open(node_attributes_file, 'r') as f:
                node_attributes_raw = [list(map(float, line.strip().split(','))) for line in f]
            node_attributes = {i: attr for i, attr in enumerate(node_attributes_raw)}
            print(f"Loaded node attributes for {len(node_attributes)} nodes.")

    # Try to load edge attributes if available
    edge_attributes = None
    if USE_EDGE_ATTRIBUTES and use_edge_attributes is not None:
        edge_attributes_file = os.path.join(data_dir, f"{dataset_name}_edge_attributes.txt")
        if os.path.exists(edge_attributes_file):
            with open(edge_attributes_file, 'r') as f:
                edge_attributes_raw = [list(map(float, line.strip().split(','))) for line in f]
            edge_attributes = {i: attr for i, attr in enumerate(edge_attributes_raw)}
            print(f"Loaded edge attributes for {len(edge_attributes)} edges.")

    num_graphs = np.max(node_to_graph_map) + 1
    nx_graphs = []
    if DEBUG:
        iterable= tqdm(range(num_graphs), desc="Converting graphs to NetworkX format")
    else:
        iterable = range(num_graphs)
    for i in iterable:
        G = nx.Graph()
        nodes_in_graph_i = np.where(node_to_graph_map == i)[0]

        if len(nodes_in_graph_i) == 0:  # Handle empty graphs
            nx_graphs.append(G)
            continue
        # Add nodes with optional labels and attributes
        if USE_NODE_LABELS or USE_NODE_ATTRIBUTES and (use_node_attributes is not None or use_node_labels is not None):
            for node_idx_global in nodes_in_graph_i:
                node_data = {}
                if node_labels:
                    node_data["label"] = str(node_labels[node_idx_global])
                if node_attributes:
                    node_data[use_node_attributes] = str(node_attributes[node_idx_global])
                G.add_node(node_idx_global, **node_data)

        # Add edges within the current graph with optional labels and attributes
        if USE_EDGE_LABELS or USE_EDGE_ATTRIBUTES and (use_edge_attributes is not None or use_edge_labels is not None):
            for u, v in edges:
                if u in nodes_in_graph_i and v in nodes_in_graph_i:
                    edge_data = {}
                    if edge_labels:
                        edge_data["label"] = str(edge_labels.get((u, v), None))
                    if edge_attributes:
                        edge_data[use_edge_attributes] = str(edge_attributes.get((u, v), None))
                    G.add_edge(u, v, **edge_data)

        nx_graphs.append(G)
    print(f"Converted {num_graphs} graphs to NetworkX format.")
    return nx_graphs, y




def load_dataset(source,name,use_node_labels=None,use_node_attributes=None,use_edge_labels=None,use_edge_attributes=None) -> tuple[list, list, np.ndarray]:

    global DATASET_NAME
    DATASET_NAME = name
   
    dataset_path= os.path.join(LOCAL_DATA_PATH,source ,name)
    # dataset_path = f"{LOCAL_DATA_PATH}/{source}/{name}"
    if source == 'TUD':
        try:       
            nx_graphs, y = load_dataset_into_networkx(dataset_path, name, use_node_labels=use_node_labels, use_node_attributes=use_node_attributes, use_edge_labels=use_edge_labels, use_edge_attributes=use_edge_attributes)  
        except FileNotFoundError as e:
            print(f"Error: {e}")
            print(f"The Dataset '{name}' could not be laoded from directory: {dataset_path}")
            raise e
        return nx_graphs, y
    else:
        # TODO if other sources are needing to be parsed diffrently
        raise ValueError(f"Unsupported dataset source: {source}. Supported sources: ['TUD']")

def calculate_dataset_attributes(data: tuple, source: str, domain: str = None, Name: str = None, save=False) -> dict:
    """
    Calculate basic attributes of the dataset using NetworkX graphs.
    Args:
        data (tuple): The dataset to analyze, expected to be a tuple (graph_list, targets).
        source (str): The source of the dataset.
        domain (str, optional): Optional domain information.
        Name (str, optional): Name of the dataset.
    Returns:
        dict: A dictionary with dataset attributes.
    """
    graph_list: list = data[0]  # List of NetworkX graphs
    targets: np.ndarray = data[1]  # Numpy array of target labels
    num_graphs = len(graph_list)
    if num_graphs == 0:
        raise ValueError("The graph list is empty. Cannot extract features.")
    num_classes = len(set(targets))
    if num_classes == 0:
        raise ValueError("No classes found in the target labels. Cannot extract features.")

    # Check for node and edge labels on the first graph
    try:
        first_graph = graph_list[0]
        has_node_labels = any('label' in data for _, data in first_graph.nodes(data=True))
        has_edge_labels = any('label' in data for _, _, data in first_graph.edges(data=True))
    except Exception as e:
        print(f"Error checking graph attributes: {e}")
        has_node_labels = False
        has_edge_labels = False

    mean_nodes = np.mean([G.number_of_nodes() for G in graph_list])
    mean_edges = np.mean([G.number_of_edges() for G in graph_list])
    label_distribution = {label: np.sum(targets == label) / num_graphs for label in set(targets)}

    dataset_characteristics = {
        'name': Name,
        'source': source,
        'domain': domain,
        'num_graphs': num_graphs,
        'num_classes': num_classes,
        'has_node_labels': has_node_labels,
        'has_edge_labels': has_edge_labels,
        'mean_nodes': mean_nodes,
        'mean_edges': mean_edges,
        'label_distribution': str(label_distribution)
    }
    if save:
        datasets_log = pd.read_excel(DATASTE_LOG_FILE, index_col=0)
        if Name not in datasets_log['name'].values:
            new_entry = pd.Series(dataset_characteristics)
            datasets_log = datasets_log.append(new_entry, ignore_index=True)
            datasets_log.to_excel(DATASTE_LOG_FILE)
        else:
            if DEBUG:
                print(f"Dataset '{Name}' already exists in the log file. Skipping save.")
    return dataset_characteristics
def extract_simple_graph_features(data): 
    _,graph_list, targets = data  
    # Extract simple features
    features = []
    for G in graph_list:
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        
        # Avoid division by zero if graph has no nodes or edges
        avg_degree = np.mean([d for n, d in G.degree()]) if num_nodes > 0 else 0
        
        # Add more features as desired
        # e.g., density = nx.density(G) if num_nodes > 1 else 0
        
        return np.array(features)


class Dataset:

    def __init__(self, name,source, domain=None,ged_calculator=None, use_node_labels="label", use_node_attributes="label", use_edge_labels=None, use_edge_attributes=None):
        self.name = name
        self.data= load_dataset(source,name,use_node_labels=use_node_labels,use_node_attributes=use_node_attributes,use_edge_labels=use_edge_labels,use_edge_attributes=use_edge_attributes)
        self.nx_graphs= self.data[0]
        self.target = self.data[1]
        self.source = source
        self.domain = domain
        self.Node_label_name = use_node_labels
        self.Edge_label_name = use_edge_labels
        try:
            self.characteristics = calculate_dataset_attributes(self.data, source, domain, name)
        except Exception as e:
            print(f"An unexpected error occurred while initializing the dataset: {e}")
            print("happend while loading dataset attributes")
            print(f"Problematic Dataset: {name} from source: {source}")
            self.characteristics = {
                'name': name,
                'source': source,
                'domain': domain,
                'num_graphs': 0,
                'num_classes': 0,
                'has_node_labels': False,
                'has_edge_labels': False,
                'mean_nodes': 0,
                'mean_edges': 0,
                'label_distribution': 'N/A'
            }
            raise e
        if DEBUG:
            print(f"Now setting up the Calculator")
        if ged_calculator is not None:
            self.ged_calculator = ged_calculator
            self.ged_calculator.add_graphs(self.nx_graphs.copy(), self.target)
            if DEBUG:
                print(f"Calculating GED for between graphs")
            self.nx_graphs=self.ged_calculator.activate()
            start_time = pd.Timestamp.now()
            self.ged_calculator.calculate()
            end_time = pd.Timestamp.now()
            self.ged_calculator.runtime = (end_time - start_time).total_seconds()
    def change_dataset_params(self,new_ged_calculator=False, use_node_labels=None, use_node_attributes=None, use_edge_labels=None, use_edge_attributes=None):
        # realoads the dataset with the new parameters
        self.data = load_dataset(self.source, self.name, use_node_labels=use_node_labels, use_node_attributes=use_node_attributes, use_edge_labels=use_edge_labels, use_edge_attributes=use_edge_attributes)
        self.nx_graphs = self.data[0]
        self.target = self.data[1]
        # TODO: probalpy a reaload of the ged_calculator
        if new_ged_calculator and self.ged_calculator is not None:
            self.ged_calculator.add_graphs(self.nx_graphs.copy(), self.target)
            self.nx_graphs = self.ged_calculator.activate()
            start_time = pd.Timestamp.now()
            self.ged_calculator.calculate()
            end_time = pd.Timestamp.now()
            self.ged_calculator.runtime = (end_time - start_time).total_seconds()



    def __str__(self):
        return f"Dataset(name={self.name}, num_graphs={len(self.data)}, num_labels={len(set(self.target))})"

    def __getattribute__(self, name):
        if name == 'data':
            return super().__getattribute__(name)
        elif name == 'graphs':
            return self.data[0]
        elif name == 'target':
            return self.data[1]
        else:
            return super().__getattribute__(name)
    def attributes(self):
        return self.characteristics
    
    def train_test_split(self, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=None,shuffle=SHUFFLE,saveSplit=False):
        """
        Splits the dataset into training and testing sets.
        if saveSplit is True, saves the split into the object
        """
        if saveSplit:
            self.split_available = True
            self.is_stratified = stratify is not None
            self.shuffle = shuffle
            self.test_size = test_size
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.nx_graphs, self.target, test_size=test_size, random_state=random_state, stratify=stratify, shuffle=shuffle)
            self.X_train = [g for g in self.X_train]
            self.X_test = [g for g in self.X_test]
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            X_train, X_test, y_train, y_test = train_test_split(self.nx_graphs, self.target, test_size=test_size, random_state=random_state, stratify=stratify, shuffle=shuffle)
            X_train = [g for g in X_train]
            X_test = [g for g in X_test]
            return X_train, X_test, y_train, y_test
    def split_fold(self,train_index, test_index):
        """
        Splits the dataset into training and testing sets based on provided indices.
        """
        X_train = [self.nx_graphs[i] for i in train_index]
        X_test = [self.nx_graphs[i] for i in test_index]
        y_train_fold = [self.target[i] for i in train_index]
        y_test_fold = [self.target[i] for i in test_index]
        return X_train, X_test, y_train_fold, y_test_fold 
    
    def split_k_fold(self, k=5, random_state=RANDOM_STATE):
        """
        Splits the dataset into k folds for cross-validation.
        Returns a list of tuples (train_index, test_index) for each fold.
        """
        
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        splits = []
        for train_index, test_index in kf.split(self.nx_graphs):
            splits.append(self.split_fold(train_index, test_index))
        return splits
    def get_param_grid(self):
        """
        Returns a dictionary of parameters for the dataset.
        This can be used for hyperparameter tuning.
        """
        return {
            'use_node_labels': [self.Node_label_name,None],
            'use_edge_labels': [self.Edge_label_name,None],
        }
   
