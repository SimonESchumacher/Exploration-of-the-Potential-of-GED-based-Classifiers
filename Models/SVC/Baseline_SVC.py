from grakel.kernels import VertexHistogram, EdgeHistogram
import sys
import os
# Combined Kernel
from grakel.kernels import Kernel
from grakel.graph import Graph
from grakel.kernels import VertexHistogram, EdgeHistogram
import numpy as np
# imports from custom modules
sys.path.append(os.getcwd())

from Models.SupportVectorMachine_Classifier import SupportVectorMachine

# get Started with more kernels
# Example usage of WeisfeilerLehman kernel
DEBUG = False # Set to False to disable debug prints

class VertexHistogram_SVC(SupportVectorMachine):
    model_specific_iterations = 25
    def __init__(self, attributes=None, **kwargs):
        kernel = VertexHistogram(sparse=True)
        super().__init__( kernelfunction=kernel, kernel_name="VertexHistogram", attributes=attributes, **kwargs)
        if DEBUG:
            print(f"Initialized VertexHistogram_SVC in child class")
    

class EdgeHistogram_SVC(SupportVectorMachine):
    model_specific_iterations = 25
    def __init__(self,attributes=None,**kwargs):
        kernel = EdgeHistogram(sparse=True)
        super().__init__( kernelfunction=kernel, kernel_name="EdgeHistogram", attributes=attributes, **kwargs)
        if DEBUG:
            print(f"Initialized EdgeHistogram_SVC in child class")
    
    
class CombinedHistogram_SVC(SupportVectorMachine):
    model_specific_iterations = 50

    def __init__(self, attributes=None,weights=[1,1], **kwargs):
        self.weights = weights
        kernel = Combined_Kernel(kernels=[EdgeHistogram(sparse=False), VertexHistogram(sparse=False)],weights=self.weights)
        super().__init__( kernelfunction=kernel, kernel_name="CombinedHistogram", attributes=attributes, **kwargs)
        if DEBUG:
            print(f"Initialized CombinedHistogram_SVC  in child class")
            print(f"Model Name: {self.get_name}")
    @classmethod
    def get_param_grid(cls):
        param_grid = SupportVectorMachine.get_param_grid()
        param_grid.update({
            'weights': [[1, 1], [1, 0.5], [0.5, 1],[0,1],[0,1]],  # Different weight combinations for the histograms
        })
        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_space = SupportVectorMachine.get_random_param_space()
        param_space.update({
            'weights': [[1, 1], [1, 0.5], [0.5, 1],[0,1],[0,1]],  # Different weight combinations for the histograms
        })
        return param_space
    

# Kernels 
class Combined_Kernel(Kernel):
    model_specific_iterations = 50

    """Combined Kernel that Combines Kernels that and concatenates their feature Vectors.
    """
    def __init__(self, kernels: list[Kernel]=VertexHistogram(),weights: list[float]=[1.0]):
        self.kernels = kernels
        self.weights = weights
        super().__init__()

    def fit_transform(self, X, y=None):
        """Fit the combined kernel."""
        combined_transformation = None
        X_list = list(X)
        
        size = len(X_list)
        combined_transformation = np.zeros((size,size)) 
       
        for kernel in self.kernels:        
            if hasattr(kernel, 'fit_transform'):
                combined_transformation += kernel.fit_transform(X_list, y) * self.weights[self.kernels.index(kernel)]
            else:
                raise ValueError(f"Kernel {kernel} does not have fit_transform method.")
        self.X_fit_ = X_list
        # Concatenate the results from all kernels
        if DEBUG:
            print(combined_transformation.shape)
        return combined_transformation

    def transform(self, X):
        """Transform the input data using both kernels and concatenate the results."""
        X = list(X)
        combined_transformation = np.zeros((len(X), len(self.X_fit_)))
        for kernel in self.kernels:
            if hasattr(kernel, 'transform'):
                combined_transformation += kernel.transform(X) * self.weights[self.kernels.index(kernel)]
            else:
                raise ValueError(f"Kernel {kernel} does not have transform method.")
        # Concatenate the results from all kernels
        return combined_transformation
    
class NX_Histogram_SVC(SupportVectorMachine):
    model_specific_iterations = 10
    def __init__(self,Histogram_Type="node+1", attributes:dict=dict(),get_node_labels:callable=None,get_edge_labels:callable=None, **kwargs):
        self.Histogram_Type = Histogram_Type
        self.get_node_labels = get_node_labels
        self.get_edge_labels = get_edge_labels
        if Histogram_Type == "node"or Histogram_Type == "combined" or Histogram_Type == "node+1":
            if get_node_labels is None:
                raise ValueError("get_node_labels must be provided for node histogram.")
            else:
                # call the get function to get the node labels
                self.node_labels = get_node_labels()
        elif self.Histogram_Type == "edge+1":
            self.node_labels = [0]
        else:
            self.node_labels = []
        if Histogram_Type == "edge" or Histogram_Type == "edge+1" or Histogram_Type == "combined":
            if get_edge_labels is None:
                raise ValueError("get_edge_labels must be provided for edge histogram.")
            else:
                # call the get function to get the edge labels
                self.edge_labels = get_edge_labels()
        elif self.Histogram_Type == "node+1":
            self.node_labels = [0]
        else:
            self.node_labels = []
        attributes.update({
            "Histogram_Type": self.Histogram_Type
        })
        super().__init__(kernelfunction=None,kernel_name=f"NX_{self.Histogram_Type}_Histogram",
                         attributes=attributes,**kwargs)
        
    def fit_transform(self, X, y=None):
        """Fit the combined kernel."""
        return self.transform(X)
    
    def transform(self, X):
        """Transform the input data using both kernels and concatenate the results."""
        num_node_labels = len(self.node_labels)
        num_edge_labels = len(self.edge_labels)
        feature_vectors = np.zeros((len(X), num_node_labels + num_edge_labels))
        for i, G in enumerate(X):           
            for _, data in G.nodes(data=True):
                if self.Histogram_Type == "combined" or self.Histogram_Type == "node" or self.Histogram_Type == "node+1":
                    if 'label' in data:
                        feature_vectors[i][self.node_labels.index(data['label'])] += 1
                    else:
                        feature_vectors[i][0] += 1
                elif self.Histogram_Type == "edge+1":
                    feature_vectors[i][0] += 1
            for _, _, data in G.edges(data=True):
                if self.Histogram_Type == "combined" or self.Histogram_Type == "edge" or self.Histogram_Type == "edge+1":
                    if 'label' in data:
                        feature_vectors[i][num_node_labels+self.edge_labels.index(data['label'])] += 1
                    else:
                        feature_vectors[i][num_node_labels ] += 1
                elif self.Histogram_Type == "node+1":
                    feature_vectors[i][num_node_labels] += 1
        if DEBUG:
            print(feature_vectors.shape)
        return feature_vectors
    @classmethod
    def get_param_grid(cls):
        param_grid = SupportVectorMachine.get_param_grid()
        param_grid.update({
            'kernel_type': ['rbf'],
            'Histogram_Type': [ 'node+1', 'edge+1'],

        })
        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_space = SupportVectorMachine.get_random_param_space()
        param_space.update({
            'kernel_type': ['rbf', 'linear', 'poly'],
            'Histogram_Type': ["combined","node","edge","node+1", "edge+1"],
        })
        return param_space

