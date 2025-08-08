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
    def __init__(self,kernel_type,attributes=None,**kwargs):
        kernel = VertexHistogram()
        super().__init__(kernel_type=kernel_type, kernelfunction=kernel, kernel_name="VertexHistogram", attributes=attributes, **kwargs)
        if DEBUG:
            print(f"Initialized VertexHistogram_SVC in child class")
    

class EdgeHistogram_SVC(SupportVectorMachine):
    def __init__(self,kernel_type,attributes=None,**kwargs):
        kernel = EdgeHistogram()
        super().__init__(kernel_type=kernel_type, kernelfunction=kernel, kernel_name="EdgeHistogram", attributes=attributes, **kwargs)
        if DEBUG:
            print(f"Initialized EdgeHistogram_SVC in child class")
    
    
class CombinedHistogram_SVC(SupportVectorMachine):
    def __init__(self,kernel_type, attributes=None, **kwargs):
        kernel = Combined_Kernel(kernels=[VertexHistogram(), EdgeHistogram()])
        super().__init__(kernel_type=kernel_type, kernelfunction=kernel, kernel_name="CombinedHistogram", attributes=attributes, **kwargs)
        if DEBUG:
            print(f"Initialized CombinedHistogram_SVC  in child class")
            print(f"Model Name: {self.get_name}")
    

# Kernels 
class Combined_Kernel(Kernel):
    """Combined Kernel that Combines Kernels that and concatenates their feature Vectors.
    """
    def __init__(self, kernels: list[Kernel]=VertexHistogram()):
        self.kernels = kernels
        super().__init__()

    def fit_transform(self, X, y=None):
        """Fit the combined kernel."""
        combined_transformation = []
        for kernel in self.kernels:
            if hasattr(kernel, 'fit_transform'):
                combined_transformation.append(kernel.fit_transform(X, y))
            else:
                raise ValueError(f"Kernel {kernel} does not have fit_transform method.")
        # Concatenate the results from all kernels
        X_transformed = np.concatenate(combined_transformation, axis=1)
        return X_transformed

    def transform(self, X):
        """Transform the input data using both kernels and concatenate the results."""
        combined_transformation = []
        for kernel in self.kernels:
            if hasattr(kernel, 'transform'):
                combined_transformation.append(kernel.transform(X))
            else:
                raise ValueError(f"Kernel {kernel} does not have transform method.")
        # Concatenate the results from all kernels
        X_transformed = np.concatenate(combined_transformation, axis=1)
        return X_transformed