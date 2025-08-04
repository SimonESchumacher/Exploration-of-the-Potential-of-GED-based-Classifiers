# Combined Kernel
from grakel.kernels import Kernel
from grakel.graph import Graph
from grakel.kernels import VertexHistogram, EdgeHistogram
import numpy as np
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
   