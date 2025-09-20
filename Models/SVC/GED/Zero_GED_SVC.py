# Class for Graph Edit Distance Kernel
# imports
import sys
import os
import numpy as np
import tqdm

from Calculators.Prototype_Selction import Prototype_Selector
sys.path.append(os.getcwd())
from grakel.kernels import Kernel
from Calculators.Base_Calculator import Base_Calculator
from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Models.SVC.Base_GED_SVC import Base_GED_SVC
from Models.SVC.GED.simiple_prototype_GED_SVC import Simple_Prototype_GED_SVC
from Calculators.Prototype_Selction import Prototype_Selector, Select_Prototypes, buffered_prototype_selection
DEBUG = False  # Set to True for debug prints

class ZERO_GED_SVC(Base_GED_SVC):
    def __init__(self,
                    aggregation_method,
                    prototype_size,
                    selection_split,
                    selection_method,
                    dataset_name,
                    attributes:dict=dict(),
                    **kwargs):
        self.aggregation_method = aggregation_method
        self.prototype_size = prototype_size
        self.selection_split = selection_split
        self.selection_method = selection_method

        self.dataset_name = dataset_name
        self.kernel_name = "Zero-GED"
        attributes.update({
            "aggregation_method": self.aggregation_method,
            "prototype_size": self.prototype_size,
            "selection_split": self.selection_split,
            "selection_method": self.selection_method,
            "dataset_name": self.dataset_name
        })
        super().__init__(attributes=attributes, name=self.kernel_name, **kwargs)
    
    def compare(self, g1, g2):
        distance=0
        d_g1_g2 =self.ged_calculator.compare(g1, g2, method=self.ged_bound)**2
        for g0 in self.prototypes:
            d_k0_g1 = self.ged_calculator.compare(g0, g2, method=self.ged_bound)**2
            d_k0_g2 = self.ged_calculator.compare(g1, g0, method=self.ged_bound)**2
            if self.aggregation_method == "sum":
                distance += 0.5*(d_k0_g1 + d_k0_g2 - d_g1_g2)
            elif self.aggregation_method == "prod":
                distance *= 0.5*(d_k0_g1 + d_k0_g2 - d_g1_g2)
        return distance
    
    def fit_transform(self, X, y=None):
        """ Fit the kernel to the data and return the kernel matrix.
        """
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Fitting GED_SVC with {len(X)} graphs")
        self.X_fit_graphs_ = X
        self.prototypes = buffered_prototype_selection(X, y=y, ged_calculator=self.ged_calculator, size=self.prototype_size, selection_split=self.selection_split,
                                                        selection_method=self.selection_method,
                                                          comparison_method=self.ged_bound, dataset_name=self.dataset_name)
        return self.build_matrix(X)
    def transform(self, X):
        """ Transform the data using the fitted kernel.
        """
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Transforming with GED_SVC with {len(X)} graphs")
        return self.build_matrix(X)
    def build_matrix(self, X):
        """ Transform the data using the fitted kernel.
        """
        if self.prototypes is None:
            raise RuntimeError("Kernel has not been fitted yet.")
        n = len(X)
        m = len(self.X_fit_graphs_)
        K = np.zeros((n, m))
        if DEBUG:
            iters = tqdm.tqdm(range(n), desc="Computing kernel matrix")
        else:
            iters = range(n)
        for i in iters:
            for j in range(m):
                K[i, j] = self.compare(X[i], self.X_fit_graphs_[j])
        return K
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # add the parameters of the ged_calculator with the prefix "GED_"
        params.update({
            "aggregation_method": self.aggregation_method,
            "prototype_size": self.prototype_size,
            "selection_split": self.selection_split,
            "selection_method": self.selection_method,
            "ged_bound": self.ged_bound
        })
       
        return params

    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update(
            {
                "prototype_size": [1, 3, 5],
                "aggregation_method": ["sum", "prod"],
                "selection_split": ["all", "classwise", "single_class"],
                "selection_method": ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"]
            }
        )
        return param_grid


