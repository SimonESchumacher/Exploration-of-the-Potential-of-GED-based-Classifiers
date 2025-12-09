# Class for Graph Edit Distance Kernel
# imports
import sys
import os
import numpy as np
import pandas as pd
import tqdm
sys.path.append(os.getcwd())
from Calculators.prototype_selector import Select_Prototypes
from Calculators.Base_Calculator import Base_Calculator
from Models.support_vector_models.GED_SVC import GED_SVC
from config_loader import get_conifg_param
DEBUG = get_conifg_param('GED_models', 'debuging_prints', type='bool')

# Zero-Prototype Graph Edit Distance Support Vector Classifier
# Only works for Random Prototype Selection currently
class Zero_GED_SVC(GED_SVC):
    model_specific_iterations = get_conifg_param('Hyperparameter_fields', 'tuning_iterations', type='int')
    def __init__(self,
                    aggregation_method,
                    prototype_size,
                    selection_split,
                    selection_method,
                    attributes:dict=dict(),
                    **kwargs):
        self.aggregation_method = aggregation_method
        self.prototype_size = prototype_size
        self.selection_split = selection_split
        self.selection_method = selection_method

        self.kernel_name = "Zero-GED"
        attributes.update({
            "aggregation_method": self.aggregation_method,
            "prototype_size": self.prototype_size,
            "selection_split": self.selection_split,
            "selection_method": self.selection_method,
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
        # print("Selecting prototypes...")
        # print(f"Selection method:{ self.selection_method}, Selection split: {self.selection_split}, Prototype size: {self.prototype_size}")
        self.prototypes = Select_Prototypes(X,  ged_calculator=self.ged_calculator,y=y,
                                            size=self.prototype_size, selection_split=self.selection_split,
                                            selection_method=self.selection_method, comparison_method=self.ged_bound)
        n = len(X)
        K = np.zeros((n, n))
        if DEBUG:
            iters = tqdm.tqdm(range(n), desc="Computing kernel matrix")
        else:
            iters = range(n)
        self.feature_vectors_X_fit = np.zeros((n, len(self.prototypes)))
        start_time = pd.Timestamp.now()
        self.D_g1_g2 = np.square(self.ged_calculator.get_complete_matrix(method=self.ged_bound))
        for i in iters:
            for k, g0 in enumerate(self.prototypes):
                self.feature_vectors_X_fit[i, k] = self.D_g1_g2[i, g0]
        # Efficiently square every entry in the upperbound_matrix
        d_proto = np.zeros((n,n,len(self.prototypes)))
        for i in iters:
            for j in range(i, n):
                d_proto[i, j, :] = self.feature_vectors_X_fit[i, :] + self.feature_vectors_X_fit[j, :] - self.D_g1_g2[i, j]
                d_proto[j, i, :] = d_proto[i, j, :]
        if self.aggregation_method == "sum":
            K = np.sum(d_proto,axis=2)/2
        elif self.aggregation_method == "prod":
            K = np.prod(d_proto,axis=2)/2
         # because the kernel matrix is symmetric
        end_time = pd.Timestamp.now()
        duration = end_time - start_time
        # print(f"Kernel matrix computed in {duration}")
        return K
    def transform(self, X):
        """ Transform the data using the fitted kernel.
        """
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Transforming with GED_SVC with {len(X)} graphs")
        n = len(X)
        m = len(self.X_fit_graphs_)
        K = np.zeros((n, m))
        d_proto = np.zeros((n,m,len(self.prototypes)))
        feature_vectors_X = np.zeros(len(self.prototypes))
        for i in range(n):
            for k, g0 in enumerate(self.prototypes):
                feature_vectors_X[k] = self.D_g1_g2[X[i], g0]
            for j in range(m):
                d_proto[i, j, :] = feature_vectors_X[:] + self.feature_vectors_X_fit[j, :] - self.D_g1_g2[X[i], self.X_fit_graphs_[j]]
        if self.aggregation_method == "sum":
            K = np.sum(d_proto, axis=2)/2
        elif self.aggregation_method == "prod":
            K = np.prod(d_proto, axis=2)/2
        return K
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
    def build_matrix_fast(self, X):
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
        feature_vectors_X = np.zeros((n, self.prototype_size))
        for i in iters:
            for k, g0 in enumerate(self.prototypes):
                feature_vectors_X[i, k] = self.ged_calculator.compare(X[i], g0, method=self.ged_bound)**2
        for i in iters:
            for j in range( m):
                d_g1_g2 =self.ged_calculator.compare(X[i], self.X_fit_graphs_[j], method=self.ged_bound)**2
                d_proto = feature_vectors_X[i, :] + self.feature_vectors_X_fit[j, :] - d_g1_g2
                if self.aggregation_method == "sum":
                    K[i, j] = np.sum(d_proto)/2
                elif self.aggregation_method == "prod":
                    K[i, j] = np.prod(d_proto)/2
                # K[j, i] = K[i, j]  # because the kernel matrix is symmetric
                

        return K
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # add the parameters of the ged_calculator with the prefix "GED_"
        params.update({
            "aggregation_method": self.aggregation_method,
            "prototype_size": self.prototype_size,
            "selection_split": self.selection_split,
            "selection_method": self.selection_method,
            "ged_bound": self.ged_bound,
        })
       
        return params

    @classmethod
    def get_param_grid(cls):
        param_grid = GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update(
            {
                "prototype_size": [1, 2, 3],
                "aggregation_method": ["sum"],
                "selection_split": ["all", "classwise", "single_class"],
                # "selection_method": ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"], # Since Claculator_rebuild, only RPS works
                "C": [0.1]
            }
        )
        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_space = GED_SVC.get_random_param_space()
        param_space.update(
            {
                # "prototype_size": [1, 2],
                "aggregation_method": ["sum","prod","sum"],
                "selection_split": ["all", "classwise", "single_class"],
                # "selection_method": ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"], # Since Claculator_rebuild, only RPS works
            }
        )
        return param_space


