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
from Models.SVC.Base_GED_SVC import Base_GED_SVC, Base_Kernel
from Models.SVC.GED.simiple_prototype_GED_SVC import Simple_Prototype_GED_SVC
from Calculators.Prototype_Selction import Prototype_Selector, Select_Prototypes
DEBUG = False  # Set to True for debug prints

class ZERO_GED_SVC(Base_GED_SVC):
    
    def initKernel(self, ged_calculator: Base_Calculator = None, **kernel_kwargs):
        self.kernel = ZERO_GED_Kernel(ged_calculator=ged_calculator, **kernel_kwargs)

    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update(ZERO_GED_Kernel.get_param_grid())
        return param_grid


class ZERO_GED_Kernel(Base_Kernel):
    def __init__(self, ged_calculator: Base_Calculator = None,
                  KERNEL_aggregation_method="sum", KERNEL_comparison_method="Mean-Distance",
                  KERNEL_prototype_size=8,
                  KERNEL_classwise=False, KERNEL_single_class=False,
                  KERNEL_selection_method="RPS",
                  attributes: dict = dict(), **kwargs):
        self.aggregation_method = KERNEL_aggregation_method
        self.comparison_method = KERNEL_comparison_method
        self.prototype_size = KERNEL_prototype_size
        self.classwise = KERNEL_classwise
        self.single_class = KERNEL_single_class
        self.selection_method = KERNEL_selection_method
        attributes.update({"aggregation_method": self.aggregation_method})
        attributes.update({"KERNEL_prototype_size": self.prototype_size,
                           "KERNEL_classwise": self.classwise,
                           "KERNEL_single_class": self.single_class,
                           "KERNEL_selection_method": self.selection_method,
                           "KERNEL_comparison_method": self.comparison_method})
        super().__init__(ged_calculator=ged_calculator, attributes=attributes, **kwargs)
        if DEBUG:
            print(f"Initialized ZERO_GED_Kernel with I_size={self.I_size}, aggregation_method={self.aggregation_method}")
    
    def compare(self, g1, g2):
        distance=0
        d_g1_g2 =self.ged_calculator.compare(g1, g2, method=self.comparison_method)**2
        for g0 in self.prototypes:
            d_k0_g1 = self.ged_calculator.compare(g0, g2, method=self.comparison_method)**2
            d_k0_g2 = self.ged_calculator.compare(g1, g0, method=self.comparison_method)**2
            if self.aggregation_method == "sum":
                distance += 0.5*(d_k0_g1 + d_k0_g2 - d_g1_g2)
            elif self.aggregation_method == "prod":
                distance *= 0.5*(d_k0_g1 + d_k0_g2 - d_g1_g2)
        return distance

    def fit_transform(self, X, y=None):
        """ Fit the kernel to the data and return the kernel matrix.
        """
        self.X_fit_graphs_ = X
        self.prototypes = Select_Prototypes(X, y=y, ged_calculator=self.ged_calculator, size=self.prototype_size, classwise=self.classwise, single_class=self.single_class, selection_method=self.selection_method)
        return self.transform(X)
    def transform(self, X):
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
    @classmethod
    def get_param_grid(cls):
        param_grid = Base_Kernel.get_param_grid()
        param_grid.update({
            "KERNEL_I_size": [1,  3,  5,10],
            "KERNEL_aggregation_method": ["sum", "prod"],
            "KERNEL_classwise": [False, True],
            "KERNEL_single_class": [False, True],
            "KERNEL_selection_method": ["RPS", "CPS","BPS","TPS","SPS","k-CPS"],
            "selection_method": ["random", "statified_random"]
        })
        return param_grid

