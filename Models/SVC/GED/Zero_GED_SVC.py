# Class for Graph Edit Distance Kernel
# imports
import sys
import os
import numpy as np
import tqdm
sys.path.append(os.getcwd())
from grakel.kernels import Kernel
from Calculators.Base_Calculator import Base_Calculator
from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Models.SVC.Base_GED_SVC import Base_GED_SVC, Base_Kernel
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
    def __init__(self, ged_calculator: Base_Calculator = None, I_size=1, aggregation_method="sum",attributes: dict = dict(), **kwargs):
        self.I_size = I_size
        self.aggregation_method = aggregation_method
        self.seletion_method ="random"
        attributes.update({"I_size": I_size, "aggregation_method": aggregation_method})
        super().__init__(ged_calculator=ged_calculator, attributes=attributes, **kwargs)
        if DEBUG:
            print(f"Initialized ZERO_GED_Kernel with I_size={self.I_size}, aggregation_method={self.aggregation_method}")
    
    def compare(self, g1, g2):
        K=np.zeros(self.I_size)
        d_g1_g2 =self.ged_calculator.compare(g1, g2, method=self.comparison_method)**2
        for i in range(self.I_size):
            d_k0_g1 = self.ged_calculator.compare(self.I[i], g2, method=self.comparison_method)**2
            d_k0_g2 = self.ged_calculator.compare(g1, self.I[i], method=self.comparison_method)**2
            K[i] = 0.5*(d_k0_g1 + d_k0_g2 - d_g1_g2)
        if self.aggregation_method == "sum":
            k = np.sum(K)
        elif self.aggregation_method == "prod":
            k = np.prod(K)
        return k

    def fit_transform(self, X, y=None):
        """ Fit the kernel to the data and return the kernel matrix.
        """
        self.I=np.zeros(self.I_size)
        # pick random indices of the graphs of X
        if self.seletion_method == "random":
            self.I = np.random.choice(X, size=self.I_size, replace=False)
        elif self.seletion_method == "statified_random":
            pass
        else:
            raise ValueError(f"Unknown selection method: {self.seletion_method}")
            
        return super().fit_transform(X, y)
    @classmethod
    def get_param_grid(cls):
        param_grid = Base_Kernel.get_param_grid()
        param_grid.update({
            "KERNEL_I_size": [1,  3,  5,10],
            "KERNEL_aggregation_method": ["sum", "prod"],
            # "seletion_method": ["random", "statified_random"]
        })
        return param_grid

