# GED Diffusion Classifier
import sys
import os
import numpy as np
sys.path.append(os.getcwd())

from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Calculators.Base_Calculator import Base_Calculator
from Models.SVC.Base_GED_SVC import Base_GED_SVC
DEBUG = False
class DIFFUSION_GED_SVC(Base_GED_SVC):
    def __init__(self,
                llambda:float,
                t_iterations:int,
                diffusion_function:str,
                attributes:dict=dict(),
                **kwargs):
        self.llambda = llambda
        self.t_iterations = t_iterations
        self.diffusion_function = diffusion_function
        if self.diffusion_function not in ["exp_diff_kernel", "von_Neumann_diff_kernel"]:
            raise ValueError(f"Unknown diffusion function: {self.diffusion_function}")
        self.name="Diffusion-GED"
        attributes.update({
            "llambda": self.llambda,
            "t_iterations": self.t_iterations,
            "diffusion_function": self.diffusion_function
        })
        super().__init__(attributes=attributes, name=self.name, **kwargs)

    # definition can't be changed here, because the Kernel class requires it
    def compare(self, g1, g2):
        raise ValueError("Diffusion Kernel doesn not support direct comparison of two graphs. Use the _calculate_kernel_matrix method instead.")
    def _calculate_kernel_matrix(self, X_graphs,Y_graphs=None):
        # first get the distance Matrix D
        D = self.ged_calculator.get_complete_matrix(method=self.ged_bound,x_graphindexes=X_graphs,y_graphindexes=Y_graphs)
        # first we create Base Similarity Matrix B
        # for that first we need the max distance in the matrix
        if not hasattr(self, 'MaxD') or self.MaxD is None:
            self.MaxD = np.max(D)
        B = self.MaxD - D
        if self.diffusion_function == "exp_diff_kernel":
            K=np.zeros(B.shape)
            llambda_exp = 1.0
            B_exp = 1
            k_factorial = 1
            for i in range(0,self.t_iterations):
                K += llambda_exp * B_exp * (1/k_factorial)
                B_exp = B_exp * B
                llambda_exp = llambda_exp * self.llambda
                k_factorial = k_factorial * (i + 1)

        elif self.diffusion_function == "von_Neumann_diff_kernel":
            B_exp = 1
            llambda_exp = 1.0
            K = np.zeros(B.shape)
            for i in range(0, self.t_iterations):  # Example for 10 iterations
                K += llambda_exp * B_exp
                B_exp = B_exp * B
                llambda_exp = llambda_exp * self.llambda
        else:
            raise ValueError(f"Unknown diffusion kernel: {self.diffusion_function}")
        self.K=K
        return K
    def get_params(self, deep=True):
        """
        Get the parameters of the kernel.
        """
        params = super().get_params()
        params.update({
            "llambda": self.llambda,
            "t_iterations": self.t_iterations,
            "diffusion_function": self.diffusion_function
        })
        return params
    

    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        param_grid.update({
            "llambda": [0.1, 0.5, 1.0],
            "diffusion_function": ["exp_diff_kernel", "von_Neumann_diff_kernel"],
            "t_iterations": [3,5]
        })
        # extra grid, to narrow the search space
        # param_grid.update({
        #     "ged_bound": ['Mean-Distance', 'UpperBound-Distance'],
        #     "C": [0.1, 0.5]
        # })
        return param_grid

