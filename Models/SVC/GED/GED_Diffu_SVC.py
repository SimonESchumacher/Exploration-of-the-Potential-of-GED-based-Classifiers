# GED Diffusion Classifier
import sys
import os
import numpy as np
sys.path.append(os.getcwd())

from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Calculators.Base_Calculator import Base_Calculator
from Models.SVC.Base_GED_SVC import Base_GED_SVC, Base_Kernel
DEBUG = False
class DIFFUSION_GED_SVC(Base_GED_SVC):
       
    def initKernel(self, ged_calculator:Base_Calculator=None, **kernel_kwargs):
        self.kernel = Diffusion_Kernel(ged_calculator=ged_calculator, **kernel_kwargs)

    @classmethod
    def get_param_grid(cls):
        param_grid = SupportVectorMachine.get_param_grid()
        param_grid.update(Diffusion_Kernel.get_param_grid())
        return param_grid

class Diffusion_Kernel(Base_Kernel):
    def __init__(self, ged_calculator:Base_Calculator=None, llambda=0.5,t_iterations=10, diffusion_function="exp_diff_kernel", attributes:dict=dict(), **kwargs):
        self.KERNEL_llambda = llambda
        self.KERNEL_t_iterations = t_iterations
        self.KERNEL_diffusion_function = diffusion_function # Lafferty, 2002  ; aternative "von_Neumann_diff_kernel"  Kandola et al., 2002
        attributes.update({"KERNEL_llambda": llambda,
                           "KERNEL_t_iterations": t_iterations,
                           "KERNEL_diffusion_Kernel": diffusion_function})
        super().__init__(ged_calculator=ged_calculator, attributes=attributes,KERNEL_name="Diffusion-GED", **kwargs)
        if DEBUG:
            print(f"Initialized Diffusion Kernel with KERNEL_llambda={self.KERNEL_llambda}, KERNEL_diffusion_function={self.KERNEL_diffusion_function}")

    def compare(self, g1, g2):
        """
        Compare two graphs g1 and g2 using the diffusion kernel.
        """
        raise ValueError("Diffusion Kernel doesn not support direct comparison of two graphs. Use the _calculate_kernel_matrix method instead.")

    def _calculate_kernel_matrix(self, X_graphs,Y_graphs=None):
        # first get the distance Matrix D
        D = self.ged_calculator.get_complete_matrix(method=self.comparison_method,x_graphindexes=X_graphs,y_graphindexes=Y_graphs)
        # first we create Base Similarity Matrix B
        # for that first we need the max distance in the matrix
        if not hasattr(self, 'MaxD') or self.MaxD is None:
            self.MaxD = np.max(D)
        B = self.MaxD - D
        if self.KERNEL_diffusion_function == "exp_diff_kernel":
            K=0
            llambda_exp = 1.0
            B_exp = 1
            k_factorial = 1
            for i in range(0,self.KERNEL_t_iterations):
                K += llambda_exp * B_exp * (1/k_factorial)
                B_exp = B_exp * B
                llambda_exp = llambda_exp * self.KERNEL_llambda
                k_factorial = k_factorial * (i + 1)

        elif self.KERNEL_diffusion_function == "von_Neumann_diff_kernel":
            B_exp = 1
            llambda_exp = 1.0
            K = 0
            for i in range(0, self.KERNEL_t_iterations):  # Example for 10 iterations
                K += llambda_exp * B_exp
                B_exp = B_exp * B
                llambda_exp = llambda_exp * self.KERNEL_llambda
        else:
            raise ValueError(f"Unknown diffusion kernel: {self.KERNEL_diffusion_function}")
        self.K=K
        return K
    @classmethod
    def get_param_grid(cls):
        """
        Get the parameter grid for hyperparameter tuning.
        """
        param_grid = Base_Kernel.get_param_grid()
        param_grid.update({
            "KERNEL_llambda": [0.1, 0.5, 1.0],
            "KERNEL_diffusion_function": ["exp_diff_kernel", "von_Neumann_diff_kernel"],
            "KERNEL_iteration_t": [5, 10, 20]
        })
        return param_grid