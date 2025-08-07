# GED Diffusion Classifier
import sys
import os
import numpy as np
sys.path.append(os.getcwd())

from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Custom_Kernels.Trivial_GED_Kernel import Trivial_GED_Kernel
from Base_Calculator import Base_Calculator
DEBUG = True
class DIFFUSION_GED_SVC(SupportVectorMachine):
    def __init__(self,
                 C=1.0,
                llambda=1.0,
                class_weight=None,
                ged_calculator:Base_Calculator=None,
                Ged_distance_bound="Mean-Distance",
                diffusion_Kernel="exp_diff_kernel", # Lafferty, 2002  ; aternative "von_Neumann_diff_kernel"  Kandola et al., 2002
                attributes:dict=dict()):
        self.llambda = llambda
        self.ged_calculator = ged_calculator
        self.diffusion_Kernel = diffusion_Kernel
        self.Ged_distance_bound = Ged_distance_bound
        self.MaxD = None # the max diffrence, of the training data, used to turn distance to similarity inthe first step
        if ged_calculator is None:
            raise ValueError("ged_calculator must be provided")
        attributes.update({"llambda": llambda,
                           "Ged_distance_bound": Ged_distance_bound,
                           "diffusion_Kernel": diffusion_Kernel,})
        attributes.update(ged_calculator.get_params())
        super().__init__(C=C,
                          class_weight=class_weight,
                          kernelfunction="None",
                          kernel_name="DIFFUSION_GED",
                            attributes=attributes) 
    def get_kernel(self, X, Y=None):
        pass
    def get_calculator(self):
        return self.ged_calculator
    def add_graphs(self, graphs, labels=None):
        """
        Add graphs to the GED calculator.
        """
        if DEBUG:
            print(f"Adding {len(graphs)} graphs to GED calculator")
        self.ged_calculator.add_graphs(graphs, labels)

    def create_diffusion_matrix(self, X,Y=None):
        # first get the distance Matrix D
        D = self.ged_calculator.get_complete_matrix(method=self.Ged_distance_bound,x_graphindexes=X,y_graphindexes=Y)
        # first we create Base Similarity Matrix B
        # for that first we need the max distance in the matrix
        if self.MaxD is None:
            self.MaxD = np.max(D)
        B = self.MaxD - D
        if self.diffusion_Kernel == "exp_diff_kernel":
            K =np.exp(self.llambda * B)
        elif self.diffusion_Kernel == "von_Neumann_diff_kernel":
            B_exp =B
            llambda_exp = self.llambda
            K = llambda_exp * B_exp
            for i in range(1, 10):  # Example for 10 iterations
                B_exp = B_exp * B
                llambda_exp = llambda_exp * self.llambda
                K += llambda_exp * B_exp
        else:
            raise ValueError(f"Unknown diffusion kernel: {self.diffusion_Kernel}")
        return K
    def fit_transform(self, X, y=None):
        self.X_fit =X
        K = self.create_diffusion_matrix(X)
        return K
    def transform(self, X):
        K = self.create_diffusion_matrix(X, self.X_fit)
        return K 
    def set_params(self, **params):
        need_new_GED =False
        calculator_params = {}
        for parameter, value in params.items():
            if parameter.startswith("GED_"):
                calculator_params[parameter]= value
                need_new_GED = True
            else:
                setattr(self, parameter, value)
        if need_new_GED:
            if DEBUG:
                print(f"Creating new GED calculator with parameters: {calculator_params}")
            self.ged_calculator = self.ged_calculator.set_params(**calculator_params)
        return self
    @classmethod
    def get_param_grid(cls):
        param_grid = SupportVectorMachine.get_param_grid()
        param_grid.update({
            "llambda": [0.1, 0.2,0.5, 1.0],
            # "Ged_distance_bound": ["Mean-Distance", "LowerBound-Distance", "UpperBound-Distance"],
            "diffusion_Kernel": ["exp_diff_kernel", "von_Neumann_diff_kernel"],
        })
        return param_grid
        
    