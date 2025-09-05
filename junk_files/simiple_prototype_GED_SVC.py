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
from Calculators.Prototype_Selction import select_Prototype
DEBUG = False  # Set to True for debug prints

class Simple_Prototype_GED_SVC(Base_GED_SVC):
    def __init__(self,
                 I_size=1,
                 attributes: dict = dict(),
                 ged_calculator: Base_Calculator = None,
                 selection_method="random",
                 **kwargs):
        self.I_size = I_size
        self.selection_method = selection_method
        attributes.update({"I_size": I_size})
        self.kernel_name = "Simple_Prototype_GED"
        super().__init__(attributes=attributes, ged_calculator=ged_calculator, **kwargs)
    
    def initKernel(self, ged_calculator: Base_Calculator = None, KERNEL_comparison_method="Mean-Distance", **kernel_kwargs):
        self.kernel = simple_prototype_GED_Kernel(ged_calculator=ged_calculator, KERNEL_comparison_method=KERNEL_comparison_method, KERNEL_protype_size=self.I_size, KERNEL_selection_method=self.selection_method, **kernel_kwargs)
 
    
    
    @classmethod
    def get_param_grid(cls):
        param_grid = Base_GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update({            
            'kernel_type': ['poly', 'rbf', 'linear'],
            'I_size': [ 3, 5, 10],
            # 'selection_method': ['random', 'stratified_random']
        })
        return param_grid

class simple_prototype_GED_Kernel(Base_Kernel):
    def __init__(self, ged_calculator: Base_Calculator = None, KERNEL_protype_size=1,KERNEL_selection_method="random", attributes: dict = dict(), **kwargs):
        self.prototypes_size = KERNEL_protype_size
        self.selection_method = KERNEL_selection_method
        attributes.update({"KERNEL_protype_size": KERNEL_protype_size, "KERNEL_selection_method": KERNEL_selection_method})
        super().__init__(ged_calculator=ged_calculator, attributes=attributes,KERNEL_protype_size=KERNEL_protype_size, KERNEL_selection_method=KERNEL_selection_method,KERNEL_name="Simple_Prototype_GED", **kwargs)
        if DEBUG:
            print(f"Initialized simple_prototype_GED_Kernel with prototypes_size={self.prototypes_size}, selection_method={self.selection_method}")
        # select the Prototypes


    def fit_transform(self, X, y=None):
        # select the Prototypes
        self.X_fit_graphs_ = X # Store the training graphs for transform method

        self.prototypes = select_Prototype(G=X,ged_calculator=self.ged_calculator,selection_method=self.selection_method,size=self.prototypes_size)
        return self.transform(X)   
    def transform(self, X):
        feature_vectors = np.zeros((len(X), self.prototypes_size))
        for i, g in enumerate(X):
            feature_vectors[i, :] = self.build_feature_vector(g)
        return feature_vectors

    def build_feature_vector(self, g):
        feature_vector = np.zeros(self.prototypes_size)
        for i, g0 in enumerate(self.prototypes):
            feature_vector[i] = self.ged_calculator.compare(g, g0, method=self.comparison_method)
        return feature_vector
    
    def compare(self, g1, g2):
        return self.ged_calculator.compare(g1, g2, method=self.comparison_method)
