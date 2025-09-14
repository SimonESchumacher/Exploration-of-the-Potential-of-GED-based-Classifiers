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
from Calculators.Prototype_Selction import select_Prototype, Prototype_Selector,Select_Prototypes
DEBUG = False  # Set to True for debug prints

class Simple_Prototype_GED_SVC(Base_GED_SVC):
    def __init__(self,
                 attributes: dict = dict(),
                 ged_calculator: Base_Calculator = None,
                 **kwargs):

        self.kernel_name = "Simple_Prototype_GED"
        super().__init__(attributes=attributes, ged_calculator=ged_calculator, **kwargs)
        attributes.update(self.feature_extractor.attributes)
    
    def initKernel(self, ged_calculator: Base_Calculator = None, KERNEL_comparison_method="Mean-Distance", **kernel_kwargs):
        self.kernel = None
        self.feature_extractor = simple_prototype_GED_Feature_Extractor(ged_calculator=ged_calculator, KERNEL_comparison_method=KERNEL_comparison_method, **kernel_kwargs)

    def fit_transform(self, X, y=None):
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Fitting GED_SVC with {len(X)} graphs")
        k_train=self.feature_extractor.fit_transform(X, y)
        return  k_train
    
    def transform(self, X):
        X=[int(X[i].name) for i in range(len(X))]
        if DEBUG:
            print(f"Transforming with GED_SVC with {len(X)} graphs")
        k_test = self.feature_extractor.transform(X)
        return k_test
    def set_params(self, **params):
        for parameter, value in params.items():
            if DEBUG:
                print(f"SVC: set_params: Setting {parameter} to {value}")
            # Directly set attribute if it exists
            if hasattr(self, parameter):
                setattr(self, parameter, value)
                # If the parameter also exists in the classifier, update it there too
                if hasattr(self.classifier, parameter):
                    self.classifier.set_params(**{parameter: value})
            # Pass classifier__* params to classifier
            elif parameter.startswith('classifier_'):
                self.classifier.set_params(**{parameter.split('_', 1)[1]: value})
            # Pass kernel__* params to kernel
            elif parameter.startswith('KERNEL_') and hasattr(self.feature_extractor, 'set_params'):
                self.feature_extractor.set_params(**{parameter.split('_', 1)[1]: value})
            else:
                # Fallback to parent class
                super().set_params(**{parameter: value})
        if DEBUG:
            print(f"SVC: set_params: Set parameters for SupportVectorMachine.")
        return self
    def get_calculator(self):
        """
        Returns the GED calculator instance.
        """  
        return self.feature_extractor.ged_calculator
    @classmethod
    def get_param_grid(cls):
        param_grid = simple_prototype_GED_Feature_Extractor.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update({            
            'kernel_type': ['poly', 'rbf', 'linear'],
            # 'selection_method': ['random', 'stratified_random']
        })
        return param_grid

class simple_prototype_GED_Feature_Extractor:
    def __init__(self, ged_calculator: Base_Calculator = None,
                  KERNEL_comparison_method="Mean-Distance",
                  KERNEL_prototype_size=8,
                  KERNEL_classwise=False, KERNEL_single_class=False,
                  KERNEL_selection_method="RPS",
                  attributes: dict = dict(), **kwargs):
        self.ged_calculator = ged_calculator
        self.comparison_method = KERNEL_comparison_method
        self.prototypes_size = KERNEL_prototype_size
        self.classwise = KERNEL_classwise
        self.single_class = KERNEL_single_class
        self.selection_method = KERNEL_selection_method
        self.prototypes = None
        attributes.update({"KERNEL_comparison_method": KERNEL_comparison_method})
        attributes.update({"KERNEL_prototype_size": self.prototypes_size,
                           "KERNEL_classwise": self.classwise,
                           "KERNEL_single_class": self.single_class,
                           "KERNEL_selection_method": self.selection_method})
        
        self.attributes = attributes
        if DEBUG:
            print(f"Initialized simple_prototype_GED_Kernel with prototypes_size={self.prototypes_size}, selection_method={self.selection_method}")
    

    def fit_transform(self, X, y=None):
        # select the Prototypes
        self.X_fit_graphs_ = X # Store the training graphs for transform method

        self.prototypes = Select_Prototypes(X, y=y, ged_calculator=self.ged_calculator, size=self.prototypes_size, classwise=self.classwise, single_class=self.single_class, selection_method=self.selection_method)
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
    
    def set_params(self, **params):
        """
        Set parameters for the GED kernel.
        will probably be called by the SVC using this kernel.

        """
        for key, value in params.items():
            if key.startswith("KERNEL_"):
                key = key[len("KERNEL_"):]  # Remove the prefix
                # set the parameter in the kernel
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    print(f"Warning: Parameter {key} not found in GEDKernel. Skipping.")
    @classmethod
    def get_param_grid(cls):
        param_grid = {
            "KERNEL_prototype_size": [1, 3, 5, 8, 10],
            "KERNEL_classwise": [False, True],
            "KERNEL_single_class": [False, True],
            "KERNEL_selection_method": ["RPS", "CPS", "BPS", "TPS", "SPS", "k-CPS"]
        }
        return param_grid
