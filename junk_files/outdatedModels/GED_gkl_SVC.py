# Class for Graph Edit Distance Kernel
# imports
import sys
import os
sys.path.append(os.getcwd())

from Models.SupportVectorMachine_Classifier import SupportVectorMachine

from Custom_Kernels.GED_kernel_gkl import GEDKernel
DEBUG = False  # Set to True for debug prints

class GED_SVC(SupportVectorMachine):
    """
    Support Vector Machine with Graph Edit Distance Kernel
    """
    def __init__(self, gamma=0.1, C=1.0,
            method="BRANCH_FAST",
            normalize_ged=True,
            similarity=False,
            kernel_type="precomputed",
                attributes:dict=None):
        
        self.gamma = gamma
        self.kernel = GEDKernel(gamma=self.gamma,method=method,normalize_ged=normalize_ged,similarity=similarity, 
                                )
        if attributes is None:
            attributes = {
                "gamma": self.gamma,
                "method": method,
                "normalize_ged": normalize_ged,
                "similarity": similarity,

            }
        else:
            attributes["gamma"] = self.gamma
            attributes["method"] = method
            attributes["normalize_ged"] = normalize_ged
            attributes["similarity"] = similarity
        # Initialize the Support Vector Machine with the GED kernel
        super().__init__(kernel_type=kernel_type, 
                         C=C, 
                         kernelfunction=self.kernel,
                         kernel_name="GED_kernel",
                         attributes=attributes)	
        if DEBUG:
            print(f"Initialized GED_SVC")
    # Override fit_transform and transform methods
    # NO grakel conversion
    def get_calculator(self):
        """
        Returns the GED calculator instance.
        """
        if self.kernel is None:
            raise RuntimeError("GED calculator is not initialized.")
        return self.kernel.get_calculator()
    
    def fit_transform(self, X, y=None):
        if DEBUG:
            print(f"Fitting GED_SVC with {len(X)} graphs")
        k_train=self.kernel.fit_transform(X, y)
        return  k_train
    def transform(self, X):
        if DEBUG:
            print(f"Transforming with GED_SVC with {len(X)} graphs")
        k_test = self.kernel.transform(X)
        return k_test
    def set_params(self, **params):
        need_new_GED =False
        if DEBUG:
            print(f"Setting parameters for GED_SVC")
        for parameter, value in params.items():
            if parameter in ["gamma", "method", "normalize_ged", "similarity"]:
                if DEBUG:
                    print(f"Setting parameter {parameter} to {value}")
                if parameter == "method":
                    self.kernel.set_method(value, recalculate=True)
                else:
                    setattr(self.kernel, parameter, value)
                    if DEBUG:
                        print(f"Setting parameter {parameter} to {value} in kernel")
                self.kernel.set_params(**{parameter: value})
            else:
                setattr(self, parameter, value)
                if DEBUG:
                    print(f"passing down parameter {parameter} to super")
                super().set_params(**{parameter: value})
        return self
    @classmethod
    def get_param_grid(cls):
        param_grid = SupportVectorMachine.get_param_grid()
        param_grid.update({
            "gamma": [0.01, 0.1, 1.0],
            # "method": ['BRANCH_FAST', 'BIPARTITE', 'HED','BRANCH_COMPACT','WALKS'],
            "normalize_ged": [True, False],
            "similarity": [True, False],
        })

         
   