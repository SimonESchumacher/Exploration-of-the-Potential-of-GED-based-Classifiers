# Maximum similarity edit path SVC
import sys
import os
sys.path.append(os.getcwd())

from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Models.GEDLIB_SVC import GED_SVC
from Base_Calculator import Base_Calculator
from Dummy_Calculator import Dummy_Calculator
DEBUG = False  # Set to True for debug prints

class MaxSimEdithPath_SVC(SupportVectorMachine):
    """Support Vector Machine with Maximum Similarity Edit Path Kernel
    """
    def __init__(self, C=1.0,
                    kernel_type="precomputed",
                    class_weight=None,
                    ged_calculator:Base_Calculator=None,
                    attributes:dict=None):
        self.kernel = None
        super().__init__(kernel_type=kernel_type,
                        C=C,
                        kernelfunction=self.kernel,
                        kernel_name="MaxSimEditPath",
                        class_weight=class_weight,
                        attributes=attributes)
    
    
    def fit_transform(self, X, y=None):
        pass
    def transform(self, X):
        pass

    
    
                