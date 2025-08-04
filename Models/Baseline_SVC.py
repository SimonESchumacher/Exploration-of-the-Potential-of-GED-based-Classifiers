from grakel.kernels import VertexHistogram, EdgeHistogram
import sys
import os
# impoets from cutom modules
sys.path.append(os.getcwd())

from Custom_Kernels.Combined_Kernel import Combined_Kernel
from Models.SupportVectorMachine_Classifier import SupportVectorMachine

from Graph_Tools import convert_nx_to_grakel_graph
# get Started with more kernels
# Example usage of WeisfeilerLehman kernel
DEBUG = False # Set to False to disable debug prints

class VertrexHistogram_SVC(SupportVectorMachine):
    def __init__(self,kernel_type, C=1.0, random_state=None,attributes=None):
        kernel = VertexHistogram()
        super().__init__(kernel_type=kernel_type, C=C, random_state=random_state, kernelfunction=kernel, kernel_name="VertexHistogram", attributes=attributes)
        if DEBUG:
            print(f"Initialized VertexHistogram_SVC in child class")
    

class EdgeHistogram_SVC(SupportVectorMachine):
    def __init__(self,kernel_type, C=1.0, random_state=None,attributes=None):
        kernel = EdgeHistogram()
        super().__init__(kernel_type=kernel_type, C=C, random_state=random_state, kernelfunction=kernel, kernel_name="EdgeHistogram", attributes=attributes)
        if DEBUG:
            print(f"Initialized EdgeHistogram_SVC in child class")
    
    
class CombinedHistogram_SVC(SupportVectorMachine):
    def __init__(self,kernel_type, C=1.0, random_state=None, attributes=None):
        kernel = Combined_Kernel(kernels=[VertexHistogram(), EdgeHistogram()])
        super().__init__(kernel_type=kernel_type, C=C, random_state=random_state, kernelfunction=kernel, kernel_name="CombinedHistogram", attributes=attributes)
        if DEBUG:
            print(f"Initialized CombinedHistogram_SVC  in child class")
            print(f"Model Name: {self.get_name}")
    

