# Class for Graph Edit Distance Kernel
# imports
import sys
import os
sys.path.append(os.getcwd())

from Models.SupportVectorMachine_Classifier import SupportVectorMachine

from Custom_Kernels.GED_kernel import GEDKernel
DEBUG = False  # Set to True for debug prints

class GED_SVC(SupportVectorMachine):
    """
    Support Vector Machine with Graph Edit Distance Kernel
    """
    def __init__(self, gamma=0.1, C=1.0,
                node_del_cost=1.0, node_ins_cost=1.0,
                edge_del_cost=1.0, edge_ins_cost=1.0,
                approximation=None,kernel_type="precomputed",
                attributes:dict=None):
        
        self.gamma = gamma
        self.node_del_cost = node_del_cost
        self.node_ins_cost = node_ins_cost
        self.edge_del_cost = edge_del_cost
        self.edge_ins_cost = edge_ins_cost
        self.approximation = approximation
        self.kernel = GEDKernel(gamma=self.gamma, 
                                node_del_cost=self.node_del_cost,
                                node_ins_cost=self.node_ins_cost,
                                edge_del_cost=self.edge_del_cost,
                                edge_ins_cost=self.edge_ins_cost,
                                approximation=self.approximation)
        if attributes is None:
            attributes = {
                "gamma": self.gamma,
                "node_del_cost": self.node_del_cost,
                "node_ins_cost": self.node_ins_cost,
                "edge_del_cost": self.edge_del_cost,
                "edge_ins_cost": self.edge_ins_cost,
                "approximation": self.approximation
            }
        else:
            attributes["gamma"] = self.gamma
            attributes["node_del_cost"] = self.node_del_cost
            attributes["node_ins_cost"] = self.node_ins_cost
            attributes["edge_del_cost"] = self.edge_del_cost
            attributes["edge_ins_cost"] = self.edge_ins_cost
            attributes["approximation"] = self.approximation
        super().__init__(kernel_type=kernel_type, 
                         C=C, 
                         kernelfunction=self.kernel,
                         kernel_name="GED_kernel",
                         attributes=attributes)	
        if DEBUG:
            print(f"Initialized GED_SVC")
    # Override fit_transform and transform methods
    # NO grakel conversion
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
            if parameter == "gamma":
                self.gamma = value
            elif parameter == "node_del_cost":
                self.node_del_cost = value
                need_new_GED = True
            elif parameter == "node_ins_cost":
                self.node_ins_cost = value
                need_new_GED = True
            elif parameter == "edge_del_cost":
                self.edge_del_cost = value
                need_new_GED = True
            elif parameter == "edge_ins_cost":
                self.edge_ins_cost = value
                need_new_GED = True
            elif parameter == "approximation":
                self.approximation = value
                need_new_GED = True
            else:
                if DEBUG:
                    print(f"passing down parameter {parameter} to super")
                super().set_params(**{parameter: value})
        if need_new_GED:
            if DEBUG:
                print(f"Creating new GED kernel with updated parameters")
            self.kernel = GEDKernel(gamma=self.gamma, 
                                    node_del_cost=self.node_del_cost,
                                    node_ins_cost=self.node_ins_cost,
                                    edge_del_cost=self.edge_del_cost,
                                    edge_ins_cost=self.edge_ins_cost,
                                    approximation=self.approximation)
        return self
    @classmethod
    def get_param_grid(cls):
        param_grid = SupportVectorMachine.get_param_grid()
        param_grid.update({
            "gamma": [0.01, 0.1, 1.0],
            "node_del_cost": [0.5, 1.0, 2.0],
            "node_ins_cost": [0.5, 1.0, 2.0],
            "edge_del_cost": [0.5, 1.0, 2.0],
            "edge_ins_cost": [0.5, 1.0, 2.0],
            "approximation": [None, "greedy", "Hausdorff","Bipartite"]
        })

         
   