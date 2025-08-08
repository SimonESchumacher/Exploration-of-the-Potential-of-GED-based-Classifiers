# GED K-NN Classifier
# imports
import traceback
from sklearn.neighbors import KNeighborsClassifier


# imports from custom modules
import sys
import os
sys.path.append(os.getcwd())
from Models.Graph_Classifier import GraphClassifier
from Models.KNN_Classifer import KNN
from Graph_Tools import convert_nx_to_grakel_graph
from GED import GraphEditDistanceCalculator
DEBUG = False  # Set to False to disable debug prints

class GED_KNN(KNN):

    def __init__(self,approximation=None,
                 node_del_cost=1.0,node_ins_cost=1.0,
                 edge_del_cost=1.0,edge_ins_cost=1.0,
                 n_neighbors=1, weights='uniform', algorithm='auto',leaf_size=30,
                 attributes : dict=None ):
        """
        Initialize the GED K-NN Classifier with the given parameters.
        """
        self.approximation = approximation
        self.node_del_cost = node_del_cost
        self.node_ins_cost = node_ins_cost
        self.edge_del_cost = edge_del_cost
        self.edge_ins_cost = edge_ins_cost
        self.ged_calculator = GraphEditDistanceCalculator(
            approximation=approximation,
            node_deletion_cost=node_del_cost,
            node_insertion_cost=node_ins_cost,
            edge_deletion_cost=edge_del_cost,
            edge_insertion_cost=edge_ins_cost
        )
        if attributes is None:
            attributes = {
                "approximation": approximation,
                "node_del_cost": node_del_cost,
                "node_ins_cost": node_ins_cost,
                "edge_del_cost": edge_del_cost,
                "edge_ins_cost": edge_ins_cost
            }
        else:
            attributes["approximation"] = approximation
            attributes["node_del_cost"] = node_del_cost
            attributes["node_ins_cost"] = node_ins_cost
            attributes["edge_del_cost"] = edge_del_cost
            attributes["edge_ins_cost"] = edge_ins_cost
        super().__init__(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm,
            leaf_size=leaf_size,
            metric="precomputed",
            metric_name="GED",
            random_state=None,
            attributes=attributes
        )
        if DEBUG:
            print(f"Initialized GED_KNNClassifie")
        
    def fit_transform(self, X, y=None):
        """
        save the traiing Graphs and transform Data into matrix.
        """
        self.X_fit =X.copy()
        distance_matrix=self.ged_calculator.similarity_matrix(X)
        if DEBUG:
            print(f"Fitting {len(X)} graphs into distance matrix.")
            print("Fitted Graphs:")
            print(self.X_fit)
            print("Distance Matrix:")
            print(distance_matrix)
            print("Initial Data")
            print(X)
        
        return distance_matrix
    def transform(self, X):
        """
        Transform the input graphs into a distance matrix using the GED calculator.
        """
        if DEBUG:
            print(f"Transforming {len(X)} graphs into distance matrix.")
            print(self.X_fit)
            print(X)
        distance_matrix = self.ged_calculator.similarity_matrix(X,self.X_fit)
        return distance_matrix
    def get_params(self,deep=True):
        """
        Get the parameters of the GED K-NN Classifier.
        """
        params = super().get_params(deep=deep)
        params.update({
            "approximation": self.approximation,
            "node_del_cost": self.node_del_cost,
            "node_ins_cost": self.node_ins_cost,
            "edge_del_cost": self.edge_del_cost,
            "edge_ins_cost": self.edge_ins_cost
        })
        return params
    def set_params(self, **params):
        """
        Set the parameters of the GED K-NN Classifier.
        """
        need_new_GED_calculator = False
        for parameter, value in params.items():
            if DEBUG:
                print(f"Setting parameter {parameter} to {value}")
            if parameter == "approximation":
                self.approximation = value
                need_new_GED_calculator = True
            elif parameter == "node_del_cost":
                self.node_del_cost = value
                need_new_GED_calculator = True
            elif parameter == "node_ins_cost":
                self.node_ins_cost = value
                need_new_GED_calculator = True
            elif parameter == "edge_del_cost":
                self.edge_del_cost = value
                need_new_GED_calculator = True
            elif parameter == "edge_ins_cost":
                self.edge_ins_cost = value
                need_new_GED_calculator = True
            else:
                super().set_params(**{parameter: value})
        if need_new_GED_calculator:
            self.ged_calculator = GraphEditDistanceCalculator(
                approximation=self.approximation,
                node_deletion_cost=self.node_del_cost,
                node_insertion_cost=self.node_ins_cost,
                edge_deletion_cost=self.edge_del_cost,
                edge_insertion_cost=self.edge_ins_cost
            )
            if DEBUG:
                print("Reinitialized GED calculator with new parameters.")
        return self
    @classmethod
    def get_param_grid(cls):
        """
        Get the parameter grid for hyperparameter tuning.
        """
        param_grid = super().get_param_grid()
        param_grid.update({
            "approximation": [None, "greedy", "Hausdorff","Bipartite"],
            "node_del_cost": [1.0, 2.0],
            "node_ins_cost": [1.0, 2.0],
            "edge_del_cost": [1.0, 2.0],
            "edge_ins_cost": [1.0, 2.0]
        })
        return param_grid

        

        
        
