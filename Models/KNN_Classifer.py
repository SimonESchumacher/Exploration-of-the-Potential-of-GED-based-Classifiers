# K-NN Classifer 
# imports 
import traceback
import joblib
from sklearn.neighbors import KNeighborsClassifier


# imports from custom modules
import sys
import os
sys.path.append(os.getcwd())
from Models.Graph_Classifier import GraphClassifier
from Graph_Tools import convert_nx_to_grakel_graph
DEBUG = False # Set to False to disable debug prints

class KNN(GraphClassifier):
    def __init__(self, n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30,
                  metric=None,metric_name="unspecified",random_state=None,
                  attributes:dict=None): 
        
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_name = metric_name
        self.random_state = random_state
        classifier = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, algorithm=self.algorithm,
                                          leaf_size=self.leaf_size, metric=self.metric)
        if attributes is None:
            attributes = {
                "n_neighbors": self.n_neighbors,
                "weights": self.weights,
                "algorithm": self.algorithm,
                "metric": self.metric_name,
            }
        else:
            attributes["n_neighbors"] = self.n_neighbors
            attributes["weights"] = self.weights
            attributes["algorithm"] = self.algorithm
            attributes["metric"] = self.metric_name
        super().__init__(
            classifier=classifier,
            model_name=f"({n_neighbors})-NN_Classifier_{self.metric_name}",
            modelattributes=attributes
        )
        
        if DEBUG:
            print(f"Initialized KNNClassifier with n_neighbors={self.n_neighbors}, weights={self.weights}, algorithm={self.algorithm}")
            print(f"Model Name: {self.get_name}")
    
    def fit_transform(self, X, y=None):
        return self.metric.fit_transform(X, y)
    def transform(self, X):
        return self.metric.transform(X)
    def get_params(self, deep=True):
        return super().get_params(deep)
    def set_params(self, **params):
        # Iterate over provided parameters
        for parameter, value in params.items():
            if DEBUG:
                print(f"KNN: set_params: Setting {parameter} to {value}")
            # Handle parameters that belong to the KNNClassifier itself
            if parameter == 'n_neighbors':
                self.n_neighbors = value
                self.classifier.set_params(n_neighbors=self.n_neighbors)
            elif parameter == 'weights':
                self.weights = value
                self.classifier.set_params(weights=self.weights)
            elif parameter == 'algorithm':
                self.algorithm = value
                self.classifier.set_params(algorithm=self.algorithm)
            elif parameter == 'leaf_size':
                self.leaf_size = value
                self.classifier.set_params(leaf_size=self.leaf_size)
            elif parameter == 'metric':
                self.metric = value
                self.classifier.set_params(metric=self.metric)
            else:
                
                super().set_params(**{parameter: value})
        if DEBUG:
            print(f"KNN: set_params: Set parameters for KNNClassifier.")
        return self
    def fit(self, X, y=None):
        """
        Fits the KNN classifier model to the provided graph data.
        """
        if DEBUG:
            print("Fitting KNNClassifier...")
        self.prepare_fit(X, y)
        X =self.fit_transform(X, y)

        self.classifier.fit(X, y)
        self.post_fit(X, y)
        if DEBUG:
            print("KNNClassifier fitted successfully.")
        return self
    def predict(self, X):
        """
        Predicts class labels for graphs in X using the fitted KNN classifier.
        """
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"Model is not fitted yet: {e}")
            traceback.print_exc()
            raise e
        X = self.transform(X)
        try:
            y_pred = self.classifier.predict(X)
        except Exception as e:
            print(f"Error during KNN prediction: {e}")
            traceback.print_exc()
            raise e
        return y_pred
    def predict_proba(self, X):
        """
        Predicts class probabilities for graphs in X using the fitted KNN classifier.
        """
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"Model is not fitted yet: {e}")
            traceback.print_exc()
            raise e
        X = self.transform(X)
        try:
            y_proba = self.classifier.predict_proba(X)
        except Exception as e:
            print(f"Error during probability prediction: {e}")
            traceback.print_exc()
            raise e
        return y_proba
    def save(self, filename):
        """
        Saves the fitted KNN classifier model to a file.
        """
        if DEBUG:
            print(f"Saving KNNClassifier model to {filename}")
        joblib.dump(self, filename=filename)
    @classmethod
    def load(cls, filename):
        """
        Loads a KNN classifier model from a file.
        """
        if DEBUG:
            print(f"Loading KNNClassifier model from {filename}")
        return joblib.load(filename)
    def __str__(self):
        if self.algorithm == 'auto':
            return f"{self.metric_name} - ({self.n_neighbors})-NN"
        else:
            return f"{self.metric_name} - ({self.n_neighbors})-NN ({self.algorithm})"
    def to_string(self):
        """
        Returns a string representation of the KNN classifier.
        """
        if self.algorithm == 'auto':
            return f"{self.metric_name} - ({self.n_neighbors})-NN"
        else:
            return f"{self.metric_name} - ({self.n_neighbors})-NN ({self.algorithm})"

    @classmethod
    def get_param_grid(cls):
        param_grid = GraphClassifier.get_param_grid()
        param_grid.update({
            'n_neighbors': [1, 3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [10, 20, 30, 40, 50],
            'metric': ['euclidean', 'manhattan', 'minkowski', 'chebyshev']
        })

        return param_grid
