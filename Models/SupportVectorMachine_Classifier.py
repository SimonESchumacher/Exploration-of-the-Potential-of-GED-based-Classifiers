# Kernels
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array

# liabry to save Model:
import joblib
import sys
import os
import traceback
import abc
sys.path.append(os.getcwd())
from Graph_Tools import convert_nx_to_grakel_graph
from Models.Graph_Classifier import GraphClassifier
DEBUG = False # Set to False to disable debug prints

class SupportVectorMachine(GraphClassifier):
    # Support Vector Machine Classifier for Graphs
    # with different Kernels
    def __init__(self, kernel_type="precomputed", C=1.0, random_state=None,kernelfunction=None,kernel_name="unspecified",class_weight=None,attributes=None):
        if kernelfunction is None:
            raise ValueError("kernelfunction must be provided.")
        self.kernel = kernelfunction
        self.kernel_name = kernel_name
        self.kernel_type = kernel_type
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        classifier = SVC(kernel=self.kernel_type, C=self.C, random_state=self.random_state,class_weight=class_weight)
        default_attributes = {
            "Kernel_type": self.kernel_type,
            "Kernel": self.kernel_name,
            "Classifier_C": self.C,
            "model_random_state": self.random_state,
            "class_weight": self.class_weight
        }
        if attributes is None:
            attributes = default_attributes
        else:
            attributes.update(default_attributes)
        super().__init__(
            classifier=classifier,
            model_name=f"SVC_{self.kernel_name}_{self.kernel_type}",
            modelattributes=attributes
        )
        self.is_fitted_ = False
        if DEBUG:
            print(f"Initialized SupportVectorMachine with kernel={self.kernel_type}, C={self.C}, in child class")
            print(f"Model Name: {self.get_name}")
        
    def fit_transform(self, X, y=None):
        X = [convert_nx_to_grakel_graph(g) for g in X]
        return self.kernel.fit_transform(X)

    def transform(self, X):
        X = [convert_nx_to_grakel_graph(g) for g in X]
        return self.kernel.transform(X)
    def get_params(self, deep=True):
        return super().get_params(deep)
    def set_params(self, **params):
        # Iterate over provided parameters
        for parameter, value in params.items():
            if DEBUG:
                print(f"SVC: set_params: Setting {parameter} to {value}")
            # Handle parameters that belong to the SVC_WeisfeilerLehman itself
            if parameter == 'C':
                self.C = value
                # Update the C parameter of the internal SVC classifier
                self.classifier.set_params(C=self.C)
            elif parameter == 'kernel_type':
                self.kernel_type = value
                # If kernel_type changes, we need to update the SVC
                self.classifier.set_params(kernel=self.kernel_type)
            elif parameter == 'random_state':
                self.random_state = value
                self.classifier.set_params(random_state=self.random_state)
            elif parameter.startswith('classifier_'):
                # Pass it to the underlying classifier
                self.classifier.set_params(**{parameter.split('_')[1]: value})
            # Handle parameters that might be passed to the underlying kernel (WeisfeilerLehman)
            elif parameter.startswith('kernel_'):
                # Pass it to the underlying kernel
                self.kernel.set_params(**{parameter.split('_')[1]: value})
            # Handle parameters that might be passed to the underlying classifier (SVC)
            # This is robust if GraphClassifier's set_params correctly handles 'classifier__'
            else:
                
                super().set_params(**{parameter: value})

        # Call the parent's set_params, ensuring it handles its own parameters
        # and potentially classifier__ params if it's set up to do so.
        # This is where your GraphClassifier's handling of its 'classifier' member comes into play.
        if DEBUG:
            print(f"SVC: set_params: Set parameters for SupportVectorMachine.")
        return self
    def fit(self, X, y=None):
        """
        Fits the SVC model to the graph data.
        """
        if DEBUG:
            print("Fitting SVC model...")
        self.prepare_fit(X, y)
        X = self.fit_transform(X, y)
        self.classifier.fit(X, y)
        self.post_fit(X, y)
        if DEBUG:
            print("SVC model fitted successfully.")
        return self
    def predict(self, X):
        """
        Predicts the labels for the input graph data.
        """
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"SVC is not fitted yet: {e}")
            traceback.print_exc()
            raise e
        X = self.transform(X)
        try:
            y_pred = self.classifier.predict(X)
        except Exception as e:
            print(f"Error during prediction: {e}")
            traceback.print_exc()
            raise e
        return y_pred
    def predict_proba(self, X):
        """
        Predicts the class probabilities for the input graph data.
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
        Saves the SVC to a file.
        """
        if DEBUG:
            print(f"Saving KNNClassifier model to {filename}")
        joblib.dump(self, filename=filename)
    @classmethod
    def load(cls, filename):
        """
        Loads the SVC from a file.
        """
        if DEBUG:
            print(f"Loading model from {filename}")
        return joblib.load(filename)
    def __str__(self):
        return (f"SVC_{self.kernel_name}(kernel_type={self.kernel_type}, C={self.C}, "
                f"random_state={self.random_state})")
    def to_string(self):
            return (f"SVC_{self.kernel_name}(kernel_type={self.kernel_type}, C={self.C}, "
                f"random_state={self.random_state})")
    @classmethod
    def get_param_grid(cls):
        param_grid = GraphClassifier.get_param_grid()
        param_grid.update({
            # 'C': [0.1, 0.5, 1.0, 10.0],
            'kernl_type': ['linear', 'poly', 'rbf', 'precomputed'],
            # 'kernel_type': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
            'class_weight': [None, 'balanced'],
        })
        return param_grid
        

     
