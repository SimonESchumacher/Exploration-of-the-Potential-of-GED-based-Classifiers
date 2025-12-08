from grakel import VertexHistogram, EdgeHistogram, ShortestPath
from grakel.kernels import WeisfeilerLehman
from sklearn.svm import SVC
from sklearn.utils.validation import check_X_y, check_array

# liabry to save Model:
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_array
import numpy as np
import abc
import sys
import os
import traceback
sys.path.append(os.getcwd())
from scipy.stats import randint
from Models.Graph_Classifier import GraphClassifier
from Models.SupportVectorMachine_Classifier import SupportVectorMachine
from config_loader import get_conifg_param

DEBUG = get_conifg_param('baseline_SVC', 'debuging_prints')  # Set to False to disable debug prints
# import GraphClassifier as gc
# Class of the SVC_WeisfeilerLehman is an extension of the SVC class
class WeisfeilerLehman_SVC(SupportVectorMachine):
    model_specific_iterations = get_conifg_param('Hyperparameter_fields', 'tuning_iterations', type='int')  # Base number of iterations for this model
    def __init__(self, n_iter=5, C=1.0,normalize_kernel=True, random_state=None,base_kernel=(VertexHistogram,{ 'sparse': True }),kernel_type="precomputed",attributes:dict=dict(),**kwargs):
        self.n_iter = n_iter
        self.normalize_kernel = normalize_kernel
        self.base_kernel = base_kernel
        self.kernel = WeisfeilerLehman(n_iter=self.n_iter, normalize=self.normalize_kernel, base_graph_kernel=self.base_kernel)
        attributes.update({
            "n_iter": self.n_iter,
            "normalize_kernel": self.normalize_kernel,
            "base_kernel": self.base_kernel.__class__.__name__
        })
        super().__init__(
            kernel_type=kernel_type,
            C=C,
            random_state=random_state,
            kernelfunction=self.kernel,
            kernel_name="WL-ST",
            
            attributes=attributes,
            **kwargs
        )
        if DEBUG:
            print(f"Initialized SVC_WeisfeilerLehman with n_iter={self.n_iter}, C={self.C}, in child class")
            print(f"Model Name: {self.get_name}")

    
    def set_params(self, **params):
        # Iterate over provided parameters
        need_new_kernel = False
        for parameter, value in params.items():
            if DEBUG:
                print(f"SVC_WeisfeilerLehman: set_params: Setting {parameter} to {value}")

            # Handle parameters that belong to the SVC_WeisfeilerLehman itself
            if parameter == 'n_iter':
                self.n_iter = value
                # If n_iter changes, we need to update the kernel
                need_new_kernel = True
            elif parameter == 'normalize_kernel':
                self.normalize_kernel = value
                # If normalize_kernel changes, we need to update the kernel
                need_new_kernel = True
            # Handle parameters that might be passed to the underlying classifier (SVC)
            # This is robust if GraphClassifier's set_params correctly handles 'classifier__'
            else:
                super().set_params(**{parameter: value})
        if need_new_kernel:
            self.kernel = WeisfeilerLehman(n_iter=self.n_iter, normalize=self.normalize_kernel, base_graph_kernel=self.base_kernel)
            self.kernel_fuct = self.kernel
        return self
    def predict(self, X):
        try:
            self.check_is_fitted()
        except ValueError as e:
            print(f"Model is not fitted: {e}")
            traceback.print_exc()
            raise e
        X = self.transform(X)
        X = check_array(X, accept_sparse=True, dtype=np.float64)
        if X.shape[1] != len(self.X_fit_): # n_features_in_ des SVM ist die Anzahl der Trainingssamples
             raise ValueError(f"Shape of X for prediction ({X.shape}) does not match "
                              f"the number of training samples used during fit ({len(self.X_fit_)}). "
                              "Ensure the input to predict() is a list of graphs compatible with the fitted kernel.")

        # Vorhersagen mit dem zugrunde liegenden SVM
        try:
            y_pred = self.classifier.predict(X)
        except Exception as e:
            # print traceback
            print("Error during prediction:")  
            traceback.print_exc()
            raise ValueError(f"Error during prediction: {e}")
        return y_pred
    
    def predict_proba(self, X):
        """
        Gibt Klassenwahrscheinlichkeiten für neue Graphen zurück.

        Parameter
        ----------
        X : Liste von GraKeL-Graph-Objekten
            Die Eingabedaten zum Vorhersagen.

        Gibt zurück
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Die vorhergesagten Klassenwahrscheinlichkeiten.
        """
        self.check_is_fitted()
        if not hasattr(self.classifier, 'predict_proba'):
            raise AttributeError("Der zugrunde liegende SVC unterstützt keine Wahrscheinlichkeitsvorhersage. "
                                 "Stellen Sie sicher, dass 'probability=True' im SVC-Konstruktor gesetzt ist, "
                                 "was jedoch die Trainingszeit erhöht.")
        X = self.transform(X)
        X = check_array(X, accept_sparse=True, dtype=np.float64)

        if X.shape[1] != len(self.X_fit_):
             raise ValueError(f"Shape of X for predict_proba ({X.shape}) does not match "
                              f"the number of training samples used during fit ({len(self.X_fit_)}).")

        return self.classifier.predict_proba(X)

    
    @classmethod
    def get_param_grid(cls):
        param_grid = SupportVectorMachine.get_param_grid()
        param_grid.update({
            'n_iter': [1,2,3,4, 5, 7]
        })
        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_space = SupportVectorMachine.get_random_param_space()
        param_space.update({
            'n_iter': randint(get_conifg_param('Hyperparameter_fields', 'wl_depth_min', type='int'),
                               get_conifg_param('Hyperparameter_fields', 'wl_depth_max', type='int')),
        })
        include_kernel_normalization_options = get_conifg_param('Hyperparameter_fields', 'include_kernel_normalization_options', type='bool')
        if include_kernel_normalization_options:
            param_space['normalize_kernel'] = [True, False]
        return param_space
       
    

