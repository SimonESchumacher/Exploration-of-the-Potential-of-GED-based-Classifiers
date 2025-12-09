from grakel.kernels import RandomWalk, RandomWalkLabeled
from sklearn.utils.validation import check_array
import numpy as np
import sys
import os
sys.path.append(os.getcwd())
from Models.SVC import support_vector_classifier 
from scipy.stats import loguniform
from config_loader import get_conifg_param
DEBUG = get_conifg_param('baseline_SVC', 'debuging_prints')  # Set to False to disable debug prints


# Support Vector Machine with Random Walk Graph Kernel
class rw_SVC(support_vector_classifier):
    model_specific_iterations = get_conifg_param('Hyperparameter_fields', 'tuning_iterations', type='int')  # Base number of iterations for this model

    def __init__(self, normalize_kernel,rw_kernel_type,p_steps,C=1.0,  kernel_type="precomputed",decay_lambda: float = 0.1, attributes: dict = dict(), **kwargs):
        self.normalize_kernel = normalize_kernel
        self.rw_kernel_type = rw_kernel_type
        self.p_steps = p_steps
        self.decay_lambda = decay_lambda
        self.kernel = RandomWalkLabeled(normalize=self.normalize_kernel, kernel_type=self.rw_kernel_type, p=self.p_steps,lamda=self.decay_lambda, method_type="fast")
        attributes.update({
            "normalize_kernel": self.normalize_kernel,
            "rw_kernel_type": self.rw_kernel_type,
            "p_steps": self.p_steps,
            "decay_lambda": self.decay_lambda
        })
        super().__init__(
            kernel_type=kernel_type,
            C=C,
            kernelfunction=self.kernel,
            kernel_name=f"RandomWalk",
            attributes=attributes,
            **kwargs
        )
        if DEBUG:
            print(f"Initialized SVC_RandomWalk with C={self.C}, in child class")
            print(f"Model Name: {self.get_name}")
    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update({
            "normalize_kernel": self.normalize_kernel,
            "rw_kernel_type": self.rw_kernel_type,
            "p_steps": self.p_steps,
            "decay_lambda": self.decay_lambda
        })
        return params

    def set_params(self, **params):
        # Iterate over provided parameters
        need_new_kernel = False
        for parameter, value in params.items():
            if DEBUG:
                print(f"SVC_RandomWalk: set_params: Setting {parameter} to {value}")

            # Handle parameters that belong to the SVC_RandomWalk itself
            if parameter == 'normalize_kernel':
                self.normalize_kernel = value
                need_new_kernel = True
            elif parameter == 'rw_kernel_type':
                self.rw_kernel_type = value
                need_new_kernel = True
            elif parameter == 'p_steps':
                self.p_steps = value
                need_new_kernel = True
            elif parameter == 'decay_lambda':
                self.decay_lambda = value
                need_new_kernel = True

            # Handle parameters that belong to the parent SVC class
            elif parameter in ['C', 'kernel', 'degree', 'gamma', 'coef0', 'shrinking', 'probability', 'tol', 'cache_size', 'class_weight', 'verbose', 'max_iter', 'decision_function_shape', 'break_ties', 'random_state']:
                if DEBUG:
                    print(f"SVC_RandomWalk: set_params: Passing {parameter} to parent SVC")
                super().set_params(**{parameter: value})
            else:
                if DEBUG:
                    print(f"SVC_RandomWalk: set_params: Unknown parameter {parameter}, ignoring.")

        # If any of the kernel-related parameters changed, update the kernel
        if need_new_kernel:
            if DEBUG:
                print("SVC_RandomWalk: set_params: Updating kernel due to parameter change.")
            self.kernel = RandomWalkLabeled(normalize=self.normalize_kernel, kernel_type=self.rw_kernel_type, p=self.p_steps,lamda=self.decay_lambda, method_type="fast")
            self.kernelfunction = self.kernel

        return self
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
        param_grid = support_vector_classifier.get_param_grid()
        param_grid.update({
            "normalize_kernel": [True, False],
            "rw_kernel_type": ["geometric", "exponential"],
            "p_steps": [-1, 3, 5, 7],
        })
        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_space = support_vector_classifier.get_random_param_space()
        param_space.update({
            "normalize_kernel": [True, False],
            "rw_kernel_type": ["geometric", "exponential"],
            "decay_lambda": loguniform(get_conifg_param('Hyperparameter_fields', 'lambdas_min', type='float'),
                                        get_conifg_param('Hyperparameter_fields', 'lambdas_max', type='float')),
            "p_steps": list(range(get_conifg_param('Hyperparameter_fields', 'random_walk_min_steps', type='int'),
                                  get_conifg_param('Hyperparameter_fields', 'random_walk_max_steps', type='int')+1)
                                  ) + [-1]  # -1 indicates infinite steps
        })
        include_kernel_normalization_options = get_conifg_param('Hyperparameter_fields', 'include_kernel_normalization_options', type='bool')
        if include_kernel_normalization_options:
            param_space['normalize_kernel'] = [True, False]
        return param_space
    

class rw_SVC_unlabeled(rw_SVC):
    def __init__(self, normalize_kernel,rw_kernel_type,p_steps,C=1.0,  kernel_type="precomputed",decay_lambda: float = 0.1, attributes: dict = dict(), **kwargs):
        self.normalize_kernel = normalize_kernel
        self.rw_kernel_type = rw_kernel_type
        self.p_steps = p_steps
        self.decay_lambda = decay_lambda
        self.kernel = RandomWalk(normalize=self.normalize_kernel, kernel_type=self.rw_kernel_type, p=self.p_steps,lamda=self.decay_lambda, method_type="fast")
        attributes.update({
            "normalize_kernel": self.normalize_kernel,
            "rw_kernel_type": self.rw_kernel_type,
            "p_steps": self.p_steps,
            "decay_lambda": self.decay_lambda
        })
        super().__init__(
            normalize_kernel=normalize_kernel,
            rw_kernel_type=rw_kernel_type,
            p_steps=p_steps,
            C=C,
            kernel_type=kernel_type,
            decay_lambda=decay_lambda,
            kernelfunction=self.kernel,
            kernel_name=f"RandomWalk_unlabeled",
            attributes=attributes,
            **kwargs
        )
        if DEBUG:
            print(f"Initialized SVC_RandomWalk_unlabeled with C={self.C}, in child class")
            print(f"Model Name: {self.get_name}")