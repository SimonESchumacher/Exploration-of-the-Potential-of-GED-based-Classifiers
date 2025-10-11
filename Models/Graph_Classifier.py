# Graph Classifier Model Interface

# imports
import joblib
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array
import abc
# import check_is_fitted from sklearn.utils.validation
from sklearn.utils.validation import check_is_fitted
DEBUG = False # Set to False to disable debug prints



class GraphClassifier(BaseEstimator, ClassifierMixin, abc.ABC):
    """
    Abstract Base Class for Graph Classifiers compatible with scikit-learn.
    All concrete subclasses must implement the abstract methods defined here.
    Common functionalities like save/load are implemented here.
    """
    _estimator_type = "classifier"
    response_method = "predict_proba"
    def __init__(self,classifier=None, modelattributes:dict=None,model_name="[NO_NAME]",**kwargs):
        """
        Initializes the GraphClassifier with a classifier and model attributes.

        :param classifier: The classifier instance (e.g., SVC, KNN).
        :param modelattributes: A dictionary of model attributes/hyperparameters.
        :param model_name: A name for the model.
        :param kwargs: Additional keyword arguments.
        """

        self.classifier = classifier # SVC or an KNN Classifer - > defined and passed down in the child classes SupportVectorMachine and KNN_Classifier
        self.modelattributes = modelattributes
        self.attributes = modelattributes
        self.is_fitted_ = False
        self.model_name = model_name
        if DEBUG:
            print(f"Initialized {self.model_name} with attributes: in Parent")

    def get_calculator(self):
        return None
    def check_is_fitted(self):
        if DEBUG:
            print("Checking if the model is fitted... in parent")
        if self.is_fitted_ is False:
            raise ValueError("This model instance semms to not be fitted yet. The attribute `is_fitted_` is set to False. ")
        else:
            return check_is_fitted(self.classifier)
    def get_parmas(self, deep=True):
        """
        Get parameters for this estimator.
        """
        return self.modelattributes
    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Parameter {key} not found in GED_Calculator.")
        return self
    
    def set_class_weights(self, class_weights):
        # Sets class weights for the classifier, if supported.
        # So it has a parameter `class_weight`
        if hasattr(self.classifier, 'set_params'):
            if 'class_weight' in self.classifier.get_params():
                self.classifier.set_params(class_weight=class_weights)
                if DEBUG:
                    print(f"Set class weights: {class_weights}")
            if DEBUG:
                print(f"set the class weights to: {self.classifier.get_params()['class_weight']}")


    @abc.abstractmethod
    def fit(self, X, y=None):
        """
        Fits the graph classifier model.
        (Must be implemented by subclasses as fitting logic varies)
        """
        pass
    
    def prepare_fit(self, X, y=None):
        """
        Runs as first line in fit method.
        """
        if DEBUG:
            print("Preparing fit in parent class...")
        y = check_array(y, ensure_2d=False, dtype=None) # Stellen Sie sicher, dass y ein gültiges Array ist
        self.classes_ = unique_labels(y) # Speichert die eindeutigen Klassen-Labels

    def post_fit(self, X, y=None):
        """
        Sets the fitted status of the model.
        """
        self.X_fit_ = X # WICHTIG: Speichert die Trainingsgraphen für spätere `transform`-Aufrufe
        self.is_fitted_ = True  
    @abc.abstractmethod
    def predict(self, X):
        """
        Predicts class labels for graphs in X.
        (Must be implemented by subclasses as prediction logic varies)
        """
        pass
    @abc.abstractmethod
    def predict_proba(self, X):
        """
        Predicts class probabilities for graphs in X.
        (Must be implemented by subclasses as probability logic varies)
        """
        pass


    @property
    def get_name(self) -> str:
        """
        Returns the name of the classifier.
        (Must be implemented by subclasses as the name is specific)
        """
        return self.model_name

    @property
    def model_attributes(self) -> dict:
        """
        Returns a dictionary of the model's key attributes/hyperparameters.
        (Must be implemented by subclasses as attributes are specific)
        """
        return self.modelattributes
    def save(self, filename):
        """
        Saves the model to a file using joblib.
        """
        if DEBUG:
            print(f"Saving model {self.get_name} to {filename}")
        joblib.dump(self, filename)
    @classmethod
    def load(cls, filename):
        """
        Loads a model from a file using joblib.
        """
        if DEBUG:
            print(f"Loading model from {filename}")
        return joblib.load(filename)
    def __str__(self):
        """
        Returns a string representation of the model.
        """
        return f"{self.get_name}_(Base_Name)"
    def to_string(self):
        """
        Returns a string representation of the model.
        """
        return self.__str__()
    @classmethod
    def get_param_grid(cls):
        """
        Returns a dictionary of hyperparameters for grid search.
        Nothing to specify on the ground level, so returns an empty dict.
        """
        return dict()
    @classmethod
    def get_random_param_space(cls):
        """
        Returns a dictionary of hyperparameters for random search.
        Nothing to specify on the ground level, so returns an empty dict.
        """
        return dict()
    @classmethod
    def random_search_iterations(cls,suggested_iterations=10_000):
        return min(suggested_iterations, cls.model_specific_iterations)

