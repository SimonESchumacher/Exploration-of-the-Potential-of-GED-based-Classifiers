import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from Models.WL_SVC_Kernel import SVC_WeisfeilerLehman
from Dataset import Dataset
from grakel.kernels import WeisfeilerLehman, VertexHistogram, EdgeHistogram, RandomWalk

# 1. Load Data
DATA = Dataset(name='MUTAG', source='TUD', domain='bioinformatics')
X, y = DATA.data, DATA.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = DATA.train_test_split(test_size=0.2, random_state=43, stratify=y, shuffle=True)
# 2. Define the Model
# initialize a Kernel

# initialize the Model
model = SVC_WeisfeilerLehman()

# 3. Define the Hyperparameter Grid
# 'C': Regularization parameter. Smaller values specify stronger regularization.
# 'kernel': Specifies the kernel type to be used in the algorithm.
# 'gamma': Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
param_grid = SVC_WeisfeilerLehman.get_param_grid()


# 4. Initialize GridSearchCV
# estimator: The model to tune
# param_grid: The dictionary of hyperparameters and their values
# scoring: The metric to evaluate performance (e.g., 'accuracy', 'f1_macro', 'roc_auc')
# cv: Number of folds for cross-validation. Common choices are 3, 5, or 10.
# verbose: Controls the verbosity: 0 (no output), 1 (some output), 2 (more output)
# n_jobs: Number of CPU cores to use. -1 means use all available cores.
grid_search = GridSearchCV(estimator=model,
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=2,
                           n_jobs=-1)

# 5. Perform the Search
print("Starting GridSearchCV...")
grid_search.fit(X_train, y_train)
print("GridSearchCV finished.")

# 6. Get the Best Hyperparameters and Model
print("\nBest parameters found:", grid_search.best_params_)
print("Best cross-validation score (accuracy):", grid_search.best_score_)

# 7. Evaluate on the Test Set using the best estimator
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test set accuracy with best model: {test_accuracy:.4f}")

# You can also inspect all results
# print("\nAll results:")
# results = pd.DataFrame(grid_search.cv_results_)
# print(results[['param_C', 'param_kernel', 'param_gamma', 'mean_test_score', 'rank_test_score']])