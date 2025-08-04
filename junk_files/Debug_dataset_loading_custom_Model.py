# Debug dataset loading and splitting in Dataset.py
# Debugs and Config
# Imports
from grakel.datasets import fetch_dataset
import joblib
# Imports
import networkx as nx
import numpy as np
from grakel.graph import Graph # To convert NetworkX to GraKeL Graph objects
import traceback
import os
from grakel.kernels import WeisfeilerLehman
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
LOCAL_DATA_PATH = 'Datasets' # Make sure this points to your extracted MUTAG folder
DOWNLOAD_SOURCE = 'TUD'
DATASET_NAME = 'MUTAG' # Placeholder, will be set later
DOMAIN = 'molecular' # Placeholder, will be set later
DEBUG = True # Set to False to disable debug prints
from Dataset import Dataset
import sys
# add the Models directory to the system path
sys.path.append(os.path.join(os.getcwd(), 'Models'))
from WL_SVC_Kernel import SVC_WeisfeilerLehman

MUTAG = Dataset(name=DATASET_NAME, source=DOWNLOAD_SOURCE, domain=DOMAIN)
X_train, X_test, y_train, y_test = MUTAG.train_test_split(test_size=0.2, random_state=42)
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
# display the attributes of the dataset
if DEBUG:
    print("Dataset Attributes:")
    print(MUTAG.attributes())


 # --- Initialize WL-Kernel and Compute Kernel Matrices ---
print("\nInitializing WeisfeilerLehman Kernel and computing kernel matrices...")
svm_classifier = SVC_WeisfeilerLehman(n_iter=5,C=1.0, normalize_kernel=True) # You can tune n_iter and normalize

# K_train will be a (n_train_samples, n_train_samples) matrix
# K_train = svm_classifier.fit_transform(X_train)
# print(f"Shape of K_train: {K_train.shape}")

# # K_test will be a (n_test_samples, n_train_samples) matrix
# K_test = svm_classifier.transform(X_test)
# print(f"Shape of K_test: {K_test.shape}")

# --- Train and Test SVM Classifier ---
print("\nTraining SVM Classifier with precomputed kernel...")
svm_classifier.fit(X_train, y_train)

print("Making predictions on the test set...")
y_pred = svm_classifier.predict(X_test)

# --- Evaluate Performance ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy of SVM with WL-Kernel: {accuracy:.4f}")

# -- Also Evalute the model with AUC and F1 Score --
# also create graphic for the AUC

from sklearn.metrics import f1_score, roc_auc_score
f1 = f1_score(y_test, y_pred, average='weighted')
roc_auc = roc_auc_score(y_test, y_pred, multi_class='ovr')

print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC Score: {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
