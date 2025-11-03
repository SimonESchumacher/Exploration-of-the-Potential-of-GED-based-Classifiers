# Array Classifier Test
from Calculators.GED_Calculator import build_GED_calculator, build_Heuristic_calculator, build_Randomwalk_GED_calculator
# from Calculators.Product_GRaphs import RandomWalkCalculator
from Dataset import Dataset
from Experiment import experiment
import sys
import os
import traceback

from Models.SVC.GED.hybrid_prototype_selector import HybridPrototype_GED_SVC
# add the current directory to the system path
sys.path.append(os.getcwd())
from Models.SVC.GED.RandomWalk_edit import Random_Walk_edit_accelerated, Random_walk_edit_SVC
from Models.SVC.WeisfeilerLehman_SVC import WeisfeilerLehman_SVC
from Models.Graph_Classifier import GraphClassifier
from Models.SVC.Baseline_SVC import VertexHistogram_SVC,EdgeHistogram_SVC, CombinedHistogram_SVC, NX_Histogram_SVC
from Models.Blind_Classifier import Blind_Classifier
from Models.Random_Classifer import Random_Classifier
from Models.KNN_Classifer import KNN
from Models.SVC.GED.Trivial_GED_SVC import Trivial_GED_SVC
from Custom_Kernels.GEDLIB_kernel import GEDKernel
from Calculators.Dummy_Calculator import Dummy_Calculator
from Calculators.Base_Calculator import Base_Calculator
from Calculators.GEDLIB_Caclulator import GEDLIB_Calculator
from Models.SVC.GED.GED_Diffu_SVC import DIFFUSION_GED_SVC
from Models.SVC.GED.Zero_GED_SVC import ZERO_GED_SVC
from Models.SVC.GED.simiple_prototype_GED_SVC import Simple_Prototype_GED_SVC
from Models.SVC.Base_GED_SVC import Base_GED_SVC, set_global_ged_calculator
from Models.KNN.GEDLIB_KNN import GED_KNN, set_global_ged_calculator_KNN
from Models.KNN.feature_KNN import Feature_KNN
import pandas as pd
from io_Manager import IO_Manager
N_JOBS = 5
SPLIT=0.2 # 10% test size alternatively 0.2 for 20%
NUM_TRIALS=3
TEST_TRIAL=False
Test_DF=pd.DataFrame()
DATASET_NAME="MUTAG"
GED_BOUND="Vertex"  # "UpperBound-Distance", "Mean-Distance", "LowerBound-Distance"
EXPERIMENT_NAME="test_llambda"
ONLY_ESTIMATE=False
PRELOAD_CALCULATORS=True
USE_NODE_LABELS="label"
USE_EDGE_LABELS="label"


def get_classifier(ged_calculator):
    set_global_ged_calculator_KNN(ged_calculator)

    # return ZERO_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="precomputed", selection_split="classwise",prototype_size=7, aggregation_method="sum",dataset_name=DATASET.name,selection_method="k-CPS")
    # return Random_walk_edit_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, decay_lambda=0.1, max_walk_length=-1, C=1.0,kernel_type="precomputed", class_weight='balanced')
    # random_walk_calculator = RandomWalkCalculator(ged_calculator=ged_calculator, llambda_samples=[0.005,0.01,0.03,0.05,0.1,0.2,0.45,0.89], dataset=DATASET,ged_method=GED_BOUND)
    random_walk_calculator = build_Randomwalk_GED_calculator(ged_calculator=ged_calculator)
    return Random_Walk_edit_accelerated(ged_calculator=ged_calculator, ged_bound=GED_BOUND, decay_lambda=0.1, max_walk_length=-1, C=1.0,kernel_type="precomputed", class_weight='balanced', random_walk_calculator=random_walk_calculator)
    # return Trivial_GED_SVC(calculator_id=ged_calculator.get_identifier_name(),ged_bound=GED_BOUND, C=1.0,kernel_type="precomputed", class_weight='balanced',similarity_function='k1',llambda=100)
    # return  Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=1, selection_method="TPS", selection_split="all",dataset_name=DATASET.name)

    # return Feature_KNN(vector_feature_list=["VertexHistogram","density","Prototype-Distance"], dataset_name=DATASET.name, prototype_size=5, selection_split="all", selection_method="TPS", metric="minkowski", calculator_id=ged_calculator.get_identifier_name(), ged_bound=GED_BOUND, n_neighbors=5, weights='uniform', algorithm='auto')

    # return HybridPrototype_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=5, selection_method="TPS", selection_split="all",dataset_name=DATASET.name, vector_feature_list=["density"], node_label_tag="label", edge_label_tag="label")
    # return GED_KNN(ged_calculator=ged_calculator, ged_bound=GED_BOUND, n_neighbors=7, weights='uniform', algorithm='auto')
    # return CombinedHistogram_SVC(kernel_type="rbf", C=1.0, class_weight='balanced')
    # classifier = Random_walk_edit_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", decay_lambda=0.1, max_walk_length=-1, C=1.0,kernel_type="precomputed", class_weight='balanced')
    # return Simple_Prototype_GED_SVC(ged_calculator=ged_calculator, ged_bound=GED_BOUND, C=1.0,kernel_type="poly", class_weight='balanced',prototype_size=8, selection_method="k-CPS", selection_split="all",dataset_name=DATASET_NAME)
# classifier = Trivial_GED_SVC(kernel_type='precomputed',ged_calculator=ged_calculator, comparison_method="Mean-Distance", KERNEL_similarity_function="k1")

# classifier = Base_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="precomputed", class_weight='balanced')
# classifier = Trivial_GED_SVC(ged_calculator=ged_calculator, ged_bound="Mean-Distance", C=1.0,kernel_type="precomputed", class_weight='balanced',similarity_function='k1')
# classifier = DIFFUSION_GED_SVC(C=1.0, llambda=1.0, ged_calculator=ged_calculator, ged_bound="Mean-Distance", diffusion_function="exp_diff_kernel", class_weight='balanced', t_iterations=7)
# classifier = GED_KNN(ged_calculator=ged_calculator, ged_bound="Mean-Distance", n_neighbors=7, weights='uniform', algorithm='auto')    
# Kernel = Trivial_GED_Kernel(ged_calculator=ged_calculator, comparison_method="Mean-Distance", similarity_function="k1")
# Kernel = GEDKernel(ged_calculator=ged_calculator, comparison_method="Mean-Similarity")
# classifier = GED_SVC(kernel=Kernel, kernel_name="GEDLIB", class_weight='balanced', C=1.0)
# classifier = NX_Histogram_SVC(kernel_type="rbf", C=1.0, class_weight='balanced',get_edge_labels=DATASET.get_edge_labels, get_node_labels=DATASET.get_node_labels,Histogram_Type="combined")
# Kernel = GEDKernel(method="BRANCH", edit_cost="CONSTANT",comparison_method="Mean-Similarity")
# classifier = GED_SVC(kernel=Kernel, kernel_name="GEDLIB", C=1.0)
# ged_calculator = classifier.get_calculator()
# Define the Dataset

# dataset = Dataset(name='ENZYMES', source='TUD', domain='Bioinformatics',)

# define the model

# classifier = WeisfeilerLehman_SVC(n_iter=5,C=1.0, normalize_kernel=True) # You can tune n_iter and normalize
# classifier = CombinedHistogram_SVC(kernel_type="rbf",C=1.0)
# classifier = GED_SVC(gamma=1.0, node_del_cost=1.0, node_ins_cost=1.0, edge_del_cost=1.0, edge_ins_cost=1.0, approximation=None, kernel_type="rbf", C=0.1)
# classifier = Blind_Classifier()
# classifier = Random_Classifier(random_state=42, strategy='uniform', constant=None)
# classifier = GED_KNN(approximation=None, node_del_cost=1.0, node_ins_cost=1.0, edge_del_cost=1.0, edge_ins_cost=1.0, n_neighbors=5, weights='uniform', algorithm='auto')
def estimate_experiment_duration(classifier: GraphClassifier, DATASET: Dataset, ged_calculator: Base_Calculator,search_method="grid"):
    expi=experiment(f"{classifier.__class__.__name__}",DATASET,dataset_name=DATASET.name,
                    model=classifier,model_name=classifier.get_name,ged_calculator=None)
    estimated_time = expi.estimate_nested_cv_time(cv=int(1/SPLIT),num_trials=NUM_TRIALS,search_method=search_method) 
    return estimated_time

def run_classifiers_new(classifier: GraphClassifier, DATASET: Dataset, ged_calculator: Base_Calculator,testDF: pd.DataFrame, experiment_name: str="unknown",search_method="grid"):        
    expi=experiment(f"{classifier.__class__.__name__}",DATASET,dataset_name=DATASET.name,
                    model=classifier,model_name=classifier.get_name,ged_calculator=ged_calculator)
    cv= int(1/SPLIT)
    try:
        instance_dict =expi.run_joblib_parallel_nested_cv(outer_cv=cv,inner_cv=cv,num_trials=NUM_TRIALS,scoring=['f1_macro','f1_weighted','accuracy','roc_auc','precision','recall'], verbose=0, n_jobs=N_JOBS, search_method=search_method,should_print=True,test_trail=TEST_TRIAL)
    except Exception as e:
        #  print the full traceback
        traceback.print_exc()
        print(f"Error running {classifier.get_name} on {DATASET.name}: {e}")
        instance_dict = {
            "Model": classifier.get_name,
            "Dataset": DATASET.name,
            "Error": str(e)
        }
    # add the values form instance_dict as the last row of testDF
    instance_df = pd.DataFrame([instance_dict])
    testDF = pd.concat([testDF, instance_df], ignore_index=True)
    return testDF
def get_Dataset(dataset_name: str, ged_calculator):
    DATASET= Dataset(name=dataset_name, source="TUD", domain="Bioinformatics", ged_calculator=ged_calculator, use_node_labels=USE_NODE_LABELS, use_edge_labels=USE_EDGE_LABELS,load_now=False)
    DATASET.load()
    return DATASET, DATASET.get_calculator()

if __name__ == "__main__":
    start_time = pd.Timestamp.now()
    print(f"Starting experiment {EXPERIMENT_NAME} on dataset {DATASET_NAME})")
    # if PRELOAD_CALCULATORS:
    #     ged_calculator = "GED_Calculator"
    # else:
    #     ged_calculator = GEDLIB_Calculator(GED_calc_method="IPFP", GED_edit_cost="CONSTANT",need_node_map=True)
    if PRELOAD_CALCULATORS:
        ged_calculator = "Heuristic_Calculator"
    else:
        ged_calculator = lambda dataset, labels: build_GED_calculator(GED_edit_cost="CONSTANT", GED_calc_methods=[("IPFP","upper")], dataset=dataset, labels=labels,dataset_name=DATASET_NAME, need_node_map=True)
    DATASET, ged_calculator = get_Dataset(DATASET_NAME, ged_calculator)
    set_global_ged_calculator(ged_calculator)
    classifier = get_classifier(ged_calculator)
    if ONLY_ESTIMATE:
        # estimate the time for all GED-based classifiers
      
        est_total_duration = pd.Timedelta(0)
        total_duration_nonGED = estimate_experiment_duration(classifier, DATASET, ged_calculator,search_method="random")
        print(f"Estimated total duration for non-GED classifiers: {total_duration_nonGED}")
    else:
        print(f"Running GED-based classifiers on {DATASET.name} dataset.")
        testDF = run_classifiers_new(classifier, DATASET, ged_calculator, Test_DF, EXPERIMENT_NAME,search_method="random")
        results_dir = os.path.join("configs", "results")
        results_path = os.path.join(results_dir, f"{EXPERIMENT_NAME}_{DATASET.name}_results.xlsx")
        testDF.to_excel(results_path, index=False)
        IO_Manager.save_prototype_selector()
        end_time = pd.Timestamp.now()
        print(f"Experiment ended at {end_time}")
        total_duration = end_time - start_time
        print(f"Total experiment duration: {total_duration}")
