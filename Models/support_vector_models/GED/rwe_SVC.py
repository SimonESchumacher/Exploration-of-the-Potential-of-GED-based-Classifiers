import time
from Models.support_vector_models.GED_SVC import GED_SVC
from Calculators.GED_Calculator import load_Randomwalk_calculator_from_id
from Calculators.Product_GRaphs import build_restricted_product_graph, limited_length_approx_random_walk_similarity, infinte_length_random_walk_similarity
from scipy.stats import loguniform
from config_loader import get_conifg_param
DEBUG = get_conifg_param('GED_models', 'debuging_prints', type='bool')
# Support Vector Machine with Random Walk Edit Graph Kernel
class rwe_SVC(GED_SVC):
    model_specific_iterations = get_conifg_param('Hyperparameter_fields', 'tuning_iterations', type='int')
    """
    Support Vector Machine with Random Walk Edit Graph Kernel
    """
    def __init__(self,
                decay_lambda,
                max_walk_length,
                attributes:dict=dict(),
                **kwargs):
        
        self.name="RWE"
        self.decay_lambda = decay_lambda
        self.max_walk_length = max_walk_length
        # inner metrics 
        self.sum_bulid_product_graph_time = 0
        self.sum_random_walk_time = 0
        attributes.update({
            "decay_lambda": decay_lambda,
            "max_walk_length": max_walk_length
        })
        super().__init__(attributes=attributes, name=self.name, **kwargs)
    def _calculate_kernel_matrix(self,X_graphs ,Y_graphs=None):
        # buffered, to see if calculation maybe has already been done
        # if DEBUG:
        #     print("Starting to calculate kernel matrix using Random Walk Edit kernel...")
        kernel_matrix = super()._calculate_kernel_matrix(X_graphs, Y_graphs)
        if DEBUG:
            print(f"Sum time to build product graphs: {self.sum_random_walk_time} seconds, for {self.max_walk_length}")
        return kernel_matrix
    def compare(self, g1, g2):
        node_map = self.ged_calculator.get_node_map(g1, g2, method=self.ged_bound)
        graph1 = self.ged_calculator.get_dataset()[g1]
        graph2 = self.ged_calculator.get_dataset()[g2]
        start_time = time.time()
        product_graph = build_restricted_product_graph(graph1, graph2, node_map)
        end_time = time.time()
        self.sum_bulid_product_graph_time += end_time - start_time
        start_time = time.time()
        similarity = 0.0
        if self.max_walk_length == 0:
            similarity = 1.0  # similarity of 1 for walk length 0
        elif self.max_walk_length >=1:
            similarity = limited_length_approx_random_walk_similarity(product_graph, llamda=self.decay_lambda, max_length=self.max_walk_length)
        elif self.max_walk_length == -1:
            similarity = infinte_length_random_walk_similarity(product_graph, llamda=self.decay_lambda)
        end_time = time.time()
        self.sum_random_walk_time += end_time - start_time
        return similarity
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # add the parameters of the ged_calculator with the prefix "GED_"
        params.update({
            "decay_lambda": self.decay_lambda,
            "max_walk_length": self.max_walk_length
        })
        return params
    
    @classmethod
    def get_param_grid(cls):
        param_grid = GED_SVC.get_param_grid()
        # this is a problem, because the kernel has its own parameters
        param_grid.update({
            "decay_lambda": [0.01, 0.1],
            "max_walk_length": [5, -1]  # -1 indicates infinite length
        })
        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_space = GED_SVC.get_random_param_space()
        param_space.update({
            "decay_lambda": loguniform(a=0.005, b=0.95),
            "max_walk_length": list(range(get_conifg_param('Hyperparameter_fields', 'iteration_depth_min'),
                                           get_conifg_param('Hyperparameter_fields', 'iteration_depth_max'))
                                           ) + [-1]  # -1 indicates infinite length
        })
        return param_space
_random_walk_calculator = None
def set_global_random_walk_calculator(calculator):
    global _random_walk_calculator
    _random_walk_calculator = calculator
class rwe_SVC_new(rwe_SVC):
    model_specific_iterations = get_conifg_param('Hyperparameter_fields', 'tuning_iterations',type="int")
    
    def __init__(self,
                decay_lambda,
                max_walk_length,
                random_walk_calculator_id: str,
                attributes:dict=dict(),
                **kwargs):
        # if DEBUG:
        #     print(f"Initializing Random_walk_edit_SVC with decay_lambda={decay_lambda}, max_walk_length={max_walk_length}")
        self.name="Random-Walk-Edit-Accelerated"
        self.random_walk_calculator_id = random_walk_calculator_id
        global _random_walk_calculator
        if _random_walk_calculator is None or _random_walk_calculator.get_identifier_name() != random_walk_calculator_id:
            if DEBUG:
                print("Warning: _random_walk_calculator is None or has a different ID. We are rebuilding")
            _random_walk_calculator = load_Randomwalk_calculator_from_id(random_walk_calculator_id)
        else:
            self.random_walk_calculator = _random_walk_calculator
        super().__init__(decay_lambda=decay_lambda, max_walk_length=max_walk_length, attributes=attributes, **kwargs)
        self.random_walk_calculator = _random_walk_calculator
        # print(f"fitting random walk function for max_walk_length={max_walk_length}")
        # start = time.time()
               # print(f"fitted random walk function in {end-start} seconds")
    def compare(self, g1, g2):
        start_time = time.time()
        if self.max_walk_length == 0:
            similarity = 1.0  # similarity of 1 for walk length 0
        elif self.max_walk_length >=1:
            similarity = self.random_walk_calculator.get_limited_length_walk(g1, g2, llambda=self.decay_lambda, max_length=self.max_walk_length, method=self.ged_bound)
        elif self.max_walk_length == -1:
            similarity = self.random_walk_calculator.get_exact_inflength_walk(g1, g2, llambda=self.decay_lambda, method=self.ged_bound)
        else: 
            raise ValueError("max_walk_length must be >=0 or -1 for infinite length")
        end_time = time.time()
        self.sum_random_walk_time += end_time - start_time
        return similarity
    def get_params(self, deep=True):
        params = super().get_params(deep=deep)
        # add the parameters of the ged_calculator with the prefix "GED_"
        params.update({
            "random_walk_calculator_id": self.random_walk_calculator_id
        })
        return params

    

        
