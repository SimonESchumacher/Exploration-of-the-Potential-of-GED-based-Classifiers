

from Calculators.Base_Calculator import Base_Calculator
from grakel.kernels import VertexHistogram, EdgeHistogram
import networkx as nx
import numpy as np
from Calculators.Prototype_Selction import buffered_prototype_selection, Select_Prototypes
from Graph_Tools import get_grakel_graphs_from_nx
class VectorCreator:
    def __init__(self, ged_calculator: Base_Calculator):
        self.ged_calculator = ged_calculator
        self.vector_extractor_functions = []
        self.need_Grakel_parse = False
    
    def add_prototype_dis_vector_extractor(self, selection_split, selection_method, size, ged_bound,dataset_name):
        def prototype_distance_vector_extractor(X, fitted=False):
            X=[int(X[i].name) for i in range(len(X))]
            if not fitted:
                self.prototypes = Select_Prototypes(X, y=None, ged_calculator=self.ged_calculator, size=size, selection_split=selection_split, selection_method=selection_method, comparison_method=ged_bound)
            feature_vectors = np.zeros((len(X), len(self.prototypes)), dtype=float)
            for i, g in enumerate(X):
                for j, g0 in enumerate(self.prototypes):
                    feature_vectors[i, j] = self.ged_calculator.compare(g, g0, method=ged_bound)
            return feature_vectors
        self.vector_extractor_functions.append(prototype_distance_vector_extractor)

    def add_edge_histogram_extractor(self):
        self.edge_kernel = EdgeHistogram(normalize=True)
        self.need_Grakel_parse = True
        
        def edge_histogram_extractor(X, fitted=False):
            if fitted:
                self.edge_kernel._method_calling = 3
            else:
                self.edge_kernel._method_calling = 2
            feature_vectors = self.edge_kernel.parse_input(self.grakelX)
            feature_vectors = feature_vectors.toarray() if hasattr(feature_vectors, "toarray") else feature_vectors
            if not fitted:
                self.edge_feature_size = feature_vectors.shape[1]
            else:
                if feature_vectors.shape[1] < self.edge_feature_size:
                    missing_cols = self.edge_feature_size - feature_vectors.shape[1]
                    filling = np.zeros((feature_vectors.shape[0], missing_cols))
                    feature_vectors = np.concatenate([feature_vectors, filling], axis=1)
                else:
                    feature_vectors = feature_vectors[:, :self.edge_feature_size]
            return feature_vectors
        self.vector_extractor_functions.append(edge_histogram_extractor)
    def add_vertex_histogram_extractor(self):
        self.vertex_kernel = VertexHistogram(normalize=True)
        self.need_Grakel_parse = True
        
        def vertex_histogram_extractor(X, fitted=False):
            if fitted:
                self.vertex_kernel._method_calling = 3
            else:
                self.vertex_kernel._method_calling = 2        
            feature_vectors = self.vertex_kernel.parse_input(self.grakelX)
            feature_vectors = feature_vectors.toarray() if hasattr(feature_vectors, "toarray") else feature_vectors
            if not fitted:
                self.vertex_feature_size = feature_vectors.shape[1]
            else:
                if feature_vectors.shape[1] < self.vertex_feature_size:
                    missing_cols = self.vertex_feature_size - feature_vectors.shape[1]
                    filling = np.zeros((feature_vectors.shape[0], missing_cols))
                    feature_vectors = np.concatenate([feature_vectors, filling], axis=1)
                else:
                    feature_vectors = feature_vectors[:, :self.vertex_feature_size]
                
            return feature_vectors
        self.vector_extractor_functions.append(vertex_histogram_extractor)

    def add_density_extractor(self):
        def density_extractor(X,is_fitted=False):
            vector = np.zeros((len(X),), dtype=float)
            for i, g in enumerate(X):
                vector[i] = nx.density(g)
            return np.array(vector).reshape(-1, 1)
        self.vector_extractor_functions.append(density_extractor)


    def create_vector(self, X, is_fitted=False,node_label_tag="label", edge_label_tag="label"):
        if self.need_Grakel_parse:
            self.grakelX = get_grakel_graphs_from_nx(X, node_label_tag=node_label_tag, edge_label_tag=edge_label_tag)
            self.grakelX = list(self.grakelX)
        self.vector = np.empty((len(X), 0), dtype=float)
        for func in self.vector_extractor_functions:
            self.vector = np.concatenate([self.vector, func(X, is_fitted)], axis=1)
        return self.vector