import networkx as nx
import numpy as np
from Graph_Tools import get_grakel_graphs_from_nx, graph_from_networkx, convert_nx_to_grakel_graph
from Models.SVC.GED.simiple_prototype_GED_SVC import Simple_Prototype_GED_SVC
from grakel.kernels import VertexHistogram, EdgeHistogram, WeisfeilerLehman


class HybridPrototype_GED_SVC(Simple_Prototype_GED_SVC):
    model_specific_iterations = 80  # Base number of iterations for this model
    def __init__(self,
                vector_feature_list: list,
                node_label_tag: str,
                edge_label_tag: str,
                dataset_name: str,
                attributes: dict = dict(),
                 **kwargs):
        self.vector_feature_list = vector_feature_list
        self.node_label_tag = node_label_tag
        self.edge_label_tag = edge_label_tag
        self.name= "Hybrid-Prototype-GED"
        attributes.update({
            "vector_feature_list": self.vector_feature_list,
            "node_label_tag": self.node_label_tag,
            "edge_label_tag": self.edge_label_tag,
            "name": self.name

        })
        self.need_Grakel_parse = False
        if "VertexHistogram" in self.vector_feature_list:
            self.vertex_kernel = VertexHistogram(normalize=True)
            self.need_Grakel_parse = True
        if "WeisfeilerLehman" in self.vector_feature_list:
            self.weisfeiler_kernel = WeisfeilerLehman(normalize=True)
            self.need_Grakel_parse = True
        if "EdgeHistogram" in self.vector_feature_list:
            self.edge_kernel = EdgeHistogram(normalize=True)
            self.need_Grakel_parse = True
        super().__init__(dataset_name=dataset_name, attributes=attributes, name=self.name, **kwargs)
    
    def fit_transform(self, X, y=None):
        if hasattr(self, "vertex_kernel"):
            self.vertex_kernel._method_calling = 1
            
        if hasattr(self, "weisfeiler_kernel"):
            self.weisfeiler_kernel._method_calling = 1
        if hasattr(self, "edge_kernel"):
            self.edge_kernel._method_calling = 1
        self.method_calling = 1
        feature_matrix = super().fit_transform(X, y)
        return feature_matrix
    def transform(self, X):
        if hasattr(self, "vertex_kernel"):
            self.vertex_kernel._method_calling = 3
        if hasattr(self, "weisfeiler_kernel"):
            self.weisfeiler_kernel._method_calling = 3
        if hasattr(self, "edge_kernel"):
            self.edge_kernel._method_calling = 3
        self.method_calling = 3
        return super().transform(X)

    def build_feature_matrix(self, X):
        prototype_features: np.ndarray = super().build_feature_matrix(X)
        if self.need_Grakel_parse:
            # grakelX = [graph_from_networkx(self.ged_calculator.dataset[x]) for x in X]
            # grakelX = convert_nx_to_grakel_graph([self.ged_calculator.dataset[x] for x in X])
            grakelX = get_grakel_graphs_from_nx([self.ged_calculator.dataset[x] for x in X], node_label_tag=self.node_label_tag, edge_label_tag=self.edge_label_tag)
            grakelX = list(grakelX)  # Ensure it's a list in case it's a generator
        if hasattr(self, "vertex_kernel"):
            vertex_features = self.vertex_kernel.parse_input(grakelX)
            # decode the sparse matrix to a dense one
            if self.method_calling == 1:
                self.vertex_feature_size = vertex_features.shape[1]
            elif self.method_calling == 3:
                vertex_features = vertex_features[:, :self.vertex_feature_size]
            if hasattr(vertex_features, "toarray"):
                vertex_features = vertex_features.toarray()
            prototype_features = np.concatenate([prototype_features, vertex_features], axis=1)
        if hasattr(self, "weisfeiler_kernel"):
            weisfeiler_features = self.weisfeiler_kernel.parse_input(grakelX)
            if self.method_calling == 1:
                self.weisfeiler_feature_size = weisfeiler_features.shape[1]
            elif self.method_calling == 3:
                weisfeiler_features = weisfeiler_features[:, :self.weisfeiler_feature_size]
            if hasattr(weisfeiler_features, "toarray"):
                weisfeiler_features = weisfeiler_features.toarray()
            prototype_features = np.concatenate([prototype_features, weisfeiler_features], axis=1)
        if hasattr(self, "edge_kernel"):
            edge_features = self.edge_kernel.parse_input(grakelX)
            if self.method_calling == 1:
                self.edge_feature_size = edge_features.shape[1]
            elif self.method_calling == 3:
                edge_features = edge_features[:, :self.edge_feature_size]
            if hasattr(edge_features, "toarray"):
                edge_features = edge_features.toarray()
            prototype_features = np.concatenate([prototype_features, edge_features], axis=1)
        for attr in self.vector_feature_list:
            if attr not in ["VertexHistogram", "WeisfeilerLehman", "EdgeHistogram"]:
                if attr == "density":
                    densities = get_graph_density([self.ged_calculator.dataset[x] for x in X])
                    # append the desity as the last value of the feature vector
                    prototype_features = np.concatenate([prototype_features, np.array(densities).reshape(-1, 1)], axis=1)
                else:
                    raise ValueError(f"Unknown vector feature: {attr}")
        return prototype_features
    def get_params(self, deep=True):
        params = super().get_params(deep)
        params.update({
            "vector_feature_list": self.vector_feature_list,
            "node_label_tag": self.node_label_tag,
            "edge_label_tag": self.edge_label_tag,
        })
        return params
    @classmethod
    def get_param_grid(cls):
        param_grid = Simple_Prototype_GED_SVC.get_param_grid()
        param_grid.update({
            'vector_feature_list': [
                ["VertexHistogram", "density"],
                # ["WeisfeilerLehman", "density"],
                ["EdgeHistogram", "density"],
                # ["VertexHistogram", "density"],
                ["VertexHistogram","EdgeHistogram"],
                # ["WeisfeilerLehman"],
                ["EdgeHistogram"],
                ["density"],
                # Different combinations of features
            ]

        })
        return param_grid
    @classmethod
    def get_random_param_space(cls):
        param_space = Simple_Prototype_GED_SVC.get_random_param_space()
        param_space.update({
            'vector_feature_list': [
                ["VertexHistogram", "density"],
                ["EdgeHistogram", "density"],
                ["VertexHistogram","EdgeHistogram"],
                ["density"],
                ["VertexHistogram"],
                ["EdgeHistogram"],
                # Different combinations of features
            ]
        })
        return param_space


def get_graph_density(G):
    """
    calculates the density of every graph in G
    """
    if len(G) <= 1:
        return 0.0
    densities = np.zeros((len(G)), dtype=float)
    for i, g in enumerate(G):
        densities[i] = nx.density(g)
    return densities


