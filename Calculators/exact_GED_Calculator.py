import sys
import os 
# append the dircetotry this gets caled from to the sys.path
sys.path.append(os.getcwd())
import subprocess
from asyncio.log import logger
import re
from tempfile import NamedTemporaryFile
from Calculators.GED_Calculator import GED_Calculator
from torch_geometric.utils import from_networkx
import os
from Dataset import Dataset
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

_ged_matrix = None
class exact_GED_Computer(GED_Calculator):
    def __init__(self, dataset_name, **kwargs):
        global _ged_matrix
        global _node_map_dict
        self.distance_matrix = _ged_matrix
        self.node_map_dict = _node_map_dict
        
        # Any specific initialization for exact GED can be added here


def build_exact_ged_calculator(dataset, **kwargs):
    dataset_name = dataset.get_name()
    graphs = dataset.get_graphs()
    targets = dataset.get_targets()
    with tqdm.tqdm(total=len(graphs)*(len(graphs)-1)//2, desc="Precomputing exact GEDs") as pbar:
        distance_matrix = np.zeros((len(graphs), len(graphs)), dtype=np.int32)
        node_map_dict = np.empty((len(graphs), len(graphs)), dtype=object)
        group_node_attrs = [] ; group_node_attrs.append(dataset.Node_label_name); group_node_attrs.append(dataset.Node_attr_name)
        group_edge_attrs = [] ; group_edge_attrs.append(dataset.Edge_label_name); group_edge_attrs.append(dataset.Edge_attr_name)
        torch_geometric_graphs = []
        for i in range(len(graphs)):
            torch_geometric_graph = from_networkx(graphs[i], group_node_attrs, group_edge_attrs)
            torch_geometric_graphs.append(torch_geometric_graph)

        for i in range(len(graphs)):
            for j in range(i+1, len(graphs)):
                graph1 = torch_geometric_graphs[i]
                graph2 = torch_geometric_graphs[j]
                result = ged_main.compute_ged_info(graph1, graph2)
                if result is not None:
                    ged, mapping, runtime = result
                else:
                    raise RuntimeError("Exact GED computation failed.")
                distance_matrix[i, j] = ged
                distance_matrix[j, i] = ged
                node_map_dict[i, j] = mapping
                node_map_dict[j, i] = {v: k for k, v in mapping.items()}
                pbar.update(1)
        global _ged_matrix
        _ged_matrix = distance_matrix
        global _node_map_dict
        _node_map_dict = node_map_dict
    calculator = exact_GED_Computer(dataset_name=dataset_name, **kwargs)
    return calculator

def convert_Dataset_to_exact_GED_format(dataset:Dataset):
    dataset_name = dataset.get_name()
    filepath = f"Datasets/ged/{dataset_name}/"
    os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

    for graph_index, G in enumerate(dataset.get_graphs()):
        graph_path = os.path.join(filepath, f"g_{graph_index}.txt")
        output_lines = []
        # 1. Create a mapping from NetworkX's original node IDs to new sequential 0-based IDs
        original_nodes = list(G.nodes())
        node_id_map = {original_id: new_id for new_id, original_id in enumerate(original_nodes)}
        
        num_nodes = G.number_of_nodes()
        
        # Determine the arbitrary strings for the 't' line
        t_val1 = G.graph.get("Name", f"#{graph_index}")
        t_val2 = num_nodes
        # 2. Generate the 't' line
        output_lines.append(f"t {t_val1} {t_val2}")
        
        # 3. Generate 'v' lines
        for original_id, new_id in node_id_map.items():
            # Safely retrieve the node label attribute
            label = G.nodes[original_id].get(dataset.Node_label_name, 0)
            output_lines.append(f"v {new_id} {label}")
            
        # 4. Generate 'e' lines
        for u_orig, v_orig, data in G.edges(data=True):
            # Get the new sequential IDs
            u_new = node_id_map[u_orig]
            v_new = node_id_map[v_orig]
            
            # Safely retrieve the edge label attribute
            label = data.get(dataset.Edge_label_name, 0)
            
            # The format expects 'e [vertex_id1] [vertex_id2] [edge_label]'
            output_lines.append(f"e {u_new} {v_new} {label}")
        # 5. Write to file
        try:
            # Ensure the directory exists
            with open(graph_path, 'w') as f:
                f.write('\n'.join(output_lines) + '\n') # Add final newline for file hygiene
        except IOError as e:
            print(f"\nError writing to file '{graph_path}': {e}")



def create_build_dataset_file(dataset:Dataset):
    output_lines = []
    filepath = "Datasets/ged/Datasets/" + dataset.get_name() + ".txt"
    for graph_index, G in enumerate(dataset.get_graphs()):
        # 1. Create a mapping from NetworkX's original node IDs to new sequential 0-based IDs
        original_nodes = list(G.nodes())
        node_id_map = {original_id: new_id for new_id, original_id in enumerate(original_nodes)}
        
        num_nodes = G.number_of_nodes()
        
        # Determine the arbitrary strings for the 't' line
        t_val1 = G.graph.get("Name", f"#{graph_index}")
        t_val2 = num_nodes
        # 2. Generate the 't' line
        output_lines.append(f"t {t_val1} {t_val2}")
        
        # 3. Generate 'v' lines
        for original_id, new_id in node_id_map.items():
            # Safely retrieve the node label attribute
            label = G.nodes[original_id].get(dataset.Node_label_name, 0)
            output_lines.append(f"v {new_id} {label}")
            
        # 4. Generate 'e' lines
        for u_orig, v_orig, data in G.edges(data=True):
            # Get the new sequential IDs
            u_new = node_id_map[u_orig]
            v_new = node_id_map[v_orig]
            
            # Safely retrieve the edge label attribute
            label = data.get(dataset.Edge_label_name, 0)
            
            # The format expects 'e [vertex_id1] [vertex_id2] [edge_label]'
            output_lines.append(f"e {u_new} {v_new} {label}")
    # 5. Write to file
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)

        with open(filepath, 'w') as f:
            f.write('\n'.join(output_lines) + '\n') # Add final newline for file hygiene
    except IOError as e:
        print(f"\nError writing to file '{filepath}': {e}")
def calculate_ged_between_two_graphs(dataset_name,g_id1, g_id2,timeout=5, lb=0):
    # load the graphs from files
    filepath1 = f"Datasets/ged/{dataset_name}/g_{g_id1}.txt"
    filepath2 = f"Datasets/ged/{dataset_name}/g_{g_id2}.txt"

    try:
        command = ["Graph_Edit_Distance/ged", "-q", filepath1, "-d", filepath2, "-g"]
        process = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
        output: str = process.stdout.decode()
        err_output: str = process.stderr.decode()
        # print(output)
        # Extract GED
        ged_match = re.search(r"\*\*\* GEDs \*\*\*\s*(\d+)", output)
        total_time_match = re.search(
            r"Total time: ([\d,]+) \(microseconds\)", output
        )
        if ged_match and total_time_match:
            ged = int(ged_match.group(1))
            # convert the time to a readable string (seconds.milliseconds)
            time_us = int(total_time_match.group(1).replace(",", "")) if total_time_match else None
            if time_us is None:
                time = None
            else:
                secs = time_us // 1_000_000
                ms = (time_us % 1_000_000) // 1000
                time = f"{secs}.{ms:03d}s"  # e.g. "1.234s" or "0.123s"
            global _ged_matrix
            _ged_matrix[g_id1, g_id2] = ged
            _ged_matrix[g_id2, g_id1] = ged
            print(f"Computed {g_id1} and graph {g_id2}: {ged} in {time}")
        else:
            raise Exception(
                "GED value not found in output:"
                + "\nSTDOUT:\n "
                + output
                + "\nSTDERR:\n "
                + err_output
            )

        # Extract mapping
        # mapping_match = re.search(r"Mapping: (.+)", output)
        # if mapping_match:
        #     mapping: dict[int, int] = {}
        #     pairs = mapping_match.group(1).split(", ")
        #     for pair in pairs:
        #         if "->" in pair:
        #             q, g = map(int, pair.split(" -> "))
        #             mapping[q] = g
        #     global _node_map_dict
        #     _node_map_dict[g_id1, g_id2] = mapping
        #     # reverse mapping
        #     reverse_mapping = {v: k for k, v in mapping.items()}
        #     _node_map_dict[g_id2, g_id1] = reverse_mapping
        # else:
        #     raise Exception(
        #         "Mapping not found in output:"
        #         + "\nSTDOUT:\n "
        #         + output
        #         + "\nSTDERR:\n "
        #         + err_output
        #     )

        # ===
        # Extract total time. For some unknown reason, time is not always
        # present in the binary's output, hence None is also accepted here.
        # ===
        



    except subprocess.TimeoutExpired:
        print(f"Timeout expired when computing GED between graph {g_id1} and graph {g_id2}.")
        # inifite distance max int
        ged = 10000
        # global _ged_matrix
        _ged_matrix[g_id1, g_id2] = ged
        _ged_matrix[g_id2, g_id1] = ged

def calculate_exact_ged_distance_matrix(dataset,n_jobs=1):
    # we assume the Dataset is already loadded, but also we have the files of the Graphs in the directories.
    dataset_name ="MUTAG"
    n = len(dataset.get_graphs())

    # we distribute the Jobs, sot that every Jobs needs to caclulate the same amount of GEDs
    global _ged_matrix
    global _node_map_dict
    _ged_matrix = np.zeros((n,n), dtype=np.int32)
    _node_map_dict = np.empty((n,n), dtype=object)
    # first we compute the diagonal
    for i in range(n):
        _ged_matrix[i,i] = 0
        _node_map_dict[i,i] = {k:k for k in range(len(dataset.get_graphs()[i].nodes()))}
    # then we compute the upper triangle
    # we distribute the Jobs so that every Jobs needs to caclulate the same amount of GEDs
    tasks = [] # will be n/2 tasks
    for i in range(n):
        for j in range(i+1, n):
            tasks.append( (i,j) )
    # start parallel processing
    # for every entry in task the GED needs to be calculated
    print(f"Starting calculation of exact GED distance matrix with {n_jobs} parallel jobs...")
    Parallel(n_jobs=n_jobs)(delayed(calculate_ged_between_two_graphs)(dataset_name,
        i,
        j
    ) for i,j in tasks)

    # process results
    print("Finished calculating exact GED distance matrix.")
    print(_ged_matrix)
    # create GED_Calculator_object 
    calculator = GED_Calculator()
    calculator.distance_matrix = _ged_matrix
    calculator.node_map_dict = _node_map_dict
    calculator.save_calculator(dataset_name)


DATASET = Dataset(name="MUTAG", source="TUD", domain="Bioinformatics", ged_calculator=None, use_node_labels="label", use_edge_labels="label",load_now=False)
DATASET.load()
calculate_exact_ged_distance_matrix(DATASET, n_jobs=8)


def load_exact_ged_calculator(dataset_name, **kwargs):
    dataset = Dataset(name=dataset_name, source="TUD", domain="Bioinformatics", ged_calculator=None, use_node_labels="label", use_edge_labels="label",load_now=True)
    calculator = build_exact_ged_calculator(dataset, **kwargs)
    return calculator