import sys
import os
# add the root directory to the sys.path
sys.path.append(os.getcwd())
from Dataset import Dataset
from Calculators.GED_Calculator import build_Heuristic_calculator, build_exact_ged_calculator, build_exact_ged_calculator_anti_leak, build_exact_ged_calculator_buffered
TIMEOUT= 1
N_JOBS=32
def convert_Dataset_to_exact_GED_format(dataset:Dataset,use_node_labels=True,use_edge_labels=True):
    dataset_name = dataset.get_name()
    label_info = f"{int(use_node_labels)}_{int(use_edge_labels)}"
    filepath = f"Datasets/ged/{dataset_name}_{label_info}/"
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
            if use_node_labels:
                label = G.nodes[original_id].get(dataset.Node_label_name, 0)
            else:
                label = 0
            output_lines.append(f"v {new_id} {label}")
            
        # 4. Generate 'e' lines
        for u_orig, v_orig, data in G.edges(data=True):
            # Get the new sequential IDs
            u_new = node_id_map[u_orig]
            v_new = node_id_map[v_orig]
            
            # Safely retrieve the edge label attribute
            if use_edge_labels:
                label = data.get(dataset.Edge_label_name, 0)
            else:
                label = 0
            # The format expects 'e [vertex_id1] [vertex_id2] [edge_label]'
            output_lines.append(f"e {u_new} {v_new} {label}")
        # 5. Write to file
        try:
            # Ensure the directory exists
            with open(graph_path, 'w') as f:
                f.write('\n'.join(output_lines) + '\n') # Add final newline for file hygiene
        except IOError as e:
            print(f"\nError writing to file '{graph_path}': {e}")
dataset_names =["IMDB-MULTI_0_0"]
# for dataset_name in dataset_names:
#     #  check for the digits at the end to determine whether to use node/edge labels
#     use_node_edge_labels = not dataset_name.endswith("_0_0")
#     dataset_name_only :str = dataset_name.rsplit("_",2)[0]
#     load = "label" if use_node_edge_labels else None
#     DATASET = Dataset(name=dataset_name_only, source="TUD", domain="Bioinformatics", ged_calculator=None, use_node_labels=load, use_edge_labels=load, load_now=False)     
#     DATASET.load()

#     convert_Dataset_to_exact_GED_format(DATASET,use_node_labels=DATASET.use_node_labels(), use_edge_labels=DATASET.use_edge_labels())
approximation_counters = {}
rel_deviations = {}
try:
    for dataset_name in dataset_names:
        use_node_edge_labels = not dataset_name.endswith("_0_0")

        load = "label" if use_node_edge_labels else None
        dataset_name_only :str = dataset_name.rsplit("_",2)[0]
        DATASET = Dataset(name=dataset_name_only, source="TUD", domain="Bioinformatics", ged_calculator=None, use_node_labels=load, use_edge_labels=load, load_now=False)
        DATASET.load()
        Dataset_name = DATASET.get_name() + f"_{int(DATASET.use_node_labels())}_{int(DATASET.use_edge_labels())}"
        ged_calculator, approximation_counter, rel_deviation = build_exact_ged_calculator(DATASET.get_graphs(),Dataset_name, n_jobs=N_JOBS,timeout=TIMEOUT)
        approximation_counters[Dataset_name] = approximation_counter
        rel_deviations[Dataset_name] = rel_deviation
        print(f"Finished dataset {Dataset_name}: Approximations={approximation_counter}, Relative Deviation={rel_deviation:.4f}")
        # save the number of approximations and relative deviation to a file
        with open(f"Calculators/exact_GED_results_summary.txt", "a") as f:
            f.write(f"{Dataset_name}: Approximations={approximation_counter}, Relative Deviation={rel_deviation:.4f}\n")
        # also load a heuristic calculator
        heuristic_calculator = build_Heuristic_calculator(GED_edit_cost="CONSTANT", dataset=DATASET.get_graphs(), labels=DATASET.get_targets())
        heuristic_calculator.save_calculator(DATASET.get_name())
except Exception as e:
    print(f"Error processing dataset {dataset_name}: {e}")
    # print all the approximation counters and relative deviations collected so far
    print("Approximation Counters so far:")
    for name, count in approximation_counters.items():
        print(f"{name}: {count}")
    print("Relative Deviations so far:")
    for name, dev in rel_deviations.items():
        print(f"{name}: {dev:.4f}")
    raise e








