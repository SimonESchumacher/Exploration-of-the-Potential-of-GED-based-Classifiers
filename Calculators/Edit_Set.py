EDIT_COST = 1
DELETE_COST = 1
INSERT_COST = 1

class EditOperation:
    """ A class representing an edit operation on a graph """
    def __init__(self,cost: float):
        self.cost = cost
    def __repr__(self):
        return "Base Edit Operation"
    def __eq__(self, other):
        return False
    def __hash__(self):
        return hash(id(self))
class NodeExchangeOperation(EditOperation):
    def __init__(self, node_from_id: int, node_to_id: int, Label1: str, Label2: str, cost: float):
        self.node_from_id = node_from_id
        self.node_to_id = node_to_id
        self.Label1 = Label1
        self.Label2 = Label2
        super().__init__(cost)

    def __repr__(self):
        return f"Exchange Labels {self.Label1} to {self.Label2} on node {self.node_from_id} with node {self.node_to_id}"
    def __eq__(self, other):
        if not isinstance(other, NodeExchangeOperation):
            return False
        return self.node_from_id == other.node_from_id and self.node_to_id == other.node_to_id
    def __hash__(self):
        return hash((self.node_from_id, self.node_to_id))    
class NodeDeleteOperation(EditOperation):
    def __init__(self, node_id: int, Label: str, cost: float):
        self.node_id = node_id
        self.Label = Label
        super().__init__(cost)

    def __repr__(self):
        return f"Delete node {self.node_id} with label {self.Label}"
    def __eq__(self, other):
        if not isinstance(other, NodeDeleteOperation):
            return False
        return self.node_id == other.node_id
    def __hash__(self):
        return hash(self.node_id)
class NodeInsertOperation(EditOperation):
    def __init__(self, node_id: int, Label: str, cost: float):
        self.node_id = node_id
        self.Label = Label
        super().__init__(cost)

    def __repr__(self):
        return f"Insert node {self.node_id} with label {self.Label}"
    def __eq__(self, other):
        if not isinstance(other, NodeInsertOperation):
            return False
        return self.node_id == other.node_id
    def __hash__(self):
        return hash(self.node_id)        

class EdgeDeletionOperation(EditOperation):
    def __init__(self, node1_id: int, node2_id: int, Label: str, cost: float):
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.Label = Label
        super().__init__(cost)

    def __repr__(self):
        return f"Delete edge from {self.node1_id} to {self.node2_id} with label {self.Label}"
    def __eq__(self, other):
        if not isinstance(other, EdgeDeletionOperation):
            return False
        return self.node1_id == other.node1_id and self.node2_id == other.node2_id
    def __hash__(self):
        return hash((self.node1_id, self.node2_id))
class EdgeInsertionOperation(EditOperation):
    def __init__(self, node1_id: int, node2_id: int, Label: str, cost: float):
        self.node1_id = node1_id
        self.node2_id = node2_id
        self.Label = Label
        super().__init__(cost)

    def __repr__(self):
        return f"Insert edge from {self.node1_id} to {self.node2_id} with label {self.Label}"
    def __eq__(self, other):
        if not isinstance(other, EdgeInsertionOperation):
            return False
        return self.node1_id == other.node1_id and self.node2_id == other.node2_id
    def __hash__(self):
        return hash((self.node1_id, self.node2_id))   

class EdgeExchangeOperation(EditOperation):
    def __init__(self, node1_from_id: int, node2_from_id: int, node1_to_id:int, node2_to_id: int, Label_from: str,Label_to:str, cost: float):
        self.node1_from_id = node1_from_id
        self.node2_from_id = node2_from_id
        self.node1_to_id = node1_to_id
        self.node2_to_id = node2_to_id
        self.Label_from = Label_from
        self.Label_to = Label_to
        super().__init__(cost)

    def __repr__(self):
        return f"Exchange edge from {(self.node1_from_id,self.node2_from_id)} to {(self.node1_to_id,self.node2_to_id)} with label {self.Label_from} to {self.Label_to}"
    def __eq__(self, other):
        if not isinstance(other, EdgeExchangeOperation):
            return False
        return self.node1_from_id == other.node1_from_id and self.node2_from_id == other.node2_from_id and self.node1_to_id == other.node1_to_id and self.node2_to_id == other.node2_to_id
    def __hash__(self):
        return hash((self.node1_from_id, self.node2_from_id, self.node1_to_id, self.node2_to_id))


class pathGenerator:
    @staticmethod
    def get_edit_ops(g1 , g2, mappings: list[(int, int)]):
        """ Get the edit operations from g1 to g2 given a node mapping from g1 to g2
        Args:
            g1 (nx.Graph): The first graph  (source graph)
            g2 (nx.Graph): The second graph (target graph)
            mappings (dict[int, int]): A dictionary mapping node ids in g1 to node ids in g2
            """
        lowest_index_g1= min(g1.nodes)
        lowest_index_g2= min(g2.nodes)
        mappings_dict = dict()
        for source_id, target_id in mappings:
            mappings_dict[source_id] = target_id
        edit_ops =[]
        for source_id, target_id in mappings:
            if source_id == 18446744073709551614:
                source_id = None
            else:
                source_id +=lowest_index_g1 # to convert from 0-indexed to correct index
                g1_label= g1.nodes[source_id].get('label', None)
            if target_id == 18446744073709551614:
                target_id = None
            else:
                target_id +=lowest_index_g2 # to convert from 0-indexed to correct index
                g2_label= g2.nodes[target_id].get('label', None)
            
            if source_id is None and target_id is None:
                # No Operation
                raise ValueError("Both source_id and target_id are None in mapping, which is invalid.")
            elif source_id is None:
                # Insert Operation
                ins_Op = NodeInsertOperation(target_id, g2_label, cost=INSERT_COST)
                edit_ops.append(ins_Op)
            elif target_id is None:
                # Delete Operation
                del_op = NodeDeleteOperation(source_id, g1_label, cost=DELETE_COST)
                edit_ops.append(del_op)
            elif g1_label != g2_label:
                # Exchange Operation
                ex_op = NodeExchangeOperation(source_id, target_id, g1_label, g2_label, cost=EDIT_COST)
                edit_ops.append(ex_op)
            else:
                # No operation needed               
                pass
        # get the edges of g1 and g2
        g1_edges = set(g1.edges)
        g2_edges = set(g2.edges)
        g1_edges_dict = {edge: g1.edges[edge] for edge in g1.edges()}
        g2_edges_dict = {edge: g2.edges[edge] for edge in g2.edges()}
        unmached_g2_edges =g2_edges.copy()
        for edge in g1_edges:
            n1, n2 = edge
            if n1 == 18446744073709551614 or n2 == 18446744073709551614:
                # One of the nodes is deleted, so the edge is deleted
                del_op = EdgeDeletionOperation(n1, n2, g1_edges_dict[edge].get('label', None), cost=DELETE_COST)
                edit_ops.append(del_op)
                continue
            else:
                n1_image_id = mappings_dict[n1-lowest_index_g1]
                n1_image_id +=lowest_index_g2
                n2_image_id = mappings_dict[n2-lowest_index_g1]
                n2_image_id +=lowest_index_g2

            if (n1_image_id, n2_image_id) in g2_edges:
                # Edge exists in both graphs, check if labels are different
                unmached_g2_edges.remove((n1_image_id, n2_image_id))
                if g1_edges_dict[edge].get('label', None) != g2_edges_dict[(n1_image_id, n2_image_id)].get('label', None):
                    ex_op = EdgeExchangeOperation(n1, n2, n1_image_id, n2_image_id, g1_edges_dict[edge].get('label', None), g2_edges_dict[(n1_image_id, n2_image_id)].get('label', None), cost=EDIT_COST)
                    edit_ops.append(ex_op)
                else:
                    # No operation needed
                    pass
            elif (n2_image_id, n1_image_id) not in g2_edges:
                # Edge does not exist in g2, so it is deleted
                del_op = EdgeDeletionOperation(n1, n2, g1_edges_dict[edge].get('label', None), cost=DELETE_COST)
                edit_ops.append(del_op)
            else:
                pass
        for edge in unmached_g2_edges:
            n1, n2 = edge
            ins_Op = EdgeInsertionOperation(n1, n2, g2_edges_dict[edge].get('label', None), cost=INSERT_COST)
            edit_ops.append(ins_Op)
        return edit_ops

    @staticmethod
    def _get_path_ops(g1 , g2, mappings: dict[int, int]):
        """ Get the path edit operations from g1 to g2 given a node mapping from g1 to g2
        Args:
            g1 (nx.Graph): The first graph 
            g2 (nx.Graph): The second graph (larger or equal in size to g1)
            mappings (dict[int, int]): A dictionary mapping node ids in g1 to node ids in g2
            """
        touched_image_node_ids = set() # node ids in g2 that have been mapped to a node in g1

        # find the Node Exchange Operations

        larger_graph_nodes_dict = {node for node in g2.nodes} # the nodes of g2

        node_exchange_ops =set()
        node_delete_ops = set()
        for node in g1.nodes:
            _ ,image_id = mappings[node-1]  # the node in g2 that node in g1 is mapped to
            image_id +=1 # to convert from 0-indexed to 1-indexed
            if image_id == 18446744073709551615: # the id for a deletion
                del_op = NodeDeleteOperation(node, node_label, cost=DELETE_COST)
                node_delete_ops.add(del_op)
            touched_image_node_ids.add(image_id)
            # mapped_node = larger_graph_nodes_dict.get(image_id, None)
            # nx_graph.nodes[node].get('label', None)
            mapped_node = g2.nodes[image_id]
            mapped_node_label = mapped_node.get('label', None)
            node_label = g1.nodes[node].get('label', None)
            if mapped_node_label != node_label:
                ex_op = NodeExchangeOperation(node, image_id, node_label, mapped_node_label, cost=EDIT_COST)
                node_exchange_ops.add(ex_op)
        
        # Compare existing edges with required edges : Delete unwanted existing edges

        larger_graph_edges_set = {edge for edge in g2.edges} # the edges of g2
        touched_image_edges  = set() # edges in g2 that have been mapped to an edge in g1
        edge_delete_ops: set[EdgeDeletionOperation] = set()
        for edge in g1.edges:
            n1,n2 = edge
            node1_image_id, _ = mappings[n1-1]
            node1_image_id +=1 # to convert from 0-indexed to 1-indexed
            touched_image_node_ids.add(node1_image_id) # not understood by the developer
            _, node2_image_id = mappings[n2-1]
            node2_image_id +=1
            touched_image_node_ids.add(node2_image_id) # not understood by the developer
            
            if (node1_image_id,node2_image_id) in larger_graph_edges_set:
                touched_image_edges.add((node1_image_id,node2_image_id))
                # edge exists in both graphs, do nothing
            elif(node2_image_id,node1_image_id) in larger_graph_edges_set:
                touched_image_edges.add((node2_image_id,node1_image_id))
                # edge exists in both graphs, do nothing
            else:
                edge_label = g1.edges[edge].get('label', None)
                del_op = EdgeDeletionOperation(edge[0], edge[1], edge_label, cost=DELETE_COST)
                edge_delete_ops.add(del_op)

            
        untoched_image_edges = set(larger_graph_edges_set) - touched_image_edges
        inverse_mapping = {v: k for k, v in mappings}
        edge_insert_ops: set[EdgeInsertionOperation] = set()
        node_insert_ops: set[NodeInsertOperation] = set()
        for (id1,id2), edge in untoched_image_edges:
            touched_image_node_ids.add(id1)
            touched_image_node_ids.add(id2)

            required_ids :list[int] = []
            if id1 not in inverse_mapping:
                required_ids.append(id1)
                ins_op =NodeInsertOperation(id1, g2.nodes[id1].get('label', None), cost=INSERT_COST)
                node_insert_ops.add(ins_op)
            else:
                id1 = inverse_mapping[id1]
            
            if id2 not in inverse_mapping:
                required_ids.append(id2)
                ins_op =NodeInsertOperation(id2, g2.nodes[id2].get('label', None), cost=INSERT_COST)
                node_insert_ops.add(ins_op)
            else:
                id2 = inverse_mapping[id2]
            
            edge_ins = EdgeInsertionOperation(id1, id2, edge.get('label', None), cost=INSERT_COST)

        untouched_nodes = {node for node in g1.nodes} - touched_image_node_ids# nodes in g1 that have not been mapped
        for node_id in untouched_nodes:
            node_label = g1.nodes[node_id].get('label', None)
            del_op = NodeInsertOperation(node_id, node_label, cost=INSERT_COST)

        path_ops = {
            "Node Exchange": node_exchange_ops,
            "Edge Deletion": edge_delete_ops,
            "Edge Insertion": edge_insert_ops,
            "Edge Exchange": set(), # Not implemented
            "Node Deletion": set(), # Not implemented
            "Node Insertion": node_insert_ops
        }