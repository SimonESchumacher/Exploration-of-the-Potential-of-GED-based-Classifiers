# Prototype Selction
import numpy as np
import networkx as nx
from Dataset import Dataset
from Calculators.Base_Calculator import Base_Calculator

def medianGraph(G, ged_calculator:Base_Calculator) -> nx.Graph:
    return_index =False
    if isinstance(G, list[nx.Graph]):

        # get the indexes of the Graphs
        graphindexes = [int(g.name) for g in G]
    elif isinstance(G, list[int]):
        graphindexes = G
        return_index = True
    # get the matrix of all distances between the graphs
    distance_matrix = ged_calculator.get_complete_matrix(method="Mean-Distance",x_graphindexes=graphindexes)
    # for every graph calculate the sum of its distances to all other graphs
    distances = np.sum(distance_matrix, axis=1)
    # the median graph is the one with the lowest sum of distances
    median_index = np.argmin(distances)
    if return_index:
        return median_index
    else:
        return G[median_index]

def centerGraph(G, ged_calculator:Base_Calculator):
    return_index =False
    if isinstance(G, list[nx.Graph]):

        # get the indexes of the Graphs
        graphindexes = [int(g.name) for g in G]
    elif isinstance(G, list[int]):
        graphindexes = G
        return_index = True
    # get the matrix of all distances between the graphs
    distance_matrix = ged_calculator.get_complete_matrix(method="Mean-Distance",x_graphindexes=graphindexes)
    # for every graph calculate the sum of its distances to all other graphs
    distances = np.max(distance_matrix, axis=1)
    # the median graph is the one with the lowest sum of distances
    median_index = np.argmin(distances)
    if return_index:
        return median_index
    else:
        return G[median_index]
    
def marignalGraph(G, ged_calculator:Base_Calculator):
    return_index =False
    if isinstance(G, list[nx.Graph]):

        # get the indexes of the Graphs
        graphindexes = [int(g.name) for g in G]
    elif isinstance(G, list[int]):
        graphindexes = G
        return_index = True
    # get the matrix of all distances between the graphs
    distance_matrix = ged_calculator.get_complete_matrix(method="Mean-Distance",x_graphindexes=graphindexes)
    # for every graph calculate the sum of its distances to all other graphs
    distances = np.sum(distance_matrix, axis=1)
    # the median graph is the one with the lowest sum of distances
    median_index = np.argmax(distances)
    if return_index:
        return median_index
    else:
        return G[median_index]
def randomGraph(G, ged_calculator:Base_Calculator):
    # get a random index in the range of the graphs
    random_index = np.random.randint(0, len(G))
    if isinstance(G, list[nx.Graph]):
        # get the indexes of the Graphs
        return G[random_index]
    elif isinstance(G, list[int]):
        return random_index

def select_Prototype(G, ged_calculator:Base_Calculator, selection_method="median"):
    pass

def get_indexor_NX(G):
    if isinstance(G, list) and len(G) > 0 and isinstance(G[0], nx.Graph):
        # get the indexes of the Graphs
        return [int(g.name) for g in G], True
    elif isinstance(G, list) and len(G) > 0 and isinstance(G[0], int):
        return G, False
    else:
        raise ValueError("G must be a list of nx.Graph or a list of integers representing graph indexes")
def return_prototype_set(G, prototype_indexes, asNX=True):
    if asNX:
        return [G[i] for i in prototype_indexes]
    else:
        return prototype_indexes
def select_RPS(G, size=3, **kwargs):
    """
    A random selection of m prototypes from T is performed. 
    Of course,the Random prototype selector (rps) can be applied
    class-independent or classwise (rps-c).
    """
    graphindexes, asNX = get_indexor_NX(G)
    if size <= 1:
        raise ValueError("Size must be greater than 1")
    if size > len(graphindexes):
        raise ValueError(f"Size {size} is greater than the number of graphs {len(graphindexes)}")
    prototypes = np.random.choice(graphindexes, size=size, replace=False)
    return return_prototype_set(G, prototypes, asNX=asNX)


def select_CPS(G, ged_calculator:Base_Calculator, ged_distance="Mean-Distance", size=3):
    """
    The Center prototype selector (cps) selects prototypes situated in, or
    near, the center of the training set T. The set of prototypes P = {p1,...,pm} is
    iteratively constructed as follows, starting with the median graph of T.
    """
    graphindexes, asNX = get_indexor_NX(G)
    if size <= 1:
        raise ValueError("Size must be greater than 1")
    if size > len(graphindexes):
        raise ValueError(f"Size {size} is greater than the number of graphs {len(graphindexes)}")
    distance_matrix = ged_calculator.get_complete_matrix(method=ged_distance,x_graphindexes=graphindexes)
    # get the first median graph
    distance_sums = np.sum(distance_matrix, axis=1)
    median_index = np.argmin(distance_sums)
    prototypes = [graphindexes[median_index]]
    distance_sums[median_index] = np.inf
    size -= 1
    while size > 0:
        # get the distance to the already selected prototypes
        for i in range(len(distance_sums)):
            distance_sums[i] -= distance_matrix[i, graphindexes[median_index]]
        # get the next median graph
        median_index = np.argmin(distance_sums)
        prototypes.append(graphindexes[median_index])
        distance_sums[median_index] = np.inf
        size -= 1
    return return_prototype_set(G, prototypes, asNX=asNX)

def select_BPS(G, ged_calculator:Base_Calculator, ged_distance="Mean-Distance", size=3):
    """
    TheBorder prototype selector (bps) selects prototypes situated at the
    border of the training set T. The set of prototypes P = {p1,...,pm} is iteratively
    constructed as follows, starting with the marginal graph of T.
    """
    graphindexes, asNX = get_indexor_NX(G)
    if size <= 1:
        raise ValueError("Size must be greater than 1")
    if size > len(graphindexes):
        raise ValueError(f"Size {size} is greater than the number of graphs {len(graphindexes)}")
    distance_matrix = ged_calculator.get_complete_matrix(method=ged_distance,x_graphindexes=graphindexes)
    # get the first median graph
    distance_sums = np.sum(distance_matrix, axis=1)
    median_index = np.argmax(distance_sums)
    prototypes = [graphindexes[median_index]]
    distance_sums[median_index] = 0
    size -= 1
    while size > 0:
        # get the distance to the already selected prototypes
        for i in range(len(distance_sums)):
            distance_sums[i] -= distance_matrix[i, median_index]
        # get the next median graph
        median_index = np.argmax(distance_sums)
        prototypes.append(graphindexes[median_index])
        distance_sums[median_index] = 0
        size -= 1
    return return_prototype_set(G, prototypes, asNX=asNX)

def select_Targetsphere(G, ged_calculator:Base_Calculator, ged_distance="Mean-Distance", size=3):
    """
    The idea of this prototype selector is to distribute the prototypes
    from the center to the border as uniformly as possible. The Targetsphere prototype
    selector tps first determines the center graph gc in T. After the center graph
    has been found, the graph furthest away from gc, i.e. the graph gf ∈ T whose
    distance to gc is maximum, is located. Both graphs, gc and gf, are selected as prototypes. The distance from gc to gf is referred to as dmax, i.e. dmax = d(gc,gf).
    The interval [0,dmax] is then divided into m − 1 equidistant subintervals of width
    (w = dmax)/(m-1).The m−2 graphs for which the corresponding distances to the center
    graph gc are located nearest to the interval borders in terms of edit distance are
    selected as prototypes:"""
    graphindexes, asNX = get_indexor_NX(G)
    if size <= 1:
        raise ValueError("Size must be greater than 1")
    if size > len(graphindexes):
        raise ValueError(f"Size {size} is greater than the number of graphs {len(graphindexes)}")
    distance_matrix = ged_calculator.get_complete_matrix(method=ged_distance,x_graphindexes=graphindexes)
    # we get the center graph
    distances = np.max(distance_matrix, axis=1)
    # the median graph is the one with the lowest sum of distances
    center_index = np.argmin(distances)
    distances_to_center = distance_matrix[center_index]
    # get the furthest graph from the center
    furthest_index = np.argmax(distances_to_center)
    furthest_distance = distance_matrix[center_index, furthest_index]
    prototypes = [graphindexes[center_index], graphindexes[furthest_index]]
    # divide the distance into m-1 intervals
    if size >= 2:
        interval_width = furthest_distance / (size - 1)
        for i in range(1, size - 1):
            # find the graph whose distance to the center is closest to the interval borders
            target_distance = i * interval_width
            closest_index = np.argmin(np.abs(distances_to_center - target_distance))
            prototypes.append(graphindexes[closest_index])
    return return_prototype_set(G, prototypes, asNX=asNX)

def select_SpanningTree(G, ged_calculator:Base_Calculator, ged_distance="Mean-Distance", size=3):
    """
    TheSpanning prototype selector sps considers all distances to the pro
    totypes selected before. The first prototype is the set median graph. Each additional
    prototype selected by the spanning prototype selector is the graph furthest away
    from the already selected prototype graphs.
    """
    graphindexes, asNX = get_indexor_NX(G)
    if size <= 1:
        raise ValueError("Size must be greater than 1")
    if size > len(graphindexes):
        raise ValueError(f"Size {size} is greater than the number of graphs {len(graphindexes)}")
    distance_matrix = ged_calculator.get_complete_matrix(method=ged_distance,x_graphindexes=graphindexes)
    # we get the center graph
    distances = np.max(distance_matrix, axis=1)
    # the median graph is the one with the lowest sum of distances
    center_index = np.argmin(distances)
    distances_to_center = distance_matrix[center_index]
    # get the furthest graph from the center
    furthest_index = np.argmax(distances_to_center)
    furthest_distance = distance_matrix[center_index, furthest_index]
    prototypes_indexes = [center_index, furthest_index]
    prototypes = [graphindexes[center_index], graphindexes[furthest_index]]
    not_prototypes_idx = set(range(len(graphindexes)))
    not_prototypes_idx.remove(center_index)
    not_prototypes_idx.remove(furthest_index)
    # divide the distance into m-1 intervals
    size -= 2
    while size > 0:
        # find the the graph which has the maximum distance to the already selected prototypes
        distances_to_prototypes = np.zeros(len(graphindexes))
        for i in not_prototypes_idx:
            distances_to_prototypes[i] = np.min([distance_matrix[i, p] for p in prototypes_indexes])
        # select the graph with the maximum distance to the prototypes
        new_prototype = np.argmax(distances_to_prototypes)
        prototypes.append(graphindexes[new_prototype])
        prototypes_indexes.append(new_prototype)
        not_prototypes_idx.remove(new_prototype)
        size -= 1
    return return_prototype_set(G, prototypes, asNX=asNX)

def select_k_Centers(G, ged_calculator:Base_Calculator, ged_distance="Mean-Distance", size=3,initial_prototype_selector=select_RPS):
    """
    The k-Centers prototype selector k-cps tries to choose m graphs
    from T so that they are evenly distributed with respect to the dissimilarity information given by d.
    """
    T = G
    distance_matrix = ged_calculator.get_complete_matrix(method=ged_distance,x_graphindexes=T)
    # 1. 
    # Select an initial set of m prototypes: P0 = {p1,...,pm}. One can choose the
    # initial prototypes randomly or by a more sophisticated procedure, for example,
    # the spanning prototype selector mentioned above.
    P = initial_prototype_selector(T, ged_calculator=ged_calculator, size=size)

    # 2.
    # Construct m sets Si where each set consists of one prototype: S1 ={p1},...,Sm = {pm}.
    # For each graph g ∈ T\P find its nearest neighbor pi ∈ P
    # and add the graph under consideration to the set Si corresponding to prototype
    # pi. This step results in m disjoint sets with T = 1≤i≤m Si.
    another_iteration = True
    num_iterations = 0
    MAX_Iterations = 20
    while another_iteration:
        S_list = [[p] for p in P]
        for g in T:
            if g not in P:
                # find the nearest prototype
                distances_to_prototypes = [distance_matrix[g, p] for p in P]
                nearest_prototype_index = np.argmin(distances_to_prototypes)
                S_list[nearest_prototype_index].append(g)

        # 3.
        # For each set Si find its center ci, that is, the graph for which the maximum
        # distance to all other objects in Si is minimum.

        another_iteration = False
         # 4.
        #  For each center ci, ifci= pi, replace pi by ci in Si. If any replacement is done,
        #  return to step 2, otherwise stop.
        for Si in S_list:
            # find center of the set Si
            max_distances = np.zeros(len(Si))
            for index, gs in enumerate(Si):
                max_distances[index] = np.max([distance_matrix[gs, g] for g in Si])
            center_index = np.argmin(max_distances)
            center = Si[center_index]
            # check if the center is already in P
            if center != P[S_list.index(Si)]:
                # replace the prototype with the center
                P[S_list.index(Si)] = center
                another_iteration = True
        num_iterations += 1
        if num_iterations > MAX_Iterations:
            another_iteration = False
    return P

def select_Prototype(G, ged_calculator:Base_Calculator, selection_method="CPS",size=3):
    """
    Selects a prototype from the given graphs G using the specified selection method.
    """
    if selection_method == "CPS":
        return select_CPS(G, ged_calculator=ged_calculator, size=size)
    elif selection_method == "RPS":
        return select_RPS(G, ged_calculator=ged_calculator, size=size)
    elif selection_method == "BPS":
        return select_BPS(G, ged_calculator=ged_calculator, size=size)
    elif selection_method == "TPS":
        return select_Targetsphere(G, ged_calculator=ged_calculator, size=size)
    elif selection_method == "SPS":
        return select_SpanningTree(G, ged_calculator=ged_calculator, size=size)
    elif selection_method == "k-CPS":
        return select_k_Centers(G, ged_calculator=ged_calculator, size=size)
    
def Composite_Selection(G, ged_calculator:Base_Calculator, composite_set:set=set(), size=3):
    """
    Selects a composite prototype from the given graphs G using the specified composite set.
    """
    if not composite_set:
        # If no composite set is provided, use a default selection method
        return select_CPS(G, ged_calculator=ged_calculator, size=size)
    else:
        # the set must have a the psoibilites a long  with the number of graphs for that method
        # the sum of the numbers must add up to the size
        prototypes=[]
        if size != sum(composite_set.values()):
            raise ValueError(f"Size {size} does not match the sum of the composite set {sum(composite_set.values())}")
        for method, count in composite_set.items():
            prototypes.extend(select_Prototype(G, ged_calculator=ged_calculator, selection_method=method, size=count))
        return prototypes