# front matter
import networkx as nx
def build_graph(duration_dict, edge_list):
    """
    inputs: duration_dict → {label: duration}, edge_list → list of (source, target) edges
    output: G → a networkx.DiGraph object
    """
    G = nx.DiGraph()
    for label, dur in duration_dict.items():
        G.add_node(label, duration = dur)
    for source, target in edge_list:
        G.add_edge(source, target)
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("The input graph is not a DAG! Cannot proceed.")
    return G 


