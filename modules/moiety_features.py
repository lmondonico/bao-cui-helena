"""
Functions to generate features for a molecule on a global (moiety) level.
Functions here are used in smiles preprocessing.
"""

import numpy as np
import networkx as nx


def map_atom_to_moiety(clique_atom_ids):
    """
    Maps each atom to its corresponding moiety index.
    """
    atom_to_moiety = {}
    for moiety_index, atoms_in_moiety in enumerate(clique_atom_ids):
        for atom_idx in atoms_in_moiety:
            atom_to_moiety[atom_idx] = moiety_index
    return atom_to_moiety


def build_moiety_edges(edges, atom_to_moiety):
    """
    Constructs a set of moiety edges from the given edges and atom-to-moiety mapping.
    """
    moiety_edges = set()
    for edge in edges:
        u, v = edge
        moiety_u = atom_to_moiety.get(u)
        moiety_v = atom_to_moiety.get(v)
        if moiety_u is not None and moiety_v is not None and moiety_u != moiety_v:
            moiety_edge = (moiety_u, moiety_v)
            moiety_edges.add(moiety_edge)
    return moiety_edges


def get_adj_matrix(node_list, edges):
    """
    Computes the adjacency matrix of a molecule based on its graph edges.
    This function will be used to generate the adjacency matrix when we replace each moiety with a single node
    and we connect these nodes based on the edges between the moieties.
    """
    adjoin_matrix = np.eye(len(node_list))
    for edge in edges:
        u = edge[0]
        v = edge[1]
        adjoin_matrix[u, v] = 1.0
        adjoin_matrix[v, u] = 1.0
    return adjoin_matrix


def get_dist_matrix(node_list, edges):
    """
    Computes the shortest path distance matrix between atoms in a molecule based on its graph edges.
    This function will be used to generate the distance matrix when we replace each moiety with a single node
    and we connect these nodes based on the edges between the moieties.
    """
    make_graph = nx.Graph()
    make_graph.add_edges_from(edges)
    dist_matrix = np.zeros((len(node_list), len(node_list)))
    dist_matrix.fill(1e9)
    row, col = np.diag_indices_from(dist_matrix)
    dist_matrix[row, col] = 0  # Distance of a node from itself is zero
    graph_nodes = sorted(make_graph.nodes.keys())
    all_distance = dict(nx.all_pairs_shortest_path_length(make_graph))
    # Returns an iterator of (source, dict(target: shortest_path_length)) pairs
    for node in graph_nodes:
        node_relative_distance = dict(sorted(all_distance[node].items(), key=lambda x: x[0]))
        temp_node_dist_dict = {i: node_relative_distance.get(i) if node_relative_distance.get(i) != None else 1e9 for i in graph_nodes}
        temp_node_dist_list = list(temp_node_dist_dict.values())
        dist_matrix[node][graph_nodes] = temp_node_dist_list
    return dist_matrix.astype(np.float32)
