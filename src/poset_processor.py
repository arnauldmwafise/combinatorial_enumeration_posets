import math
import time
import numpy as np
import networkx as nx
from joblib import Parallel, delayed

# =============================================================================
# ORDER STRUCTURAL FILTER ENGINE LAYER (PosetProcessor)
# =============================================================================
class PosetProcessor:
    """
    Unified classification properties workbench for finite partially ordered sets.
    
    Provides high-performance vectorized operations, linear algebraic property
    filters, and parallelized reduction maps to evaluate poset properties 
    on adjacency/incidence matrices using lower-triangular rules.
    """

    @classmethod
    def get_maximal_elements(cls, poset_matrix):
        """Identifies all maximal elements using lower-triangular row constraints."""
        # Rows track precursors; row sum == 1 means it has no strict predecessors
        row_sums = poset_matrix.sum(axis=1)
        return np.where(row_sums == 1)[0].tolist()

    @classmethod
    def get_minimal_elements(cls, poset_matrix):
        """Identifies all minimal elements using lower-triangular column constraints."""
        # Columns track successors; col sum == 1 means it has no strict successors
        column_sums = poset_matrix.sum(axis=0)
        return np.where(column_sums == 1)[0].tolist()

    @classmethod
    def compute_dual_poset_matrix(cls, poset_matrix):
        """Computes the dual (order-inverted) poset incidence matrix via anti-diagonal reflection."""
        return np.flip(poset_matrix).T

    @classmethod
    def is_not_self_dual(cls, poset_matrix):
        """Determines if the poset is asymmetric under order dualization."""
        dual = cls.compute_dual_poset_matrix(poset_matrix)
        return not np.array_equal(dual, poset_matrix)

    @classmethod
    def is_trivial_poset(cls, poset_matrix):
        """Evaluates whether the poset matrix represents a singleton element."""
        return poset_matrix.shape == (1, 1) and poset_matrix[0, 0] == 1

    @classmethod
    def is_connected_poset(cls, poset_matrix):
        """Evaluates the undirected connectivity characteristics of the poset."""
        if poset_matrix.size == 0:
            return False
        if poset_matrix.shape[0] == 1:
            return cls.is_trivial_poset(poset_matrix)
        # Symmetrize the lower-triangular relation graph layout for undirected processing
        symmetric_adj = np.maximum(poset_matrix, poset_matrix.T)
        graph = nx.from_numpy_array(symmetric_adj)
        return nx.is_connected(graph)

    @classmethod
    def is_semi_right_dualizable(cls, poset_matrix):
        """Classifies if a matrix qualifies as a semi-right dualizable poset structure."""
        n = poset_matrix.shape[0]
        depth = 0
        while depth < n:
            if np.all(poset_matrix[depth:, depth]):
                depth += 1
            else:
                break
        if depth == 0 or depth == n:
            return False
        submatrix = poset_matrix[depth:, depth:]
        if not cls.is_connected_poset(submatrix):
            is_self_dual = np.array_equal(cls.compute_dual_poset_matrix(submatrix), submatrix)
            if not is_self_dual:
                return True
        return False

    @classmethod
    def is_semi_left_dualizable(cls, poset_matrix):
        """Classifies if a matrix qualifies as a semi-left dualizable poset structure."""
        n = poset_matrix.shape[0]
        depth = 0
        while depth < n:
            row_idx = n - depth - 1
            if np.all(poset_matrix[row_idx, :row_idx + 1]):
                depth += 1
            else:
                break
        if depth == 0 or depth == n:
            return False
        submatrix = poset_matrix[:-depth, :-depth]
        if not cls.is_connected_poset(submatrix):
            dual_sub = cls.compute_dual_poset_matrix(submatrix)
            if not np.array_equal(dual_sub, submatrix):
                return True
        return False

    @classmethod
    def is_double_dualizable_poset(cls, poset_matrix):
        """Evaluates the simultaneous presence of bilateral semi-dualizability features."""
        if poset_matrix.shape[0] <= 2:
            return False
        last_row = poset_matrix[-1]
        first_col = poset_matrix[:, 0]
        if not (np.array_equal(last_row, first_col[::-1]) and
                len(cls.get_maximal_elements(poset_matrix)) == len(cls.get_minimal_elements(poset_matrix))):
            return False
        top_left_sub = poset_matrix[:-1, :-1]
        bottom_right_sub = poset_matrix[1:, 1:]
        if cls.is_semi_right_dualizable(top_left_sub) and cls.is_semi_left_dualizable(bottom_right_sub):
            shared_core_tl = top_left_sub[1:, 1:]
            shared_core_br = bottom_right_sub[:-1, :-1]
            return np.array_equal(shared_core_tl, shared_core_br)
        return False

    @classmethod
    def visualize_poset_from_triangular_matrix(cls, poset_matrix, labels=None):
        """Generates and displays the Hasse diagram for a poset given its relation matrix."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("[ERROR] Matplotlib is missing. Please run: pip install matplotlib")

        num_indices = poset_matrix.shape[0]
        if labels is None:
            labels = list(range(1, (num_indices + 1)))
        elif len(labels) != num_indices:
            raise ValueError("Number of labels must match the number of elements in the matrix.")

        covering_relations = []
        n = len(labels)
        for i in range(n):
            for j in range(n):
                if poset_matrix[i, j] == 1 and i != j:
                    is_covering = True
                    for k in range(n):
                        if k != i and k != j and poset_matrix[i, k] == 1 and poset_matrix[k, j] == 1:
                            is_covering = False
                            break
                    if is_covering:
                        covering_relations.append((labels[i], labels[j]))

        G = nx.DiGraph()
        G.add_nodes_from(labels)
        G.add_edges_from(covering_relations)

        try:
            for layer, nodes in enumerate(nx.topological_generations(G)):
                for node in nodes:
                    G.nodes[node]["layer"] = layer
            pos = nx.multipartite_layout(G, subset_key="layer", align="horizontal")
        except nx.NetworkXUnfeasible:
            pos = nx.spring_layout(G, seed=42)

        plt.figure(figsize=(6, 5), dpi=150)
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=700, edgecolors='grey')
        nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='black', width=1.2)
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')

        plt.title(f"Hasse Diagram Representation\nDimension: {num_indices}x{num_indices}", fontsize=11, pad=10)
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    @classmethod
    def extract_metrics_bundle(cls, matrix: np.ndarray) -> dict:
        trivial = cls.is_trivial_poset(matrix)
        connected = cls.is_connected_poset(matrix)
        not_self_dual = cls.is_not_self_dual(matrix)
        semi_right = cls.is_semi_right_dualizable(matrix)
        semi_left = cls.is_semi_left_dualizable(matrix)
        double_dual = cls.is_double_dualizable_poset(matrix)

        return {
            "trivial": trivial,
            "connected": connected,
            "disconnected": not connected if matrix.size > 0 else False,
            "non_self_dual": not_self_dual,
            "self_dual": not not_self_dual,
            "semi_right": semi_right,
            "semi_left": semi_left,
            "double_dual": double_dual
        }

    @classmethod
    def parallel_analyze_dataset(cls, matrices: list[np.ndarray], n_jobs: int = -1) -> list[dict]:
        return Parallel(n_jobs=n_jobs)(delayed(cls.extract_metrics_bundle)(mat) for mat in matrices)
