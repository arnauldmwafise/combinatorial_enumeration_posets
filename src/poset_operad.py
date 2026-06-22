import math
import time
import numpy as np
import networkx as nx
from joblib import Parallel, delayed

# =============================================================================
# OPERAD SIGNATURE BITPACKING LAYER (PosetOperad)
# =============================================================================
class PosetOperad:
    """
    High-performance, memory-optimized computing pipeline for algebraic
    poset operads on finite poset matrices supporting all 9 boundary partition variants.
    """

    @staticmethod
    def matrix_to_bits(matrix: np.ndarray) -> int:
        flat_bin = matrix.ravel()
        weights = 1 << np.arange(flat_bin.size, dtype=object)
        return int(np.sum(flat_bin * weights))

    @staticmethod
    def bits_to_matrix(signature: int, side_dim: int) -> np.ndarray:
        n_elements = side_dim * side_dim
        flat_bin = np.array([(signature >> idx) & 1 for idx in range(n_elements)], dtype='int64')
        return flat_bin.reshape(side_dim, side_dim)

    @classmethod
    def _get_extremal_indices(cls, matrix: np.ndarray) -> tuple[list[int], list[int]]:
        col_sums = matrix.sum(axis=0)
        row_sums = matrix.sum(axis=1)
        max_elements = np.where(col_sums == 1)[0].tolist()
        min_elements = np.where(row_sums == 1)[0].tolist()
        return max_elements, min_elements

    @classmethod
    def compute_operadic_submatrix_V_bitpacked(cls, l_mat, r_mat, i, p, q):
        n, m = p, q
        base_relations_col = l_mat[i:n, i-1]
        v_block = np.zeros((n - i, m), dtype='int64')
        _, min_elements = cls._get_extremal_indices(r_mat)
        if min_elements:
            v_block[:, min_elements] = base_relations_col[:, np.newaxis]
        return v_block

    @classmethod
    def compute_operadic_submatrix_U_bitpacked(cls, l_mat, r_mat, i, p, q):
        row_idx = i - 1
        base_relations = l_mat[row_idx, :row_idx]
        m = q
        u_block = np.zeros((m, row_idx), dtype='int64')
        max_elements, _ = cls._get_extremal_indices(r_mat)
        if max_elements:
            u_block[max_elements] = base_relations
        return u_block

    @classmethod
    def generate_compositions_bitpacked(cls, l_signature: int, r_signature: int,
                                        p: int, q: int, comp_type: str = "general") -> list[int]:
        l_mat = cls.bits_to_matrix(l_signature, p)
        r_mat = cls.bits_to_matrix(r_signature, q)
        output_signatures = []

        for i in range(1, p + 1):
            idx = i - 1
            A11 = l_mat[:idx, :idx]
            A21 = l_mat[i:, i:]

            y1 = np.zeros((idx, q + p - i), dtype='int64')
            y2 = np.zeros((q, p - i), dtype='int64')

            if comp_type == "general":
                A22 = l_mat[i:, :idx]
                row_Ai = l_mat[idx, :idx]
                col_Ai = l_mat[i:, idx]
                Ui = np.outer(np.ones(q, dtype='int64'), row_Ai)
                Vi = np.outer(col_Ai, np.ones(q, dtype='int64'))

            elif comp_type == "minimal":
                A22 = l_mat[i:, :idx]
                row_Ai = l_mat[idx, :idx]
                Ui = np.tile(row_Ai, (q, 1))
                Vi = cls.compute_operadic_submatrix_V_bitpacked(l_mat, r_mat, i, p, q)

            elif comp_type == "maximal":
                A22 = l_mat[i:, :idx]
                # Reverted logic strictly to original row slice metrics mapping
                Ui = cls.compute_operadic_submatrix_U_bitpacked(l_mat, r_mat, i, p, q)
                col_Ai = l_mat[i:, idx]
                Vi = np.tile(col_Ai[:, np.newaxis], (1, q))

            elif comp_type == "boundary":
                A22 = l_mat[i:, :idx]
                # Reverted logic strictly to original row slice metrics mapping
                Ui = cls.compute_operadic_submatrix_U_bitpacked(l_mat, r_mat, i, p, q)
                Vi = cls.compute_operadic_submatrix_V_bitpacked(l_mat, r_mat, i, p, q)

            elif comp_type == "total":
                A22 = l_mat[i:, :idx]
                Ui = np.ones((q, idx), dtype='int64')
                Vi = np.ones((p - i, q), dtype='int64')

            elif comp_type == "isolated":
                A22 = np.ones((p - i, idx), dtype='int64')
                Ui = np.zeros((q, idx), dtype='int64')
                Vi = np.zeros((p - i, q), dtype='int64')

            elif comp_type == "pred_dominant":
                A22 = np.ones((p - i, idx), dtype='int64')
                Ui = np.ones((q, idx), dtype='int64')
                Vi = np.zeros((p - i, q), dtype='int64')

            elif comp_type == "succ_dominant":
                A22 = np.ones((p - i, idx), dtype='int64')
                Ui = np.zeros((q, idx), dtype='int64')
                Vi = np.ones((p - i, q), dtype='int64')

            elif comp_type == "fully_isolated":
                A22 = np.zeros((p - i, idx), dtype='int64')
                Ui = np.zeros((q, idx), dtype='int64')
                Vi = np.zeros((p - i, q), dtype='int64')

            full_composed_matrix = np.block([
                [A11, y1],
                [Ui, r_mat, y2],
                [A22, Vi, A21]
            ])
            output_signatures.append(cls.matrix_to_bits(full_composed_matrix))

        return output_signatures
