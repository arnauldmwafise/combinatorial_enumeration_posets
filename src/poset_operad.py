import math
import time

# =============================================================================
# HARDWARE COMPATIBILITY ACCELERATION LAYER (CuPy / NumPy Dynamic Swap)
# =============================================================================
try:
    import cupy as xp
    USING_GPU = True
except ImportError:
    import numpy as xp
    USING_GPU = False

# =============================================================================
# OPERAD SIGNATURE GPU BITPACKING ENGINE (PosetOperad)
# =============================================================================
class PosetOperad:
    """
    GPU-accelerated, high-performance computing pipeline for algebraic poset 
    operads on finite matrices, optimized with CuPy vector parameters.
    """

    @staticmethod
    def is_gpu_active() -> bool:
        """Returns True if execution kernels are routing to CUDA cores."""
        return USING_GPU

    @staticmethod
    def matrix_to_bits(matrix) -> int:
        """Vectorized conversion of a 2D matrix into a large bitmask scalar."""
        flat_bin = matrix.ravel()
        # Allocate bit-shifting weight parameters directly inside VRAM/RAM
        weights = 1 << xp.arange(flat_bin.size, dtype=object)
        return int(xp.sum(flat_bin * weights))

    @staticmethod
    def bits_to_matrix(signature: int, side_dim: int):
        """Deserializes an integer bitmask signature back into a square 2D array."""
        n_elements = side_dim * side_dim
        # Parallel coordinate shift mapping
        flat_bin = xp.array([(signature >> idx) & 1 for idx in range(n_elements)], dtype='int64')
        return flat_bin.reshape(side_dim, side_dim)

    @classmethod
    def _get_extremal_indices(cls, matrix) -> tuple[list[int], list[int]]:
        """Computes maximal and minimal indices using vectorized sum constraints."""
        col_sums = matrix.sum(axis=0)
        row_sums = matrix.sum(axis=1)
        
        # Pull data back to host CPU memory only when list export is required
        if USING_GPU:
            max_elements = xp.where(col_sums == 1)[0].get().tolist()
            min_elements = xp.where(row_sums == 1)[0].get().tolist()
        else:
            max_elements = xp.where(col_sums == 1)[0].tolist()
            min_elements = xp.where(row_sums == 1)[0].tolist()
            
        return max_elements, min_elements

    @classmethod
    def compute_operadic_submatrix_V_bitpacked(cls, l_mat, r_mat, i, p, q):
        n, m = p, q
        base_relations_col = l_mat[i:n, i-1]
        v_block = xp.zeros((n - i, m), dtype='int64')
        _, min_elements = cls._get_extremal_indices(r_mat)
        if min_elements:
            # Vectorized broadcasting matrix expansion inside VRAM
            v_block[:, min_elements] = base_relations_col[:, xp.newaxis]
        return v_block

    @classmethod
    def compute_operadic_submatrix_U_bitpacked(cls, l_mat, r_mat, i, p, q):
        row_idx = i - 1
        base_relations = l_mat[row_idx, :row_idx]
        m = q
        u_block = xp.zeros((m, row_idx), dtype='int64')
        max_elements, _ = cls._get_extremal_indices(r_mat)
        if max_elements:
            u_block[max_elements] = base_relations
        return u_block

    @classmethod
    def generate_compositions_bitpacked(cls, l_signature: int, r_signature: int,
                                        p: int, q: int, comp_type: str = "general") -> list[int]:
        """Runs the primary operadic block-tensor substitution engine inside VRAM."""
        l_mat = cls.bits_to_matrix(l_signature, p)
        r_mat = cls.bits_to_matrix(r_signature, q)
        output_signatures = []

        for i in range(1, p + 1):
            idx = i - 1
            A11 = l_mat[:idx, :idx]
            A21 = l_mat[i:, i:]

            y1 = xp.zeros((idx, q + p - i), dtype='int64')
            y2 = xp.zeros((q, p - i), dtype='int64')

            if comp_type == "general":
                A22 = l_mat[i:, :idx]
                row_Ai = l_mat[idx, :idx]
                col_Ai = l_mat[i:, idx]
                Ui = xp.outer(xp.ones(q, dtype='int64'), row_Ai)
                Vi = xp.outer(col_Ai, xp.ones(q, dtype='int64'))

            elif comp_type == "minimal":
                A22 = l_mat[i:, :idx]
                row_Ai = l_mat[idx, :idx]
                Ui = xp.tile(row_Ai, (q, 1))
                Vi = cls.compute_operadic_submatrix_V_bitpacked(l_mat, r_mat, i, p, q)

            elif comp_type == "maximal":
                A22 = l_mat[i:, :idx]
                Ui = cls.compute_operadic_submatrix_U_bitpacked(l_mat, r_mat, i, p, q)
                col_Ai = l_mat[i:, idx]
                Vi = xp.tile(col_Ai[:, xp.newaxis], (1, q))

            elif comp_type == "boundary":
                A22 = l_mat[i:, :idx]
                Ui = cls.compute_operadic_submatrix_U_bitpacked(l_mat, r_mat, i, p, q)
                Vi = cls.compute_operadic_submatrix_V_bitpacked(l_mat, r_mat, i, p, q)

            elif comp_type == "total":
                A22 = l_mat[i:, :idx]
                Ui = xp.ones((q, idx), dtype='int64')
                Vi = xp.ones((p - i, q), dtype='int64')

            elif comp_type == "isolated":
                A22 = xp.ones((p - i, idx), dtype='int64')
                Ui = xp.zeros((q, idx), dtype='int64')
                Vi = xp.zeros((p - i, q), dtype='int64')

            elif comp_type == "pred_dominant":
                A22 = xp.ones((p - i, idx), dtype='int64')
                Ui = xp.ones((q, idx), dtype='int64')
                Vi = xp.zeros((p - i, q), dtype='int64')

            elif comp_type == "succ_dominant":
                A22 = xp.ones((p - i, idx), dtype='int64')
                Ui = xp.zeros((q, idx), dtype='int64')
                Vi = xp.ones((p - i, q), dtype='int64')

            elif comp_type == "fully_isolated":
                A22 = xp.zeros((p - i, idx), dtype='int64')
                Ui = xp.zeros((q, idx), dtype='int64')
                Vi = xp.zeros((p - i, q), dtype='int64')

            # Monolithic zero-copy hardware acceleration block matrix compilation
            full_composed_matrix = xp.block([
                [A11, y1],
                [Ui, r_mat, y2],
                [A22, Vi, A21]
            ])
            output_signatures.append(cls.matrix_to_bits(full_composed_matrix))

        return output_signatures

