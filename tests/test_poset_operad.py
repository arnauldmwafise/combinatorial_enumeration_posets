import numpy as np
import pytest
from poset_operad import PosetOperad

def get_clean_matrices():
    identity_3x3 = np.eye(3, dtype='int64')
    chain_3x3 = np.zeros((3, 3), dtype='int64')
    chain_3x3[0,0] = 1; chain_3x3[1,1] = 1; chain_3x3[2,2] = 1
    chain_3x3[1,0] = 1; chain_3x3[2,1] = 1; chain_3x3[2,0] = 1
    identity_2x2 = np.eye(2, dtype='int64')
    return identity_3x3, chain_3x3, identity_2x2

def test_bit_packing_roundtrip():
    id3, ch3, _ = get_clean_matrices()
    for mat in [id3, ch3]:
        sig = PosetOperad.matrix_to_bits(mat)
        recovered = PosetOperad.bits_to_matrix(sig, 3)
        np.testing.assert_array_equal(mat, recovered)

def test_get_extremal_indices():
    id3, ch3, _ = get_clean_matrices()
    max_idx, min_idx = PosetOperad._get_extremal_indices(id3)
    assert len(max_idx) == 3
    assert len(min_idx) == 3

    max_idx, min_idx = PosetOperad._get_extremal_indices(ch3)
    assert 2 in max_idx or 0 in max_idx
    assert 0 in min_idx or 2 in min_idx

def test_compute_submatrices():
    _, ch3, id2 = get_clean_matrices()
    u_block = PosetOperad.compute_operadic_submatrix_U_bitpacked(ch3, id2, i=2, p=3, q=2)
    v_block = PosetOperad.compute_operadic_submatrix_V_bitpacked(ch3, id2, i=2, p=3, q=2)
    assert u_block.shape == (2, 1)
    assert v_block.shape == (1, 2)

@pytest.mark.parametrize("comp_type", [
    "general", "minimal", "maximal", "boundary", "total",
    "isolated", "pred_dominant", "succ_dominant", "fully_isolated"
])
def test_all_composition_variants_execution(comp_type):
    id3, id2, _ = get_clean_matrices()
    l_sig = PosetOperad.matrix_to_bits(id3)
    r_sig = PosetOperad.matrix_to_bits(id2)
    signatures = PosetOperad.generate_compositions_bitpacked(l_sig, r_sig, p=3, q=2, comp_type=comp_type)
    assert len(signatures) == 3

def test_total_composition_structure():
    id2 = np.eye(2, dtype='int64')
    sig2 = PosetOperad.matrix_to_bits(id2)
    signatures = PosetOperad.generate_compositions_bitpacked(sig2, sig2, p=2, q=2, comp_type="total")
    matrix_result = PosetOperad.bits_to_matrix(signatures[0], 3)
    assert np.all(matrix_result[2:, :2] == 1)
