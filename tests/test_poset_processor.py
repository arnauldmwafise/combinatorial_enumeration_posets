import numpy as np
import pytest
from poset_processor import PosetProcessor

# =============================================================================
# PROGRAMMATIC CONFIGURATION OF MANUSCRIPT LOWER-TRIANGULAR TESTING MODELS
# =============================================================================

@pytest.fixture
def trivial_poset():
    """1x1 singleton poset identity matrix."""
    return np.eye(1, dtype='int64')

@pytest.fixture
def lower_chain_3x3():
    """Generates a perfect 3x3 lower-triangular chain (3 < 2 < 1) programmatically."""
    mat = np.eye(3, dtype='int64')
    mat[1, 0] = 1
    mat[2, 0] = 1
    mat[2, 1] = 1
    return mat

@pytest.fixture
def canonical_semi_right_5x5():
    """Generates the 5x5 M_SR(LT) lower-triangular example matrix from the manuscript."""
    mat = np.eye(5, dtype='int64')
    mat[1:, 0] = 1
    mat[2, 1] = 1
    mat[3, 1] = 1
    mat[3, 2] = 1
    return mat

@pytest.fixture
def canonical_semi_left_5x5():
    """Generates the 5x5 M_SL(LT) lower-triangular example matrix from the manuscript."""
    mat = np.eye(5, dtype='int64')
    mat[4, :4] = 1
    mat[1, 0] = 1
    mat[2, 0] = 1
    mat[2, 1] = 1
    return mat

@pytest.fixture
def canonical_double_dual_6x6():
    """Generates the 6x6 M_DD(LT) lower-triangular double dual matrix from the manuscript."""
    mat = np.eye(6, dtype='int64')
    mat[1:, 0] = 1
    mat[5, :5] = 1
    mat[2, 1] = 1
    mat[3, 1] = 1
    mat[3, 2] = 1
    return mat

@pytest.fixture
def permuted_double_dual_6x6():
    """Generates the alternative 6x6 M_DD(P) variant from the manuscript."""
    mat = np.eye(6, dtype='int64')
    mat[1:, 0] = 1
    mat[5, :5] = 1
    mat[3, 2] = 1
    mat[4, 2] = 1
    mat[4, 3] = 1
    return mat


# =============================================================================
# EXPLICIT VERIFICATION ASSERTIONS (LOWER-TRIANGULAR AXIS FRAMEWORK)
# =============================================================================

def test_lower_triangular_maximal_elements(lower_chain_3x3):
    maximals = PosetProcessor.get_maximal_elements(lower_chain_3x3)
    # Lower-Triangular: Row 0 sums to 1 -> index 0 has no predecessors (Maximal)
    assert len(maximals) == 1
    assert 0 in maximals

def test_lower_triangular_minimal_elements(lower_chain_3x3):
    minimals = PosetProcessor.get_minimal_elements(lower_chain_3x3)
    # Lower-Triangular: Col 2 sums to 1 -> index 2 has no successors (Minimal)
    assert len(minimals) == 1
    assert 2 in minimals

def test_manuscript_semi_right_dualizable(canonical_semi_right_5x5):
    assert PosetProcessor.is_semi_right_dualizable(canonical_semi_right_5x5)
    assert not PosetProcessor.is_semi_left_dualizable(canonical_semi_right_5x5)

def test_manuscript_semi_left_dualizable(canonical_semi_left_5x5):
    assert PosetProcessor.is_semi_left_dualizable(canonical_semi_left_5x5)
    assert not PosetProcessor.is_semi_right_dualizable(canonical_semi_left_5x5)

def test_manuscript_double_dualizable_cases(canonical_double_dual_6x6, permuted_double_dual_6x6):
    assert PosetProcessor.is_double_dualizable_poset(canonical_double_dual_6x6)
    assert PosetProcessor.is_double_dualizable_poset(permuted_double_dual_6x6)

def test_parallel_analysis_pipeline(canonical_semi_right_5x5, canonical_semi_left_5x5):
    dataset = [canonical_semi_right_5x5, canonical_semi_left_5x5]
    results = PosetProcessor.parallel_analyze_dataset(dataset, n_jobs=2)
    assert len(results) == 2
    assert results[0]["semi_right"] is True
    assert results[1]["semi_left"] is True

def test_hasse_diagram_visualization_pipeline():
    doublemat1 = np.eye(5, dtype='int64')
    doublemat1[1:, 0] = 1
    doublemat1[2, 1] = 1
    doublemat1[3, 1] = 1
    doublemat1[3, 2] = 1
    doublemat1[4, 1:4] = 1
    
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        PosetProcessor.visualize_poset_from_triangular_matrix(doublemat1)
        pipeline_passed = True
    except Exception as e:
        pipeline_passed = False
        print(f"Visualization pipeline error: {str(e)}")
    assert pipeline_passed is True
    plt.close('all')
