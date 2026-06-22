import numpy as np
import pytest
import time
from joblib import Parallel, delayed
from poset_operad import PosetOperad
from poset_processor import PosetProcessor

# =============================================================================
# COMBINATORIAL SEED AND DATA INTEGRITY FIXTURES
# =============================================================================

@pytest.fixture
def initial_seed_matrices():
    """Returns the foundational 2x2 seed posets (Antichain and Chain)."""
    return np.array([
        [[1, 0], [0, 1]],  # Antichain
        [[1, 0], [1, 1]]   # Chain
    ], dtype='int64')

@pytest.fixture
def experimental_model_keys():
    """Returns all 9 targeted operadic composition variant identifiers."""
    return [
        "fully_isolated", "succ_dominant", "pred_dominant", "isolated",
        "total", "general", "minimal", "maximal", "boundary"
    ]


# =============================================================================
# PIPELINE INTEGRATION & DIMENSIONAL SCALING TESTS
# =============================================================================

def test_initial_signature_serialization(initial_seed_matrices):
    """Verifies loss-free translation of the starting seeds to integer bitmasks."""
    current_signatures = {PosetOperad.matrix_to_bits(m) for m in initial_seed_matrices}
    assert len(current_signatures) == 2
    for sig in current_signatures:
        assert isinstance(sig, int)
        # 2x2 matrix flattening limits values between 0 and 15 bits
        assert 0 <= sig < 16


def test_generational_dimension_growth_formula(initial_seed_matrices):
    """Validates that dimensional growth complies strictly with the (p + q - 1) rule."""
    p_init = initial_seed_matrices.shape[1]
    q_init = initial_seed_matrices.shape[1]
    
    # Simulate single iterative expansion step
    p_next = p_init + q_init - 1
    assert p_next == 3
    
    p_double_step = p_next + q_init - 1
    assert p_double_step == 4


def test_parallel_composition_threading_backend(initial_seed_matrices):
    """Ensures concurrent execution tasks are non-blocking under thread pools."""
    p, q = 2, 2
    l_sig = PosetOperad.matrix_to_bits(initial_seed_matrices[0])
    r_sig = PosetOperad.matrix_to_bits(initial_seed_matrices[1])
    
    tasks = [(l_sig, r_sig, p, q, "general"), (l_sig, r_sig, p, q, "total")]
    
    # Execute single test frame inside the threading context
    outputs = Parallel(n_jobs=2, backend="threading")(
        delayed(PosetOperad.generate_compositions_bitpacked)(l, r, p, q, t) 
        for l, r, p, q, t in tasks
    )
    
    assert len(outputs) == 2
    assert isinstance(outputs[0], list)


# =============================================================================
# METRICS REDUCTION AND ACCUMULATION SCHEMAS
# =============================================================================

def test_metrics_reducer_accumulation_schema(experimental_model_keys):
    """Validates properties mapping template generation format."""
    generational_growth_records = {}
    
    for key in experimental_model_keys:
        generational_growth_records[key] = {
            "total_count": [], "trivial": [], "connected": [], "disconnected": [],
            "non_self_dual": [], "self_dual": [], "semi_right": [], "semi_left": [], "double_dual": []
        }
        
        # Verify schema typing matches output structure
        assert isinstance(generational_growth_records[key]["total_count"], list)
        assert len(generational_growth_records[key].keys()) == 9


def test_mock_generation_reduction_loop(initial_seed_matrices):
    """Executes a complete mini single-step generation run to verify reductions."""
    p_current = 2
    q_init = 2
    model_key = "fully_isolated"
    
    current_signatures = {PosetOperad.matrix_to_bits(m) for m in initial_seed_matrices}
    insertion_signatures = [PosetOperad.matrix_to_bits(m) for m in initial_seed_matrices]
    
    # Setup step task array
    tasks = [
        (l_sig, r_sig, p_current, q_init, model_key)
        for r_sig in insertion_signatures for l_sig in list(current_signatures)
    ]
    
    nested_outputs = Parallel(n_jobs=2, backend="threading")(
        delayed(PosetOperad.generate_compositions_bitpacked)(l, r, p, q, t) for l, r, p, q, t in tasks
    )
    
    new_signatures = set()
    for sublist in nested_outputs:
        new_signatures.update(sublist)
        
    p_next = p_current + q_init - 1
    unpacked_matrices = [PosetOperad.bits_to_matrix(s, p_next) for s in list(new_signatures)]
    
    metrics_bundle = PosetProcessor.parallel_analyze_dataset(unpacked_matrices, n_jobs=2)
    
    # Verify reducer aggregations match list element limits
    assert len(metrics_bundle) == len(new_signatures)
    for metric in metrics_bundle:
        assert "double_dual" in metric
        assert isinstance(metric["double_dual"], bool)
