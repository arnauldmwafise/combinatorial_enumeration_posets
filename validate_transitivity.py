import numpy as np
from poset_operad import PosetOperad

def compute_floyd_warshall_closure(matrix: np.ndarray) -> np.ndarray:
    """Computes the transitive closure of a boolean relation matrix using Floyd-Warshall."""
    closure = matrix.copy().astype(bool)
    n = closure.shape[0]
    for k in range(n):
        # Broadcast logical operations to discover indirect structural links
        closure |= closure[:, [k]] & closure[[k], :]
    return closure.astype('int64')

def verify_operad_transitivity():
    print("========================================================================")
    print("      MATHEMATICAL TRANSITIVITY AXIOM VALIDATION UTILITY               ")
    print("========================================================================")
    
    # Define an active lower-triangular base poset (p=3) and substitution poset (q=2)
    p, q = 3, 2
    l_mat = np.array([[1, 0, 0], [1, 1, 0], [1, 1, 1]], dtype='int64')
    r_mat = np.array([[1, 0], [1, 1]], dtype='int64')
    
    l_sig = PosetOperad.matrix_to_bits(l_mat)
    r_sig = PosetOperad.matrix_to_bits(r_mat)
    
    variants = ["general", "minimal", "maximal", "boundary", "total", 
                "isolated", "pred_dominant", "succ_dominant", "fully_isolated"]
    
    all_valid = True
    expected_dim = p + q - 1
    
    for variant in variants:
        signatures = PosetOperad.generate_compositions_bitpacked(l_sig, r_sig, p, q, comp_type=variant)
        
        for idx, sig in enumerate(signatures):
            composite_matrix = PosetOperad.bits_to_matrix(sig, expected_dim)
            closure_matrix = compute_floyd_warshall_closure(composite_matrix)
            
            # Check if the generated matrix matches its own transitive closure
            is_transitive = np.array_equal(composite_matrix, closure_matrix)
            
            if is_transitive:
                print(f"[PASSED] Variant '{variant}' at index {idx} satisfies partial-order transitivity.")
            else:
                print(f"[FAILED] Variant '{variant}' at index {idx} violates transitivity!")
                all_valid = False
                
    print("------------------------------------------------------------------------")
    if all_valid:
        print("SUCCESS: All 9 boundary partition variants preserve strict poset axioms!")
    else:
        print("WARNING: Transitivity breaks on certain structural composition paths.")
    print("========================================================================")

if __name__ == "__main__":
    verify_operad_transitivity()
