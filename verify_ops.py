import time
import numpy as np
from joblib import Parallel, delayed
from src.poset_operad import PosetOperad
from src.poset_processor import PosetProcessor
import csv


# =============================================================================
# EXECUTABLE BLUEPRINT BENCHMARK WORKBENCH RUNNER WITH AUTOMATED LOGGER
# =============================================================================
if __name__ == "__main__":
    right_side_matrices = np.array([
        [[1, 0], [0, 1]],  # Antichain
        [[1, 0], [1, 1]]   # Chain
    ], dtype='int64')
    left_side_matrices = right_side_matrices

    # Set to 4 to prevent exponential memory/RAM explosion while capturing metrics
    generation_depth = 4 
    n_cpu_cores = -1

    print("======================================================================")
    print("   OPERADIC POSET COMPOSITION: COMBINATORIAL GROWTH WORKBENCH")
    print("======================================================================")
    print(f"Initial Base Element Pool Size : {len(left_side_matrices)}")
    print(f"Active Substitution Pool Size  : {len(right_side_matrices)}")
    print(f"Target Generational Execution Depth: {generation_depth} iterations")
    print("-" * 70)

    experimental_models = {
        "fully_isolated": "Fully Isolated Model  (Theorem 2.1(e))",
        "succ_dominant":  "Successor Dominant Model (Theorem 2.1(d))",
        "pred_dominant":  "Predecessor Dominant Model (Theorem 2.1(c))",
        "isolated":       "Isolated Variant Model (Theorem 2.1(b))",
        "total":          "Total Propagation Model     (Theorem 2.1(a))",
        "general":        "General Composition Model   (Theorem 2.2)",
        "minimal":        "Minimal Variant Model       (Theorem 2.6)",
        "maximal":        "Maximal Variant Model       (Theorem 2.7)",
        "boundary":       "Boundary-Restricted Model   (Theorem 2.8)"
    }

    generational_growth_records = {}

    for model_key, model_title in experimental_models.items():
        print(f"\n>>> Launching: {model_title}")
        print("." * 60)
        start_time = time.time()

        p_init = left_side_matrices.shape[1]
        q_init = right_side_matrices.shape[1]

        current_signatures = {PosetOperad.matrix_to_bits(m) for m in left_side_matrices}
        insertion_signatures = [PosetOperad.matrix_to_bits(m) for m in right_side_matrices]

        p_current = p_init

        generational_growth_records[model_key] = {
            "total_count": [], "trivial": [], "connected": [], "disconnected": [],
            "non_self_dual": [], "self_dual": [], "semi_right": [], "semi_left": [], "double_dual": []
        }

        for step in range(generation_depth):
            bases_list = list(current_signatures)
            tasks = [
                (l_sig, r_sig, p_current, q_init, model_key)
                for r_sig in insertion_signatures for l_sig in bases_list
            ]
            if not tasks:
                break

            nested_outputs = Parallel(n_jobs=n_cpu_cores, backend="threading")(
                delayed(PosetOperad.generate_compositions_bitpacked)(l, r, p, q, t) for l, r, p, q, t in tasks
            )

            current_signatures = set()
            for sublist in nested_outputs:
                current_signatures.update(sublist)

            p_current = p_current + q_init - 1

            sigs_list = list(current_signatures)
            unpacked_matrices = [PosetOperad.bits_to_matrix(s, p_current) for s in sigs_list]

            metrics_bundle = PosetProcessor.parallel_analyze_dataset(unpacked_matrices, n_jobs=n_cpu_cores)

            generational_growth_records[model_key]["total_count"].append(len(metrics_bundle))
            generational_growth_records[model_key]["trivial"].append(sum(1 for m in metrics_bundle if m["trivial"]))
            generational_growth_records[model_key]["connected"].append(sum(1 for m in metrics_bundle if m["connected"]))
            generational_growth_records[model_key]["disconnected"].append(sum(1 for m in metrics_bundle if m["disconnected"]))
            generational_growth_records[model_key]["non_self_dual"].append(sum(1 for m in metrics_bundle if m["non_self_dual"]))
            generational_growth_records[model_key]["self_dual"].append(sum(1 for m in metrics_bundle if m["self_dual"]))
            generational_growth_records[model_key]["semi_right"].append(sum(1 for m in metrics_bundle if m["semi_right"]))
            generational_growth_records[model_key]["semi_left"].append(sum(1 for m in metrics_bundle if m["semi_left"]))
            generational_growth_records[model_key]["double_dual"].append(sum(1 for m in metrics_bundle if m["double_dual"]))

            print(f" -> [Iteration {step + 1}] Size: {p_current}x{p_current} | Unique Pool: {len(current_signatures)}")

        elapsed = time.time() - start_time
        generational_growth_records[model_key]["runtime"] = elapsed
        print(f"Status: Complete | Time Elapsed: {elapsed:.4f} seconds")

    display_mapping = [
        ("fully_isolated", "Theorem 2.1(e) (Fully_Isol)"),
        ("succ_dominant",  "Theorem 2.1(d) (Succ_Dominant)"),
        ("pred_dominant",  "Theorem 2.1(c) (Pred_Dominant)"),
        ("isolated",       "Theorem 2.1(b) (Isolated)"),
        ("total",          "Theorem 2.1(a) (Total)"),
        ("general",        "Theorem 2.2    (General)"),
        ("minimal",        "Theorem 2.6    (Minimal)"),
        ("maximal",        "Theorem 2.7    (Maximal)"),
        ("boundary",       "Theorem 2.8    (Boundary)")
    ]

    # =============================================================================
    # AUTOMATED EXPORT TASK 1: CLEAN CSV LOGGER PIPELINE
    # =============================================================================
    csv_filename = "poset_generational_growth_log.csv"
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write clean multi-column spreadsheet headers
        writer.writerow(["Model Strategy", "Order Property Criterion"] + [f"Generation {i}" for i in range(1, generation_depth + 1)] + ["Runtime (s)"])
        
        for model_key, label in display_mapping:
            data = generational_growth_records[model_key]
            for prop in ["total_count", "trivial", "connected", "disconnected", "non_self_dual", "self_dual", "semi_right", "semi_left", "double_dual"]:
                writer.writerow([label, prop.replace('_', ' ').title()] + data[prop] + [f"{data['runtime']:.4f}" if prop == "total_count" else ""])
    print(f"\n[LOGGER] Spreadsheets successfully written and synced to -> {csv_filename}")

    # =============================================================================
    # AUTOMATED EXPORT TASK 2: LATEX BOOKTABS SOURCE GENERATOR
    # =============================================================================
    print("\n" + "=" * 95)
    print("         GENERATED JOURNAL-READY LATEX BOOKTABS ENUMERATION SOURCE CODE")
    print("=" * 95)
    
    latex_alignment = "l l " + " ".join(["r" for _ in range(generation_depth)]) + " r"
    latex_header = " & ".join([f"\\textbf{{Gen {i}}}" for i in range(1, generation_depth + 1)])
    
    print("\\begin{table}[htbp]")
    print("  \\centering")
    print("  \\small")
    print(f"  \\begin{{tabular}}{{{latex_alignment}}}")
    print("    \\toprule")
    print(f"    \\textbf{{Operadic Model Strategy}} & \\textbf{{Structural Axiom Criterion}} & {latex_header} & \\textbf{{Runtime}} \\\\")
    print("    \\midrule")
    
    for model_key, label in display_mapping:
        data = generational_growth_records[model_key]
        cleaned_label = label.replace("_", "\\_")
        
        # Display main metric row for the current theorem block
        prop = "total_count"
        cols = " & ".join([str(val) for val in data[prop]])
        print(f"    \\rule{{0pt}}{{3ex}}\\textbf{{{cleaned_label}}} & {prop.replace('_', ' ').title()} & {cols} & {data['runtime']:.2f}s \\\\")
        
        # Secondary sub-criterion details
        for prop in ["connected", "non_self_dual", "semi_right", "semi_left", "double_dual"]:
            cols = " & ".join([str(val) for val in data[prop]])
            print(f"                                   & \\quad\\textit{{{prop.replace('_', ' ').title()}}} & {cols} & \\\\")
        print("    \\midrule")
        
    print("    \\bottomrule")
    print("  \\end{tabular}")
    print("  \\caption{Combinatorial Growth and Structural Axiom Evolution of Inductive Lower-Triangular Poset Operads.}")
    print("  \\label{tab:poset_growth_metrics}")
    print("\\end{table}")
    print("=" * 95)

