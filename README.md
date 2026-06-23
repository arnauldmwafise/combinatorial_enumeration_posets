# Combinatorial Enumeration of Finite Lower-Triangular Poset Matrices

[![Poset Research Pipeline CI](https://github.com)](https://github.com)
[![Python Version](https://shields.io)](https://python.org)
[![License: MIT](https://shields.io)](LICENSE)

A high-performance computational framework designed to automate the generation, tracking, and algebraic classification of finite partially ordered sets (posets) under operadic compositions. Optimized for top-tier theoretical computer science and combinatorics journal protocols.

---

## 1. Repository Architecture Layout

The framework utilizes a standardized package design layout to guarantee complete execution reproducibility:

```text
my_research_project/
├── .github/workflows/ci.yml       # Automated GitHub Actions continuous integration 
├── data/logs/                     # Export directory for growth spreadsheets (.csv)
├── docs/                          # Publication assets, manuals, and latency line plots
├── notebooks/                     # Exploratory research and derivation notebooks
├── src/                           # Core operational algorithms engine modules
│   ├── __init__.py
│   ├── poset_operad.py            # High-performance, bitpacked algebraic composition matrix engine
│   └── poset_processor.py         # Structural filter classifying dualizable properties
├── tests/                         # Complete 25-case unit testing engine suite
│   ├── __init__.py
│   ├── test_growth_workbench.py   # Threading and scale validation loops
│   ├── test_poset_operad.py       # Composition model verification tests
│   └── test_poset_processor.py    # Classification and Hasse diagram checks
├── benchmark_poset.py             # Throughput latency tracking entrypoint
├── plot_benchmark.py              # High-res line plot rendering pipeline
├── pyproject.toml                 # Root path directory mapping declarations
├── requirements.txt               # Explicit package library dependencies
└── verify_ops.py                  # Main orchestration driver tracking all 9 theorems
```

---

## 2. Mathematical Context & Complexity Specifications

## Overview

The computational and combinatorial enumeration of poset matrices primarily bridges algebraic combinatorics, order theory, and theoretical computer science. By representing partially ordered sets (posets) as Boolean matrices, this interdisciplinary area translates abstract ordering relations into quantitative, graph-theoretic, and algorithmic structures. The resulting framework enables the study of combinatorial enumeration, structural properties of posets, and computational methods for analyzing ordered systems.

This workspace formalizes the **partial operadic composition**

\[
\circ_i : \mathcal{P}(p)\times\mathcal{P}(q)\rightarrow\mathcal{P}(p+q-1)
\]

over Boolean incidence matrices stored in lower-triangular form, where
entries satisfy \(M_{j,j}=1\) and \(M_{j,k}=1\) encodes the order relation
\(k \preceq j\).

The implementation realizes the operadic substitution mechanisms introduced in *Operads of Poset Matrices*, where a poset matrix is expanded by grafting a second poset matrix into a specified insertion position while preserving the defining poset axioms of reflexivity, antisymmetry, and transitivity through structured block-matrix constructions. The framework supports several operadic composition variants, including unrestricted, minimal-boundary, maximal-boundary, and symmetric boundary-restricted propagations, providing constructive generators for broad classes of finite posets.

In addition, the codebase implements high-performance enumeration pipelines based on compact bit-packed matrix encodings, canonical hashing schemes, and parallelized generation algorithms. These techniques enable large-scale computational experiments on operad-generated poset families and facilitate comparisons with classical combinatorial sequences, including the large Schröder, super-Catalan, and Fibonacci families. The resulting environment serves both as a computational laboratory for poset operad-theoretic investigations and as a verification platform for structural classifications involving connectivity, duality invariants, interval decompositions, and series-parallel constructions.

## References

1. **Operads of Poset Matrices**
   - DOI: https://doi.org/10.37236/14396

2. **Computational and Combinatorial Enumeration of Poset Matrices**
   - https://www.researchgate.net/publication/385137523_Computational_and_Combinatorial_Enumeration_of_Poset_Matrices
### Asymptotic Performance Boundaries
* **Bitmask Serialization**: Maps relation blocks $\mathcal{M}_{n \times n}(\mathbb{F}_2) \longleftrightarrow \mathbb{Z}_{2^{n^2}}$ in $\mathcal{O}(n^2)$ time via arbitrary-precision vectorized masks.
* **Property Classification Workbench**: Validates connectivity, self-duality mapping, and complex bilateral **Double-Dualizability** constraints within strict structural complexity limits of $\mathcal{O}(n^3)$ utilizing NetworkX undirected graphs.
* **Transitive Reduction Mapping**: Isolates immediate covering relations to automatically project clean topological **Hasse Diagrams** in $\mathcal{O}(n^3)$ time complexity boundaries.

---

## 3. Quickstart & Installation Setup

Ensure you have a local environment with Python 3.12+ active on your computer.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Execute the Test Suite Suite
To check all 25 automated testing logic paths locally, trigger the pytest package:
```bash
python -m pytest -v
```

### 3. Track Global Combinatorial Growth
To regenerate the full theorem metrics matrix spreadsheet and extract raw journal-ready LaTeX booktabs tables directly onto your terminal console, run:
```bash
python verify_ops.py
```

### 4. Run Scaling Benchmarks
Measure throughput execution times up to $64 \times 64$ dimensions and export a publication chart plot:
```bash
python benchmark_poset.py
python plot_benchmark.py
```
*Outputs are written directly to `docs/poset_scaling_analysis.png`.*

---

## 4. Citation Format

If you utilize this computational framework or its underlying operadic partition algorithms in your academic research, please cite the repository software package as follows:

### BibTeX Format
```bibtex
@software{mwafise2026posets,
  author       = {Mesinga Mwafise, Arnauld},
  title        = {Combinatorial Enumeration of Finite Lower-Triangular Poset Matrices},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub Repository},
  howpublished \(= {\url{https://github.com/arnauldmwafise/combinatorial_enumeration_posets}} \)}
```

### APA Format
Mesinga Mwafise, A. (2026). *Combinatorial Enumeration of Finite Lower-Triangular Poset Matrices* [Computer software]. GitHub Repository. https://github.com

---

## 5. License

This project is open-source software licensed under the terms of the [MIT License](LICENSE).
