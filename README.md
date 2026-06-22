# Combinatorial Enumeration of Finite Lower-Triangular Poset Matrices

[![Poset Research Pipeline CI](https://github.com)](https://github.com)
[![Python Version](https://shields.io)](https://python.org)

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
│   ├── poset_operad.py            # High-performance, bitpacked algebraic composition matrix engine
│   └── poset_processor.py         # Structural filter classifying dualizable properties
├── tests/                         # Complete 25-case unit testing engine suite
├── benchmark_poset.py             # Throughput latency tracking entrypoint
├── plot_benchmark.py              # High-res line plot rendering pipeline
├── pyproject.toml                 # Root path directory mapping declarations
├── requirements.txt               # Explicit package library dependencies
└── verify_ops.py                  # Main orchestration driver tracking all 9 theorems
```

---

## 2. Mathematical Context & Complexity Specifications

This workspace bridges algebraic topology and order theory by formalizing the **partial operadic composition** $\circ_i: \mathcal{P}(p) \times \mathcal{P}(q) \rightarrow \mathcal{P}(p+q-1)$ over boolean matrices sitting in a lower-triangular format ($M_{j,k} = 1 \iff j \ge k$).

### Asymptotic Performance Boundaries
* **Bitmask Serialization**: Maps relation blocks $\mathcal{M}_{n \times n}(\mathbb{F}_2) \longleftrightarrow \mathbb{Z}_{2^{n^2}}$ in $\mathcal{O}(n^2)$ time via arbitrary-precision vectorized masks.
* **Property ClassificationWorkbench**: Validates connectivity, self-duality mapping, and complex bilateral **Double-Dualizability** constraints within strict structural complexity limits of $\mathcal{O}(n^3)$ utilizing NetworkX undirected graphs.
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
