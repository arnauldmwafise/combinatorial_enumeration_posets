import re
import matplotlib.pyplot as plt

def generate_publication_chart():
    log_file = "benchmark_results.txt"
    output_image = "poset_scaling_analysis.png"
    
    # Storage maps for parsing metrics
    dimensions = []
    data_series = {}
    
    # Regular expression pattern to clean benchmark layout fields
    line_pattern = re.compile(r"^\s*(\d+)\s+(\w+)\s+([\d\.]+)\s+([\d\.]+)")
    
    with open(log_file, "r") as f:
        for line in f:
            match = line_pattern.match(line)
            if match:
                p_dim = int(match.group(1))
                variant = match.group(2)
                avg_time_ms = float(match.group(4))
                
                if p_dim not in dimensions:
                    dimensions.append(p_dim)
                    
                if variant not in data_series:
                    data_series[variant] = []
                data_series[variant].append(avg_time_ms)

    # Style optimization configuration for academic journals
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    
    # Plot distinct line markers for the evaluation series
    markers = {'general': 'o', 'boundary': 's', 'total': '^', 'fully_isolated': 'd'}
    colors = {'general': '#1f77b4', 'boundary': '#ff7f0e', 'total': '#2ca02c', 'fully_isolated': '#d62728'}
    
    for variant, times in data_series.items():
        ax.plot(dimensions, times, label=f"Variant: {variant}", 
                marker=markers.get(variant, 'x'), color=colors.get(variant, '#7f7f7f'),
                linewidth=1.8, markersize=6)
        
    # Apply standard scaling titles and layout parameters
    ax.set_title("Computational Scaling Characteristics of Bitpacked Poset Operads", fontsize=11, fontweight='bold', pad=12)
    ax.set_xlabel("Base Poset Dimension Size ($p \\times p$ Matrix)", fontsize=10)
    ax.set_ylabel("Average Latency per Composition ($ms$)", fontsize=10)
    
    ax.set_xscale('log', base=2)
    ax.set_xticks(dimensions)
    ax.set_xticklabels([str(d) for d in dimensions])
    
    ax.tick_params(axis='both', which='major', labelsize=9)
    ax.legend(frameon=True, facecolor='white', edgecolor='#e0e0e0', fontsize=9, loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6, color='#cccccc')
    
    plt.tight_layout()
    plt.savefig(output_image, bbox_inches='tight')
    print(f"Chart generated and saved to -> {output_image}")

if __name__ == "__main__":
    generate_publication_chart()
