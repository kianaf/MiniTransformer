#!/usr/bin/env python3
"""
Script to parse simulation results files and generate LaTeX table.
"""

import re
import os
from pathlib import Path

def parse_result_file(filepath):
    """Parse a simulation results file and extract metrics."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Extract n_train from filename
    filename = os.path.basename(filepath)
    n_match = re.search(r'n=(\d+)', filename)
    n_train = int(n_match.group(1)) if n_match else None
    
    # Extract metrics using regex
    metrics = {}
    
    # Total MSE metrics
    baseline_avg_match = re.search(r'baseline average loss: ([\d.]+) ± ([\d.]+)', content)
    regression_total_match = re.search(r'regression loss total: ([\d.]+) ± ([\d.]+)', content)
    informed_match = re.search(r'baseline informed loss: ([\d.]+) ± ([\d.]+)', content)
    model_total_match = re.search(r'model loss total: ([\d.]+) ± ([\d.]+)', content)
    
    # Predindex MSE metrics (MSE_j3)
    baseline_avg_pred_match = re.search(r'baseline average loss predindex: ([\d.]+) ± ([\d.]+)', content)
    regression_pred_match = re.search(r'regression loss predindex: ([\d.]+) ± ([\d.]+)', content)
    informed_pred_match = re.search(r'baseline informed loss predindex: ([\d.]+) ± ([\d.]+)', content)
    model_pred_match = re.search(r'model loss predindex: ([\d.]+) ± ([\d.]+)', content)
    
    if baseline_avg_match:
        metrics['average_mse'] = (baseline_avg_match.group(1), baseline_avg_match.group(2))
    if baseline_avg_pred_match:
        metrics['average_mse_j3'] = (baseline_avg_pred_match.group(1), baseline_avg_pred_match.group(2))
    
    if regression_total_match:
        metrics['regression_mse'] = (regression_total_match.group(1), regression_total_match.group(2))
    if regression_pred_match:
        metrics['regression_mse_j3'] = (regression_pred_match.group(1), regression_pred_match.group(2))
    
    if informed_match:
        metrics['informed_mse'] = (informed_match.group(1), informed_match.group(2))
    if informed_pred_match:
        metrics['informed_mse_j3'] = (informed_pred_match.group(1), informed_pred_match.group(2))
    
    if model_total_match:
        metrics['model_mse'] = (model_total_match.group(1), model_total_match.group(2))
    if model_pred_match:
        metrics['model_mse_j3'] = (model_pred_match.group(1), model_pred_match.group(2))
    
    return n_train, metrics

def format_latex_value(value, std, is_bold=False):
    """Format a value for LaTeX table."""
    # Round to 3 decimal places
    val = float(value)
    std_val = float(std)
    
    # Determine precision based on value
    if val >= 0.1:
        val_str = f"{val:.3f}"
        std_str = f"{std_val:.3f}"
    else:
        val_str = f"{val:.3f}"
        std_str = f"{std_val:.3f}"
    
    # Remove trailing zeros
    val_str = val_str.rstrip('0').rstrip('.')
    std_str = std_str.rstrip('0').rstrip('.')
    
    result = f"${val_str} \\pm {std_str}$"
    if is_bold:
        result = f"$\\mathbf{{{val_str} \\pm {std_str}}}$"
    
    return result

def find_best_values(results_dict, n_values):
    """Find the best (minimum) values for each metric and n_train."""
    best_values = {}
    
    for n in n_values:
        if n not in results_dict:
            continue
        
        metrics = results_dict[n]
        
        # For each metric type, find minimum
        metric_types = ['mse', 'mse_j3']
        for metric_type in metric_types:
            values = []
            keys = []
            
            if metric_type == 'mse':
                if 'average_mse' in metrics:
                    values.append(float(metrics['average_mse'][0]))
                    keys.append('average')
                if 'regression_mse' in metrics:
                    values.append(float(metrics['regression_mse'][0]))
                    keys.append('regression')
                if 'informed_mse' in metrics:
                    values.append(float(metrics['informed_mse'][0]))
                    keys.append('informed')
                if 'model_mse' in metrics:
                    values.append(float(metrics['model_mse'][0]))
                    keys.append('model')
            else:  # mse_j3
                if 'average_mse_j3' in metrics:
                    values.append(float(metrics['average_mse_j3'][0]))
                    keys.append('average')
                if 'regression_mse_j3' in metrics:
                    values.append(float(metrics['regression_mse_j3'][0]))
                    keys.append('regression')
                if 'informed_mse_j3' in metrics:
                    values.append(float(metrics['informed_mse_j3'][0]))
                    keys.append('informed')
                if 'model_mse_j3' in metrics:
                    values.append(float(metrics['model_mse_j3'][0]))
                    keys.append('model')
            
            if values:
                min_idx = values.index(min(values))
                best_key = f"{n}_{metric_type}_{keys[min_idx]}"
                best_values[best_key] = True
    
    return best_values

def generate_latex_table(results_dict):
    """Generate LaTeX table from parsed results."""
    
    # Order of n_train values
    n_values = [100, 200, 500, 1000]
    
    # Find best values for bolding
    best_values = find_best_values(results_dict, n_values)
    
    # Build table rows
    rows = []
    
    # Average row
    avg_mse_row = ["\\multirow{2}{*}{\\makecell{Average}}", "MSE"]
    avg_mse_j3_row = ["", "MSE$_{j_3}$"]
    
    for n in n_values:
        if n in results_dict:
            metrics = results_dict[n]
            if 'average_mse' in metrics:
                is_bold = best_values.get(f"{n}_mse_average", False)
                avg_mse_row.append(format_latex_value(*metrics['average_mse'], is_bold=is_bold))
            else:
                avg_mse_row.append("--")
            
            if 'average_mse_j3' in metrics:
                is_bold = best_values.get(f"{n}_mse_j3_average", False)
                avg_mse_j3_row.append(format_latex_value(*metrics['average_mse_j3'], is_bold=is_bold))
            else:
                avg_mse_j3_row.append("--")
        else:
            avg_mse_row.append("--")
            avg_mse_j3_row.append("--")
    
    rows.append(" & ".join(avg_mse_row) + " \\\\")
    rows.append(" & ".join(avg_mse_j3_row) + " \\\\")
    rows.append("\\addlinespace")
    
    # Regression row
    reg_mse_row = ["\\multirow{2}{*}{Regression}", "MSE"]
    reg_mse_j3_row = ["", "MSE$_{j_3}$"]
    
    for n in n_values:
        if n in results_dict:
            metrics = results_dict[n]
            if 'regression_mse' in metrics:
                is_bold = best_values.get(f"{n}_mse_regression", False)
                reg_mse_row.append(format_latex_value(*metrics['regression_mse'], is_bold=is_bold))
            else:
                reg_mse_row.append("--")
            
            if 'regression_mse_j3' in metrics:
                is_bold = best_values.get(f"{n}_mse_j3_regression", False)
                reg_mse_j3_row.append(format_latex_value(*metrics['regression_mse_j3'], is_bold=is_bold))
            else:
                reg_mse_j3_row.append("--")
        else:
            reg_mse_row.append("--")
            reg_mse_j3_row.append("--")
    
    rows.append(" & ".join(reg_mse_row) + " \\\\")
    rows.append(" & ".join(reg_mse_j3_row) + " \\\\")
    rows.append("\\addlinespace")
    
    # Informed row
    inf_mse_row = ["\\multirow{2}{*}{Informed}", "MSE"]
    inf_mse_j3_row = ["", "MSE$_{j_3}$"]
    
    for n in n_values:
        if n in results_dict:
            metrics = results_dict[n]
            if 'informed_mse' in metrics:
                is_bold = best_values.get(f"{n}_mse_informed", False)
                inf_mse_row.append(format_latex_value(*metrics['informed_mse'], is_bold=is_bold))
            else:
                inf_mse_row.append("--")
            
            if 'informed_mse_j3' in metrics:
                is_bold = best_values.get(f"{n}_mse_j3_informed", False)
                inf_mse_j3_row.append(format_latex_value(*metrics['informed_mse_j3'], is_bold=is_bold))
            else:
                inf_mse_j3_row.append("--")
        else:
            inf_mse_row.append("--")
            inf_mse_j3_row.append("--")
    
    rows.append(" & ".join(inf_mse_row) + " \\\\")
    rows.append(" & ".join(inf_mse_j3_row) + " \\\\")
    rows.append("\\addlinespace")
    
    # MiniTransformer row
    model_mse_row = ["\\multirow{2}{*}{MiniTransformer}", "MSE"]
    model_mse_j3_row = ["", "MSE$_{j_3}$"]
    
    for n in n_values:
        if n in results_dict:
            metrics = results_dict[n]
            if 'model_mse' in metrics:
                is_bold = best_values.get(f"{n}_mse_model", False)
                model_mse_row.append(format_latex_value(*metrics['model_mse'], is_bold=is_bold))
            else:
                model_mse_row.append("--")
            
            if 'model_mse_j3' in metrics:
                is_bold = best_values.get(f"{n}_mse_j3_model", False)
                model_mse_j3_row.append(format_latex_value(*metrics['model_mse_j3'], is_bold=is_bold))
            else:
                model_mse_j3_row.append("--")
        else:
            model_mse_row.append("--")
            model_mse_j3_row.append("--")
    
    rows.append(" & ".join(model_mse_row) + " \\\\")
    rows.append(" & ".join(model_mse_j3_row) + " \\\\")
    
    # Generate full table
    table = """\\begin{table}[h]
\\centering
\\label{tab:sim}
    \\begin{tabular}{llcccc} 
        \\toprule
        \\textbf{Approach} & \\textbf{Metric} & $n_\\text{train} = 100 $ & $n_\\text{train} = 200$ & $n_\\text{train} = 500$ & $n_\\text{train} = 1000$ \\\\
        & & & & & \\\\ 
        \\midrule
"""
    
    table += "\n".join(rows)
    
    table += """
        \\bottomrule
    \\end{tabular}
\\end{table}"""
    
    return table

def main():
    # Find all simulation result files. This script lives at
    # notebooks/experiments/tables/, but the .txt inputs and .tex output live
    # directly under notebooks/, so resolve two levels up.
    notebooks_dir = Path(__file__).parents[2]
    result_files = list(notebooks_dir.glob("simulation_results_n=*.txt"))
    
    # Filter to only use epochs=100 files (prefer epochs=100 over epochs=200)
    # Sort files so epochs=100 comes before epochs=200
    result_files_sorted = sorted(result_files, key=lambda x: (
        int(re.search(r'n=(\d+)', x.name).group(1)) if re.search(r'n=(\d+)', x.name) else 0,
        int(re.search(r'epochs=(\d+)', x.name).group(1)) if re.search(r'epochs=(\d+)', x.name) else 999
    ))
    
    # Parse all files, keeping only the first occurrence of each n_train
    results_dict = {}
    seen_n_values = set()
    for filepath in result_files_sorted:
        n_train, metrics = parse_result_file(filepath)
        if n_train and n_train not in seen_n_values:
            results_dict[n_train] = metrics
            seen_n_values.add(n_train)
            print(f"Parsed n={n_train}: {len(metrics)} metrics found from {filepath.name}")
    
    # Generate LaTeX table
    latex_table = generate_latex_table(results_dict)
    
    # Print table
    print("\n" + "="*80)
    print("Generated LaTeX Table:")
    print("="*80 + "\n")
    print(latex_table)
    
    # Save to file
    output_file = notebooks_dir / "simulation_table.tex"
    with open(output_file, 'w') as f:
        f.write(latex_table)
    
    print(f"\n\nTable saved to: {output_file}")

if __name__ == "__main__":
    main()
