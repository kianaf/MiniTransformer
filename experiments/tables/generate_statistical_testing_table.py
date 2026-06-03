#!/usr/bin/env python3
"""
Script to parse statistical testing simulation results files and generate a LaTeX table.

The input file has blocks of the form:
    For combination sample size: <V>
    Variable 1: <p-value> ± <std>
    Variable 2: <p-value> ± <std>
    ...

The output is a LaTeX table with V values as columns and variables as rows.
Values below a threshold (default 0.001) are displayed as '<0.001'.
"""

import re
import sys
from pathlib import Path
from collections import OrderedDict


def parse_results_file(filepath):
    """
    Parse a statistical testing results file.

    Handles both numeric V values ("For combination sample size: 8")
    and the exact case ("For combination sample size: all (exact)").
    For "all (exact)", V is set to 2^p where p is extracted from the filename.

    Returns:
        dict: {V: {var_num: (p_val, std_val, is_exact)}}
    """
    # Extract p from the filename (e.g. "p=4")
    p_match = re.search(r"p=(\d+)", str(filepath))
    p = int(p_match.group(1)) if p_match else None

    data = OrderedDict()
    current_v = None
    current_exact = False

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match "For combination sample size: all (exact)"
            exact_match = re.match(
                r"For combination sample size:\s*all\s*\(exact\)", line
            )
            if exact_match:
                if p is None:
                    print(
                        f"Warning: 'all (exact)' found but p not in filename, skipping.",
                        file=sys.stderr,
                    )
                    current_v = None
                    continue
                current_v = 2**p
                current_exact = True
                if current_v not in data:
                    data[current_v] = OrderedDict()
                continue

            # Match "For combination sample size: <V>"
            v_match = re.match(r"For combination sample size:\s*(\d+)", line)
            if v_match:
                current_v = int(v_match.group(1))
                current_exact = False
                if current_v not in data:
                    data[current_v] = OrderedDict()
                continue

            # Match "Variable <num>: <p_value> ± <std>"
            var_match = re.match(
                r"Variable\s+(\d+):\s*([\d.]+(?:e[+-]?\d+)?)\s*±\s*([\d.]+(?:e[+-]?\d+)?)",
                line,
            )
            if var_match and current_v is not None:
                var_num = int(var_match.group(1))
                p_val = float(var_match.group(2))
                std_val = float(var_match.group(3))
                data[current_v][var_num] = (p_val, std_val, current_exact)

    return data


def format_value(p_val, std_val, is_exact=False, threshold=0.001):
    """
    Format a (p_value, std) pair for LaTeX.

    For exact values (is_exact=True), only the p-value is shown (no ±).
    Values below threshold are displayed as '<0.001'.
    """
    def fmt_single(val):
        if val < threshold:
            return "<0.001"
        # Use 3 decimal places, strip trailing zeros
        s = f"{val:.3f}".rstrip("0").rstrip(".")
        return s

    p_str = fmt_single(p_val)

    if is_exact:
        return f"\\({p_str}\\)"

    s_str = fmt_single(std_val)
    return f"\\({p_str} \\pm {s_str}\\)"


def generate_latex_table(data, label="tab:statistical_testing_results_sim"):
    """
    Generate a LaTeX table from the parsed data.

    Args:
        data: dict {V: {var_num: (p_val, std)}} as returned by parse_results_file
        label: LaTeX label for the table

    Returns:
        str: LaTeX table source
    """
    # Sort V values in ascending order for the columns
    v_values = sorted(data.keys())

    # Collect all variable numbers across all V groups
    all_vars = sorted(set(v for vd in data.values() for v in vd.keys()))

    # Number of columns: 1 (Variable) + len(v_values)
    n_cols = 1 + len(v_values)
    col_spec = "c" * n_cols

    lines = []
    lines.append(f"\\label{{{label}}}")
    lines.append("    % \\footnotesize")
    lines.append(f"    \\begin{{tabular}}{{{col_spec}}}")
    lines.append("        \\toprule")

    # Header row
    header_parts = ["Variable"]
    for v in v_values:
        header_parts.append(f"$V={v}$")
    lines.append("        " + " & \n        ".join(header_parts) + "\\\\  ")

    lines.append("        \\midrule")

    # Data rows
    for var_num in all_vars:
        row_parts = [f"         {var_num} "]
        for v in v_values:
            if var_num in data[v]:
                p_val, std_val, is_exact = data[v][var_num]
                cell = format_value(p_val, std_val, is_exact=is_exact)
            else:
                cell = "-- "
            row_parts.append(f" {cell}  ")
        # Join with & and add newlines for readability
        row_line = " &".join(row_parts)
        lines.append(row_line)
        lines.append("            \\\\")

    lines.append("        \\bottomrule")
    lines.append("    \\end{tabular}")

    return "\n".join(lines)


def main():
    # This script lives at experiments/tables/, but the statistical_testing/
    # inputs and the .tex output live under notebooks/ at the repo root
    # (parents[2]).
    notebooks_dir = Path(__file__).parents[2] / "notebooks"
    results_dir = notebooks_dir / "statistical_testing"

    # --- Configuration: input file and output path ---
    input_file = results_dir / "statistical_testing_simulation_results_n=50_batch_size=1_p=4_ncum=4_nheads=4_epochs=200_seed=12345.txt"
    output_file = notebooks_dir / "statistical_testing_table.tex"

    if not input_file.exists():
        print(f"Error: file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Parse the file
    data = parse_results_file(input_file)

    if not data:
        print("Error: no data found in the file.", file=sys.stderr)
        sys.exit(1)

    # Print summary
    print(f"Parsed: {input_file.name}", file=sys.stderr)
    for v, vars_dict in sorted(data.items()):
        exact_flag = " (exact)" if any(e for _, _, e in vars_dict.values()) else ""
        print(f"  V={v}{exact_flag}: {len(vars_dict)} variables", file=sys.stderr)

    # Generate table
    latex_table = generate_latex_table(data)

    # Save
    with open(output_file, "w") as f:
        f.write(latex_table)
    print(f"\nSaved to: {output_file}", file=sys.stderr)

    # Also print to stdout
    print(latex_table)


if __name__ == "__main__":
    main()
