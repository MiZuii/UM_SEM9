import json
import os
import sys

# Configuration matches your original script
RESULTS_DIR = "./results/single"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")

def load_json(filename):
    """Safely load a JSON file."""
    path = os.path.join(METRICS_DIR, filename)
    if not os.path.exists(path):
        print(f"Error: Could not find {filename} at {path}")
        return None
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"Error: {filename} is not a valid JSON file.")
        return None

def format_value(val, fmt="{:.4f}"):
    """Helper to format None values gracefully."""
    return fmt.format(val) if val is not None else "N/A"

def print_pretty_report():
    print(f"Loading metrics from: {METRICS_DIR} ...\n")
    
    summary_results = load_json("summary_metrics.json")
    comparison_report = load_json("comparison_report.json")

    if not summary_results or not comparison_report:
        print("Could not load necessary files. Run the evaluation script first.")
        sys.exit(1)

    print(f"{'='*80}")
    print(f"{'MODEL EVALUATION REPORT':^80}")
    print(f"{'='*80}")

    # Iterate through datasets sorted alphabetically
    for dataset_name in sorted(summary_results.keys()):
        print(f"\nDataset: {dataset_name}")
        print(f"{'-'*80}")
        
        # Iterate through models sorted alphabetically
        for model_name in sorted(summary_results[dataset_name].keys()):
            model_data = summary_results[dataset_name][model_name]
            
            # Check for missing data
            if model_data.get("baseline") is None:
                print(f"  • {model_name:<15} | ⚠ Baseline data missing")
                continue
            if model_data.get("diet") is None:
                print(f"  • {model_name:<15} | ⚠ DiET data missing")
                continue

            # Extract data
            baseline = model_data["baseline"]
            diet = model_data["diet"]
            comp = comparison_report.get(dataset_name, {}).get(model_name, {})
            faithfulness = model_data.get("faithfulness")

            # Determine improvement/degradation symbol
            acc_diff = comp.get("accuracy_difference", 0)
            diff_symbol = "↑" if acc_diff > 0 else "↓"
            if abs(acc_diff) < 1e-5: diff_symbol = "="

            # === PRINT BLOCK ===
            print(f"  Model: {model_name}")
            
            # 1. Accuracy Section
            print(f"    {'Accuracy':<18} | Baseline: {format_value(baseline['accuracy'])} "
                  f"→ DiET: {format_value(diet['accuracy'])}")
            
            pct_imp = comp.get('accuracy_improvement_percent', 0)
            print(f"    {'Difference':<18} | {diff_symbol} {format_value(acc_diff, '{:+.4f}')} ({format_value(pct_imp, '{:+.2f}')}%)")
            
            # 2. Faithfulness Section
            if faithfulness is not None:
                print(f"    {'Faithfulness':<18} | {format_value(faithfulness)}")
            
            # 3. Detailed Metrics Compact Row
            print(f"    {'Other Metrics':<18} | "
                  f"F1(Mac): {format_value(diet['f1_macro'])} | "
                  f"Bal.Acc: {format_value(diet['balanced_accuracy'])}")
            
            print("") # Empty line between models
        
        print(f"{'.'*80}") # Separator between datasets

    print(f"\n{'='*80}")
    print("End of Report")
    print(f"{'='*80}")

if __name__ == "__main__":
    print_pretty_report()