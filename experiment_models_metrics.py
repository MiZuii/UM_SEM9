import torch
import os
import json
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef,
    confusion_matrix, classification_report, roc_auc_score, log_loss
)
from utils.data_loader import get_dataloader
from utils.model_loader import get_model

HIGHEST_STEP_NUM = 4

def compute_metrics(model, test_loader, num_classes, device):
    """
    Compute comprehensive metrics for a model.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Basic metrics
    metrics = {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "balanced_accuracy": float(balanced_accuracy_score(all_labels, all_preds)),
    }

    # Store raw predictions temporarily to calculate faithfulness later
    # We will pop this out before saving to JSON to avoid massive file sizes
    metrics["predictions"] = all_preds
    
    # Precision, Recall, F1 (macro and weighted)
    metrics["precision_macro"] = float(precision_score(all_labels, all_preds, average='macro', zero_division=0))
    metrics["precision_weighted"] = float(precision_score(all_labels, all_preds, average='weighted', zero_division=0))
    metrics["recall_macro"] = float(recall_score(all_labels, all_preds, average='macro', zero_division=0))
    metrics["recall_weighted"] = float(recall_score(all_labels, all_preds, average='weighted', zero_division=0))
    metrics["f1_macro"] = float(f1_score(all_labels, all_preds, average='macro', zero_division=0))
    metrics["f1_weighted"] = float(f1_score(all_labels, all_preds, average='weighted', zero_division=0))
    
    # Per-class metrics
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)
    
    metrics["precision_per_class"] = [float(p) for p in precision_per_class]
    metrics["recall_per_class"] = [float(r) for r in recall_per_class]
    metrics["f1_per_class"] = [float(f) for f in f1_per_class]
    
    # Additional metrics
    try:
        metrics["cohen_kappa"] = float(cohen_kappa_score(all_labels, all_preds))
    except:
        metrics["cohen_kappa"] = None
    
    try:
        metrics["matthews_corrcoef"] = float(matthews_corrcoef(all_labels, all_preds))
    except:
        metrics["matthews_corrcoef"] = None
    
    # Log loss (cross-entropy)
    try:
        metrics["log_loss"] = float(log_loss(all_labels, all_probs))
    except:
        metrics["log_loss"] = None
    
    # ROC-AUC (for multi-class, use ovr)
    try:
        if num_classes == 2:
            metrics["roc_auc"] = float(roc_auc_score(all_labels, all_probs[:, 1]))
        else:
            metrics["roc_auc_ovr"] = float(roc_auc_score(all_labels, all_probs, multi_class='ovr', zero_division=0))
            metrics["roc_auc_ovo"] = float(roc_auc_score(all_labels, all_probs, multi_class='ovo', zero_division=0))
    except:
        metrics["roc_auc"] = None
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    metrics["confusion_matrix"] = cm.tolist()
    
    # Classification report
    clf_report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    metrics["classification_report"] = clf_report
    
    # Additional statistics
    metrics["total_samples"] = int(len(all_labels))
    metrics["correct_predictions"] = int(np.sum(all_preds == all_labels))
    metrics["incorrect_predictions"] = int(len(all_labels) - np.sum(all_preds == all_labels))
    
    return metrics

def evaluate_all_models():
    """
    Evaluate all baseline and diet models across all datasets and models.
    Save comprehensive metrics to JSON files.
    """
    MODELS = ['resnet18', 'resnet34']
    DATASETS = ['CIFAR10', 'CIFAR100', 'HardMNIST']
    MODEL_DIR = "./models"
    RESULTS_DIR = "./results/single"
    METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
    
    os.makedirs(METRICS_DIR, exist_ok=True)
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {DEVICE}\n")
    
    # Store all results
    all_results = {}
    summary_results = {}
    
    for dataset_name in DATASETS:
        print(f"{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")
        
        _, test_loader, num_classes = get_dataloader(dataset_name, batch_size=32)
        
        all_results[dataset_name] = {}
        summary_results[dataset_name] = {}
        
        for model_name in MODELS:
            print(f"\n  Model: {model_name}")
            all_results[dataset_name][model_name] = {}
            summary_results[dataset_name][model_name] = {}
            
            # Variables to hold predictions for faithfulness calculation
            baseline_preds = None
            diet_preds = None

            # ============ BASELINE MODEL ============
            print(f"    > Evaluating baseline model...", end=" ", flush=True)
            baseline_weights = os.path.join(MODEL_DIR, f"{model_name}_{dataset_name}_model.pth")
            
            if not os.path.exists(baseline_weights):
                print(f"✗ Weights not found: {baseline_weights}")
                all_results[dataset_name][model_name]["baseline"] = None
                summary_results[dataset_name][model_name]["baseline"] = None
            else:
                baseline_model = get_model(model_name, num_classes, baseline_weights, DEVICE)
                
                if baseline_model is None:
                    print("✗ Failed to load model")
                    all_results[dataset_name][model_name]["baseline"] = None
                    summary_results[dataset_name][model_name]["baseline"] = None
                else:
                    baseline_metrics = compute_metrics(baseline_model, test_loader, num_classes, DEVICE)
                    
                    # Extract predictions for faithfulness check and remove from dict to save space
                    baseline_preds = baseline_metrics.pop("predictions", None)
                    
                    all_results[dataset_name][model_name]["baseline"] = baseline_metrics
                    
                    # Create summary for baseline
                    summary_results[dataset_name][model_name]["baseline"] = {
                        "accuracy": baseline_metrics["accuracy"],
                        "balanced_accuracy": baseline_metrics["balanced_accuracy"],
                        "f1_macro": baseline_metrics["f1_macro"],
                        "f1_weighted": baseline_metrics["f1_weighted"],
                        "precision_macro": baseline_metrics["precision_macro"],
                        "recall_macro": baseline_metrics["recall_macro"],
                    }
                    
                    print(f"✓ Accuracy: {baseline_metrics['accuracy']:.4f}")
            
            # ============ DIET MODEL ============
            print(f"    > Evaluating DiET model...", end=" ", flush=True)
            diet_weights = os.path.join(
                MODEL_DIR, 
                f"{model_name}_{dataset_name}_{HIGHEST_STEP_NUM}_diet_model.pth"
            )
            
            if not os.path.exists(diet_weights):
                print(f"✗ Weights not found: {diet_weights}")
                all_results[dataset_name][model_name]["diet"] = None
                summary_results[dataset_name][model_name]["diet"] = None
            else:
                diet_model = get_model(model_name, num_classes, diet_weights, DEVICE)
                
                if diet_model is None:
                    print("✗ Failed to load model")
                    all_results[dataset_name][model_name]["diet"] = None
                    summary_results[dataset_name][model_name]["diet"] = None
                else:
                    diet_metrics = compute_metrics(diet_model, test_loader, num_classes, DEVICE)
                    
                    # Extract predictions for faithfulness check and remove from dict
                    diet_preds = diet_metrics.pop("predictions", None)

                    all_results[dataset_name][model_name]["diet"] = diet_metrics
                    
                    # Create summary for diet
                    summary_results[dataset_name][model_name]["diet"] = {
                        "accuracy": diet_metrics["accuracy"],
                        "balanced_accuracy": diet_metrics["balanced_accuracy"],
                        "f1_macro": diet_metrics["f1_macro"],
                        "f1_weighted": diet_metrics["f1_weighted"],
                        "precision_macro": diet_metrics["precision_macro"],
                        "recall_macro": diet_metrics["recall_macro"],
                    }
                    
                    print(f"✓ Accuracy: {diet_metrics['accuracy']:.4f}")
            
            # ============ COMPARISON & FAITHFULNESS ============
            if all_results[dataset_name][model_name]["baseline"] and all_results[dataset_name][model_name]["diet"]:
                # Accuracy Diff
                acc_diff = diet_metrics['accuracy'] - baseline_metrics['accuracy']
                print(f"    > Accuracy difference (DiET - Baseline): {acc_diff:+.4f}")
                
                # Faithfulness Calculation
                # Agreement between Baseline (Ground Truth) and DiET
                if baseline_preds is not None and diet_preds is not None:
                    faithfulness = float(accuracy_score(baseline_preds, diet_preds))
                    summary_results[dataset_name][model_name]["faithfulness"] = faithfulness
                    print(f"    > Faithfulness (Agreement with Baseline): {faithfulness:.4f}")

    # ============ SAVE RESULTS ============
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}\n")
    
    # Save detailed results
    detailed_path = os.path.join(METRICS_DIR, "detailed_metrics.json")
    with open(detailed_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Detailed metrics saved to: {detailed_path}")
    
    # Save summary results
    summary_path = os.path.join(METRICS_DIR, "summary_metrics.json")
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    print(f"✓ Summary metrics saved to: {summary_path}")
    
    # Create comparison report
    comparison_report = create_comparison_report(all_results)
    comparison_path = os.path.join(METRICS_DIR, "comparison_report.json")
    with open(comparison_path, 'w') as f:
        json.dump(comparison_report, f, indent=2)
    print(f"✓ Comparison report saved to: {comparison_path}")
    
    # Print summary
    print_summary(summary_results, comparison_report)

def create_comparison_report(all_results):
    """
    Create a detailed comparison report between baseline and diet models.
    """
    report = {}
    
    for dataset_name, dataset_results in all_results.items():
        report[dataset_name] = {}
        
        for model_name, model_results in dataset_results.items():
            baseline = model_results.get("baseline")
            diet = model_results.get("diet")
            
            if baseline is None or diet is None:
                report[dataset_name][model_name] = None
                continue
            
            comparison = {
                "baseline_accuracy": baseline["accuracy"],
                "diet_accuracy": diet["accuracy"],
                "accuracy_difference": diet["accuracy"] - baseline["accuracy"],
                "accuracy_improvement_percent": ((diet["accuracy"] - baseline["accuracy"]) / baseline["accuracy"] * 100) if baseline["accuracy"] > 0 else 0,
                
                "baseline_f1_macro": baseline["f1_macro"],
                "diet_f1_macro": diet["f1_macro"],
                "f1_macro_difference": diet["f1_macro"] - baseline["f1_macro"],
                
                "baseline_f1_weighted": baseline["f1_weighted"],
                "diet_f1_weighted": diet["f1_weighted"],
                "f1_weighted_difference": diet["f1_weighted"] - baseline["f1_weighted"],
                
                "baseline_balanced_accuracy": baseline["balanced_accuracy"],
                "diet_balanced_accuracy": diet["balanced_accuracy"],
                "balanced_accuracy_difference": diet["balanced_accuracy"] - baseline["balanced_accuracy"],
                
                "baseline_precision_macro": baseline["precision_macro"],
                "diet_precision_macro": diet["precision_macro"],
                "precision_macro_difference": diet["precision_macro"] - baseline["precision_macro"],
                
                "baseline_recall_macro": baseline["recall_macro"],
                "diet_recall_macro": diet["recall_macro"],
                "recall_macro_difference": diet["recall_macro"] - baseline["recall_macro"],
            }
            
            # Add Cohen's Kappa if available
            if baseline.get("cohen_kappa") is not None and diet.get("cohen_kappa") is not None:
                comparison["baseline_cohen_kappa"] = baseline["cohen_kappa"]
                comparison["diet_cohen_kappa"] = diet["cohen_kappa"]
                comparison["cohen_kappa_difference"] = diet["cohen_kappa"] - baseline["cohen_kappa"]
            
            # Add Matthews correlation if available
            if baseline.get("matthews_corrcoef") is not None and diet.get("matthews_corrcoef") is not None:
                comparison["baseline_matthews_corrcoef"] = baseline["matthews_corrcoef"]
                comparison["diet_matthews_corrcoef"] = diet["matthews_corrcoef"]
                comparison["matthews_corrcoef_difference"] = diet["matthews_corrcoef"] - baseline["matthews_corrcoef"]
            
            report[dataset_name][model_name] = comparison
    
    return report

def print_summary(summary_results, comparison_report):
    """
    Print a formatted summary of the results.
    """
    print(f"\n{'='*60}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*60}\n")
    
    for dataset_name in sorted(summary_results.keys()):
        print(f"Dataset: {dataset_name}")
        print("-" * 60)
        
        for model_name in sorted(summary_results[dataset_name].keys()):
            model_data = summary_results[dataset_name][model_name]
            
            if model_data.get("baseline") is None:
                print(f"  {model_name}: Baseline model not found")
                continue
            
            baseline = model_data["baseline"]
            diet = model_data["diet"]
            comparison = comparison_report.get(dataset_name, {}).get(model_name)
            
            print(f"\n  {model_name}:")
            print(f"    Baseline Accuracy: {baseline['accuracy']:.4f}")
            print(f"    DiET Accuracy:     {diet['accuracy']:.4f}")
            
            if comparison:
                acc_diff = comparison.get("accuracy_difference", 0)
                print(f"    Difference:        {acc_diff:+.4f} ({comparison.get('accuracy_improvement_percent', 0):+.2f}%)")
            
            print(f"    F1-Score (macro):  {baseline['f1_macro']:.4f} → {diet['f1_macro']:.4f}")
            
            # === PRINT FAITHFULNESS HERE ===
            faithfulness = model_data.get("faithfulness")
            if faithfulness is not None:
                print(f"    Faithfulness:      {faithfulness:.4f}")
        
        print()

if __name__ == "__main__":
    evaluate_all_models()