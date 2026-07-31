"""
Evaluation script to test trained models.
Generates academic/forensic evidence: ROC curves, confusion matrices,
precision, recall, f1-score, and Equal Error Rate (EER) reports.
"""
import json
import argparse
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import create_data_loaders
from train_video import create_model as create_video_model
from train_audio import AudioSpoofModel


def compute_eer(fpr, tpr, thresholds):
    """Compute Equal Error Rate (EER) from ROC curve statistics."""
    fnr = 1 - tpr
    # Find the threshold where FNR and FPR are closest
    idx = np.nanargmin(np.absolute(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]
    return eer, eer_threshold


def evaluate_model(model, loader, device, model_type="video"):
    """Evaluate model and collect predictions and targets."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch
            
            # For video, extract middle frame
            if model_type == "video" and inputs.dim() == 5:
                inputs = inputs[:, inputs.shape[1] // 2]
                
            inputs = inputs.to(device)
            outputs = model(inputs)
            
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy().flatten())
            
    return np.array(all_preds), np.array(all_targets)


def generate_reports(preds, targets, output_dir: Path, model_type: str):
    """Generate and save evaluation metrics and charts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not SKLEARN_AVAILABLE:
        print("scikit-learn is required to generate metrics report.")
        return
        
    # Calculate binary predictions
    binary_preds = (preds > 0.5).astype(int)
    
    # Generate classification report
    report_dict = classification_report(targets, binary_preds, output_dict=True)
    report_txt = classification_report(targets, binary_preds)
    
    print("\n" + "="*40)
    print(f"EVALUATION REPORT: {model_type.upper()}")
    print("="*40)
    print(report_txt)
    
    # Save text report
    with open(output_dir / f"{model_type}_classification_report.txt", "w") as f:
        f.write(report_txt)
        
    # Save json metrics
    metrics = {
        "accuracy": float(report_dict["accuracy"]),
        "precision": float(report_dict["macro avg"]["precision"]),
        "recall": float(report_dict["macro avg"]["recall"]),
        "f1_score": float(report_dict["macro avg"]["f1-score"]),
    }
    
    # Compute ROC Curve and AUC
    fpr, tpr, thresholds = roc_curve(targets, preds)
    roc_auc = auc(fpr, tpr)
    metrics["auc"] = float(roc_auc)
    
    # Compute EER
    eer, eer_thresh = compute_eer(fpr, tpr, thresholds)
    metrics["eer"] = float(eer)
    metrics["eer_threshold"] = float(eer_thresh)
    
    print(f"AUC: {roc_auc:.4f}")
    print(f"Equal Error Rate (EER): {eer:.4f} (at threshold {eer_thresh:.4f})")
    
    with open(output_dir / f"{model_type}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
        
    # Plotting
    if PLT_AVAILABLE:
        # 1. ROC Curve
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f}, EER = {eer:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic - {model_type.title()} Detection')
        plt.legend(loc="lower right")
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(output_dir / f"{model_type}_roc_curve.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion Matrix
        cm = confusion_matrix(targets, binary_preds)
        plt.figure(figsize=(5, 4))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix - {model_type.title()} Detection')
        plt.colorbar()
        tick_marks = np.arange(2)
        classes = ['Real/Bonafide', 'Fake/Spoof']
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes, rotation=90)
        
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
                     
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(output_dir / f"{model_type}_confusion_matrix.png", dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved evaluation charts to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection models")
    parser.add_argument("--model-type", type=str, required=True, choices=["video", "audio"], help="Model type to evaluate")
    parser.add_argument("--weights", type=str, required=True, help="Path to weights file (.pt/.pth)")
    parser.add_argument("--data-dir", type=str, default="./ml/data", help="Data directory")
    parser.add_argument("--output-dir", type=str, default="./ml/evaluation", help="Output directory for reports")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu")
    args = parser.parse_args()
    
    if not TORCH_AVAILABLE:
        print("PyTorch is required for evaluation.")
        return
        
    device = torch.device(args.device)
    print(f"Evaluating using device: {device}")
    
    # Load model
    if args.model_type == "video":
        model = create_video_model(pretrained=False).to(device)
    else:
        model = AudioSpoofModel().to(device)
        
    checkpoint = torch.load(args.weights, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    # Data loader
    _, _, test_loader = create_data_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=0, modality=args.model_type
    )
    
    preds, targets = evaluate_model(model, test_loader, device, args.model_type)
    generate_reports(preds, targets, Path(args.output_dir), args.model_type)


if __name__ == "__main__":
    main()
