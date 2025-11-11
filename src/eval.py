"""
Evaluation Script for Multimodal Sensor Fusion

Provides framework for:
- Standard evaluation on test set
- Missing modality robustness testing
- Generating results for experiments/ directory
- (NEW) Inference latency measurement (ms per sample)
"""

import time
import itertools
import argparse
import json
from pathlib import Path
from typing import Tuple, Dict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from train import MultimodalFusionModule
from data import create_dataloaders
from uncertainty import CalibrationMetrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = "cpu",
    return_predictions: bool = False,
    measure_inference: bool = True,
) -> Tuple[Dict, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Evaluate model on a dataset.

    Args:
        model: Trained model
        dataloader: Data loader (test/val)
        device: 'cpu' or 'cuda'
        return_predictions: If True, also return (preds, labels, confidences)
        measure_inference: If True, measure ms/sample during forward passes

    Returns:
        metrics: Dict with accuracy, loss, f1_macro, num_samples, (optional) inference_ms
        predictions: Optional tuple of (preds, labels, confidences)
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_confidences = []
    total_loss = 0.0
    num_batches = 0

    # timing
    total_forward_time = 0.0
    total_samples = 0
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    for batch in tqdm(dataloader, desc="Evaluating"):
        features, labels, mask = batch

        # Move to device
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)
        mask = mask.to(device)

        # Forward pass (measure if enabled)
        if measure_inference:
            if use_cuda:
                torch.cuda.synchronize()
            t0 = time.time()

        logits = model(features, mask)

        if measure_inference:
            if use_cuda:
                torch.cuda.synchronize()
            t1 = time.time()
            bs = labels.size(0)
            total_forward_time += (t1 - t0)
            total_samples += bs

        # Loss
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item()
        num_batches += 1

        # Predictions & confidences
        probs = F.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
        all_confidences.append(confidences.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_confidences = torch.cat(all_confidences)

    # Metrics
    accuracy = (all_preds == all_labels).float().mean().item()
    avg_loss = total_loss / max(1, num_batches)

    # F1 (macro)
    from sklearn.metrics import f1_score
    f1_macro = f1_score(
        all_labels.numpy(),
        all_preds.numpy(),
        average="macro",
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "loss": float(avg_loss),
        "num_samples": int(len(all_labels)),
    }

    if measure_inference and total_samples > 0:
        ms_per_sample = (total_forward_time / total_samples) * 1000.0
        metrics["inference_ms"] = float(ms_per_sample)

    if return_predictions:
        return metrics, (all_preds, all_labels, all_confidences)
    else:
        return metrics


@torch.no_grad()
def _evaluate_with_modality_subset(
    model,
    dataloader,
    available_indices,
    total_modalities,
    device,
):
    """Evaluate model with specific subset of modalities available."""
    model.eval()

    all_preds = []
    all_labels = []

    for batch in dataloader:
        features, labels, _ = batch

        # Move to device
        features = {k: v.to(device) for k, v in features.items()}
        labels = labels.to(device)

        # Create mask for available modalities
        batch_size = labels.size(0)
        mask = torch.zeros(batch_size, total_modalities, device=device)
        for idx in available_indices:
            mask[:, idx] = 1

        # Zero out unavailable modalities
        modality_names = list(features.keys())
        for i, modality in enumerate(modality_names):
            if i not in available_indices:
                features[modality] = torch.zeros_like(features[modality])

        # Forward pass
        logits = model(features, mask)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    accuracy = (all_preds == all_labels).float().mean().item()

    from sklearn.metrics import f1_score
    f1_macro = f1_score(
        all_labels.numpy(),
        all_preds.numpy(),
        average="macro",
        zero_division=0,
    )

    return {"accuracy": float(accuracy), "f1_macro": float(f1_macro)}


def _compute_modality_importance(results, modality_names):
    """
    Compute relative importance of each modality.

    Importance = average performance with modality - average without modality
    """
    importance = {}

    for modality in modality_names:
        with_scores = []
        without_scores = []

        for combo_name, metrics in results["all_combinations"].items():
            if modality in combo_name:
                with_scores.append(metrics["accuracy"])
            else:
                without_scores.append(metrics["accuracy"])

        if with_scores and without_scores:
            importance[modality] = float(np.mean(with_scores) - np.mean(without_scores))
        else:
            importance[modality] = 0.0

    # Normalize to [0, 1]
    total = sum(abs(v) for v in importance.values())
    if total > 0:
        importance = {k: v / total for k, v in importance.items()}

    return importance


def evaluate_missing_modalities(
    model,
    dataloader,
    modality_names,
    device="cpu",
):
    """
    Test model robustness to missing modalities.

    Tests all possible subsets of modalities (2^M - 1 combinations).
    """
    model.eval()
    model.to(device)

    num_modalities = len(modality_names)
    results = {
        "full_modalities": {},
        "single_modalities": {},
        "all_combinations": {},
    }

    print("\nTesting missing modality robustness...")

    for num_available in range(1, num_modalities + 1):
        print(f"\n{num_available}/{num_modalities} modalities available:")

        for modality_indices in itertools.combinations(range(num_modalities), num_available):
            modality_subset = [modality_names[i] for i in modality_indices]
            subset_name = "+".join(modality_subset)

            print(f"  Testing: {subset_name}")

            metrics = _evaluate_with_modality_subset(
                model, dataloader, modality_indices, num_modalities, device
            )
            results["all_combinations"][subset_name] = metrics

            if num_available == 1:
                results["single_modalities"][modality_subset[0]] = metrics

            if num_available == num_modalities:
                results["full_modalities"] = metrics

    results["modality_importance"] = _compute_modality_importance(results, modality_names)
    return results


def save_results_json(results, output_path):
    """Save results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multimodal fusion model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/base.yaml", help="Path to config file")
    parser.add_argument("--output_dir", type=str, default="experiments", help="Directory to save results")
    parser.add_argument("--missing_modality_test", action="store_true", help="Run missing modality robustness test")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run on")
    parser.add_argument(
        "--no_measure_inference",
        action="store_true",
        help="Disable inference latency measurement",
    )

    args = parser.parse_args()
    measure_inference = not args.no_measure_inference

    # Load model
    print(f"Loading model from: {args.checkpoint}")
    model = MultimodalFusionModule.load_from_checkpoint(args.checkpoint)
    model.eval()
    model.to(args.device)

    # Config from model
    config = model.config

    # Dataloaders
    print("Creating dataloaders...")
    _, _, test_loader = create_dataloaders(
        dataset_name=config.dataset.name,
        data_dir=config.dataset.data_dir,
        modalities=config.dataset.modalities,
        batch_size=config.dataset.batch_size,
        num_workers=config.dataset.num_workers,
    )

    # Standard evaluation
    print("\n" + "=" * 80)
    print("Standard Evaluation")
    print("=" * 80)

    metrics, (preds, labels, confidences) = evaluate_model(
        model,
        test_loader,
        device=args.device,
        return_predictions=True,
        measure_inference=measure_inference,
    )

    print(f"\nTest Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
    print(f"Test Loss: {metrics['loss']:.4f}")
    if "inference_ms" in metrics:
        print(f"Inference (ms/sample): {metrics['inference_ms']:.3f}")

    # Calibration metrics (ECE)
    print("\nComputing calibration metrics...")
    ece = CalibrationMetrics.expected_calibration_error(confidences, preds, labels)
    print(f"ECE: {ece:.4f}")

    # Save standard results
    standard_results = {
        "dataset": config.dataset.name,
        "fusion_type": config.model.fusion_type,
        "test_accuracy": metrics["accuracy"],
        "test_f1_macro": metrics["f1_macro"],
        "test_loss": metrics["loss"],
        "ece": float(ece),
    }
    if "inference_ms" in metrics:
        standard_results["inference_ms"] = metrics["inference_ms"]

    # Missing modality test
    if args.missing_modality_test:
        print("\n" + "=" * 80)
        print("Missing Modality Robustness Test")
        print("=" * 80)

        missing_results = evaluate_missing_modalities(
            model, test_loader, config.dataset.modalities, args.device
        )

        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(f"\nFull modalities: {missing_results['full_modalities']['accuracy']:.4f}")
        print("\nSingle modality performance:")
        for modality, m in missing_results["single_modalities"].items():
            print(f"  {modality}: {m['accuracy']:.4f}")

        print("\nModality importance scores:")
        for modality, score in missing_results["modality_importance"].items():
            print(f"  {modality}: {score:.4f}")

        # Save missing modality results
        output_path = Path(args.output_dir) / "missing_modality.json"
        save_results_json(missing_results, output_path)

    # Save all results
    output_path = Path(args.output_dir) / "evaluation_results.json"
    save_results_json(standard_results, output_path)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
