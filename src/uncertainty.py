"""
Uncertainty Quantification for Multimodal Fusion

Implements methods for estimating and calibrating confidence scores:
1. MC Dropout for epistemic uncertainty
2. Calibration metrics (ECE, reliability diagrams)
3. Uncertainty-weighted fusion
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict


class MCDropoutUncertainty(nn.Module):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    Runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty via variance.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 10):
        """
        Args:
            model: The model to estimate uncertainty for
            num_samples: Number of MC dropout samples
        """
        super().__init__()
        self.model = model
        self.num_samples = num_samples
    
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.
        
        Returns:
            mean_logits: (batch_size, num_classes) - mean prediction
            uncertainty: (batch_size,) - prediction uncertainty (variance)
        """
        # TODO: Implement MC Dropout
        # Steps:
        #   1. Enable dropout in model (model.train())
        #   2. Run num_samples forward passes
        #   3. Compute mean and variance of predictions
        #   4. Return mean prediction and uncertainty

        # Enable dropout sampling
        prev_mode = self.model.training
        self.model.train()
        logits_list = []
        probs_list = []

        with torch.no_grad():
            for _ in range(self.num_samples):
                logits = self.model(*args, **kwargs)  # (B, C)
                probs = F.softmax(logits, dim=1)
                logits_list.append(logits)
                probs_list.append(probs)

        # Restore original mode
        self.model.train(prev_mode)

        # Stack along sample dimension
        logits_s = torch.stack(logits_list, dim=0)  # (S, B, C)
        probs_s = torch.stack(probs_list, dim=0)    # (S, B, C)

        mean_logits = logits_s.mean(dim=0)          # (B, C)
        # Epistemic uncertainty proxy: variance across MC samples (avg over classes)
        uncertainty = probs_s.var(dim=0).mean(dim=1)  # (B,)

        return mean_logits, uncertainty
        raise NotImplementedError("Implement MC Dropout uncertainty")


class CalibrationMetrics:
    """
    Compute calibration metrics for confidence estimates.
    
    Key metrics:
    - Expected Calibration Error (ECE)
    - Maximum Calibration Error (MCE)  
    - Negative Log-Likelihood (NLL)
    """
    
    
    @staticmethod
    def _bin_stats(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int
    ):
        # Ensure cpu tensors
        conf = confidences.detach().float().cpu()
        preds = predictions.detach().long().cpu()
        labs = labels.detach().long().cpu()

        bin_edges = torch.linspace(0, 1, steps=num_bins + 1)
        bin_lowers = bin_edges[:-1]
        bin_uppers = bin_edges[1:]
        total = conf.numel()

        ece = 0.0
        mce = 0.0
        acc_per_bin = []
        conf_per_bin = []
        frac_per_bin = []

        for lower, upper in zip(bin_lowers, bin_uppers):
            # Include right edge on the last bin
            if upper == 1.0:
                in_bin = (conf >= lower) & (conf <= upper)
            else:
                in_bin = (conf >= lower) & (conf < upper)

            count = in_bin.sum().item()
            if count > 0:
                bin_conf = conf[in_bin].mean().item()
                bin_acc = (preds[in_bin] == labs[in_bin]).float().mean().item()
                gap = abs(bin_acc - bin_conf)
                w = count / total

                ece += gap * w
                mce = max(mce, gap)

                acc_per_bin.append(bin_acc)
                conf_per_bin.append(bin_conf)
                frac_per_bin.append(w)
            else:
                acc_per_bin.append(0.0)
                conf_per_bin.append(0.0)
                frac_per_bin.append(0.0)

        return ece, mce, np.array(acc_per_bin), np.array(conf_per_bin), np.array(frac_per_bin)

    @staticmethod
    def expected_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE = Σ (|bin_accuracy - bin_confidence|) * (bin_size / total_size)
        
        Args:
            confidences: (N,) - predicted confidence scores [0, 1]
            predictions: (N,) - predicted class indices
            labels: (N,) - ground truth class indices
            num_bins: Number of bins for calibration
            
        Returns:
            ece: Expected Calibration Error (lower is better)
        """
        # TODO: Implement ECE calculation
        # Steps:
        #   1. Bin predictions by confidence level
        #   2. For each bin, compute accuracy and average confidence
        #   3. Compute weighted difference |accuracy - confidence|
        #   4. Return ECE
        
        # Hint: Use np.histogram or torch.histc to bin confidences
        ece, _, _, _, _ = CalibrationMetrics._bin_stats(confidences, predictions, labels, num_bins)
        return float(ece)
        raise NotImplementedError("Implement ECE calculation")
    
    @staticmethod
    def maximum_calibration_error(
        confidences: torch.Tensor,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        num_bins: int = 15
    ) -> float:
        """
        Compute Maximum Calibration Error (MCE).
        
        MCE = max_bin |bin_accuracy - bin_confidence|
        
        Returns:
            mce: Maximum calibration error across bins
        """
        # TODO: Implement MCE
        # Similar to ECE but take max instead of average
        
        _, mce, _, _, _ = CalibrationMetrics._bin_stats(confidences, predictions, labels, num_bins)
        return float(mce)
        raise NotImplementedError("Implement MCE calculation")
    
    @staticmethod
    def negative_log_likelihood(
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> float:
        """
        Compute average Negative Log-Likelihood (NLL).
        
        NLL = -log P(y_true | x)
        
        Args:
            logits: (N, num_classes) - predicted logits
            labels: (N,) - ground truth labels
            
        Returns:
            nll: Average negative log-likelihood
        """
        # TODO: Implement NLL
        # Hint: Use F.cross_entropy which computes -log(softmax(logits)[label])
        
        nll = F.cross_entropy(logits, labels, reduction='mean')
        return float(nll.item())
        raise NotImplementedError("Implement NLL calculation")
    
    @staticmethod
    def reliability_diagram(
        confidences: np.ndarray,
        predictions: np.ndarray,
        labels: np.ndarray,
        num_bins: int = 15,
        save_path: str = None
    ) -> None:
        """
        Plot reliability diagram showing calibration.
        
        X-axis: Predicted confidence
        Y-axis: Actual accuracy
        Perfect calibration: y = x (diagonal line)
        
        Args:
            confidences: (N,) - confidence scores
            predictions: (N,) - predicted classes
            labels: (N,) - ground truth
            num_bins: Number of bins
            save_path: Optional path to save plot
        """
        import matplotlib.pyplot as plt
        
        # TODO: Implement reliability diagram
        # Steps:
        #   1. Bin predictions by confidence
        #   2. Compute accuracy per bin
        #   3. Plot bar chart: confidence vs accuracy
        #   4. Add diagonal line for perfect calibration
        #   5. Add ECE to plot

        confidences_t = torch.from_numpy(confidences)
        predictions_t = torch.from_numpy(predictions)
        labels_t = torch.from_numpy(labels)

        ece, _, acc_bins, conf_bins, _ = CalibrationMetrics._bin_stats(
            confidences_t, predictions_t, labels_t, num_bins
        )

        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        width = 1.0 / num_bins

        plt.figure(figsize=(6, 5))
        # Bars: accuracy per bin
        plt.bar(bin_centers, acc_bins, width=width, edgecolor='black', alpha=0.7, label='Accuracy')
        # Line: mean confidence per bin
        plt.plot(bin_centers, conf_bins, marker='o', linestyle='--', label='Confidence')
        # Diagonal: perfect calibration
        plt.plot([0, 1], [0, 1], 'k-', linewidth=1, label='Perfect')

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xlabel('Predicted confidence')
        plt.ylabel('Empirical accuracy')
        plt.title(f'Reliability Diagram (ECE={ece:.4f})')
        plt.legend(loc='best')
        plt.tight_layout()

        if save_path is not None:
            plt.savefig(save_path, dpi=200, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
        return
        raise NotImplementedError("Implement reliability diagram")


class UncertaintyWeightedFusion(nn.Module):
    """
    Fuse modalities weighted by inverse uncertainty.
    
    Intuition: More uncertain modalities get lower weight.
    Weight_i ∝ 1 / (uncertainty_i + ε)
    """
    
    def __init__(self, epsilon: float = 1e-6):
        """
        Args:
            epsilon: Small constant to avoid division by zero
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(
        self,
        modality_predictions: Dict[str, torch.Tensor],
        modality_uncertainties: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse modality predictions weighted by inverse uncertainty.
        
        Args:
            modality_predictions: Dict of {modality: logits}
                                Each tensor: (batch_size, num_classes)
            modality_uncertainties: Dict of {modality: uncertainty}
                                   Each tensor: (batch_size,)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            fused_logits: (batch_size, num_classes) - weighted fusion
            fusion_weights: (batch_size, num_modalities) - used weights
        """
        # TODO: Implement uncertainty-weighted fusion
        # Steps:
        #   1. Compute inverse uncertainty weights: w_i = 1/(σ_i + ε)
        #   2. Normalize weights to sum to 1
        #   3. Apply modality mask (zero weight for missing modalities)
        #   4. Fuse predictions: Σ w_i * pred_i
        #   5. Return fused predictions and weights
        
        names = list(modality_predictions.keys())
        B = next(iter(modality_predictions.values())).size(0)
        C = next(iter(modality_predictions.values())).size(1)

        # Stack logits (B, M, C) and uncertainties (B, M)
        logits_stack = torch.stack([modality_predictions[n] for n in names], dim=1)
        sigma_stack = torch.stack([modality_uncertainties[n] for n in names], dim=1)

        # Inverse-uncertainty weights with mask
        w_raw = 1.0 / (sigma_stack + self.epsilon)  # (B, M)
        w_raw = w_raw * modality_mask.to(w_raw.dtype)

        denom = w_raw.sum(dim=1, keepdim=True).clamp_min(1e-9)
        weights = w_raw / denom  # (B, M)

        fused_logits = (weights.unsqueeze(-1) * logits_stack).sum(dim=1)  # (B, C)
        return fused_logits, weights
        raise NotImplementedError("Implement uncertainty-weighted fusion")


class TemperatureScaling(nn.Module):
    """
    Post-hoc calibration via temperature scaling.
    
    Learns a single temperature parameter T that scales logits:
    P_calibrated = softmax(logits / T)
    
    Reference: Guo et al. "On Calibration of Modern Neural Networks", ICML 2017
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: (batch_size, num_classes) - model outputs
            
        Returns:
            scaled_logits: (batch_size, num_classes) - temperature-scaled logits
        """
        return logits / self.temperature
    
    def calibrate(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lr: float = 0.01,
        max_iter: int = 50
    ) -> None:
        """
        Learn optimal temperature on validation set.
        
        Args:
            logits: (N, num_classes) - validation set logits
            labels: (N,) - validation set labels
            lr: Learning rate
            max_iter: Maximum optimization iterations
        """
        # TODO: Implement temperature calibration
        # Steps:
        #   1. Initialize temperature = 1.0
        #   2. Optimize temperature to minimize NLL on validation set
        #   3. Use LBFGS or Adam optimizer

        device = logits.device
        self.to(device)
        self.temperature.data.fill_(1.0)

        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

        def _nll_with_temp():
            scaled = self.forward(logits)
            return F.cross_entropy(scaled, labels)

        def closure():
            optimizer.zero_grad(set_to_none=True)
            loss = _nll_with_temp()
            loss.backward()
            return loss

        optimizer.step(closure)
        # Safety clamp
        with torch.no_grad():
            self.temperature.data = self.temperature.data.clamp_min(1e-6)
        return
        raise NotImplementedError("Implement temperature calibration")


class EnsembleUncertainty:
    """
    Estimate uncertainty via ensemble of models.
    
    Train multiple models with different initializations/data splits.
    Uncertainty = variance across ensemble predictions.
    """
    
    def __init__(self, models: list):
        """
        Args:
            models: List of trained models (same architecture)
        """
        self.models = models
        self.num_models = len(models)
    
    def predict_with_uncertainty(
        self,
        inputs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get predictions and uncertainty from ensemble.
        
        Args:
            inputs: Model inputs
            
        Returns:
            mean_predictions: (batch_size, num_classes) - average prediction
            uncertainty: (batch_size,) - prediction variance
        """
        # TODO: Implement ensemble prediction
        # Steps:
        #   1. Get predictions from all models
        #   2. Compute mean prediction
        #   3. Compute variance as uncertainty measure
        #   4. Return mean and uncertainty
        
        probs_list = []
        with torch.no_grad():
            for m in self.models:
                m.eval()
                logits = m(inputs)
                probs = F.softmax(logits, dim=1)
                probs_list.append(probs)

        probs_s = torch.stack(probs_list, dim=0)  # (M, B, C)
        mean_probs = probs_s.mean(dim=0)          # (B, C)
        uncertainty = probs_s.var(dim=0).mean(dim=1)  # (B,)
        return mean_probs, uncertainty
        raise NotImplementedError("Implement ensemble uncertainty")


def compute_calibration_metrics(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Compute all calibration metrics on a dataset.
    
    Args:
        model: Trained model
        dataloader: Test/validation dataloader
        device: Device to run on
        
    Returns:
        metrics: Dict with ECE, MCE, NLL, accuracy
    """
    model.eval()
    all_confidences = []
    all_predictions = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            all_logits.append(logits.cpu())
            all_confidences.append(confidences.cpu())
            all_predictions.append(predictions.cpu())
            all_labels.append(labels.cpu())
    
    logits_cat = torch.cat(all_logits)
    confidences = torch.cat(all_confidences)
    predictions = torch.cat(all_predictions)
    labels = torch.cat(all_labels)
    
    # TODO: Compute and return all metrics
    # - ECE
    # - MCE
    # - NLL
    # - Accuracy

    ece = CalibrationMetrics.expected_calibration_error(confidences, predictions, labels)
    mce = CalibrationMetrics.maximum_calibration_error(confidences, predictions, labels)
    nll = CalibrationMetrics.negative_log_likelihood(logits_cat, labels)
    accuracy = (predictions == labels).float().mean().item()

    return {
        'ECE': float(ece),
        'MCE': float(mce),
        'NLL': float(nll),
        'Accuracy': float(accuracy),
    }
    raise NotImplementedError("Implement calibration metrics computation")


if __name__ == '__main__':
    # Test calibration metrics
    print("Testing calibration metrics...")
    
    # Generate fake predictions
    num_samples = 1000
    num_classes = 10
    
    # Well-calibrated predictions
    logits = torch.randn(num_samples, num_classes)
    labels = torch.randint(0, num_classes, (num_samples,))
    probs = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    
    # Test ECE
    try:
        ece = CalibrationMetrics.expected_calibration_error(
            confidences, predictions, labels
        )
        print(f"✓ ECE computed: {ece:.4f}")
    except NotImplementedError:
        print("✗ ECE not implemented yet")
    
    # Test reliability diagram
    try:
        CalibrationMetrics.reliability_diagram(
            confidences.numpy(),
            predictions.numpy(),
            labels.numpy(),
            save_path='test_reliability.png'
        )
        print("✓ Reliability diagram created")
    except NotImplementedError:
        print("✗ Reliability diagram not implemented yet")

