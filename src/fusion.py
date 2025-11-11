"""
Multimodal Fusion Architectures for Sensor Integration

This module implements three fusion strategies:
1. Early Fusion: Concatenate features before processing
2. Late Fusion: Independent processing, combine predictions
3. Hybrid Fusion: Cross-modal attention + learned weighting
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from attention import CrossModalAttention

def _stack_by_names(
    features: Dict[str, torch.Tensor],
    names: list
) -> torch.Tensor:
    """Return stacked tensor in name order -> (B, M, D_i?) not used for concat but a helper if needed."""
    return torch.stack([features[n] for n in names], dim=1)

class EarlyFusion(nn.Module):
    """
    Early fusion: Concatenate encoder outputs and process jointly.
    
    Pros: Joint representation learning across modalities
    Cons: Requires temporal alignment, sensitive to missing modalities
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
                          Example: {'video': 512, 'imu': 64}
            hidden_dim: Hidden dimension for fusion network
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        
        # TODO: Implement early fusion architecture
        # Hint: Concatenate all modality features, pass through MLP
        # Architecture suggestion:
        #   concat_dim = sum(modality_dims.values())
        #   Linear(concat_dim, hidden_dim) -> ReLU -> Dropout
        #   Linear(hidden_dim, hidden_dim) -> ReLU -> Dropout
        #   Linear(hidden_dim, num_classes)

        concat_dim = sum(modality_dims.values())
        self.net = nn.Sequential(
            nn.Linear(concat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        return
        raise NotImplementedError("Implement early fusion architecture")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with early fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor shape: (batch_size, feature_dim)
            modality_mask: Binary mask (batch_size, num_modalities)
                          1 = available, 0 = missing
                          
        Returns:
            logits: (batch_size, num_classes)
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Extract features for each modality from dict
        #   2. Handle missing modalities (use zeros or learned embeddings)
        #   3. Concatenate all features
        #   4. Pass through fusion network
        
        B = next(iter(modality_features.values())).size(0)

        feats = []
        for i, name in enumerate(self.modality_names):
            f = modality_features[name]  # (B, Di)
            if modality_mask is not None:
                m = modality_mask[:, i].view(B, 1).to(f.dtype)  # (B,1)
                f = f * m  # zero out missing modality
            feats.append(f)
        x = torch.cat(feats, dim=1)  # (B, sum Di)
        logits = self.net(x)
        return logits
        raise NotImplementedError("Implement early fusion forward pass")


class LateFusion(nn.Module):
    """
    Late fusion: Independent classifiers per modality, combine predictions.
    
    Pros: Handles asynchronous sensors, modular per-modality training
    Cons: Limited cross-modal interaction, fusion only at decision level
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for per-modality classifiers
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        
        # TODO: Create separate classifier for each modality
        # Hint: Use nn.ModuleDict to store per-modality classifiers
        # Each classifier: Linear(modality_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes)

        self.classifiers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(modality_dims[name], hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            for name in self.modality_names
        })
        
        # TODO: Learn fusion weights (how to combine predictions)
        # Option 1: Learnable weights (nn.Parameter)
        # Option 2: Attention over predictions
        # Option 3: Simple averaging

        self.weight_scorers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(modality_dims[name], max(32, hidden_dim // 8)),
                nn.ReLU(inplace=True),
                nn.Linear(max(32, hidden_dim // 8), 1),
            )
            for name in self.modality_names
        })
        return
        raise NotImplementedError("Implement late fusion architecture")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with late fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            
        Returns:
            logits: (batch_size, num_classes) - fused predictions
            per_modality_logits: Dict of individual modality predictions
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Get predictions from each modality classifier
        #   2. Handle missing modalities (mask out or skip)
        #   3. Combine predictions using fusion weights
        #   4. Return both fused and per-modality predictions
        
        B = next(iter(modality_features.values())).size(0)
        if modality_mask is None:
            modality_mask = torch.ones(B, self.num_modalities, device=next(self.parameters()).device, dtype=torch.bool)

        # 1) Per-modality predictions and scores
        per_logits: Dict[str, torch.Tensor] = {}
        scores = []
        for i, name in enumerate(self.modality_names):
            x = modality_features[name]
            per_logits[name] = self.classifiers[name](x)  # (B,C)

            s = self.weight_scorers[name](x).squeeze(-1)  # (B,)
            # mask: if missing, set to -inf before softmax
            s = s.masked_fill(~modality_mask[:, i].bool(), float('-inf'))
            scores.append(s)

        scores = torch.stack(scores, dim=1)  # (B,M)
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)  # when all-masked in a row

        # 2) Combine predictions
        stacked = torch.stack([per_logits[name] for name in self.modality_names], dim=1)  # (B,M,C)
        fused = (weights.unsqueeze(-1) * stacked).sum(dim=1)  # (B,C)
        return fused, per_logits
        raise NotImplementedError("Implement late fusion forward pass")


class HybridFusion(nn.Module):
    """
    Hybrid fusion: Cross-modal attention + learned fusion weights.
    
    Pros: Rich cross-modal interaction, robust to missing modalities
    Cons: More complex, higher computation cost
    
    This is the main focus of the assignment!
    """
    
    def __init__(
        self,
        modality_dims: Dict[str, int],
        hidden_dim: int = 256,
        num_classes: int = 11,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dictionary mapping modality name to feature dimension
            hidden_dim: Hidden dimension for fusion
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # TODO: Project each modality to common hidden dimension
        # Hint: Use nn.ModuleDict with Linear layers per modality

        self.proj = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(modality_dims[name], hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            for name in self.modality_names
        })
        self.norm = nn.ModuleDict({name: nn.LayerNorm(hidden_dim) for name in self.modality_names})
        
        # TODO: Implement cross-modal attention
        # Use CrossModalAttention from attention.py
        # Each modality should attend to all other modalities
        
        self.attn = nn.ModuleDict()
        for a in self.modality_names:
            for b in self.modality_names:
                if a == b:
                    continue
                self.attn[f"{a}_to_{b}"] = CrossModalAttention(
                    query_dim=hidden_dim, key_dim=hidden_dim,
                    hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout
                )

        # TODO: Learn adaptive fusion weights based on modality availability
        # Hint: Small MLP that takes modality mask and outputs weights
        
        self.weight_scorers = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(hidden_dim, max(32, hidden_dim // 8)),
                nn.ReLU(inplace=True),
                nn.Linear(max(32, hidden_dim // 8), 1),
            )
            for name in self.modality_names
        })

        # TODO: Final classifier
        # Takes fused representation -> num_classes logits

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        return
        raise NotImplementedError("Implement hybrid fusion architecture")
    
    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        Forward pass with hybrid fusion.
        
        Args:
            modality_features: Dict of {modality_name: features}
            modality_mask: Binary mask for available modalities
            return_attention: If True, return attention weights for visualization
            
        Returns:
            logits: (batch_size, num_classes)
            attention_info: Optional dict with attention weights and fusion weights
        """
        # TODO: Implement forward pass
        # Steps:
        #   1. Project all modalities to common hidden dimension
        #   2. Apply cross-modal attention between modality pairs
        #   3. Compute adaptive fusion weights based on modality_mask
        #   4. Fuse attended representations with learned weights
        #   5. Pass through final classifier
        #   6. Optionally return attention weights for visualization
        
        B = next(iter(modality_features.values())).size(0)
        device = next(self.parameters()).device
        if modality_mask is None:
            modality_mask = torch.ones(B, self.num_modalities, device=device, dtype=torch.bool)

        # 1) Project to hidden space
        hidden = {name: self.proj[name](modality_features[name]) for name in self.modality_names}

        # 2) Cross-modal attention per modality (aggregate influences from others)
        attended = {}
        attn_maps = {}
        name_to_idx = {n: i for i, n in enumerate(self.modality_names)}
        for a in self.modality_names:
            others_ctx = []
            for b in self.modality_names:
                if a == b:
                    continue
                # key/value available?
                key_mask = modality_mask[:, name_to_idx[b]]  # (B,)
                ctx, w = self.attn[f"{a}_to_{b}"](
                    query=hidden[a], key=hidden[b], value=hidden[b], mask=key_mask
                )  # ctx: (B,H)
                others_ctx.append(ctx)
                attn_maps[f"{a}_to_{b}"] = w  # (B, heads, 1, 1)

            if len(others_ctx) > 0:
                agg = torch.stack(others_ctx, dim=0).mean(dim=0)  # (B,H)
            else:
                agg = torch.zeros_like(hidden[a])

            combined = self.norm[a](hidden[a] + agg)
            # If the query modality is missing, zero it out
            qmask = modality_mask[:, name_to_idx[a]].view(B, 1).to(combined.dtype)
            combined = combined * qmask
            attended[a] = combined

        # 3) Adaptive weights from attended features + mask
        weights = self.compute_adaptive_weights(attended, modality_mask)  # (B,M)

        # 4) Fuse representations
        Hstack = torch.stack([attended[n] for n in self.modality_names], dim=1)  # (B,M,H)
        fused = (weights.unsqueeze(-1) * Hstack).sum(dim=1)  # (B,H)

        # 5) Classify
        logits = self.classifier(fused)

        if return_attention:
            info = {
                "attention_maps": attn_maps,
                "fusion_weights": {n: weights[:, i] for i, n in enumerate(self.modality_names)},
                "fused_representation": fused,
            }
            return logits, info
        else:
            return logits, None
        raise NotImplementedError("Implement hybrid fusion forward pass")
    
    def compute_adaptive_weights(
        self,
        modality_features: Dict[str, torch.Tensor],
        modality_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute adaptive fusion weights based on modality availability.
        
        Args:
            modality_features: Dict of modality features
            modality_mask: (batch_size, num_modalities) binary mask
            
        Returns:
            weights: (batch_size, num_modalities) normalized fusion weights
        """
        # TODO: Implement adaptive weighting
        # Ideas:
        #   1. Learn weight predictor from modality features + mask
        #   2. Higher weights for more reliable/informative modalities
        #   3. Ensure weights sum to 1 (softmax) and respect mask
        
        B = modality_mask.size(0)
        scores = []
        for i, name in enumerate(self.modality_names):
            s = self.weight_scorers[name](modality_features[name]).squeeze(-1)  # (B,)
            s = s.masked_fill(~modality_mask[:, i].bool(), float('-inf'))
            scores.append(s)
        scores = torch.stack(scores, dim=1)  # (B,M)
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
        return weights
        raise NotImplementedError("Implement adaptive weight computation")


# Helper functions

def build_fusion_model(
    fusion_type: str,
    modality_dims: Dict[str, int],
    num_classes: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to build fusion models.
    
    Args:
        fusion_type: One of ['early', 'late', 'hybrid']
        modality_dims: Dictionary mapping modality names to dimensions
        num_classes: Number of output classes
        **kwargs: Additional arguments for fusion model
        
    Returns:
        Fusion model instance
    """
    fusion_classes = {
        'early': EarlyFusion,
        'late': LateFusion,
        'hybrid': HybridFusion,
    }
    
    if fusion_type != 'hybrid' and 'num_heads' in kwargs:
        del kwargs['num_heads']

    if fusion_type not in fusion_classes:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    return fusion_classes[fusion_type](
        modality_dims=modality_dims,
        num_classes=num_classes,
        **kwargs
    )


if __name__ == '__main__':
    # Simple test to verify implementation
    print("Testing fusion architectures...")
    
    # Test configuration
    modality_dims = {'video': 512, 'imu': 64}
    num_classes = 11
    batch_size = 4
    
    # Create dummy features
    features = {
        'video': torch.randn(batch_size, 512),
        'imu': torch.randn(batch_size, 64)
    }
    mask = torch.tensor([[1, 1], [1, 0], [0, 1], [1, 1]])  # Different availability patterns
    
    # Test each fusion type
    for fusion_type in ['early', 'late', 'hybrid']:
        print(f"\nTesting {fusion_type} fusion...")
        try:
            model = build_fusion_model(fusion_type, modality_dims, num_classes)
            
            if fusion_type == 'late':
                logits, per_mod_logits = model(features, mask)
            else:
                logits = model(features, mask)
            
            assert logits.shape == (batch_size, num_classes), \
                f"Expected shape ({batch_size}, {num_classes}), got {logits.shape}"
            print(f"✓ {fusion_type} fusion working! Output shape: {logits.shape}")
            
        except NotImplementedError:
            print(f"✗ {fusion_type} fusion not implemented yet")
        except Exception as e:
            print(f"✗ {fusion_type} fusion error: {e}")

