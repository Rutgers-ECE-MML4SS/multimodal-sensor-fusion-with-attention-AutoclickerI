"""
Attention Mechanisms for Multimodal Fusion

Implements:
1. CrossModalAttention: Attention between different modalities
2. TemporalAttention: Attention over time steps in sequences
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import math

def _to_sequence(x: torch.Tensor) -> Tuple[torch.Tensor, bool]:
    """
    Ensure tensor has shape (B, L, D). If input is (B, D), add L=1 and remember to squeeze later.
    Returns: (x3d, squeezed_flag)
    """
    if x.dim() == 2:
        return x.unsqueeze(1), True
    elif x.dim() == 3:
        return x, False
    else:
        raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

def _split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    (B, L, H*Dh) -> (B, Heads, L, Dh)
    """
    B, L, Hx = x.shape
    assert Hx % num_heads == 0, "Hidden dim must be divisible by num_heads"
    Dh = Hx // num_heads
    x = x.view(B, L, num_heads, Dh).permute(0, 2, 1, 3)  # (B, heads, L, Dh)
    return x


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    """
    (B, Heads, L, Dh) -> (B, L, Heads*Dh)
    """
    B, H, L, Dh = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * Dh)

class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: Modality A attends to Modality B.
    
    Example: Video features attend to IMU features to incorporate
    relevant motion information at each timestep.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            query_dim: Dimension of query modality features
            key_dim: Dimension of key/value modality features  
            hidden_dim: Hidden dimension for attention computation
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # TODO: Implement multi-head attention projections
        # Hint: Use nn.Linear for Q, K, V projections
        # Query from modality A, Key and Value from modality B
                # Multi-head projections
        self.q_proj = nn.Linear(query_dim, hidden_dim)
        self.k_proj = nn.Linear(key_dim, hidden_dim)
        self.v_proj = nn.Linear(key_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        return
        raise NotImplementedError("Implement cross-modal attention projections")
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            query: (batch_size, query_dim) - features from modality A
            key: (batch_size, key_dim) - features from modality B
            value: (batch_size, key_dim) - features from modality B
            mask: Optional (batch_size,) - binary mask for valid keys
            
        Returns:
            attended_features: (batch_size, hidden_dim) - query attended by key/value
            attention_weights: (batch_size, num_heads, 1, 1) - attention scores
        """
        B = query.size(0)
        
        # TODO: Implement multi-head attention computation
        # Steps:
        #   1. Project query, key, value to (batch, num_heads, seq_len, head_dim)
        #   2. Compute attention scores: Q @ K^T / sqrt(head_dim)
        #   3. Apply mask if provided (set masked positions to -inf before softmax)
        #   4. Apply softmax to get attention weights
        #   5. Apply attention to values: attn_weights @ V
        #   6. Reshape and project back to hidden_dim
        # Ensure 3D
        q, squeeze_q = _to_sequence(query)
        k, _ = _to_sequence(key)
        v, _ = _to_sequence(value)

        # Project and split heads
        Q = _split_heads(self.q_proj(q), self.num_heads)  # (B, H, Lq, Dh)
        K = _split_heads(self.k_proj(k), self.num_heads)  # (B, H, Lk, Dh)
        V = _split_heads(self.v_proj(v), self.num_heads)  # (B, H, Lk, Dh)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,Lq,Lk)

        # Apply key mask if provided
        if mask is not None:
            if mask.dim() == 1:
                key_mask = mask.view(B, 1, 1, 1)  # single key case
            elif mask.dim() == 2:
                key_mask = mask.view(B, 1, 1, -1)  # (B,1,1,Lk)
            else:
                raise ValueError("mask must be (B,) or (B, Lk)")
            scores = scores.masked_fill(~key_mask.bool(), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        # guard all-masked rows -> NaN
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.attn_drop(attn)

        context = torch.matmul(attn, V)  # (B,H,Lq,Dh)
        context = _merge_heads(context)  # (B,Lq,H*Dh=hidden_dim)
        out = self.out_proj(context)
        out = self.out_drop(out)

        if squeeze_q:
            out = out.squeeze(1)  # (B, hidden_dim)

        return out, attn
        raise NotImplementedError("Implement cross-modal attention forward pass")


class TemporalAttention(nn.Module):
    """
    Temporal attention: Attend over sequence of time steps.
    
    Useful for: Variable-length sequences, weighting important timesteps
    """
    
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            feature_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        
        # TODO: Implement self-attention over temporal dimension
        # Hint: Similar to CrossModalAttention but Q, K, V from same modality

        self.q_proj = nn.Linear(feature_dim, hidden_dim)
        self.k_proj = nn.Linear(feature_dim, hidden_dim)
        self.v_proj = nn.Linear(feature_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        return
        raise NotImplementedError("Implement temporal attention")
    
    def forward(
        self,
        sequence: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for temporal attention.
        
        Args:
            sequence: (batch_size, seq_len, feature_dim) - temporal sequence
            mask: Optional (batch_size, seq_len) - binary mask for valid timesteps
            
        Returns:
            attended_sequence: (batch_size, seq_len, hidden_dim) - attended features
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
        """
        # TODO: Implement temporal self-attention
        # Steps:
        #   1. Project sequence to Q, K, V
        #   2. Compute self-attention over sequence length
        #   3. Apply mask for variable-length sequences
        #   4. Return attended sequence and weights
        
        if sequence.dim() != 3:
            raise ValueError("sequence must be (B, T, D)")

        B, T, _ = sequence.shape

        Q = _split_heads(self.q_proj(sequence), self.num_heads)  # (B,H,T,Dh)
        K = _split_heads(self.k_proj(sequence), self.num_heads)  # (B,H,T,Dh)
        V = _split_heads(self.v_proj(sequence), self.num_heads)  # (B,H,T,Dh)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B,H,T,T)

        if mask is not None:
            if mask.shape != (B, T):
                raise ValueError(f"mask must be (B, T); got {tuple(mask.shape)}")
            key_mask = mask.view(B, 1, 1, T)  # broadcast over heads & queries
            scores = scores.masked_fill(~key_mask.bool(), float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, posinf=0.0, neginf=0.0)
        attn = self.attn_drop(attn)

        context = torch.matmul(attn, V)  # (B,H,T,Dh)
        context = _merge_heads(context)  # (B,T,H*Dh)
        out = self.out_proj(context)
        out = self.out_drop(out)

        return out, attn
        raise NotImplementedError("Implement temporal attention forward pass")
    
    def pool_sequence(
        self,
        sequence: torch.Tensor,
        attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Pool sequence to fixed-size representation using attention weights.
        
        Args:
            sequence: (batch_size, seq_len, hidden_dim)
            attention_weights: (batch_size, num_heads, seq_len, seq_len)
            
        Returns:
            pooled: (batch_size, hidden_dim) - fixed-size representation
        """
        # TODO: Implement attention-based pooling
        # Option 1: Weighted average using mean attention weights
        # Option 2: Learn pooling query vector
        # Option 3: Take output at special [CLS] token position

        if sequence.dim() != 3:
            raise ValueError("sequence must be (B, T, H)")
        if attention_weights.dim() != 4:
            raise ValueError("attention_weights must be (B, Heads, T, T)")

        B, T, Hdim = sequence.shape
        # Mean over heads and query positions -> importance over key positions
        # shape: (B, T) then normalize
        importance = attention_weights.mean(dim=1).mean(dim=1)  # (B, T)
        norm = importance.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        weights = importance / norm  # (B, T)

        pooled = (weights.unsqueeze(-1) * sequence).sum(dim=1)  # (B, Hdim)
        return pooled
        raise NotImplementedError("Implement attention-based pooling")


class PairwiseModalityAttention(nn.Module):
    """
    Pairwise attention between all modality combinations.
    
    For M modalities, computes M*(M-1)/2 pairwise attention operations.
    Example: {video, audio, IMU} -> {video<->audio, video<->IMU, audio<->IMU}
    """
    
    def __init__(
        self,
        modality_dims: dict,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Args:
            modality_dims: Dict mapping modality names to feature dimensions
                          Example: {'video': 512, 'audio': 128, 'imu': 64}
            hidden_dim: Hidden dimension for attention
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()
        self.modality_names = list(modality_dims.keys())
        self.num_modalities = len(self.modality_names)
        self.hidden_dim = hidden_dim
        
        # TODO: Create CrossModalAttention for each modality pair
        # Hint: Use nn.ModuleDict with keys like "video_to_audio"
        # For each pair (A, B), create attention A->B and B->A
        # Cross attention modules for each ordered pair A->B (A != B)
        self.attn = nn.ModuleDict()
        for i, a in enumerate(self.modality_names):
            for j, b in enumerate(self.modality_names):
                if i == j:
                    continue
                self.attn[f"{a}_to_{b}"] = CrossModalAttention(
                    query_dim=modality_dims[a],
                    key_dim=modality_dims[b],
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )

        # Per-modality input projection & normalization
        self.in_proj = nn.ModuleDict({
            m: nn.Linear(modality_dims[m], hidden_dim) for m in self.modality_names
        })
        self.norm = nn.ModuleDict({
            m: nn.LayerNorm(hidden_dim) for m in self.modality_names
        })
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        return
        raise NotImplementedError("Implement pairwise modality attention")
    
    def forward(
        self,
        modality_features: dict,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[dict, dict]:
        """
        Apply pairwise attention between all modalities.
        
        Args:
            modality_features: Dict of {modality_name: features}
                             Each tensor: (batch_size, feature_dim)
            modality_mask: (batch_size, num_modalities) - availability mask
            
        Returns:
            attended_features: Dict of {modality_name: attended_features}
            attention_maps: Dict of {f"{mod_a}_to_{mod_b}": attention_weights}
        """
        # TODO: Implement pairwise attention
        # Steps:
        #   1. For each modality pair (A, B):
        #      - Apply attention A->B (A attends to B)
        #      - Apply attention B->A (B attends to A)
        #   2. Aggregate attended features (options: sum, concat, gating)
        #   3. Handle missing modalities using mask
        #   4. Return attended features and attention maps for visualization
        
        B = next(iter(modality_features.values())).size(0)
        name_to_idx = {m: i for i, m in enumerate(self.modality_names)}

        # Collect pairwise attentions
        pair_outputs = {}
        attention_maps = {}

        for a in self.modality_names:
            for b in self.modality_names:
                if a == b:
                    continue
                key_mask = None
                if modality_mask is not None:
                    key_mask = modality_mask[:, name_to_idx[b]]  # (B,)

                out_ab, w_ab = self.attn[f"{a}_to_{b}"](
                    modality_features[a], modality_features[b], modality_features[b], mask=key_mask
                )
                pair_outputs[(a, b)] = out_ab  # (B, hidden_dim) if queries are vectors
                attention_maps[f"{a}_to_{b}"] = w_ab

        # Aggregate per modality: projected self + mean of attended others
        attended_features = {}
        for a in self.modality_names:
            base = self.in_proj[a](modality_features[a])  # (B, hidden_dim)

            others = [pair_outputs[(a, b)] for b in self.modality_names if b != a]
            if len(others) > 0:
                agg = torch.stack(others, dim=0).mean(dim=0)  # (B, hidden_dim)
            else:
                agg = torch.zeros_like(base)

            combined = base + agg
            combined = self.norm[a](combined)
            combined = self.act(combined)
            combined = self.drop(combined)

            # If the query modality 'a' is missing for a sample, zero-out its output
            if modality_mask is not None:
                qmask = modality_mask[:, name_to_idx[a]].view(B, 1).float()
                combined = combined * qmask

            attended_features[a] = combined

        return attended_features, attention_maps
        raise NotImplementedError("Implement pairwise attention forward pass")


def visualize_attention(
    attention_weights: torch.Tensor,
    modality_names: list,
    save_path: str = None
) -> None:
    """
    Visualize attention weights between modalities.
    
    Args:
        attention_weights: (num_heads, num_queries, num_keys) or similar
        modality_names: List of modality names for labeling
        save_path: Optional path to save visualization
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # TODO: Implement attention visualization
    # Create heatmap showing which modalities attend to which
    # Useful for understanding fusion behavior
    import matplotlib.pyplot as plt
    import numpy as np

    w = attention_weights
    if w.dim() == 4:
        # average over batch
        w = w.mean(dim=0)  # (H, Q, K)
    if w.dim() != 3:
        raise ValueError("attention_weights must be (H,Q,K) or (B,H,Q,K)")

    heat = w.mean(dim=0)  # mean over heads -> (Q, K)
    heat = heat.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(heat, aspect='auto')
    ax.set_xlabel("Keys")
    ax.set_ylabel("Queries")
    if len(modality_names) == heat.shape[0]:
        ax.set_yticks(np.arange(len(modality_names)))
        ax.set_yticklabels(modality_names)
    if len(modality_names) == heat.shape[1]:
        ax.set_xticks(np.arange(len(modality_names)))
        ax.set_xticklabels(modality_names, rotation=45, ha='right')

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()
    return
    raise NotImplementedError("Implement attention visualization")


if __name__ == '__main__':
    # Simple test
    print("Testing attention mechanisms...")
    
    batch_size = 4
    query_dim = 512  # e.g., video features
    key_dim = 64     # e.g., IMU features
    hidden_dim = 256
    num_heads = 4
    
    # Test CrossModalAttention
    print("\nTesting CrossModalAttention...")
    try:
        attn = CrossModalAttention(query_dim, key_dim, hidden_dim, num_heads)
        
        query = torch.randn(batch_size, query_dim)
        key = torch.randn(batch_size, key_dim)
        value = torch.randn(batch_size, key_dim)
        
        attended, weights = attn(query, key, value)
        
        assert attended.shape == (batch_size, hidden_dim)
        print(f"✓ CrossModalAttention working! Output shape: {attended.shape}")
        
    except NotImplementedError:
        print("✗ CrossModalAttention not implemented yet")
    except Exception as e:
        print(f"✗ CrossModalAttention error: {e}")
    
    # Test TemporalAttention
    print("\nTesting TemporalAttention...")
    try:
        seq_len = 10
        feature_dim = 128
        
        temporal_attn = TemporalAttention(feature_dim, hidden_dim, num_heads)
        sequence = torch.randn(batch_size, seq_len, feature_dim)
        
        attended_seq, weights = temporal_attn(sequence)
        
        assert attended_seq.shape == (batch_size, seq_len, hidden_dim)
        print(f"✓ TemporalAttention working! Output shape: {attended_seq.shape}")
        
    except NotImplementedError:
        print("✗ TemporalAttention not implemented yet")
    except Exception as e:
        print(f"✗ TemporalAttention error: {e}")

