"""
Modality-Specific Encoders for Sensor Fusion

Implements lightweight encoders suitable for CPU training:
1. SequenceEncoder: For time-series data (IMU, audio, motion capture)
2. FrameEncoder: For frame-based data (video features)
3. SimpleMLPEncoder: For pre-extracted features
"""

import torch
import torch.nn as nn
from typing import Optional
import math

def masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor], dim: int) -> torch.Tensor:
    """
    x: (..., T, D) or (B, T, D)
    mask: (..., T) with 1 for valid, 0 for pad
    returns mean over dim with mask; if mask sum is 0, returns zeros
    """
    if mask is None:
        return x.mean(dim=dim)
    # expand mask to x shape for broadcast
    while mask.dim() < x.dim():
        mask = mask.unsqueeze(-1)
    mask = mask.to(x.dtype)
    s = (x * mask).sum(dim=dim)
    denom = mask.sum(dim=dim).clamp_min(1e-9)
    return s / denom

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class SequenceEncoder(nn.Module):
    """
    Encoder for sequential/time-series sensor data.
    
    Options: 1D CNN, LSTM, GRU, or Transformer
    Output: Fixed-size embedding per sequence
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        encoder_type: str = 'lstm',
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Dimension of input features at each timestep
            hidden_dim: Hidden dimension for RNN/Transformer
            output_dim: Output embedding dimension
            num_layers: Number of encoder layers
            encoder_type: One of ['lstm', 'gru', 'cnn', 'transformer']
            dropout: Dropout probability
        """
        super().__init__()
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # TODO: Implement sequence encoder
        # Choose ONE of the following architectures:
        
        if encoder_type == 'lstm':
            # TODO: Implement LSTM encoder
            self.rnn = nn.LSTM(input_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=dropout)
            self.projection = nn.Linear(hidden_dim, output_dim)
            pass
            
        elif encoder_type == 'gru':
            # TODO: Implement GRU encoder
            # Similar to LSTM
            self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, 
                               batch_first=True, dropout=dropout)
            self.projection = nn.Linear(hidden_dim, output_dim)
            pass
            
        elif encoder_type == 'cnn':
            # TODO: Implement 1D CNN encoder
            # Stack of Conv1d -> BatchNorm -> ReLU -> Pool
            # Example:
            self.conv = nn.Sequential(
                nn.Conv1d(input_dim, hidden_dim // 2, kernel_size=5, padding=2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Conv1d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.projection = nn.Linear(hidden_dim, output_dim)
            self.drop = nn.Dropout(dropout)
            pass
            
        elif encoder_type == 'transformer':
            # TODO: Implement Transformer encoder
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.posenc = SinusoidalPositionalEncoding(hidden_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
                dropout=dropout, batch_first=True, activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(hidden_dim)
            self.projection = nn.Linear(hidden_dim, output_dim)
            self.drop = nn.Dropout(dropout)
            pass
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        return
        raise NotImplementedError(f"Implement {encoder_type} sequence encoder")
    
    def forward(
        self,
        sequence: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode variable-length sequences.
        
        Args:
            sequence: (batch_size, seq_len, input_dim) - input sequence
            lengths: Optional (batch_size,) - actual sequence lengths (for padding)
            
        Returns:
            encoding: (batch_size, output_dim) - fixed-size embedding
        """
        # TODO: Implement forward pass based on encoder_type
        # Handle variable-length sequences if lengths provided
        # Return fixed-size embedding via pooling or taking last hidden state
        B, T, _ = sequence.shape

        if self.encoder_type in ['lstm', 'gru']:
            if lengths is not None:
                packed = nn.utils.rnn.pack_padded_sequence(
                    sequence, lengths.cpu(), batch_first=True, enforce_sorted=False
                )
                outputs, h_n = self.rnn(packed)
                # h_n: (num_layers, B, H) for GRU, (num_layers, B, H) for LSTM's hidden
                if self.encoder_type == 'lstm':
                    # h_n is a tuple (h, c)
                    # after pack, rnn returns (output, (h_n, c_n))
                    # but above we overwrote h_n; retrieve properly:
                    packed_out, (h_last, _) = self.rnn(packed)
                    last = h_last[-1]  # (B,H)
                else:
                    # GRU
                    _, h_last = self.rnn(packed)  # type: ignore[assignment]
                    last = h_last[-1]
            else:
                outputs, h = self.rnn(sequence)
                if self.encoder_type == 'lstm':
                    # h is (h_n, c_n)
                    h_n, _ = h  # type: ignore[misc]
                    last = h_n[-1]
                else:
                    last = h[-1]  # type: ignore[index]

            return self.projection(last)

        elif self.encoder_type == 'cnn':
            # (B, T, C) -> (B, C, T)
            x = sequence.transpose(1, 2)
            x = self.conv(x)                  # (B, H, T)
            x = self.pool(x).squeeze(-1)      # (B, H)
            x = self.drop(x)
            return self.projection(x)

        elif self.encoder_type == 'transformer':
            x = self.input_proj(sequence)     # (B, T, H)
            x = self.posenc(x)
            key_padding_mask = None
            if lengths is not None:
                # True where PAD
                mask = torch.arange(T, device=sequence.device).unsqueeze(0) >= lengths.unsqueeze(1)
                key_padding_mask = mask  # (B,T) bool
            x = self.transformer(x, src_key_padding_mask=key_padding_mask)
            x = self.norm(x)
            # masked mean pooling over time
            valid_mask = None
            if lengths is not None:
                valid_mask = ~key_padding_mask  # True for valid
                valid_mask = valid_mask.to(x.dtype)
            pooled = masked_mean(x, valid_mask, dim=1)
            pooled = self.drop(pooled)
            return self.projection(pooled)
        raise NotImplementedError("Implement sequence encoder forward pass")


class FrameEncoder(nn.Module):
    """
    Encoder for frame-based data (e.g., video features).
    
    Aggregates frame-level features into video-level embedding.
    """
    
    def __init__(
        self,
        frame_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        temporal_pooling: str = 'attention',
        dropout: float = 0.1
    ):
        """
        Args:
            frame_dim: Dimension of per-frame features
            hidden_dim: Hidden dimension
            output_dim: Output embedding dimension
            temporal_pooling: How to pool frames ['average', 'max', 'attention']
            dropout: Dropout probability
        """
        super().__init__()
        self.temporal_pooling = temporal_pooling
        
        # TODO: Implement frame encoder
        # 1. Frame-level processing (optional MLP)
        # 2. Temporal aggregation (pooling or attention)

        # 1) Frame-level lightweight MLP
        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        if temporal_pooling == 'attention':
            # TODO: Implement attention-based pooling
            # Learned query-style attention
            self.att_query = nn.Parameter(torch.randn(hidden_dim))
            pass
        elif temporal_pooling in ['average', 'max']:
            # Simple pooling, no learnable parameters needed
            pass
        else:
            raise ValueError(f"Unknown pooling: {temporal_pooling}")
        
        # TODO: Add projection layer
        # self.projection = nn.Sequential(...)

        # 3) Projection to output
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        return
        raise NotImplementedError("Implement frame encoder")
    
    def forward(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode sequence of frames.
        
        Args:
            frames: (batch_size, num_frames, frame_dim) - frame features
            mask: Optional (batch_size, num_frames) - valid frame mask
            
        Returns:
            encoding: (batch_size, output_dim) - video-level embedding
        """
        # TODO: Implement forward pass
        # 1. Process frames (optional)
        # 2. Apply temporal pooling
        # 3. Project to output dimension
        
        x = self.frame_mlp(frames)  # (B, F, H)

        if self.temporal_pooling == 'average':
            pooled = masked_mean(x, mask, dim=1)
        elif self.temporal_pooling == 'max':
            if mask is not None:
                # set padded to large negative so they don't win max
                m = mask.unsqueeze(-1).to(dtype=x.dtype)
                x_masked = x * m + (1.0 - m) * (-1e9)
            else:
                x_masked = x
            pooled, _ = x_masked.max(dim=1)
            pooled = torch.nan_to_num(pooled, nan=0.0, neginf=0.0, posinf=0.0)
        else:  # attention
            pooled = self.attention_pool(x, mask)

        return self.projection(pooled)
        raise NotImplementedError("Implement frame encoder forward pass")
    
    def attention_pool(
        self,
        frames: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool frames using learned attention weights.
        
        Args:
            frames: (batch_size, num_frames, frame_dim)
            mask: Optional (batch_size, num_frames) - valid frames
            
        Returns:
            pooled: (batch_size, frame_dim) - attended frame features
        """
        # TODO: Implement attention pooling
        # 1. Compute attention scores for each frame
        # 2. Apply mask if provided
        # 3. Softmax to get weights
        # 4. Weighted sum of frames
        
        B, F, H = frames.shape
        # scores = frames · q
        scores = torch.einsum('bfh,h->bf', frames, self.att_query)  # (B,F)

        if mask is not None:
            scores = scores.masked_fill(~mask.bool(), float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0, neginf=0.0, posinf=0.0)  # all-masked guard

        pooled = torch.einsum('bf,bfh->bh', attn, frames)  # weighted sum
        return pooled
        raise NotImplementedError("Implement attention pooling")


class SimpleMLPEncoder(nn.Module):
    """
    Simple MLP encoder for pre-extracted features.
    
    Use this when working with pre-computed features
    (e.g., ResNet features for images, MFCC for audio).
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output embedding dimension
            num_layers: Number of hidden layers
            dropout: Dropout probability
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        # TODO: Implement MLP encoder
        # Architecture: Input -> [Linear -> BatchNorm -> ReLU -> Dropout] x num_layers -> Output
        
        layers = []
        current_dim = input_dim
        
        # TODO: Add hidden layers
        for i in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        
        # TODO: Add output layer
        layers.append(nn.Linear(current_dim, output_dim))
        
        self.encoder = nn.Sequential(*layers)
        return
        raise NotImplementedError("Implement MLP encoder")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Encode features through MLP.
        
        Args:
            features: (batch_size, input_dim) - input features
            
        Returns:
            encoding: (batch_size, output_dim) - encoded features
        """
        # TODO: Implement forward pass
        return self.encoder(features)
        raise NotImplementedError("Implement MLP encoder forward pass")


def build_encoder(
    modality: str,
    input_dim: int,
    output_dim: int,
    encoder_config: dict = None
) -> nn.Module:
    """
    Factory function to build appropriate encoder for each modality.
    
    Args:
        modality: Modality name ('video', 'audio', 'imu', etc.)
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        encoder_config: Optional config dict with encoder hyperparameters
        
    Returns:
        Encoder module appropriate for the modality
    """
    if encoder_config is None:
        encoder_config = {}
    
    # TODO: Implement encoder selection logic
    # Example heuristics:
    # - 'video' -> FrameEncoder
    # - 'imu', 'audio', 'mocap' -> SequenceEncoder
    # - Pre-extracted features -> SimpleMLPEncoder
    
    if modality in ['video', 'frames']:
        return FrameEncoder(
            frame_dim=input_dim,
            output_dim=output_dim,
            **encoder_config
        )
    seq_keywords = ['imu', 'accelerometer', 'accel', 'gyro', 'mag', 'mocap', 'audio', 'heart_rate', 'hr', 'ecg', 'ppg']
    if any(k in modality for k in seq_keywords):
        return SequenceEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            **encoder_config
        )
    else:
        # Default to MLP for unknown modalities
        return SimpleMLPEncoder(
            input_dim=input_dim,
            output_dim=output_dim,
            **encoder_config
        )


if __name__ == '__main__':
    # Test encoders
    print("Testing encoders...")
    
    batch_size = 4
    seq_len = 100
    input_dim = 64
    output_dim = 128
    
    # Test SequenceEncoder
    print("\nTesting SequenceEncoder...")
    for enc_type in ['lstm', 'gru', 'cnn']:
        try:
            encoder = SequenceEncoder(
                input_dim=input_dim,
                output_dim=output_dim,
                encoder_type=enc_type
            )
            
            sequence = torch.randn(batch_size, seq_len, input_dim)
            output = encoder(sequence)
            
            assert output.shape == (batch_size, output_dim)
            print(f"✓ {enc_type} encoder working! Output shape: {output.shape}")
            
        except NotImplementedError:
            print(f"✗ {enc_type} encoder not implemented yet")
        except Exception as e:
            print(f"✗ {enc_type} encoder error: {e}")
    
    # Test FrameEncoder
    print("\nTesting FrameEncoder...")
    try:
        num_frames = 30
        frame_dim = 512
        
        encoder = FrameEncoder(
            frame_dim=frame_dim,
            output_dim=output_dim,
            temporal_pooling='attention'
        )
        
        frames = torch.randn(batch_size, num_frames, frame_dim)
        output = encoder(frames)
        
        assert output.shape == (batch_size, output_dim)
        print(f"✓ FrameEncoder working! Output shape: {output.shape}")
        
    except NotImplementedError:
        print("✗ FrameEncoder not implemented yet")
    except Exception as e:
        print(f"✗ FrameEncoder error: {e}")
    
    # Test SimpleMLPEncoder
    print("\nTesting SimpleMLPEncoder...")
    try:
        encoder = SimpleMLPEncoder(
            input_dim=input_dim,
            output_dim=output_dim
        )
        
        features = torch.randn(batch_size, input_dim)
        output = encoder(features)
        
        assert output.shape == (batch_size, output_dim)
        print(f"✓ SimpleMLPEncoder working! Output shape: {output.shape}")
        
    except NotImplementedError:
        print("✗ SimpleMLPEncoder not implemented yet")
    except Exception as e:
        print(f"✗ SimpleMLPEncoder error: {e}")

