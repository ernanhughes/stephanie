# stephanie/scoring/model/vpm_model.py
"""
vpm_transformer_scorer.py
=========================
Vision Transformer-based scorer for Visual Policy Maps (VPMs).

This implementation:
- Uses a custom TinyVisionTransformer architecture optimized for VPMs
- Treats VPMs as epistemic states to be scored on cognitive dimensions
- Includes both inference and training capabilities
- Integrates with Stephanie's scoring ecosystem
- Provides visual introspection via attention maps

Key improvements over heuristic VPMScorer:
- Learned patterns instead of handcrafted heuristics
- Multi-dimensional scoring with proper gradient flow
- Attention mechanisms for explainable AI
- Trainable parameters that improve over time
- Better generalization across VPM types and resolutions
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__file__)

class VPMDimension(str, Enum):
    """Cognitive dimensions for scoring VPMs"""
    CLARITY = "clarity"
    NOVELTY = "novelty"
    CONFIDENCE = "confidence"
    CONTRADICTION = "contradiction"
    COHERENCE = "coherence"
    COMPLEXITY = "complexity"
    ALIGNMENT = "alignment"

@dataclass
class AttentionMap:
    """Container for attention map data for visualization"""
    layer: int
    head: int
    attention_weights: np.ndarray  # Shape: (num_patches, num_patches)
    patch_positions: np.ndarray   # Shape: (num_patches, 2) - (row, col)
    dimension: str                 # Which dimension this attention relates to

class PatchEmbedding(nn.Module):
    """Convert image patches to embeddings for transformer processing"""
    
    def __init__(self, img_size: int = 64, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Convolutional approach to patch embedding (more efficient than linear projection)
        self.projection = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
        # Positional embeddings
        self.position_embeddings = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.dropout = nn.Dropout(0.1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        nn.init.normal_(self.position_embeddings, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert image to patch embeddings.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, n_patches + 1, embed_dim)
        """
        B, C, H, W = x.shape
        
        # Project patches
        x = self.projection(x)  # (B, embed_dim, n_patches_h, n_patches_w)
        x = x.flatten(2)        # (B, embed_dim, n_patches)
        x = x.transpose(1, 2)   # (B, n_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, n_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.position_embeddings
        
        x = self.dropout(x)
        return x

class TransformerBlock(nn.Module):
    """Single transformer block with multi-head self-attention"""
    
    def __init__(self, embed_dim: int = 128, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = dropout
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        
    def forward(self, x: torch.Tensor, return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with optional attention return for introspection.
        
        Args:
            x: Input tensor of shape (B, n_patches + 1, embed_dim)
            return_attention: Whether to return attention weights
            
        Returns:
            Processed tensor and optionally attention weights
        """
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm, need_weights=return_attention)
        
        # Residual connection
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x, None

class TinyVisionTransformer(nn.Module):
    """Compact Vision Transformer optimized for VPM scoring"""
    
    def __init__(
        self,
        img_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        num_dimensions: int = 7
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_dimensions = num_dimensions
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) 
            for _ in range(depth)
        ])
        
        # Layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Head for multi-dimensional scoring
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_dimensions),
            nn.Sigmoid()  # Output in [0,1] range for all dimensions
        )
        
        # Initialize weights
        self._init_weights()
        
        log.info(f"TinyVisionTransformer initialized: "
                f"{depth} layers, {num_heads} heads, {embed_dim} embedding dim")
    
    def _init_weights(self):
        """Initialize weights for better training stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from the transformer (without head)"""
        x = self.patch_embed(x)
        
        for block in self.blocks:
            x, _ = block(x)
            
        x = self.norm(x)
        # Use CLS token output
        return x[:, 0]
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_attention: bool = False,
        attention_layers: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional attention return.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            return_attention: Whether to return attention maps
            attention_layers: Which layers to return attention from (None = last layer)
            
        Returns:
            Dictionary with scores and optionally attention maps
        """
        x = self.patch_embed(x)
        
        attention_maps = []
        
        for i, block in enumerate(self.blocks):
            if return_attention and (attention_layers is None or i in attention_layers):
                x, attn = block(x, return_attention=True)
                attention_maps.append(attn)
            else:
                x, _ = block(x)
        
        x = self.norm(x)
        # Use CLS token output
        cls_output = x[:, 0]
        
        # Get scores
        scores = self.head(cls_output)
        
        result = {"scores": scores}
        
        if return_attention:
            result["attention_maps"] = attention_maps
            result["patch_positions"] = self._get_patch_positions(x.shape[0])
            
        return result
    
    def _get_patch_positions(self, batch_size: int) -> torch.Tensor:
        """Get positions of patches in the original image"""
        # This would be calculated based on patch size and image dimensions
        # For now, return a placeholder
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.patch_embed.n_patches, 2, device=device)
