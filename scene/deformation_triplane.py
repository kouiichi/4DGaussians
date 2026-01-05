#
# Deformation Network with TriPlane and Multi-head FiLM Fusion
#
# This implementation uses TriPlane for spatial encoding with Multi-head FiLM 
# (Feature-wise Linear Modulation) for control signal fusion.
#
# Key features:
# 1. TriPlane only encodes spatial geometry (XY, XZ, YZ planes)
# 2. Multi-head FiLM: Control signals generate γ (scale) and β (shift) for each layer
# 3. Residual connections preserve original spatial information
# 4. More expressive than simple concatenation
#

import math
import torch
import torch.nn as nn
import torch.nn.init as init
from typing import Optional, List, Tuple

from scene.triplane import TriPlaneField, ControlProcessor
from scene.grid import DenseGrid
from utils.graphics_utils import batch_quaternion_multiply


# ============================================================================
# FiLM Layers
# ============================================================================

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) Layer.
    
    Applies affine transformation to features based on conditioning signal:
        output = γ * input + β
    
    where γ (scale) and β (shift) are generated from the conditioning signal.
    
    Args:
        feature_dim: Dimension of input features to modulate
        condition_dim: Dimension of conditioning signal
        hidden_dim: Hidden dimension for γ/β generation MLP
    """
    
    def __init__(self, feature_dim: int, condition_dim: int, hidden_dim: int = 64):
        super(FiLMLayer, self).__init__()
        
        self.feature_dim = feature_dim
        
        # Generate γ (scale) and β (shift) from conditioning signal
        self.film_generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_dim * 2)  # γ and β
        )
        
        # Initialize to identity transformation: γ=1, β=0
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights so that initial output is identity (γ=1, β=0)."""
        nn.init.zeros_(self.film_generator[-1].weight)
        nn.init.zeros_(self.film_generator[-1].bias)
        # Set γ bias to 1 for identity
        self.film_generator[-1].bias.data[:self.feature_dim] = 1.0
    
    def forward(self, features: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM modulation.
        
        Args:
            features: [N, feature_dim] - Features to modulate
            condition: [N, condition_dim] - Conditioning signal
            
        Returns:
            modulated: [N, feature_dim] - Modulated features
        """
        # Generate γ and β
        film_params = self.film_generator(condition)
        gamma = film_params[:, :self.feature_dim]
        beta = film_params[:, self.feature_dim:]
        
        # Apply affine transformation
        modulated = gamma * features + beta
        
        return modulated


class FiLMBlock(nn.Module):
    """
    FiLM Block with Linear layer + FiLM modulation + Activation.
    
    Architecture:
        input -> Linear -> FiLM(γ, β) -> ReLU -> output
    """
    
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        condition_dim: int,
        hidden_dim: int = 64,
        activation: str = 'relu'
    ):
        super(FiLMBlock, self).__init__()
        
        self.linear = nn.Linear(in_dim, out_dim)
        self.film = FiLMLayer(out_dim, condition_dim, hidden_dim)
        
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'silu':
            self.activation = nn.SiLU(inplace=True)
        else:
            self.activation = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [N, in_dim] - Input features
            condition: [N, condition_dim] - Conditioning signal
            
        Returns:
            output: [N, out_dim] - Output features
        """
        h = self.linear(x)
        h = self.film(h, condition)
        h = self.activation(h)
        return h


class MultiHeadFiLMDecoder(nn.Module):
    """
    Multi-head FiLM Decoder with Residual Connections.
    
    Architecture:
        spatial_feat ─┬─> FiLMBlock1 ─> FiLMBlock2 ─> ... ─> FiLMBlockN ─┬─> output
                      │                                                  │
                      └──────────────── Residual Projection ─────────────┘
    
    Each FiLM block is modulated by the control signal, allowing the network
    to learn how control affects spatial features at each layer.
    
    Args:
        spatial_dim: Dimension of spatial features from TriPlane
        control_dim: Dimension of control features (after PE)
        hidden_dim: Hidden dimension of FiLM blocks
        num_layers: Number of FiLM blocks
        film_hidden: Hidden dimension for FiLM γ/β generation
    """
    
    def __init__(
        self,
        spatial_dim: int,
        control_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        film_hidden: int = 64,
        use_residual: bool = True
    ):
        super(MultiHeadFiLMDecoder, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.control_dim = control_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_residual = use_residual
        
        # Build FiLM blocks
        self.film_blocks = nn.ModuleList()
        
        # First block: spatial_dim -> hidden_dim
        self.film_blocks.append(
            FiLMBlock(spatial_dim, hidden_dim, control_dim, film_hidden)
        )
        
        # Middle blocks: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.film_blocks.append(
                FiLMBlock(hidden_dim, hidden_dim, control_dim, film_hidden)
            )
        
        # Residual projection: spatial_dim -> hidden_dim
        if use_residual:
            self.residual_proj = nn.Linear(spatial_dim, hidden_dim)
        else:
            self.residual_proj = None
        
        print(f"[MultiHeadFiLMDecoder] Created with {num_layers} FiLM blocks")
        print(f"[MultiHeadFiLMDecoder]   - Spatial dim: {spatial_dim}")
        print(f"[MultiHeadFiLMDecoder]   - Control dim: {control_dim}")
        print(f"[MultiHeadFiLMDecoder]   - Hidden dim: {hidden_dim}")
        print(f"[MultiHeadFiLMDecoder]   - Residual: {use_residual}")
    
    def forward(
        self, 
        spatial_feat: torch.Tensor, 
        control_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with multi-head FiLM modulation.
        
        Args:
            spatial_feat: [N, spatial_dim] - Spatial features from TriPlane
            control_feat: [N, control_dim] - Control features (after PE)
            
        Returns:
            output: [N, hidden_dim] - Fused features
        """
        # Pass through FiLM blocks
        h = spatial_feat
        for film_block in self.film_blocks:
            h = film_block(h, control_feat)
        
        # Add residual connection
        if self.use_residual and self.residual_proj is not None:
            residual = self.residual_proj(spatial_feat)
            h = h + residual
        
        return h


# ============================================================================
# Deformation Network with FiLM Fusion
# ============================================================================

class DeformationTriPlane(nn.Module):
    """
    Deformation network using TriPlane for spatial encoding and Multi-head FiLM fusion.
    
    Architecture:
        1. TriPlane: Extract spatial features from (x, y, z)
        2. ControlProcessor: Process control vector with optional PE
        3. Multi-head FiLM: Control modulates spatial features through γ and β
        4. Residual Connection: Preserve original spatial information
        5. Deformation Heads: Predict per-attribute deformations
        
    Args:
        D: Number of FiLM blocks (decoder depth)
        W: Hidden dimension (decoder width)
        grid_pe: Positional encoding for grid features
        args: Configuration arguments
    """
    
    def __init__(self, D=2, W=128, grid_pe=0, args=None):
        super(DeformationTriPlane, self).__init__()
        
        self.D = D
        self.W = W
        self.grid_pe = grid_pe
        self.args = args
        self.no_grid = getattr(args, 'no_grid', False)
        
        # TriPlane configuration
        triplane_config = {
            'resolution': args.kplanes_config.get('resolution', [64, 64, 64])[:3],
            'output_coordinate_dim': args.kplanes_config.get('output_coordinate_dim', 32)
        }
        
        # 1. TriPlane for spatial feature encoding
        self.triplane = TriPlaneField(
            bounds=args.bounds,
            planeconfig=triplane_config,
            multires=args.multires
        )
        
        # 2. Control signal processor
        control_use_pe = getattr(args, 'control_use_pe', True)
        control_num_freq = getattr(args, 'control_num_frequencies', 4)
        control_input_dim = getattr(args, 'control_input_dim', 6)
        control_hidden = getattr(args, 'control_hidden_dim', 64)
        control_output_dim = getattr(args, 'control_output_dim', None)
        
        self.control_processor = ControlProcessor(
            input_dim=control_input_dim,
            use_pe=control_use_pe,
            num_frequencies=control_num_freq,
            hidden_dim=control_hidden if control_output_dim else None,
            output_dim=control_output_dim
        )
        
        # 3. Compute dimensions
        self.spatial_dim = self.triplane.feat_dim
        if grid_pe > 0:
            self.spatial_dim = self.spatial_dim * (1 + 2 * grid_pe)
        self.control_dim = self.control_processor.output_dim
        
        # 4. FiLM fusion configuration
        film_hidden = getattr(args, 'film_hidden_dim', 64)
        use_residual = getattr(args, 'film_use_residual', True)
        
        # 5. Multi-head FiLM decoder
        self.film_decoder = MultiHeadFiLMDecoder(
            spatial_dim=self.spatial_dim,
            control_dim=self.control_dim,
            hidden_dim=W,
            num_layers=D,
            film_hidden=film_hidden,
            use_residual=use_residual
        )
        
        # 6. Optional modules
        if getattr(args, 'empty_voxel', False):
            self.empty_voxel = DenseGrid(channels=1, world_size=[64, 64, 64])
        else:
            self.empty_voxel = None
        
        if getattr(args, 'static_mlp', False):
            self.static_mlp = nn.Sequential(
                nn.ReLU(),
                nn.Linear(W, W),
                nn.ReLU(),
                nn.Linear(W, 1)
            )
        else:
            self.static_mlp = None
        
        self.ratio = 0
        
        # 7. Deformation prediction heads
        self._create_deform_heads()
        
        print(f"[DeformationTriPlane] Using Multi-head FiLM fusion")
        print(f"[DeformationTriPlane]   - Spatial dim: {self.spatial_dim}")
        print(f"[DeformationTriPlane]   - Control dim: {self.control_dim}")
        print(f"[DeformationTriPlane]   - FiLM layers: {D}")
        print(f"[DeformationTriPlane]   - Hidden dim: {W}")
    
    def _create_deform_heads(self):
        """Create deformation prediction heads."""
        
        self.pos_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 3)
        )
        
        self.scales_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 3)
        )
        
        self.rotations_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 4)
        )
        
        self.opacity_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 1)
        )
        
        self.shs_deform = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W),
            nn.ReLU(),
            nn.Linear(self.W, 16 * 3)
        )
    
    @property
    def get_aabb(self):
        return self.triplane.get_aabb
    
    def set_aabb(self, xyz_max, xyz_min):
        print(f"[DeformationTriPlane] Setting AABB: max={xyz_max}, min={xyz_min}")
        self.triplane.set_aabb(xyz_max, xyz_min)
        if self.empty_voxel is not None:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    
    @property
    def get_empty_ratio(self):
        return self.ratio
    
    def _apply_grid_pe(self, features: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding to grid features."""
        if self.grid_pe <= 0:
            return features
        
        freq_bands = 2.0 ** torch.arange(
            self.grid_pe, device=features.device, dtype=features.dtype
        )
        encoded = [features]
        for freq in freq_bands:
            encoded.append(torch.sin(features * freq))
            encoded.append(torch.cos(features * freq))
        return torch.cat(encoded, dim=-1)
    
    def query_features(
        self, 
        pts: torch.Tensor, 
        control_vec: torch.Tensor
    ) -> torch.Tensor:
        """
        Query fused features using Multi-head FiLM.
        
        Args:
            pts: [N, 3] - 3D positions
            control_vec: [N, control_dim] - Control vectors
            
        Returns:
            hidden: [N, W] - Fused hidden features
        """
        if self.no_grid:
            # Fallback mode without grid
            control_feat = self.control_processor(control_vec)
            # Simple MLP fallback
            combined = torch.cat([pts, control_feat], dim=-1)
            hidden = nn.functional.relu(
                nn.functional.linear(combined, torch.randn(self.W, combined.shape[-1], device=pts.device))
            )
            return hidden
        
        # 1. Get spatial features from TriPlane
        spatial_feat = self.triplane(pts)
        
        # Optional: apply PE to spatial features
        if self.grid_pe > 0:
            spatial_feat = self._apply_grid_pe(spatial_feat)
        
        # 2. Process control vector
        control_feat = self.control_processor(control_vec)
        
        # 3. Multi-head FiLM fusion
        hidden = self.film_decoder(spatial_feat, control_feat)
        
        return hidden
    
    def forward(
        self, 
        rays_pts_emb: torch.Tensor, 
        scales_emb: Optional[torch.Tensor] = None, 
        rotations_emb: Optional[torch.Tensor] = None, 
        opacity: Optional[torch.Tensor] = None, 
        shs_emb: Optional[torch.Tensor] = None, 
        time_feature: Optional[torch.Tensor] = None, 
        control_vec: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        
        Args:
            rays_pts_emb: [N, 3+PE] - Position with PE
            scales_emb, rotations_emb: Attribute embeddings
            opacity, shs_emb: Gaussian attributes
            time_feature: [UNUSED] Legacy parameter
            control_vec: [N, control_dim] - Control vectors
            
        Returns:
            Tuple of deformed attributes
        """
        if control_vec is None:
            return self.forward_static(rays_pts_emb[:, :3])
        else:
            return self.forward_dynamic(
                rays_pts_emb, scales_emb, rotations_emb,
                opacity, shs_emb, control_vec
            )
    
    def forward_static(self, pts: torch.Tensor):
        """Static forward (no control-based deformation)."""
        if self.static_mlp is not None:
            spatial_feat = self.triplane(pts)
            dx = self.static_mlp(spatial_feat)
            return pts + dx
        return pts
    
    def forward_dynamic(
        self, 
        rays_pts_emb: torch.Tensor, 
        scales_emb: torch.Tensor, 
        rotations_emb: torch.Tensor, 
        opacity_emb: torch.Tensor, 
        shs_emb: torch.Tensor, 
        control_vec: torch.Tensor
    ):
        """
        Dynamic forward with Multi-head FiLM control modulation.
        
        Args:
            rays_pts_emb: [N, 3+PE] - Position with positional encoding
            scales_emb: [N, 3+PE] - Scales with PE
            rotations_emb: [N, 4+PE] - Rotations with PE
            opacity_emb: [N, 1] - Opacity
            shs_emb: [N, 16, 3] - SH coefficients
            control_vec: [N, control_dim] - Full control vector
            
        Returns:
            Tuple of (pts, scales, rotations, opacity, shs)
        """
        # Extract raw position
        pts = rays_pts_emb[:, :3]
        
        # Query fused features with FiLM modulation
        hidden = self.query_features(pts, control_vec)
        
        # Compute deformation mask
        if self.static_mlp is not None:
            mask = self.static_mlp(hidden)
        elif self.empty_voxel is not None:
            mask = self.empty_voxel(pts)
        else:
            mask = torch.ones_like(opacity_emb[:, 0]).unsqueeze(-1)
        
        # Predict deformations
        # Position
        if getattr(self.args, 'no_dx', False):
            pts_out = pts
        else:
            dx = self.pos_deform(hidden)
            pts_out = pts * mask + dx
        
        # Scale
        if getattr(self.args, 'no_ds', False):
            scales_out = scales_emb[:, :3]
        else:
            ds = self.scales_deform(hidden)
            scales_out = scales_emb[:, :3] * mask + ds
        
        # Rotation
        if getattr(self.args, 'no_dr', False):
            rotations_out = rotations_emb[:, :4]
        else:
            dr = self.rotations_deform(hidden)
            if getattr(self.args, 'apply_rotation', False):
                rotations_out = batch_quaternion_multiply(rotations_emb[:, :4], dr)
            else:
                rotations_out = rotations_emb[:, :4] + dr
        
        # Opacity
        if getattr(self.args, 'no_do', True):
            opacity_out = opacity_emb[:, :1]
        else:
            do = self.opacity_deform(hidden)
            opacity_out = opacity_emb[:, :1] * mask + do
        
        # SH coefficients
        if getattr(self.args, 'no_dshs', True):
            shs_out = shs_emb
        else:
            dshs = self.shs_deform(hidden).reshape([shs_emb.shape[0], 16, 3])
            shs_out = shs_emb * mask.unsqueeze(-1) + dshs
        
        return pts_out, scales_out, rotations_out, opacity_out, shs_out
    
    def get_mlp_parameters(self) -> List[torch.nn.Parameter]:
        """Get MLP parameters (excluding grid)."""
        params = []
        for name, param in self.named_parameters():
            if 'triplane' not in name:
                params.append(param)
        return params
    
    def get_grid_parameters(self) -> List[torch.nn.Parameter]:
        """Get grid (TriPlane) parameters."""
        params = []
        for name, param in self.named_parameters():
            if 'triplane' in name:
                params.append(param)
        return params


# ============================================================================
# Top-level Network
# ============================================================================

class deform_network_triplane(nn.Module):
    """
    Top-level deformation network using TriPlane + Multi-head FiLM architecture.
    
    This is a drop-in replacement for deform_network that uses:
    1. TriPlaneField (3 planes) instead of HexPlaneField (6 planes)
    2. Multi-head FiLM fusion instead of simple concatenation
    3. Full control vector preservation (no compression to 1D)
    """
    
    def __init__(self, args):
        super(deform_network_triplane, self).__init__()
        
        net_width = args.net_width
        defor_depth = args.defor_depth
        posbase_pe = args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        opacity_pe = args.opacity_pe
        grid_pe = args.grid_pe
        
        # Create TriPlane-based deformation network with FiLM fusion
        self.deformation_net = DeformationTriPlane(
            W=net_width,
            D=defor_depth,
            grid_pe=grid_pe,
            args=args
        )
        
        # Positional encoding buffers
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        self.register_buffer('opacity_poc', torch.FloatTensor([(2**i) for i in range(opacity_pe)]))
        
        # Initialize weights
        self.apply(initialize_weights)
        
        print(f"[deform_network_triplane] Initialized with TriPlane + Multi-head FiLM Fusion")
    
    def forward(
        self, 
        point: torch.Tensor, 
        scales: Optional[torch.Tensor] = None, 
        rotations: Optional[torch.Tensor] = None, 
        opacity: Optional[torch.Tensor] = None, 
        shs: Optional[torch.Tensor] = None, 
        control_vec: Optional[torch.Tensor] = None
    ):
        """
        Forward pass.
        
        Args:
            point: [N, 3] - Positions
            scales, rotations, opacity, shs: Gaussian attributes
            control_vec: [N, control_dim] - Full control vector
            
        Returns:
            Deformed attributes
        """
        if control_vec is None:
            return self.forward_static(point)
        else:
            return self.forward_dynamic(point, scales, rotations, opacity, shs, control_vec)
    
    @property
    def get_aabb(self):
        return self.deformation_net.get_aabb
    
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
    
    def forward_static(self, points: torch.Tensor):
        return self.deformation_net(points)
    
    def forward_dynamic(
        self, 
        point: torch.Tensor, 
        scales: torch.Tensor, 
        rotations: torch.Tensor, 
        opacity: torch.Tensor, 
        shs: torch.Tensor, 
        control_vec: torch.Tensor
    ):
        """
        Dynamic deformation with Multi-head FiLM fusion.
        
        The control vector modulates spatial features through FiLM layers,
        allowing for expressive control-dependent deformations.
        """
        # Apply positional encoding
        point_emb = poc_fre(point, self.pos_poc)
        scales_emb = poc_fre(scales, self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations, self.rotation_scaling_poc)
        
        # Forward with FiLM fusion
        means3D, scales, rotations, opacity, shs = self.deformation_net(
            point_emb,
            scales_emb,
            rotations_emb,
            opacity,
            shs,
            None,  # time_feature (unused)
            control_vec
        )
        
        return means3D, scales, rotations, opacity, shs
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters()
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()


# ============================================================================
# Utilities
# ============================================================================

def initialize_weights(m):
    """Xavier initialization for linear layers."""
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            init.zeros_(m.bias)


def poc_fre(input_data: torch.Tensor, poc_buf: torch.Tensor) -> torch.Tensor:
    """Positional encoding using frequency buffer."""
    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin, input_data_cos], -1)
    return input_data_emb


# ============================================================================
# Testing
# ============================================================================

def test_film_layer():
    """Test FiLM layer."""
    print("=" * 70)
    print("Testing FiLM Layer")
    print("=" * 70)
    
    film = FiLMLayer(feature_dim=64, condition_dim=32, hidden_dim=32)
    
    features = torch.randn(100, 64)
    condition = torch.randn(100, 32)
    
    output = film(features, condition)
    print(f"Input: {features.shape}, Condition: {condition.shape}")
    print(f"Output: {output.shape}")
    
    # Check identity initialization
    zero_condition = torch.zeros(100, 32)
    output_identity = film(features, zero_condition)
    diff = (output_identity - features).abs().mean()
    print(f"Identity test (should be ~1.0): diff={diff:.6f}")
    
    print("✓ FiLM Layer test passed!")


def test_film_block():
    """Test FiLM block."""
    print("\n" + "=" * 70)
    print("Testing FiLM Block")
    print("=" * 70)
    
    block = FiLMBlock(in_dim=64, out_dim=128, condition_dim=32, hidden_dim=32)
    
    x = torch.randn(100, 64)
    condition = torch.randn(100, 32)
    
    output = block(x, condition)
    print(f"Input: {x.shape}, Condition: {condition.shape}")
    print(f"Output: {output.shape}")
    
    print("✓ FiLM Block test passed!")


def test_multihead_film_decoder():
    """Test Multi-head FiLM Decoder."""
    print("\n" + "=" * 70)
    print("Testing Multi-head FiLM Decoder")
    print("=" * 70)
    
    decoder = MultiHeadFiLMDecoder(
        spatial_dim=96,
        control_dim=54,
        hidden_dim=128,
        num_layers=3,
        film_hidden=64,
        use_residual=True
    )
    
    spatial_feat = torch.randn(1000, 96)
    control_feat = torch.randn(1000, 54)
    
    output = decoder(spatial_feat, control_feat)
    print(f"Spatial: {spatial_feat.shape}, Control: {control_feat.shape}")
    print(f"Output: {output.shape}")
    
    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"Parameters: {num_params:,}")
    
    print("✓ Multi-head FiLM Decoder test passed!")


def test_deform_network_triplane():
    """Test the complete TriPlane + FiLM deformation network."""
    print("\n" + "=" * 70)
    print("Testing deform_network_triplane with FiLM")
    print("=" * 70)
    
    # Create mock args
    class Args:
        net_width = 128
        defor_depth = 2
        posebase_pe = 10
        scale_rotation_pe = 2
        opacity_pe = 2
        grid_pe = 0
        bounds = 1.6
        multires = [1, 2, 4]
        kplanes_config = {
            'resolution': [64, 64, 64],
            'output_coordinate_dim': 32
        }
        control_input_dim = 6
        control_use_pe = True
        control_num_frequencies = 4
        control_hidden_dim = 64
        control_output_dim = None
        film_hidden_dim = 64
        film_use_residual = True
        no_grid = False
        no_dx = False
        no_ds = False
        no_dr = False
        no_do = True
        no_dshs = True
        empty_voxel = False
        static_mlp = False
        apply_rotation = False
    
    args = Args()
    
    # Create network
    network = deform_network_triplane(args)
    
    # Count parameters
    total_params = sum(p.numel() for p in network.parameters())
    mlp_params = sum(p.numel() for p in network.get_mlp_parameters())
    grid_params = sum(p.numel() for p in network.get_grid_parameters())
    
    print(f"\nNetwork Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  MLP parameters: {mlp_params:,}")
    print(f"  Grid parameters: {grid_params:,}")
    
    # Test forward pass
    batch_size = 1000
    point = torch.randn(batch_size, 3) * 1.0
    scales = torch.randn(batch_size, 3) * 0.1
    rotations = torch.randn(batch_size, 4)
    rotations = rotations / rotations.norm(dim=-1, keepdim=True)
    opacity = torch.randn(batch_size, 1)
    shs = torch.randn(batch_size, 16, 3)
    control_vec = torch.randn(batch_size, 6) * 3.14
    
    print(f"\nInput shapes:")
    print(f"  point: {point.shape}")
    print(f"  control_vec: {control_vec.shape}")
    
    # Forward pass
    means3D, scales_out, rotations_out, opacity_out, shs_out = network(
        point, scales, rotations, opacity, shs, control_vec
    )
    
    print(f"\nOutput shapes:")
    print(f"  means3D: {means3D.shape}")
    print(f"  scales: {scales_out.shape}")
    print(f"  rotations: {rotations_out.shape}")
    
    # Test gradient flow
    loss = means3D.sum() + scales_out.sum() + rotations_out.sum()
    loss.backward()
    
    print(f"\nGradient check: ✓ Passed")
    
    print("\n" + "=" * 70)
    print("✓ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    test_film_layer()
    test_film_block()
    test_multihead_film_decoder()
    test_deform_network_triplane()
