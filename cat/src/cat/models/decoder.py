import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

class SPADE(nn.Module):
    """Spatially-Adaptive Denormalization (NVlabs/SPADE): parameter-free norm modulated by gamma/beta from a semantic map."""
    def __init__(self, norm_nc, label_nc, nhidden=128, ks=3, norm_type='batch') -> None:
        """
        Build the norm and gamma/beta convs.

        :param norm_nc: int - channels to normalize
        :param label_nc: int - semantic map channels
        :param nhidden: int - hidden channels
        :param ks: int - conv kernel size
        :param norm_type: str - 'instance' for InstanceNorm2d, otherwise BatchNorm2d
        """
        super().__init__()
        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        else:
            # reference uses SyncBatchNorm; BatchNorm2d is equivalent on one gpu
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap) -> torch.Tensor:
        """
        Normalize x and modulate it with the semantic map.

        :param x: torch.Tensor - [B, norm_nc, H, W] features
        :param segmap: torch.Tensor - [B, label_nc, H', W'] semantic map
        :return: torch.Tensor - [B, norm_nc, H, W] modulated features
        """
        normalized = self.param_free_norm(x)

        segmap = F.interpolate(segmap, size=x.size()[2:], mode='bilinear', align_corners=False)
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        out = normalized * (1 + gamma) + beta
        return out

class SPADEResnetBlock(nn.Module):
    """SPADE ResNet block (NVlabs/SPADE) with spectral-normalized convs and a learned shortcut when channels change."""
    def __init__(self, fin, fout, semantic_nc, nhidden=128, norm_type='batch') -> None:
        """
        Build the main path and the shortcut.

        :param fin: int - input channels
        :param fout: int - output channels
        :param semantic_nc: int - semantic map channels
        :param nhidden: int - SPADE hidden channels
        :param norm_type: str - SPADE normalization type
        """
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))

        self.norm_0 = SPADE(fin, semantic_nc, nhidden, norm_type=norm_type)
        self.norm_1 = SPADE(fmiddle, semantic_nc, nhidden, norm_type=norm_type)

        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))
            self.norm_s = SPADE(fin, semantic_nc, nhidden, norm_type=norm_type)

    def forward(self, x, seg) -> torch.Tensor:
        """
        Apply the residual block.

        :param x: torch.Tensor - [B, fin, H, W] features
        :param seg: torch.Tensor - [B, semantic_nc, H', W'] semantic map
        :return: torch.Tensor - [B, fout, H, W] features
        """
        x_s = self._shortcut(x, seg)

        dx = self.conv_0(self._actvn(self.norm_0(x, seg)))
        dx = self.conv_1(self._actvn(self.norm_1(dx, seg)))

        return x_s + dx

    def _shortcut(self, x, seg) -> torch.Tensor:
        """
        Shortcut branch: SPADE and 1x1 conv when fin != fout, else identity.

        :param x: torch.Tensor - [B, fin, H, W] features
        :param seg: torch.Tensor - [B, semantic_nc, H', W'] semantic map
        :return: torch.Tensor - [B, fout, H, W] shortcut features
        """
        if self.learned_shortcut:
            # no activation on shortcut, as in reference
            return self.conv_s(self.norm_s(x, seg))
        return x

    def _actvn(self, x) -> torch.Tensor:
        """
        Apply LeakyReLU with negative slope 0.2.

        :param x: torch.Tensor - input
        :return: torch.Tensor - activated, same shape
        """
        return F.leaky_relu(x, 2e-1)

class CLIPSegSPADEDecoder(nn.Module):
    """SPADE decoder producing dense L2-normalized features from DINO and CLIPSeg maps."""
    def __init__(
        self,
        input_dim=384,
        semantic_channels=16,
        spade_hidden=128,
        inter_block_dropout=0.0,
        norm_type='batch',
        output_dim=64,
        use_depth=False,
        raw_embed_dim=768,
        depth_dropout=0.0,
    ) -> None:
        """
        Build the input stage, four SPADE blocks and the output head.

        :param input_dim: int - first block width after projection or depth fusion
        :param semantic_channels: int - number of CLIPSeg terrain classes (K)
        :param spade_hidden: int - SPADE hidden channels
        :param inter_block_dropout: float - Dropout2d probability between blocks, 0 = disabled
        :param norm_type: str - SPADE normalization type
        :param output_dim: int - output feature dimension (D)
        :param use_depth: bool - fuse depth embeddings through the mixer
        :param raw_embed_dim: int - channels per RGB or depth encoder map
        :param depth_dropout: float - training probability of zeroing a sample's depth features
        """
        super().__init__()
        self.output_dim = output_dim
        self.use_depth = use_depth
        self.depth_dropout = depth_dropout

        if use_depth:
            # concatenated RGB and depth embeddings are mixed down to input_dim
            self.depth_mixer = nn.Sequential(
                nn.Conv2d(raw_embed_dim * 2, raw_embed_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(raw_embed_dim, input_dim, kernel_size=1),
            )
        elif raw_embed_dim != input_dim:
            self.input_proj = nn.Conv2d(raw_embed_dim, input_dim, kernel_size=1)

        # block widths taper from input_dim to 64; every block sees the same semantic map
        self.block_0 = SPADEResnetBlock(input_dim, 256, semantic_channels, spade_hidden, norm_type=norm_type)
        self.block_1 = SPADEResnetBlock(256, 128, semantic_channels, spade_hidden, norm_type=norm_type)
        self.block_2 = SPADEResnetBlock(128, 64, semantic_channels, spade_hidden, norm_type=norm_type)
        self.block_3 = SPADEResnetBlock(64, 64, semantic_channels, spade_hidden, norm_type=norm_type)

        self.up = nn.Upsample(scale_factor=2)
        self.inter_dropout = nn.Dropout2d(inter_block_dropout) if inter_block_dropout > 0 else None

        self.conv_img = nn.Conv2d(64, output_dim, kernel_size=1)

    def forward(self, features, semantic_map, depth_features=None) -> torch.Tensor:
        """
        Decode embeddings and semantic maps into L2-normalized feature maps.

        :param features: torch.Tensor - [B, raw_embed_dim, H, W] DINO patch embeddings
        :param semantic_map: torch.Tensor - [B, K, H', W'] CLIPSeg terrain probabilities, resized in each SPADE layer
        :param depth_features: torch.Tensor | None - same shape as features; None needs use_depth=False or raw_embed_dim == input_dim
        :return: torch.Tensor - [B, output_dim, 16H, 16W], L2-normalized over channels
        """
        if self.use_depth and depth_features is not None:
            if self.training and self.depth_dropout > 0:
                keep = (torch.rand(depth_features.size(0), 1, 1, 1,
                                   device=depth_features.device) > self.depth_dropout).float()
                depth_features = depth_features * keep
            features = self.depth_mixer(torch.cat([features, depth_features], dim=1))
        elif hasattr(self, 'input_proj'):
            features = self.input_proj(features)

        seg = semantic_map

        x = self.block_0(features, seg)
        x = self.up(x)
        if self.inter_dropout: x = self.inter_dropout(x)
        x = self.block_1(x, seg)
        x = self.up(x)
        if self.inter_dropout: x = self.inter_dropout(x)
        x = self.block_2(x, seg)
        x = self.up(x)
        if self.inter_dropout: x = self.inter_dropout(x)
        x = self.block_3(x, seg)
        x = self.up(x)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.normalize(x, p=2, dim=1)
        return x
