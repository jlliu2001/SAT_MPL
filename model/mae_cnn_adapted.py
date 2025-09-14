import torch
import torch.nn as nn
from .mapseg_blocks import create_encoders, ExtResNetBlock, _ntuple, res_decoders
import numpy as np


class MAE_CNN_Adapted(nn.Module):
    """ 
    Adapted Masked Autoencoder with ResNet encoder for SAT integration
    This version is modified to work as a vision backbone in the SAT framework
    """

    def __init__(self, input_channels=3, embed_dim=512, depth=6, image_size=[288, 288, 96]):
        """
        Args:
            input_channels: number of input channels (3 for RGB-like medical images in SAT)
            embed_dim: embedding dimension 
            depth: encoder depth
            image_size: input image size [H, W, D]
        """
        super().__init__()
        
        # Store configuration
        self.embed_dim = embed_dim
        self.depth = depth
        self.image_size = image_size
        
        # Calculate feature maps for multi-scale outputs (6 scales for SAT compatibility)
        decoder_embed_dim = embed_dim // 16
        to_tuple = _ntuple(depth)
        
        # Multi-scale feature maps to match SAT's expected 6 scales
        # We'll create 6 different scales with progressively larger feature maps
        f_maps = [
            embed_dim // 8,   # Scale 0: smallest features
            embed_dim // 4,   # Scale 1
            embed_dim // 2,   # Scale 2  
            embed_dim,        # Scale 3
            embed_dim,        # Scale 4
            embed_dim         # Scale 5: largest features (bottleneck)
        ]
        
        # Create multi-scale encoders
        self.encoders = nn.ModuleList()
        
        # First encoder (input -> first downsampling)
        self.encoders.append(
            Encoder_Block(input_channels, f_maps[0], stride=1, kernel_size=3, padding=1)
        )
        
        # Subsequent encoders with downsampling
        for i in range(1, len(f_maps)):
            in_channels = f_maps[i-1]
            out_channels = f_maps[i]
            #stride = 2 if i < len(f_maps)-1 else 1  # No stride for last layer
            stride=2
            self.encoders.append(
                Encoder_Block(in_channels, out_channels, stride=stride, kernel_size=3, padding=1)
            )
        
        # Create decoders for multi-scale per-pixel features (5 scales for SAT)
        self.decoders = nn.ModuleList()
        #decoder_f_maps = [f_maps[i] for i in range(len(f_maps)-1, 0, -1)]  # Reverse order, skip bottleneck
        decoder_f_maps = [f_maps[i] for i in range(-2, -len(f_maps)-1, -1)]  # Reverse order, skip bottleneck
        
        # Decoders from bottleneck to output
        for i, out_channels in enumerate(decoder_f_maps):
            if i == 0:
                in_channels = f_maps[-1]  # From bottleneck
            else:
                in_channels = decoder_f_maps[i-1]
                
            self.decoders.append(
                Decoder_Block(in_channels, out_channels, scale_factor=2)
            )
            
        # Final output layer 
        self.final_conv = nn.Conv3d(decoder_f_maps[-1], input_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass compatible with SAT's expected interface
        
        Args:
            x: input tensor [B, C, H, W, D]
            
        Returns:
            latent_embeddings: list of encoder features at different scales
            per_pixel_embeddings: list of decoder features for per-pixel prediction
        """
        # Store encoder features at different scales
        latent_embeddings = []
        
        # Forward through encoders
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            print('x encoder.shape:',x.shape)
            latent_embeddings.append(x)
            
        # Store per-pixel decoder features
        per_pixel_embeddings = []
        
        # Forward through decoders
        decoder_x = latent_embeddings[-1]  # Start from bottleneck
        for i, decoder in enumerate(self.decoders):
            decoder_x = decoder(decoder_x)
            print('decoder_x.shape:',decoder_x.shape)
            per_pixel_embeddings.append(decoder_x)
            
        # Final reconstruction (not used in SAT but maintained for compatibility)
        final_output = self.final_conv(decoder_x)
        #per_pixel_embeddings.append(final_output)
        
        return latent_embeddings, per_pixel_embeddings


class Encoder_Block(nn.Module):
    """Basic encoder block with configurable stride for downsampling"""
    
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=3, padding=1):
        super().__init__()
        
        self.block = ExtResNetBlock(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size,
            stride_size=stride,
            order='gcr',
            num_groups=min(32, out_channels//2) if out_channels >= 64 else 8
        )
        
    def forward(self, x):
        
        return self.block(x)


class Decoder_Block(nn.Module):
    """Basic decoder block with upsampling"""
    
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        
        self.upsample = nn.ConvTranspose3d(
            in_channels, out_channels,
            kernel_size=scale_factor,
            stride=scale_factor
        )
        
        self.conv = ExtResNetBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride_size=1,
            order='gcr', 
            num_groups=min(32, out_channels//2) if out_channels >= 64 else 8
        )
        
    def forward(self, x):
        
        x = self.upsample(x)
        x = self.conv(x)
        return x


class MAE_CNN_Original(nn.Module):
    """ 
    Original Masked Autoencoder implementation preserved for reference
    """

    def __init__(self, cfg):
        super().__init__()
        # --------------------------------------------------------------------------
        # ResNet encoder specifics
        self.cfg = cfg
        embed_dim = cfg.model.embed_dim
        depth = cfg.model.depth
        decoder_embed_dim = embed_dim // 16
        to_tuple = _ntuple(depth)
        # encoder
        self.local_encoder = create_encoders(in_channels=1, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
                                             conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
                                             num_groups=32)

        # upsample
        self.local_upsample = nn.ConvTranspose3d(in_channels=embed_dim, out_channels=decoder_embed_dim, kernel_size=4,
                                                 stride=4)

        # decoder
        self.local_decoder = res_decoders(in_channels=decoder_embed_dim, f_maps=[16],
                                          basic_module=ExtResNetBlock, conv_kernel_size=3, conv_stride_size=1,
                                          conv_padding=0, layer_order='gcr', num_groups=8)

        # norm layers
        self.final_projection_local_recon = nn.Conv3d(
            in_channels=16, out_channels=1, kernel_size=3, padding=1)
        self.final_norm_local_recon = nn.GroupNorm(
            num_groups=8, num_channels=16)

        self.avgpool = nn.AdaptiveAvgPool3d((3, 1, 1))

    def patchify(self, imgs, p):
        """
        imgs: (N, 1, H, W, D)
        x: (N, H*W*D/P***3, patch_size**3)
        """
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0
        h, w, d = [i//p for i in self.cfg.data.patch_size]

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3))
        return x

    def unpatchify(self, x, p):
        """
        x: (N, H*W*D/P***3, patch_size**3)
        imgs: (N, 1, H, W, D)
        """
        h, w, d = [i//p for i in self.cfg.data.patch_size]
        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p))
        x = torch.einsum('nhwdpqr->nhpwqdr', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, w * p, d * p))
        return imgs

    def random_masking(self, x, mask_ratio, p):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        x = self.patchify(x, p)

        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        mask_ = torch.zeros_like(x_masked)
        # generate the binary mask: 0 is keep, 1 is remove

        x_empty = torch.zeros((N, L - len_keep, D)).cuda()
        mask = torch.ones_like(x_empty)
        x_ = torch.cat([x_masked, x_empty], dim=1)
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        mask_ = torch.cat([mask_, mask], dim=1)
        mask_ = torch.gather(
            mask_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))

        x_masked = self.unpatchify(x_, p)

        mask = self.unpatchify(mask_, p)

        return x_masked, mask

    def forward_encoder(self, x, mask_ratio, p):
        # masking: length -> length * mask_ratio
        x, mask = self.random_masking(x, mask_ratio, p)

        # apply Transformer blocks
        for blk in self.local_encoder:
            x = blk(x)

        return x, mask

    def forward_local_decoder(self, x):
        x = self.local_upsample(x)
        # apply Transformer blocks
        for blk in self.local_decoder:
            x = blk(x)

        x = self.final_norm_local_recon(x)
        x = self.final_projection_local_recon(x)
        x = torch.sigmoid(x)

        return x

    def recon_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]  
        mask: [N, L], 0 is keep, 1 is remove,
        """
        loss = (pred - imgs) ** 2
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_train(self, local_patch, global_img):
        local_latent, local_mask = self.forward_encoder(
            local_patch, self.cfg.train.mask_ratio, self.cfg.train.local_mae_patch)
        local_pred = self.forward_local_decoder(local_latent)  # [N, L, p*p*3]
        local_loss = self.recon_loss(local_patch, local_pred, local_mask)

        global_latent, global_mask = self.forward_encoder(
            global_img, self.cfg.train.mask_ratio, self.cfg.train.global_mae_patch)
        global_pred = self.forward_local_decoder(
            global_latent)  # [N, L, p*p*3]
        global_loss = self.recon_loss(global_img, global_pred, global_mask)

        return local_loss, global_loss, local_pred, global_pred, local_mask, global_mask

    def forward(self, local_patch, global_img):
        local_latent, local_mask = self.forward_encoder(
            local_patch, self.cfg.train.mask_ratio, self.cfg.train.local_mae_patch)
        global_latent, global_mask = self.forward_encoder(
            global_img, self.cfg.train.mask_ratio, self.cfg.train.global_mae_patch)
        global_pred = self.forward_local_decoder(
            global_latent)  # [N, L, p*p*3]
        local_pred = self.forward_local_decoder(local_latent)  # [N, L, p*p*3]
        return local_pred, global_pred, local_mask, global_mask