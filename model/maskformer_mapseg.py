import random
from typing import Tuple, Union, List

import torch.nn as nn
import torch.nn.functional as F
import torch 
from einops import rearrange, repeat, reduce
from positional_encodings.torch_encodings import PositionalEncoding3D
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.initialization.weight_init import InitWeights_He

from .transformer_decoder import TransformerDecoder,TransformerDecoderLayer
from .SwinUNETR import SwinUNETR
from .mapseg_backbone import MAPSeg_MAE_Backbone
from .mapseg_blocks import create_encoders, ExtResNetBlock, _ntuple, res_decoders,DoubleConv, MPL
from cfg.default import get_cfg_defaults
# from .umamba_mid import UMambaMid

class Maskformer_MAPSeg(nn.Module):
    def __init__(self, vision_backbone='UNET', image_size=[288, 288, 96], patch_size=[32, 32, 32], 
                 deep_supervision=False, mapseg_pretrained_path=None, mapseg_embed_dim=512):
        """
        Args:
            vision_backbone (str, optional): visual backbone. Defaults to UNET.
            image_size (list, optional): image size. Defaults to [288, 288, 96].
            patch_size (list, optional): maxium downsample ratio of the bottleneck feature map. Defaults to [32, 32, 32].
            deep_supervision (bool, optional): seg results from mid layers of decoder. Defaults to False.
            mapseg_pretrained_path (str, optional): path to MAPSeg_MAE pretrained weights.
            mapseg_embed_dim (int, optional): embedding dimension for MAPSeg_MAE. Defaults to 512.
        """
        super().__init__()
        image_height, image_width, frames = image_size
        self.hw_patch_size = patch_size[0] 
        self.frame_patch_size = patch_size[-1]
        
        self.deep_supervision = deep_supervision
        self.vision_backbone_name = vision_backbone
        
        # backbone can be any multi-scale enc-dec vision backbone
        # the enc outputs multi-scale latent features
        # the dec outputs multi-scale per-pixel features
        if vision_backbone == 'MAPSeg_MAE':
            # Use MAPSeg_MAE backbone
            self.backbone = MAPSeg_MAE_Backbone(
                input_channels=3,
                embed_dim=mapseg_embed_dim,
                depth=6,
                image_size=image_size,
                deep_supervision=deep_supervision,
                pretrained_path=mapseg_pretrained_path
            )
        else:
            # Use original SAT backbones
            self.backbone = {
                'SwinUNETR' : SwinUNETR(
                                img_size=[288, 288, 96],    # 48, 48, 96, 192, 384, 768
                                in_channels=3,
                                feature_size=48,  
                                drop_rate=0.0,
                                attn_drop_rate=0.0,
                                dropout_path_rate=0.0,
                                use_checkpoint=False,
                                ),
                'UNET' : PlainConvUNet(input_channels=3, 
                                       n_stages=6, 
                                       features_per_stage=(64, 64, 128, 256, 512, 768), 
                                       conv_op=nn.Conv3d, 
                                       kernel_sizes=3, 
                                       strides=(1, 2, 2, 2, 2, 2), 
                                       n_conv_per_stage=(2, 2, 2, 2, 2, 2), 
                                       n_conv_per_stage_decoder=(2, 2, 2, 2, 2), 
                                       conv_bias=True, 
                                       norm_op=nn.InstanceNorm3d,
                                       norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                       dropout_op=None,
                                       dropout_op_kwargs=None,
                                       nonlin=nn.LeakyReLU, 
                                       nonlin_kwargs=None,
                                       deep_supervision=deep_supervision,
                                       nonlin_first=False
                                       ),
                'UNET-L' : PlainConvUNet(input_channels=3, 
                                       n_stages=6, 
                                       features_per_stage=(128, 128, 256, 512, 1024, 1536), 
                                       conv_op=nn.Conv3d, 
                                       kernel_sizes=3, 
                                       strides=(1, 2, 2, 2, 2, 2), 
                                       n_conv_per_stage=(3, 3, 3, 3, 3, 3), 
                                       n_conv_per_stage_decoder=(3, 3, 3, 3, 3), 
                                       conv_bias=True, 
                                       norm_op=nn.InstanceNorm3d,
                                       norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                       dropout_op=None,
                                       dropout_op_kwargs=None,
                                       nonlin=nn.LeakyReLU, 
                                       nonlin_kwargs=None,
                                       deep_supervision=deep_supervision,
                                       nonlin_first=False
                                       ),
                'UNET-H' : PlainConvUNet(input_channels=3, 
                                       n_stages=6, 
                                       features_per_stage=(256, 256, 512, 1024, 1536, 2048), 
                                       conv_op=nn.Conv3d, 
                                       kernel_sizes=3, 
                                       strides=(1, 2, 2, 2, 2, 2), 
                                       n_conv_per_stage=(3, 3, 3, 3, 3, 3), 
                                       n_conv_per_stage_decoder=(3, 3, 3, 3, 3), 
                                       conv_bias=True, 
                                       norm_op=nn.InstanceNorm3d,
                                       norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                                       dropout_op=None,
                                       dropout_op_kwargs=None,
                                       nonlin=nn.LeakyReLU, 
                                       nonlin_kwargs=None,
                                       deep_supervision=deep_supervision,
                                       nonlin_first=False
                                       ),
                # 'UMamba' : UMambaMid(
                #             input_channels=3,
                #             n_stages=6,
                #             features_per_stage=(64, 64, 128, 256, 512, 768),
                #             conv_op=nn.Conv3d,
                #             kernel_sizes=3,
                #             strides=(1, 2, 2, 2, 2, 2),
                #             n_conv_per_stage=(1, 1, 1, 1, 1, 1),
                #             n_conv_per_stage_decoder=(1, 1, 1, 1, 1),
                #             conv_bias=True,
                #             norm_op=nn.InstanceNorm3d,
                #             norm_op_kwargs={'eps': 1e-5, 'affine': True}, 
                #             dropout_op=None,
                #             dropout_op_kwargs=None,
                #             nonlin=nn.LeakyReLU, 
                #             nonlin_kwargs=None,
                #     ),
            }[vision_backbone]
            
            self.backbone.apply(InitWeights_He(1e-2))
        
        # fixed to text encoder out dim
        query_dim = 1536 if vision_backbone == 'UNET-H' else 768

        # all backbones are 6-depth, thus the first 5 scale latent feature outputs need to be down-sampled
        self.avg_pool_ls = [    
            nn.AvgPool3d(32, 32),
            nn.AvgPool3d(16, 16),
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(4, 4),
            nn.AvgPool3d(2, 2),
            ]
            
        # multi-scale latent feature are projected to query_dim before query decoder
        if vision_backbone == 'MAPSeg_MAE':
            # For MAPSeg_MAE, calculate concatenated feature dimension
            feature_dims = self.backbone.get_feature_dims()
            total_dim = feature_dims['total_concat']
            self.projection_layer = nn.Sequential(
                        nn.Linear(total_dim, 1536),  
                        nn.GELU(),
                        nn.Linear(1536, query_dim),
                        nn.GELU()
                    )
        else:
            # Original SAT projection layers
            self.projection_layer = {
                'SwinUNETR' : nn.Sequential(
                            nn.Linear(1536, 768),
                            nn.GELU(),
                            nn.Linear(768, query_dim),
                            nn.GELU()
                        ),
                'UNET' : nn.Sequential(
                            nn.Linear(1792, 768),
                            nn.GELU(),
                            nn.Linear(768, query_dim),
                            nn.GELU()
                        ),
                'UNET-L' : nn.Sequential(
                            nn.Linear(3584, 1536),  # 128, 128, 256, 512, 1024, 1536 --> 3584 --> 768
                            nn.GELU(),
                            nn.Linear(1536, query_dim),
                            nn.GELU()
                        ),
                'UNET-H' : nn.Sequential(
                            nn.Linear(5632, 3072),
                            nn.GELU(),
                            nn.Linear(3072, query_dim),
                            nn.GELU()
                        ),
                'UMamba' : nn.Sequential(
                            nn.Linear(1792, 768),  # 64, 64, 128, 256, 512, 768 --> 768
                            nn.GELU(),
                            nn.Linear(768, query_dim),
                            nn.GELU()
                        ),
            }[vision_backbone]
        
        # positional encoding
        pos_embedding = PositionalEncoding3D(query_dim)(torch.zeros(1, (image_height//self.hw_patch_size), (image_width//self.hw_patch_size), (frames//self.frame_patch_size), query_dim)) # b h/p w/p d/p dim
        self.pos_embedding = rearrange(pos_embedding, 'b h w d c -> (h w d) b c')   # n b dim
        
        # (fused latent embeddings + pe) x query prompts
        decoder_layer = TransformerDecoderLayer(d_model=query_dim, nhead=8, normalize_before=True)
        decoder_norm = nn.LayerNorm(query_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=6, norm=decoder_norm)
        
        self.query_proj = nn.Sequential(
                        nn.Linear(768, 1536),  # 64, 64, 128, 256, 512, 768 --> 768
                        nn.GELU(),
                        nn.Linear(1536, query_dim),
                        nn.GELU()
                    ) if query_dim > 768 else nn.Identity()
        
        # mask embedding are projected to perpixel_dim
        # mid stage output (only consider the last 3 mid layers of decoder, i.e. feature maps with resolution /2 /4 /8)
        if self.deep_supervision:
            if vision_backbone == 'MAPSeg_MAE':
                # For MAPSeg_MAE, use dimensions based on embed_dim
                feature_per_stage = [
                    mapseg_embed_dim // 8,   # 64 for embed_dim=512
                    mapseg_embed_dim // 4,   # 128
                    mapseg_embed_dim // 2,   # 256
                ]
                mid_dim = [256, 384, 512]
            else:
                feature_per_stage = {
                    'SwinUNETR':[48, 96, 192],
                    'UNET':[64, 128, 256],
                    'UNET-L':[128, 256, 512],
                    'UNET-H':[256, 512, 1024],
                    'UMamba':[64, 128, 256]
                    }[vision_backbone]
                mid_dim = {
                    'SwinUNETR':[256, 384, 512],
                    'UNET':[256, 384, 512],
                    'UNET-L':[384, 512, 512],
                    'UNET-H':[768, 1024, 1024],
                    'UMamba':[256, 384, 512]
                    }[vision_backbone]
                    
            self.mid_mask_embed_proj = []
            for hidden_dim, per_pixel_dim in zip(mid_dim, feature_per_stage):
                self.mid_mask_embed_proj.append(
                    nn.Sequential(
                        nn.Linear(query_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, per_pixel_dim),
                        nn.GELU(),
                        ),
                    )
            self.mid_mask_embed_proj = nn.ModuleList(self.mid_mask_embed_proj)
                
        # largest output        
        if vision_backbone == 'MAPSeg_MAE':
            mid_dim, per_pixel_dim = [256, mapseg_embed_dim // 8]  # [256, 64] for embed_dim=512
        else:
            mid_dim, per_pixel_dim = {
                'SwinUNETR' : [256, 48],
                'UNET' : [256, 64],
                'UNET-L' : [384, 128],
                'UNET-H' : [768, 256],
                'UMamba':[256, 64]
            }[vision_backbone]
            
        self.mask_embed_proj = nn.Sequential(
            nn.Linear(query_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, per_pixel_dim),
            nn.GELU(),
            )
            
    def vision_backbone_forward(self, image_input):
        """
        Visual backbone forward

        Args:
            image_input (torch.tensor): C,H,W,D (C=1)

        Returns:
            image_embedding (torch.tensor): multiscale image features from encoder layers. N,B,d
            pos (torch.tensor): position encoding. N,B,d
            per_pixel_embedding_ls (List of torch.tensor): perpixel embeddings from decoder layers. B,d,H,W,D
        """

        # Image Encoder and Pixel Decoder
        latent_embedding_ls, per_pixel_embedding_ls = self.backbone(image_input) # B Dim H/P W/P D/P
        for per_pixel_embedding in per_pixel_embedding_ls:
            print('per_pixel_embedding.shape:',per_pixel_embedding.shape)
        #print('per_pixel_embedding_ls[0].shape:',per_pixel_embedding_ls[0].shape)
        
        
        # avg pooling each multiscale feature to H/P W/P D/P
        image_embedding = []
        for latent_embedding, avg_pool in zip(latent_embedding_ls, self.avg_pool_ls):
            print('latent_embedding.shape:',latent_embedding.shape)
            tmp = avg_pool(latent_embedding)
            image_embedding.append(tmp)   # B ? H/P W/P D/P
            print('tmp.shape',tmp.shape)
        image_embedding.append(latent_embedding_ls[-1])
    
        # aggregate multiscale features into image embedding (and proj to align with query dim)
        image_embedding = torch.cat(image_embedding, dim=1)
        image_embedding = rearrange(image_embedding, 'b d h w depth -> b h w depth d')
        image_embedding = self.projection_layer(image_embedding)   # B H/P W/P D/P Dim
        image_embedding = rearrange(image_embedding, 'b h w d dim -> (h w d) b dim') # (H/P W/P D/P) B Dim
            
        # add pe to image embedding
        pos = self.pos_embedding.to(latent_embedding_ls[-1].device)   # (H/P W/P D/P) B Dim
        
        print('image_embedding.shape:',image_embedding.shape)
        print('pos.shape:',pos.shape)
            
        return image_embedding, pos, per_pixel_embedding_ls 
    
    def infer_forward(self, queries, image_embedding, pos, per_pixel_embedding_ls):
        """
        infer batches of queries (a list) on a batch of patches
        
        Args:
            queries (List of torch.tensor): N,d

        Returns:
            logits (torch.tensor): concat seg output of all queries. B,N_all,H,W,D
        """
        _, B, _ = image_embedding.shape
        
        logits_ls = []
        for q in queries:      
            # query decoder
            N,_ = q.shape    # N is the num of query
            q = repeat(q, 'n dim -> n b dim', b=B) # N B Dim NOTE:By default, attention in torch is not batch_first
            q = self.query_proj(q)
            
            # 计算text_query和image_embedding的余弦相似度，判断语义对齐程度
            # 假设q为text_query嵌入，image_embedding为视觉特征嵌入
            # q: N B Dim, image_embedding: (HWD) B Dim
            # 先对q和image_embedding做归一化
            q_norm = F.normalize(q, dim=-1)  # N B Dim
            img_emb_norm = F.normalize(image_embedding, dim=-1)  # (HWD) B Dim

            # 取平均池化后的image_embedding作为全局视觉语义
            global_img_emb = img_emb_norm.mean(dim=0)  # B Dim

            # 计算每个query与全局视觉语义的相似度
            # q_norm: N B Dim, global_img_emb: B Dim
            # 先交换q_norm为B N Dim
            q_norm_b = q_norm.permute(1, 0, 2)  # B N Dim
            # 计算余弦相似度
            sim = torch.einsum('bnd,bd->bn', q_norm_b, global_img_emb)  # B N

            # 可以根据sim的均值或阈值判断对齐程度
            avg_sim = sim.mean().item()
            if avg_sim > 0.5:
                print(f"[Info] Text queries and {self.vision_backbone_name} backbone are semantically aligned. Avg cosine sim: {avg_sim:.3f}")
            else:
                print(f"[Warning] Low semantic alignment between text queries and {self.vision_backbone_name} backbone. Avg cosine sim: {avg_sim:.3f}")

            mask_embedding,_ = self.transformer_decoder(q, image_embedding, pos = pos) # N B Dim
            mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
            # Dot product
            mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
            mask_embedding = rearrange(mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
            per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
            
            print('per_pixel_embedding.shape:',per_pixel_embedding.shape)
            print('mask_embedding.shape:',mask_embedding.shape)
            logits_ls.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mask_embedding)) # bnhwd
        logits = torch.concat(logits_ls, dim=1)  # bNhwd
        
        return logits
    
    def train_forward(self, queries, image_embedding, pos, per_pixel_embedding_ls):
        """
        Args:
            queries (torch.tensor): B,N,d

        Returns:
            logits (List of torch.tensor): list of seg results. B,N,H,W,D
        """
        _, B, _ = image_embedding.shape
        
        # query decoder
        _, N, _ = queries.shape    # N is the num of query
        queries = rearrange(queries, 'b n dim -> n b dim') # N B Dim NOTE:By default, attention in torch is not batch_first
        queries = self.query_proj(queries)
        mask_embedding,_ = self.transformer_decoder(queries, image_embedding, pos = pos) # N B Dim
        mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
        # Dot product
        last_mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
        last_mask_embedding = rearrange(last_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
        logits = [torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, last_mask_embedding)]
        
        # deep supervision
        if self.deep_supervision:
            for mask_embed_proj, per_pixel_embedding in zip(self.mid_mask_embed_proj, per_pixel_embedding_ls[1:]):  # H/2 --> H/16
                mid_mask_embedding = mask_embed_proj(mask_embedding)
                mid_mask_embedding = rearrange(mid_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
                logits.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mid_mask_embedding))
                
        return logits
    
    def forward(self, queries, image_input):
        # get vision features
        image_embedding, pos, per_pixel_embedding_ls = self.vision_backbone_forward(image_input)
        
        # Infer / Evaluate Forward ------------------------------------------------------------
        if isinstance(queries, List):
            del image_input
            torch.cuda.empty_cache()
            logits = self.infer_forward(queries, image_embedding, pos, per_pixel_embedding_ls)
            
        # Train Forward -----------------------------------------------------------------------
        else:
            logits = self.train_forward(queries, image_embedding, pos, per_pixel_embedding_ls)
        
        return logits
    
    def freeze_backbone_encoder(self):
        """Freeze the vision backbone encoder for fine-tuning"""
        if hasattr(self.backbone, 'freeze_encoder'):
            self.backbone.freeze_encoder()
        else:
            print(f"Warning: {self.vision_backbone_name} backbone doesn't support encoder freezing")
    
    def unfreeze_backbone_encoder(self):
        """Unfreeze the vision backbone encoder"""
        if hasattr(self.backbone, 'unfreeze_encoder'):
            self.backbone.unfreeze_encoder()
        else:
            print(f"Warning: {self.vision_backbone_name} backbone doesn't support encoder unfreezing")
    
    def get_backbone_encoder_params(self):
        """Get backbone encoder parameters for differential learning rates"""
        if hasattr(self.backbone, 'get_encoder_params'):
            return self.backbone.get_encoder_params()
        else:
            # For original SAT backbones, return all backbone parameters
            return self.backbone.parameters()
    
    def get_decoder_params(self):
        """Get all parameters except backbone encoder"""
        decoder_params = []
        
        # Add projection layer params
        decoder_params.extend(list(self.projection_layer.parameters()))
        
        # Add transformer decoder params
        decoder_params.extend(list(self.transformer_decoder.parameters()))
        decoder_params.extend(list(self.query_proj.parameters()))
        decoder_params.extend(list(self.mask_embed_proj.parameters()))
        
        if self.deep_supervision:
            for module in self.mid_mask_embed_proj:
                decoder_params.extend(list(module.parameters()))
        
        # Add backbone decoder params if available
        if hasattr(self.backbone, 'get_decoder_params'):
            decoder_params.extend(list(self.backbone.get_decoder_params()))
        
        return decoder_params



class Maskformer_MAPSegV2(nn.Module):
    def __init__(self, vision_backbone='UNET', image_size=[288, 288, 96], patch_size=[32, 32, 32], 
                 deep_supervision=False, mapseg_pretrained_path=None, mapseg_embed_dim=512):
        """
        Args:
            vision_backbone (str, optional): visual backbone. Defaults to UNET.
            image_size (list, optional): image size. Defaults to [288, 288, 96].
            patch_size (list, optional): maxium downsample ratio of the bottleneck feature map. Defaults to [32, 32, 32].
            deep_supervision (bool, optional): seg results from mid layers of decoder. Defaults to False.
            mapseg_pretrained_path (str, optional): path to MAPSeg_MAE pretrained weights.
            mapseg_embed_dim (int, optional): embedding dimension for MAPSeg_MAE. Defaults to 512.
        """
        super().__init__()
        image_height, image_width, frames = image_size
        # self.patch_size=patch_size
        self.patch_size=image_size
        self.hw_patch_size = patch_size[0] 
        self.frame_patch_size = patch_size[-1]
        
        self.deep_supervision = deep_supervision
        self.vision_backbone_name = vision_backbone
        
        # backbone can be any multi-scale enc-dec vision backbone
        # the enc outputs multi-scale latent features
        # the dec outputs multi-scale per-pixel features
        if vision_backbone == 'MAPSeg_MAE':
            # Use MAPSeg_MAE backbone
            # --------------------------------------------------------------------------
            # ResNet encoder specifics
            # self.cfg = cfg
            embed_dim = mapseg_embed_dim
            
                
        
            # Calculate expected feature dimensions based on embed_dim
            # feature_dims = {
            #     'scale_0': self.embed_dim // 8,   # 64 for embed_dim=512
            #     'scale_1': self.embed_dim // 4,   # 128
            #     'scale_2': self.embed_dim // 2,   # 256  
            #     'scale_3': self.embed_dim,        # 512
            #     'scale_4': self.embed_dim,        # 512
            #     'scale_5': self.embed_dim,        # 512 (bottleneck)
            #     'total_concat': (self.embed_dim // 8) + (self.embed_dim // 4) + (self.embed_dim // 2) + 
            #                 (self.embed_dim * 3)  # Total when concatenated
            # }
        
            # depth = cfg.model.depth
            depth=8
            # image_size=image_size,
            decoder_embed_dim = embed_dim // 16
            to_tuple = _ntuple(depth)
            print('to_tuple(embed_dim):',to_tuple(embed_dim))
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
        
        
        # fixed to text encoder out dim
        query_dim = 1536 if vision_backbone == 'UNET-H' else 768

        # all backbones are 6-depth, thus the first 5 scale latent feature outputs need to be down-sampled
        # self.avg_pool_ls = [    
        #     nn.AvgPool3d(32, 32),
        #     nn.AvgPool3d(16, 16),
        #     nn.AvgPool3d(8, 8),
        #     nn.AvgPool3d(4, 4),
        #     nn.AvgPool3d(2, 2),
        #     ]
        self.avg_pool_ls = [    
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(8, 8),
            ]
            
        # multi-scale latent feature are projected to query_dim before query decoder
        if vision_backbone == 'MAPSeg_MAE':
            # For MAPSeg_MAE, calculate concatenated feature dimension
            # feature_dims = self.backbone.get_feature_dims()
            # total_dim = feature_dims['total_concat']
            # self.projection_layer = nn.Sequential(
            #             nn.Linear(total_dim, 1536),  
            #             nn.GELU(),
            #             nn.Linear(1536, query_dim),
            #             nn.GELU()
            #         )
            self.projection_layer = nn.Sequential(
                            nn.LayerNorm(2560),
                            nn.Linear(2560, 1536),
                            nn.GELU(),
                            nn.Linear(1536, query_dim),
                            nn.GELU()
                        )
            # self.new_channel_transfer_module= nn.Sequential(
            #                 nn.Linear(2560, 1792),
            #                 nn.GELU()
            #             )
            # print('****')
            # self.vision_channel_expand_module=DoubleConv(16, 64, False, kernel_size=3, order='gcr', num_groups=8, padding=1)
            self.vision_channel_expand_module=nn.Sequential(
                                                nn.Conv3d(in_channels=16, out_channels=64, kernel_size=3, padding=1),
                                                nn.BatchNorm3d(64),
                                                nn.GELU(),
                                                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                                nn.BatchNorm3d(64),
                                                nn.GELU(),
                                                SqueezeExcite(64),
                                                nn.Conv3d(in_channels=64, out_channels=64, kernel_size=1),
                                                nn.BatchNorm3d(64),
                                                nn.GELU()
            )
            
       
        # positional encoding
        pos_embedding = PositionalEncoding3D(query_dim)(torch.zeros(1, (image_height//self.hw_patch_size), (image_width//self.hw_patch_size), (frames//self.frame_patch_size), query_dim)) # b h/p w/p d/p dim
        self.pos_embedding = rearrange(pos_embedding, 'b h w d c -> (h w d) b c')   # n b dim
        
        # (fused latent embeddings + pe) x query prompts
        decoder_layer = TransformerDecoderLayer(d_model=query_dim, nhead=8, normalize_before=True)
        decoder_norm = nn.LayerNorm(query_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=6, norm=decoder_norm)
        
        self.query_proj = nn.Sequential(
                        nn.Linear(768, 1536),  # 64, 64, 128, 256, 512, 768 --> 768
                        nn.GELU(),
                        nn.Linear(1536, query_dim),
                        nn.GELU()
                    ) if query_dim > 768 else nn.Identity()
        
        # mask embedding are projected to perpixel_dim
        # mid stage output (only consider the last 3 mid layers of decoder, i.e. feature maps with resolution /2 /4 /8)
        if self.deep_supervision:
            if vision_backbone == 'MAPSeg_MAE':
                # For MAPSeg_MAE, use dimensions based on embed_dim
                feature_per_stage = [
                    mapseg_embed_dim,   # 64 for embed_dim=512
                    mapseg_embed_dim,   # 128
                    mapseg_embed_dim,   # 256
                ]
                mid_dim = [1024, 1024, 1024]
            
                    
            self.mid_mask_embed_proj = []
            for hidden_dim, per_pixel_dim in zip(mid_dim, feature_per_stage):
                self.mid_mask_embed_proj.append(
                    nn.Sequential(
                        nn.LayerNorm(query_dim),
                        nn.Linear(query_dim, hidden_dim),
                        nn.GELU(),
                        nn.Linear(hidden_dim, per_pixel_dim),
                        nn.GELU(),
                        ),
                    )
            self.mid_mask_embed_proj = nn.ModuleList(self.mid_mask_embed_proj)
                
        # largest output        
        if vision_backbone == 'MAPSeg_MAE':
            mid_dim, per_pixel_dim = [256, mapseg_embed_dim // 8]  # [256, 64] for embed_dim=512
        
            
        self.mask_embed_proj = nn.Sequential(
            nn.Linear(query_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, per_pixel_dim),
            nn.GELU(),
            )
            
    def vision_backbone_forward(self, image_input):
        """
        Visual backbone forward

        Args:
            image_input (torch.tensor): C,H,W,D (C=1)

        Returns:
            image_embedding (torch.tensor): multiscale image features from encoder layers. N,B,d
            pos (torch.tensor): position encoding. N,B,d
            per_pixel_embedding_ls (List of torch.tensor): perpixel embeddings from decoder layers. B,d,H,W,D
        """

        
        global_latent, global_mask,latent_embedding_ls = self.forward_encoder(
            image_input, 0.7, 4)
        global_pred,per_pixel_embedding_ls = self.forward_local_decoder(global_latent)  # [N, L, p*p*3]
        # print('global_pred.shape:',global_pred.shape)
        # uni,uni_count=torch.unique(global_pred,return_counts=True)
        # mix_count=0
        # all_count=0
        # for uu,uu_cou in zip(uni,uni_count):
        #     if uu>0.5 and uu<0.65:
        #         # print(f'near 0.5:{uu}-number:{uu_cou}')
        #         mix_count+=uu_cou
        #     if uu>0.8:
        #         print(f'near 1:{uu}-number:{uu_cou}')
        #     if uu>0.5:
        #         all_count+=uu_cou
        # print('radios for mix:',mix_count/all_count)

        # Image Encoder and Pixel Decoder
        # latent_embedding_ls, per_pixel_embedding_ls = self.backbone(image_input) # B Dim H/P W/P D/P
        for per_pixel_embedding in per_pixel_embedding_ls:
            print('per_pixel_embedding encoder.shape:',per_pixel_embedding.shape)
        #print('per_pixel_embedding_ls[0].shape:',per_pixel_embedding_ls[0].shape)
        
        
        # avg pooling each multiscale feature to H/P W/P D/P
        image_embedding = []
        for latent_embedding, avg_pool in zip(latent_embedding_ls, self.avg_pool_ls):
            # print('latent_embedding.shape:',latent_embedding.shape)
            tmp = avg_pool(latent_embedding)
            image_embedding.append(tmp)   # B ? H/P W/P D/P
            # print('tmp.shape',tmp.shape)
        # image_embedding.append(latent_embedding_ls[-1])
        # print('latent_embedding_ls[-1]:',latent_embedding_ls[-1].shape)
    
        # aggregate multiscale features into image embedding (and proj to align with query dim)
        image_embedding = torch.cat(image_embedding, dim=1)
        image_embedding = rearrange(image_embedding, 'b d h w depth -> b h w depth d')
        # print('image_embedding:',image_embedding.shape)
        # ******add for channel project*****
        # image_embedding=self.new_channel_transfer_module(image_embedding)
        # **********
        image_embedding = self.projection_layer(image_embedding)   # B H/P W/P D/P Dim
        image_embedding = rearrange(image_embedding, 'b h w d dim -> (h w d) b dim') # (H/P W/P D/P) B Dim
            
        # add pe to image embedding
        pos = self.pos_embedding.to(latent_embedding_ls[-1].device)   # (H/P W/P D/P) B Dim
        
        # print('image_embedding.shape:',image_embedding.shape)
        # print('pos.shape:',pos.shape)
            
        return image_embedding, pos, per_pixel_embedding_ls, global_pred
    
    def infer_forward(self, queries, image_embedding, pos, per_pixel_embedding_ls):
        """
        infer batches of queries (a list) on a batch of patches
        
        Args:
            queries (List of torch.tensor): N,d

        Returns:
            logits (torch.tensor): concat seg output of all queries. B,N_all,H,W,D
        """
        _, B, _ = image_embedding.shape
        
        logits_ls = []
        for q in queries:      
            # query decoder
            N,_ = q.shape    # N is the num of query
            q = repeat(q, 'n dim -> n b dim', b=B) # N B Dim NOTE:By default, attention in torch is not batch_first
            q = self.query_proj(q)
            
            # 计算text_query和image_embedding的余弦相似度，判断语义对齐程度
            # 假设q为text_query嵌入，image_embedding为视觉特征嵌入
            # q: N B Dim, image_embedding: (HWD) B Dim
            # 先对q和image_embedding做归一化
            q_norm = F.normalize(q, dim=-1)  # N B Dim
            img_emb_norm = F.normalize(image_embedding, dim=-1)  # (HWD) B Dim

            # 取平均池化后的image_embedding作为全局视觉语义
            global_img_emb = img_emb_norm.mean(dim=0)  # B Dim

            # 计算每个query与全局视觉语义的相似度
            # q_norm: N B Dim, global_img_emb: B Dim
            # 先交换q_norm为B N Dim
            q_norm_b = q_norm.permute(1, 0, 2)  # B N Dim
            # 计算余弦相似度
            sim = torch.einsum('bnd,bd->bn', q_norm_b, global_img_emb)  # B N

            # 可以根据sim的均值或阈值判断对齐程度
            avg_sim = sim.mean().item()
            if avg_sim > 0.5:
                print(f"[Info] Text queries and {self.vision_backbone_name} backbone are semantically aligned. Avg cosine sim: {avg_sim:.3f}")
            else:
                print(f"[Warning] Low semantic alignment between text queries and {self.vision_backbone_name} backbone. Avg cosine sim: {avg_sim:.3f}")

            mask_embedding,_ = self.transformer_decoder(q, image_embedding, pos = pos) # N B Dim
            mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
            print('mask_embedding.shape:',mask_embedding.shape)
            # Dot product
            mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
            print('mask_embedding.shape:',mask_embedding.shape)
            mask_embedding = rearrange(mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
            per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
            print('per_pixel_embedding.shape:',per_pixel_embedding.shape)
            per_pixel_embedding=self.vision_channel_expand_module(per_pixel_embedding)
            # for per_pixel in per_pixel_embedding_ls:
            #     print('per_pixel.shape:',per_pixel.shape)
            print('per_pixel_embedding.shape:',per_pixel_embedding.shape)
            
            logits_ls.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mask_embedding)) # bnhwd
        logits = torch.concat(logits_ls, dim=1)  # bNhwd

        
        return logits
    
    def train_forward(self, queries, image_embedding, pos, per_pixel_embedding_ls):
        """
        Args:
            queries (torch.tensor): B,N,d

        Returns:
            logits (List of torch.tensor): list of seg results. B,N,H,W,D
        """
        _, B, _ = image_embedding.shape
        
        # query decoder
        _, N, _ = queries.shape    # N is the num of query
        queries = rearrange(queries, 'b n dim -> n b dim') # N B Dim NOTE:By default, attention in torch is not batch_first
        queries = self.query_proj(queries)
        mask_embedding,_ = self.transformer_decoder(queries, image_embedding, pos = pos) # N B Dim
        mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
        # Dot product
        last_mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
        last_mask_embedding = rearrange(last_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        per_pixel_embedding = per_pixel_embedding_ls[0] # decoder最后一层的输出
        # print('last per_pixel_embedding,shape:',per_pixel_embedding.shape)
        # print('last_mask_embedding,shape:',last_mask_embedding.shape)
        per_pixel_embedding=self.vision_channel_expand_module(per_pixel_embedding)
        
        logits = [torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, last_mask_embedding)]
        
        # deep supervision
        if self.deep_supervision:
            for mask_embed_proj, per_pixel_embedding in zip(self.mid_mask_embed_proj, per_pixel_embedding_ls[1:]):  # H/2 --> H/16
                mid_mask_embedding = mask_embed_proj(mask_embedding)
                mid_mask_embedding = rearrange(mid_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
                # print('deep per_pixel_embedding,shape:',per_pixel_embedding.shape)
                # print('last_mask_embedding,shape:',mid_mask_embedding.shape)
                logits.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mid_mask_embedding))
                
        return logits
    
    def forward(self, queries, image_input):
        # get vision features
        # print('image_input.shape:',image_input.shape)
        image_embedding, pos, per_pixel_embedding_ls, global_pred = self.vision_backbone_forward(image_input)
        
        # Infer / Evaluate Forward ------------------------------------------------------------
        if isinstance(queries, List):
            del image_input
            torch.cuda.empty_cache()
            logits = self.infer_forward(queries, image_embedding, pos, per_pixel_embedding_ls)
            
        # Train Forward -----------------------------------------------------------------------
        else:
            logits = self.train_forward(queries, image_embedding, pos, per_pixel_embedding_ls)
        
        return logits, global_pred
    
    # def freeze_backbone_encoder(self):
    #     """Freeze the vision backbone encoder for fine-tuning"""
    #     if hasattr(self.backbone, 'freeze_encoder'):
    #         self.backbone.freeze_encoder()
    #     else:
    #         print(f"Warning: {self.vision_backbone_name} backbone doesn't support encoder freezing")
    
    # def unfreeze_backbone_encoder(self):
    #     """Unfreeze the vision backbone encoder"""
    #     if hasattr(self.backbone, 'unfreeze_encoder'):
    #         self.backbone.unfreeze_encoder()
    #     else:
    #         print(f"Warning: {self.vision_backbone_name} backbone doesn't support encoder unfreezing")
    
    # def get_backbone_encoder_params(self):
    #     """Get backbone encoder parameters for differential learning rates"""
    #     if hasattr(self.backbone, 'get_encoder_params'):
    #         return self.backbone.get_encoder_params()
    #     else:
    #         # For original SAT backbones, return all backbone parameters
    #         return self.backbone.parameters()
    
    # def get_decoder_params(self):
    #     """Get all parameters except backbone encoder"""
    #     decoder_params = []
        
    #     # Add projection layer params
    #     decoder_params.extend(list(self.projection_layer.parameters()))
        
    #     # Add transformer decoder params
    #     decoder_params.extend(list(self.transformer_decoder.parameters()))
    #     decoder_params.extend(list(self.query_proj.parameters()))
    #     decoder_params.extend(list(self.mask_embed_proj.parameters()))
        
    #     if self.deep_supervision:
    #         for module in self.mid_mask_embed_proj:
    #             decoder_params.extend(list(module.parameters()))
        
    #     # Add backbone decoder params if available
    #     if hasattr(self.backbone, 'get_decoder_params'):
    #         decoder_params.extend(list(self.backbone.get_decoder_params()))
        
    #     return decoder_params

    # def __init__(self, cfg):
        # super().__init__()
        # # --------------------------------------------------------------------------
        # # ResNet encoder specifics
        # self.cfg = cfg
        # embed_dim = cfg.model.embed_dim
        # depth = cfg.model.depth
        # decoder_embed_dim = embed_dim // 16
        # to_tuple = _ntuple(depth)
        # # encoder
        # self.local_encoder = create_encoders(in_channels=1, f_maps=to_tuple(embed_dim), basic_module=ExtResNetBlock,
        #                                      conv_kernel_size=4, conv_stride_size=4, conv_padding=0, layer_order='gcr',
        #                                      num_groups=32)

        # # upsample
        # self.local_upsample = nn.ConvTranspose3d(in_channels=embed_dim, out_channels=decoder_embed_dim, kernel_size=4,
        #                                          stride=4)

        # # decoder
        # self.local_decoder = res_decoders(in_channels=decoder_embed_dim, f_maps=[16],
        #                                   basic_module=ExtResNetBlock, conv_kernel_size=3, conv_stride_size=1,
        #                                   conv_padding=0, layer_order='gcr', num_groups=8)

        # # norm layers
        # self.final_projection_local_recon = nn.Conv3d(
        #     in_channels=16, out_channels=1, kernel_size=3, padding=1)
        # self.final_norm_local_recon = nn.GroupNorm(
        #     num_groups=8, num_channels=16)

        # self.avgpool = nn.AdaptiveAvgPool3d((3, 1, 1))

    def patchify(self, imgs, p):
        """
        imgs: (N, 1, H, W, D)
        x: (N, H*W*D/P***3, patch_size**3)
        """
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0
        print('imgs.shape:',imgs.shape)
        h, w, d = [i//p for i in self.patch_size]
        print('h, w, d:',h, w, d)

        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p, d, p))
        x = torch.einsum('nchpwqdr->nhwdpqrc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p ** 3))
        return x

    def unpatchify(self, x, p):
        """
        x: (N, H*W*D/P***3, patch_size**3)
        imgs: (N, 1, H, W, D)
        """
        h, w, d = [i//p for i in self.patch_size]
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
        encoder_embedding=[]
        for blk in self.local_encoder:
            x = blk(x)
            encoder_embedding.append(x)

        return x, mask, encoder_embedding

    def forward_local_decoder(self, x):
        x = self.local_upsample(x)
        # apply Transformer blocks
        decoder_embedding=[]
        for blk in self.local_decoder:
            x = blk(x)
            decoder_embedding.append(x)

        
        x = self.final_norm_local_recon(x)
        
        x = self.final_projection_local_recon(x)
        
        x = torch.sigmoid(x)
        

        return x,decoder_embedding

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
            local_patch, 0.7, 8)
        local_pred = self.forward_local_decoder(local_latent)  # [N, L, p*p*3]
        local_loss = self.recon_loss(local_patch, local_pred, local_mask)

        global_latent, global_mask = self.forward_encoder(
            global_img, 0.7, 4)
        global_pred = self.forward_local_decoder(
            global_latent)  # [N, L, p*p*3]
        global_loss = self.recon_loss(global_img, global_pred, global_mask)

        return local_loss, global_loss, local_pred, global_pred, local_mask, global_mask

    # def forward(self, local_patch, global_img):
    #     local_latent, local_mask = self.forward_encoder(
    #         local_patch, self.cfg.train.mask_ratio, self.cfg.train.local_mae_patch)
    #     global_latent, global_mask = self.forward_encoder(
    #         global_img, self.cfg.train.mask_ratio, self.cfg.train.global_mae_patch)
    #     global_pred = self.forward_local_decoder(
    #         global_latent)  # [N, L, p*p*3]
    #     local_pred = self.forward_local_decoder(local_latent)  # [N, L, p*p*3]
    #     return local_pred, global_pred, local_mask, global_mask
    

class Maskformer_MPLSeg(nn.Module):
    def __init__(self, vision_backbone='UNET', image_size=[288, 288, 96], patch_size=[32, 32, 32], 
                 deep_supervision=False, mapseg_pretrained_path=None, mapseg_embed_dim=512,cfg_file="/cfg/example_yaml/X_site_finetune70_testtime.yaml"):
        """
        Args:
            vision_backbone (str, optional): visual backbone. Defaults to UNET.
            image_size (list, optional): image size. Defaults to [288, 288, 96].
            patch_size (list, optional): maxium downsample ratio of the bottleneck feature map. Defaults to [32, 32, 32].
            deep_supervision (bool, optional): seg results from mid layers of decoder. Defaults to False.
            mapseg_pretrained_path (str, optional): path to MAPSeg_MAE pretrained weights.
            mapseg_embed_dim (int, optional): embedding dimension for MAPSeg_MAE. Defaults to 512.
        """
        super().__init__()
        image_height, image_width, frames = image_size
        # self.patch_size=patch_size
        self.patch_size=image_size
        self.hw_patch_size = patch_size[0] 
        self.frame_patch_size = patch_size[-1]
        
        self.deep_supervision = deep_supervision
        self.vision_backbone_name = vision_backbone
        
        # backbone can be any multi-scale enc-dec vision backbone
        # the enc outputs multi-scale latent features
        # the dec outputs multi-scale per-pixel features
        if vision_backbone == 'MAPSeg_MAE':
            # Use MAPSeg_MAE backbone
            # --------------------------------------------------------------------------
            # ResNet encoder specifics
            # self.cfg = cfg
            embed_dim = mapseg_embed_dim
            
                
        

        
            # depth = cfg.model.depth
            depth=8
            # image_size=image_size,
            decoder_embed_dim = embed_dim // 16
            to_tuple = _ntuple(depth)
            print('to_tuple(embed_dim):',to_tuple(embed_dim))
            cfg = get_cfg_defaults()
            cfg.merge_from_file(cfg_file)
            self.mpl = MPL(cfg=cfg)

            self.avgpool = nn.AdaptiveAvgPool3d((3, 1, 1))
        
        
        # fixed to text encoder out dim
        query_dim = 1536 if vision_backbone == 'UNET-H' else 768

        # all backbones are 6-depth, thus the first 5 scale latent feature outputs need to be down-sampled
        # self.avg_pool_ls = [    
        #     nn.AvgPool3d(32, 32),
        #     nn.AvgPool3d(16, 16),
        #     nn.AvgPool3d(8, 8),
        #     nn.AvgPool3d(4, 4),
        #     nn.AvgPool3d(2, 2),
        #     ]
        self.avg_pool_ls = [    
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(8, 8),
            nn.AvgPool3d(8, 8),
            ]
            
        # multi-scale latent feature are projected to query_dim before query decoder
        if vision_backbone == 'MAPSeg_MAE':

            self.projection_layer = nn.Sequential(
                            nn.LayerNorm(2560),
                            nn.Linear(2560, 1536),
                            nn.GELU(),
                            nn.Linear(1536, query_dim),
                            nn.GELU()
                        )

       
        # positional encoding
        pos_embedding = PositionalEncoding3D(query_dim)(torch.zeros(1, (image_height//self.hw_patch_size), (image_width//self.hw_patch_size), (frames//self.frame_patch_size), query_dim)) # b h/p w/p d/p dim
        self.pos_embedding = rearrange(pos_embedding, 'b h w d c -> (h w d) b c')   # n b dim
        
        # (fused latent embeddings + pe) x query prompts
        decoder_layer = TransformerDecoderLayer(d_model=query_dim, nhead=8, normalize_before=True)
        decoder_norm = nn.LayerNorm(query_dim)
        self.transformer_decoder = TransformerDecoder(decoder_layer=decoder_layer, num_layers=6, norm=decoder_norm)
        
        self.query_proj = nn.Sequential(
                        nn.Linear(768, 1536),  # 64, 64, 128, 256, 512, 768 --> 768
                        nn.GELU(),
                        nn.Linear(1536, query_dim),
                        nn.GELU()
                    ) if query_dim > 768 else nn.Identity()
        
        # mask embedding are projected to perpixel_dim
        # mid stage output (only consider the last 3 mid layers of decoder, i.e. feature maps with resolution /2 /4 /8)
        # if self.deep_supervision:
        #     if vision_backbone == 'MAPSeg_MAE':
        #         # For MAPSeg_MAE, use dimensions based on embed_dim
        #         feature_per_stage = [
        #             mapseg_embed_dim,   # 64 for embed_dim=512
        #             mapseg_embed_dim,   # 128
        #             mapseg_embed_dim,   # 256
        #         ]
        #         mid_dim = [1024, 1024, 1024]
            
                    
        #     self.mid_mask_embed_proj = []
        #     for hidden_dim, per_pixel_dim in zip(mid_dim, feature_per_stage):
        #         self.mid_mask_embed_proj.append(
        #             nn.Sequential(
        #                 nn.LayerNorm(query_dim),
        #                 nn.Linear(query_dim, hidden_dim),
        #                 nn.GELU(),
        #                 nn.Linear(hidden_dim, per_pixel_dim),
        #                 nn.GELU(),
        #                 ),
        #             )
        #     self.mid_mask_embed_proj = nn.ModuleList(self.mid_mask_embed_proj)
                
        # largest output        
        if vision_backbone == 'MAPSeg_MAE':
            # mid_dim, per_pixel_dim = [256, mapseg_embed_dim // 8]  # [256, 64] for embed_dim=512
            mid_dim, per_pixel_dim = [512, mapseg_embed_dim // 8]  # [256, 64] 
        
            
        self.mask_embed_proj = nn.Sequential(
            nn.Linear(query_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, per_pixel_dim),
            nn.GELU(),
            )
            
    def vision_backbone_forward(self, image_input,local_patch, coordinates):
        """
        Visual backbone forward

        Args:
            image_input (torch.tensor): C,H,W,D (C=1)

        Returns:
            image_embedding (torch.tensor): multiscale image features from encoder layers. N,B,d
            pos (torch.tensor): position encoding. N,B,d
            per_pixel_embedding_ls (List of torch.tensor): perpixel embeddings from decoder layers. B,d,H,W,D
        """

        local_mask, global_aux,latent_embedding_ls,per_pixel_embedding_ls = self.mpl(local_patch=local_patch, global_img=image_input,coordinates=coordinates)
        print('local_mask.shape:',local_mask.shape)

        # avg pooling each multiscale feature to H/P W/P D/P
        image_embedding = []
        for latent_embedding, avg_pool in zip(latent_embedding_ls, self.avg_pool_ls):
            # print('latent_embedding.shape:',latent_embedding.shape)
            tmp = avg_pool(latent_embedding)
            image_embedding.append(tmp)   # B ? H/P W/P D/P

        # aggregate multiscale features into image embedding (and proj to align with query dim)
        image_embedding = torch.cat(image_embedding, dim=1)
        image_embedding = rearrange(image_embedding, 'b d h w depth -> b h w depth d')

        image_embedding = self.projection_layer(image_embedding)   # B H/P W/P D/P Dim

        image_embedding = rearrange(image_embedding, 'b h w d dim -> (h w d) b dim') # (H/P W/P D/P) B Dim
            
        # add pe to image embedding
        pos = self.pos_embedding.to(latent_embedding_ls[-1].device)   # (H/P W/P D/P) B Dim
        
            
        return image_embedding, pos, per_pixel_embedding_ls,local_mask
    
    def infer_forward(self, queries, image_embedding, pos, per_pixel_embedding_ls):
        """
        infer batches of queries (a list) on a batch of patches
        
        Args:
            queries (List of torch.tensor): N,d

        Returns:
            logits (torch.tensor): concat seg output of all queries. B,N_all,H,W,D
        """
        _, B, _ = image_embedding.shape
        
        logits_ls = []
        for q in queries:      
            # query decoder
            N,_ = q.shape    # N is the num of query
            q = repeat(q, 'n dim -> n b dim', b=B) # N B Dim NOTE:By default, attention in torch is not batch_first
            q = self.query_proj(q)
 
            mask_embedding,_ = self.transformer_decoder(q, image_embedding, pos = pos) # N B Dim
            mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
            # print('mask_embedding.shape:',mask_embedding.shape)
            # Dot product
            mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
            # print('mask_embedding.shape:',mask_embedding.shape)
            mask_embedding = rearrange(mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
            per_pixel_embedding = per_pixel_embedding_ls[-1] # decoder最后一层的输出

            logits_ls.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mask_embedding)) # bnhwd
        logits = torch.concat(logits_ls, dim=1)  # bNhwd

        
        return logits
    
    def train_forward(self, queries, image_embedding, pos, per_pixel_embedding_ls):
        """
        Args:
            queries (torch.tensor): B,N,d

        Returns:
            logits (List of torch.tensor): list of seg results. B,N,H,W,D
        """
        _, B, _ = image_embedding.shape
        
        # query decoder
        _, N, _ = queries.shape    # N is the num of query
        queries = rearrange(queries, 'b n dim -> n b dim') # N B Dim NOTE:By default, attention in torch is not batch_first
        queries = self.query_proj(queries)
        # print('queries.shape:',queries.shape)
        mask_embedding,_ = self.transformer_decoder(queries, image_embedding, pos = pos) # N B Dim
        mask_embedding = rearrange(mask_embedding, 'n b dim -> (b n) dim') # (B N) Dim
        # Dot product
        last_mask_embedding = self.mask_embed_proj(mask_embedding)   # 768 -> 128/64/48
        last_mask_embedding = rearrange(last_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        per_pixel_embedding = per_pixel_embedding_ls[-1] # decoder最后一层的输出

        logits = [torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, last_mask_embedding)]
        
        # deep supervision
        # if self.deep_supervision:
        #     for mask_embed_proj, per_pixel_embedding in zip(self.mid_mask_embed_proj, per_pixel_embedding_ls[1:]):  # H/2 --> H/16
        #         mid_mask_embedding = mask_embed_proj(mask_embedding)
        #         mid_mask_embedding = rearrange(mid_mask_embedding, '(b n) dim -> b n dim', b=B, n=N)
        #         # print('deep per_pixel_embedding,shape:',per_pixel_embedding.shape)
        #         # print('last_mask_embedding,shape:',mid_mask_embedding.shape)
        #         logits.append(torch.einsum('bchwd,bnc->bnhwd', per_pixel_embedding, mid_mask_embedding))
                
        return logits
    
    def forward(self, queries, image_input,local_patch, coordinates):
        # get vision features
        # print('image_input.shape:',image_input.shape)
        image_embedding, pos, per_pixel_embedding_ls,local_mask= self.vision_backbone_forward(image_input,local_patch, coordinates)
        
        # Infer / Evaluate Forward ------------------------------------------------------------
        if isinstance(queries, List):
            del image_input
            torch.cuda.empty_cache()
            logits = self.infer_forward(queries, image_embedding, pos, per_pixel_embedding_ls)
            
        # Train Forward -----------------------------------------------------------------------
        else:
            logits = self.train_forward(queries, image_embedding, pos, per_pixel_embedding_ls)
        
        return logits,local_mask


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # self.is_3d = is_3d
        Pool = nn.AdaptiveAvgPool3d
        self.pool = Pool(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, max(channels // reduction, 8)),
            nn.GELU(),
            nn.Linear(max(channels // reduction, 8), channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[:2]
        w = self.pool(x).view(b, c)
        w = self.fc(w).view(b, c, *([1] * (x.dim() - 2)))
        return x * w

if __name__ == '__main__':
    # Test original backbone
    model = Maskformer_MAPSeg(vision_backbone='UNET').cuda()    
    image = torch.rand((1, 3, 288, 288, 96)).cuda()
    query = torch.rand((2, 10, 768)).cuda()
    segmentations = model(query, image)
    print("UNET backbone:", segmentations.shape if isinstance(segmentations, torch.Tensor) else [s.shape for s in segmentations])
    
    # Test MAPSeg_MAE backbone
    model_mapseg = Maskformer_MAPSeg(vision_backbone='MAPSeg_MAE', mapseg_embed_dim=512).cuda()
    segmentations_mapseg = model_mapseg(query, image)
    print("MAPSeg_MAE backbone:", segmentations_mapseg.shape if isinstance(segmentations_mapseg, torch.Tensor) else [s.shape for s in segmentations_mapseg])