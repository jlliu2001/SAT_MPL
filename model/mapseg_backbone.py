import torch
import torch.nn as nn
from .mae_cnn_adapted import MAE_CNN_Adapted
from dynamic_network_architectures.initialization.weight_init import InitWeights_He


class MAPSeg_MAE_Backbone(nn.Module):
    """
    MAPSeg_MAE backbone adapted for SAT framework
    
    This class serves as an adapter between MAPSeg_MAE's MAE_CNN and SAT's expected interface.
    It provides multi-scale feature extraction compatible with SAT's Maskformer architecture.
    """
    
    def __init__(self, input_channels=3, embed_dim=512, depth=6, image_size=[288, 288, 96], 
                 deep_supervision=False, pretrained_path=None):
        """
        Args:
            input_channels: number of input channels (3 for SAT)
            embed_dim: embedding dimension of the encoder
            depth: depth of the encoder network
            image_size: input image size [H, W, D]
            deep_supervision: whether to use deep supervision
            pretrained_path: path to pretrained MAPSeg_MAE weights
        """
        super().__init__()
        
        self.deep_supervision = deep_supervision
        self.embed_dim = embed_dim
        self.image_size = image_size
        
        # Initialize the adapted MAE_CNN
        self.mae_cnn = MAE_CNN_Adapted(
            input_channels=input_channels,
            embed_dim=embed_dim,
            depth=depth,
            image_size=image_size
        )
        
        # Apply He initialization
        self.mae_cnn.apply(InitWeights_He(1e-2))
        
        # Load pretrained weights if provided
        if pretrained_path is not None:
            print('MAP_MAE loading weights!')
            self.load_pretrained_weights(pretrained_path)
    
    def load_pretrained_weights(self, pretrained_path):
        """
        Load pretrained MAPSeg_MAE weights with proper key mapping
        """
        print(f"Loading MAPSeg_MAE pretrained weights from: {pretrained_path}")
        
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            
            # Map original MAPSeg_MAE keys to our adapted structure
            adapted_state_dict = {}
            
            for key, value in state_dict.items():
                # Remove 'module.' prefix if present (from DataParallel)
                if key.startswith('module.'):
                    key = key[7:]
                
                # Map encoder weights
                if key.startswith('local_encoder'):
                    # Map to our encoder structure
                    print('startswith local_encoder:',key)
                    new_key = key.replace('local_encoder', 'encoders')
                    adapted_state_dict[new_key] = value
                
                # Map decoder weights  
                elif key.startswith('local_decoder'):
                    print('startswith local_decoder:',key)
                    new_key = key.replace('local_decoder', 'decoders')
                    adapted_state_dict[new_key] = value
                
                # Map other components
                elif key.startswith('local_upsample'):
                    # Skip upsampling layer as our structure is different
                    continue
                    
                elif key.startswith('final_'):
                    new_key = key.replace('final_projection_local_recon', 'final_conv')
                    if 'final_conv' in new_key:
                        adapted_state_dict[new_key] = value
                
                else:
                    # Keep other keys as-is
                    adapted_state_dict[f'mae_cnn.{key}'] = value
            
            # Load the adapted weights with strict=False to handle missing keys
            missing_keys, unexpected_keys = self.mae_cnn.load_state_dict(adapted_state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys in pretrained weights: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys in pretrained weights: {unexpected_keys}")
                
            print("Successfully loaded MAPSeg_MAE pretrained weights")
            
        except Exception as e:
            print(f"Warning: Failed to load pretrained weights: {e}")
            print("Continuing with random initialization...")
    
    def forward(self, x):
        """
        Forward pass compatible with SAT's expected interface
        
        Args:
            x: input tensor [B, C, H, W, D]
            
        Returns:
            tuple: (latent_embedding_ls, per_pixel_embedding_ls)
                - latent_embedding_ls: list of encoder features at 6 different scales
                - per_pixel_embedding_ls: list of decoder features at 5 different scales
        """
        latent_embeddings, per_pixel_embeddings = self.mae_cnn(x)
        #print('per_pixel_embeddings 1.shape:',per_pixel_embeddings.shape)
        
        # Ensure we have exactly 6 latent embeddings and 5 per-pixel embeddings for SAT
        # Pad or trim if necessary
        while len(latent_embeddings) < 6:
            # Duplicate the last embedding if we don't have enough
            latent_embeddings.append(latent_embeddings[-1])
        
        while len(per_pixel_embeddings) < 5:
            # Duplicate the last embedding if we don't have enough  
            per_pixel_embeddings.append(per_pixel_embeddings[-1])
        
        # Trim to exactly the required number
        latent_embeddings = latent_embeddings[:6]
        #print('per_pixel_embeddings 2.shape:',per_pixel_embeddings.shape)
        if self.deep_supervision:
            per_pixel_embeddings = per_pixel_embeddings[:5]
        else:
            per_pixel_embeddings=[per_pixel_embeddings[-1]]
        
        return latent_embeddings, per_pixel_embeddings
    
    def freeze_encoder(self):
        """Freeze encoder parameters for fine-tuning"""
        for param in self.mae_cnn.encoders.parameters():
            param.requires_grad = False
        print("MAPSeg_MAE encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.mae_cnn.encoders.parameters():
            param.requires_grad = True
        print("MAPSeg_MAE encoder unfrozen")
    
    def freeze_decoder(self):
        """Freeze decoder parameters"""
        for param in self.mae_cnn.decoders.parameters():
            param.requires_grad = False
        print("MAPSeg_MAE decoder frozen")
    
    def unfreeze_decoder(self):
        """Unfreeze decoder parameters"""  
        for param in self.mae_cnn.decoders.parameters():
            param.requires_grad = True
        print("MAPSeg_MAE decoder unfrozen")
    
    def get_encoder_params(self):
        """Get encoder parameters for differential learning rates"""
        return self.mae_cnn.encoders.parameters()
    
    def get_decoder_params(self):
        """Get decoder parameters for differential learning rates"""
        return self.mae_cnn.decoders.parameters()
    
    def get_feature_dims(self):
        """
        Get feature dimensions for SAT projection layer configuration
        
        Returns:
            dict: feature dimensions at different scales
        """
        # Calculate expected feature dimensions based on embed_dim
        feature_dims = {
            'scale_0': self.embed_dim // 8,   # 64 for embed_dim=512
            'scale_1': self.embed_dim // 4,   # 128
            'scale_2': self.embed_dim // 2,   # 256  
            'scale_3': self.embed_dim,        # 512
            'scale_4': self.embed_dim,        # 512
            'scale_5': self.embed_dim,        # 512 (bottleneck)
            'total_concat': (self.embed_dim // 8) + (self.embed_dim // 4) + (self.embed_dim // 2) + 
                           (self.embed_dim * 3)  # Total when concatenated
        }
        return feature_dims


def create_mapseg_backbone(vision_backbone='MAPSeg_MAE', image_size=[288, 288, 96], 
                          deep_supervision=False, pretrained_path=None, **kwargs):
    """
    Factory function to create MAPSeg_MAE backbone
    
    Args:
        vision_backbone: backbone type (should be 'MAPSeg_MAE')
        image_size: input image size
        deep_supervision: whether to use deep supervision
        pretrained_path: path to pretrained weights
        **kwargs: additional arguments
        
    Returns:
        MAPSeg_MAE_Backbone: configured backbone
    """
    if vision_backbone != 'MAPSeg_MAE':
        raise ValueError(f"Expected 'MAPSeg_MAE', got {vision_backbone}")
    
    # Extract MAPSeg specific parameters
    embed_dim = kwargs.get('embed_dim', 512)
    depth = kwargs.get('depth', 6)
    input_channels = kwargs.get('input_channels', 3)
    
    backbone = MAPSeg_MAE_Backbone(
        input_channels=input_channels,
        embed_dim=embed_dim, 
        depth=depth,
        image_size=image_size,
        deep_supervision=deep_supervision,
        pretrained_path=pretrained_path
    )
    
    return backbone