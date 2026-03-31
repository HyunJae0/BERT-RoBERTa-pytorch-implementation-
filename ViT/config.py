import torch
import torch.nn as nn 

class ViTConfig:
    def __init__(
            self,
            hidden_size: int = 768,
            num_hidden_layers: int = 12,
            num_attention_heads: int = 12,
            intermediate_size: int = 3072,
            hidden_dropout_prob: float = 0.0,
            attention_probs_dropout_prob: float = 0.0,
            initializer_range: float = 0.02,
            layer_norm_eps: float = 1e-12,
            image_size: int = 224,
            patch_size: int = 16,
            num_channels: int = 3,
            qkv_bias: bool = True,
            pooler_output_size: int =768,
            pooler_act = nn.Tanh(),
            num_labels: int = 1000, # for image classification
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    ):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.qkv_bias = qkv_bias
        self.pooler_output_size = pooler_output_size
        self.pooler_act =  pooler_act
        self.num_labels = num_labels 
        self.device = device