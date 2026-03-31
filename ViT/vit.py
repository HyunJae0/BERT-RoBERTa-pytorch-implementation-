import math
import torch
import torch.nn as nn
from config import ViTConfig

class ViTPatchEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        assert self.image_size % self.patch_size == 0

        self.hidden_size = config.hidden_size 
        self.num_channels = config.num_channels
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels=self.num_channels,
            out_channels=self.hidden_size,
            kernel_size=self.patch_size,
            stride=self.patch_size
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pixel_values.shape
        assert C == self.num_channels and H == self.image_size and W == self.image_size

        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        # (B, hidden_size, H/P, W/P) -> (B, hidden_size, H/P x W/P = num_patches) -> (B, num_patches, hidden_size)
        return embeddings
    
    
class ViTEmbeddings(nn.Module):
    """
    이미지로부터 다음과 같은 시퀀스 만들기
    [CLS, x_1, x_2, ... , x_n] + position embeddings
    - CLS: 이미지 분류를 위한 special token
    - x_i: 각 패치의 임베딩
    """
    def __init__(self, config):
        super().__init__()
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.hidden_size = config.hidden_size

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_size)
        )
        self.position_embeddings = nn.Parameter(
            torch.randn(1, num_patches+1, self.hidden_size)
        )
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embeddings(pixel_values)

        B = embeddings.shape[0] # batch size
        
        # 패치 임베딩에 cls 토큰 더하기 
        cls_tokens = self.cls_token.expand(B, -1, -1)

        # 위치 임베딩 더하기
        embeddings = self.dropout(embeddings + self.position_embeddings)
        return embeddings
    

class ViTSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.hidden_size % config.num_attention_heads == 0

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.attention_probs_dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        B, seq_len, _ = x.shape

        x = x.view(B, seq_len, self.num_attention_heads, self.attention_head_size)
        x = x.permute(0, 2, 1, 3)
        return x
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query_layer = self.query(hidden_states)   # (B, seq_len, hidden_size)
        key_layer = self.key(hidden_states)       
        value_layer = self.value(hidden_states)   

        query_layer = self.transpose_for_scores(query_layer)   # (B, heads, seq_len, head_dim)
        key_layer = self.transpose_for_scores(key_layer)       
        value_layer = self.transpose_for_scores(value_layer)  

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2)
        )
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = torch.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_probs_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        B, seq_len, _, _ = context_layer.shape
        context_layer = context_layer.view(B, seq_len, -1)
        return context_layer


class ViTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states
    

class ViTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(hidden_states)
        attention_output = self.output(attention_output)
        return attention_output


class ViTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states
    

class ViTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ViTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layernorm_before = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )
        self.attention = ViTAttention(config)
        self.layernorm_after = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _residual = hidden_states
        hidden_states = self.layernorm_before(hidden_states)
        hidden_states = self.attention(hidden_states)
        hidden_states = _residual + hidden_states

        _residual = hidden_states
        hidden_states = self.layernorm_after(hidden_states)
        hidden_states = self.intermediate(hidden_states)
        hidden_states = self.output(hidden_states)
        hidden_states = _residual + hidden_states
        return hidden_states


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.layer = nn.ModuleList(
            [ViTLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states)

        return hidden_states


class ViTPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.pooler_output_size)
        self.activation = config.pooler_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        cls_token = hidden_states[:, 0]    # (B, hidden_size)
        pooled_output = self.dense(cls_token)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ViTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.embeddings = ViTEmbeddings(config)
        self.encoder = ViTEncoder(config)
        self.layernorm = nn.LayerNorm(
            config.hidden_size,
            eps=config.layer_norm_eps
        )
        self.pooler = ViTPooler(config)

    def forward(self, pixel_values):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.layernorm(hidden_states)

        pooled_output = self.pooler(hidden_states)
        return hidden_states, pooled_output
    
class ViTForImageClassification(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_labels = config.num_labels
        self.vit = ViTModel(config)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pixel_values, labels):
        sequence_output, _ = self.vit(pixel_values)
        
        cls_output = sequence_output[:, 0]   # (B, hidden_size)
        logits = self.classifier(cls_output) # (B, num_labels)

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, labels)
        return loss, logits


if __name__ == "__main__":
    config = ViTConfig(num_labels=10)
    model = ViTForImageClassification(config)

    x = torch.randn(2, 3, 224, 224)
    labels = torch.tensor([1, 3])

    loss, logits = model(x, labels)

    print(loss.shape)    # torch.Size([])
    print(logits.shape)
