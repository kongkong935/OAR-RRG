from transformers import Blip2QFormerConfig, Blip2QFormerModel
import torch
import torch.nn as nn


class SimpleQFormerWrapper(nn.Module):
    def __init__(self, num_hidden_layers=6, hidden_size=768, num_attention_heads=12, num_query_tokens=32,
                 use_separate_queries=False):
        super().__init__()
        config = Blip2QFormerConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            vocab_size=1,  # 你其实不做LM，这里随便
            cross_attention_frequency=1,  # 每层都带 cross-attn
            encoder_hidden_size=hidden_size,  # KV 的 dim
        )
        self.qformer = Blip2QFormerModel(config)
        self.use_separate_queries = use_separate_queries

        if use_separate_queries:
            # 两个独立的 query token，每个有 num_query_tokens 个 token
            self.query_tokens_front = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
            self.query_tokens_lateral = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))
        else:
            # 单个 query token
            self.query_tokens = nn.Parameter(torch.randn(1, num_query_tokens, hidden_size))

    def forward(self, encoder_hidden_states, encoder_attention_mask=None):
        """
        encoder_hidden_states: 视觉编码器输出，形状 [B, N, H]
        encoder_attention_mask: attention mask，形状 [B, N]

        如果 use_separate_queries=True，期望 encoder_hidden_states 包含两个图像的特征拼接
        """
        B = encoder_hidden_states.size(0)

        if self.use_separate_queries:
            # 分别查询正面和侧面图像
            num_patches_per_image = encoder_hidden_states.size(1) // 2  # 假设两个图像拼接

            # 第一个 query token 查询前 num_patches_per_image 个 patch（正面）
            front_features = encoder_hidden_states[:, :num_patches_per_image, :]  # [B, N/2, H]
            front_mask = encoder_attention_mask[:,
                         :num_patches_per_image] if encoder_attention_mask is not None else None
            query_tokens_front = self.query_tokens_front.expand(B, -1, -1)  # [B, num_query, H]
            outputs_front = self.qformer(
                query_embeds=query_tokens_front,
                encoder_hidden_states=front_features,
                encoder_attention_mask=front_mask,
                return_dict=True,
            )
            front_query_output = outputs_front.last_hidden_state  # [B, num_query, H]

            # 第二个 query token 查询后 num_patches_per_image 个 patch（侧面）
            lateral_features = encoder_hidden_states[:, num_patches_per_image:, :]  # [B, N/2, H]
            lateral_mask = encoder_attention_mask[:,
                           num_patches_per_image:] if encoder_attention_mask is not None else None
            query_tokens_lateral = self.query_tokens_lateral.expand(B, -1, -1)  # [B, num_query, H]
            outputs_lateral = self.qformer(
                query_embeds=query_tokens_lateral,
                encoder_hidden_states=lateral_features,
                encoder_attention_mask=lateral_mask,
                return_dict=True,
            )
            lateral_query_output = outputs_lateral.last_hidden_state  # [B, num_query, H]

            # 拼接两个 query token 的输出
            combined_output = torch.cat([front_query_output, lateral_query_output], dim=1)  # [B, 2*num_query, H]
            return combined_output
        else:
            # 原始方式：单个 query token
            query_tokens = self.query_tokens.expand(B, -1, -1)  # [B, num_query, H]

            outputs = self.qformer(
                query_embeds=query_tokens,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                return_dict=True,
            )
            # 输出的 last_hidden_state 就是更新后的 query token: [B, num_query, H]
            return outputs.last_hidden_state
