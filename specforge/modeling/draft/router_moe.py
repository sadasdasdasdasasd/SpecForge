from typing import List, Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention
from transformers.cache_utils import Cache
from transformers.activations import ACT2FN
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig

from megatron.core.transformer.moe.router import (
    apply_router_token_dropping,
    topk_routing_with_score_function,
    compute_routing_scores_for_aux_loss,
)
from megatron.core.transformer.moe.moe_utils import (
    sinkhorn,
)

from specforge.distributed import get_tp_group
from specforge.layers import ColumnParallelLinear, RowParallelLinear

from .llama3_eagle import (
    LlamaForCausalLMEagle3,
    LlamaDecoderLayer,
    LlamaAttention,
    LlamaFlexAttention,
    LlamaFlashAttention,
    LlamaUSPAttention,
    LlamaMLP,
    LlamaRMSNorm,
    prepare_decoder_attention_mask,
)
from .base import Eagle3DraftModel
from .qwen3_moe_eagle import Qwen3MoeMLP, Qwen3MoeSparseMoeBlock


class Qwen3RoutMoeSparseMoeBlock(nn.Module):
    """
    适配 Megatron-LM 路由逻辑的 Qwen3 MoE 专家层
    核心优化：
    1. 完整对齐 Megatron 路由流程（Z-Loss + TopK/Sinkhorn + Token Dropping + 辅助损失）
    2. 补充张量形状校验和分布式打印
    3. 修复 Sinkhorn 负载均衡计算逻辑
    4. 优化专家计算效率（仅遍历被命中的专家）
    """
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        # 兼容默认参数（对齐 Megatron 配置）
        self.config = config
        self.num_experts = getattr(config, "num_experts", 16)
        self.top_k = getattr(config, "num_experts_per_tok", 8)
        self.norm_topk_prob = getattr(config, "norm_topk_prob", True)
        
        # Gating 层（与 Qwen 原生一致，兼容 TP 并行）
        self.gate = ColumnParallelLinear(
            config.hidden_size, self.num_experts, bias=False, gather_output=False
        ) if getattr(config, "tensor_model_parallel", False) else nn.Linear(
            config.hidden_size, self.num_experts, bias=False
        )
        
        # 专家层（复用 Qwen 原生 MLP，兼容 Megatron 分组 GEMM）
        self.experts = nn.ModuleList(
            [
                Qwen3MoeMLP(
                    config, 
                    intermediate_size=getattr(config, "moe_intermediate_size", config.intermediate_size),
                    grouped_gemm=getattr(config, "moe_grouped_gemm", False)
                )
                for _ in range(self.num_experts)
            ]
        )
        
        # Megatron 路由补充配置
        self.expert_bias = nn.Parameter(torch.zeros(self.num_experts)) if getattr(config, "moe_router_enable_expert_bias", False) else None
        self.router_replay = getattr(config, "moe_router_replay", False)
        self.z_loss_coeff = getattr(config, "moe_z_loss_coeff", 0.01)
        self.load_balancing_loss = None  # 存储负载均衡损失

    def apply_z_loss(self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        复用 Megatron Z-Loss 逻辑：防止路由 logits 过大导致数值不稳定
        """
        if padding_mask is not None:
            logits = logits * padding_mask.unsqueeze(-1)
        
        # Z-Loss 计算（均值归一化 + L2 惩罚）
        logits_mean = logits.mean(dim=-1, keepdim=True)
        logits_centered = logits - logits_mean
        z_loss = self.z_loss_coeff * torch.sum(logits_centered **2) / logits.numel()
        logits = logits - z_loss * logits  # 仅传递梯度，不改变数值分布
        
        return logits

    def _compute_load_balancing_loss(self, expert_load: torch.Tensor) -> torch.Tensor:
        """
        计算 Megatron 风格的负载均衡辅助损失
        """
        avg_load = expert_load.float().mean()
        load_var = (expert_load - avg_load).pow(2).mean()
        self.load_balancing_loss = load_var * self.config.moe_aux_loss_coeff
        return self.load_balancing_loss

    def sinkhorn_load_balancing(self, logits: torch.Tensor):
        """Apply sinkhorn routing to the logits tensor.

        Args:
            logits (torch.Tensor): The logits tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing token assignment
            probabilities and mask.
        """

        def _sinkhorn_activation(logits):
            if self.top_k == 1:
                logits = torch.sigmoid(logits)
            else:  # k > 1
                logits = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            return logits

        assert self.config.moe_aux_loss_coeff == 0, "Sinkhorn routing does not support aux loss."
        if self.training:
            with torch.no_grad():
                norm_logits = sinkhorn(
                    logits.to(dtype=torch.float32)
                )  # explicit fp32 conversion for stability
                _, indices = torch.topk(norm_logits, k=self.top_k, dim=1)
            logits = _sinkhorn_activation(logits)
        else:
            logits = _sinkhorn_activation(logits)
            _, indices = torch.topk(logits, k=self.top_k, dim=1)
        map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
        scores = logits * map
        return scores, map

    def routing(self, logits: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        核心路由逻辑：对齐 Megatron-LM 原生实现
        支持：TopK 路由 / Sinkhorn 负载均衡 / Token Dropping / 辅助损失
        """
        # 张量形状标准化：[seq_len, batch_size, num_experts] → [num_tokens, num_experts]
        seq_len, batch_size = logits.shape[:2]
        num_tokens = seq_len * batch_size
        logits_flat = logits.reshape(num_tokens, self.num_experts)
        
        # 分布式打印：原始 Logits 分布
        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            print(
                f"[Rank {rank}] Routing Input | "
                f"Logits (mean/std/max): {logits_flat.mean().item():.4f}/{logits_flat.std().item():.4f}/{logits_flat.max().item():.4f} | "
                f"Tokens: {num_tokens}, Experts: {self.num_experts}"
            )

        # Step 1: Z-Loss 稳定 logits
        logits_flat = self.apply_z_loss(logits_flat, padding_mask)

        # Step 2: 核心路由策略
        probs, routing_map = self.sinkhorn_load_balancing(logits_flat)
            
        # Step 3: Token Dropping（超出专家容量时丢弃）
        expert_load_before_drop = routing_map.sum(dim=0)
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs,
                routing_map,
                router_topk=self.top_k,
                capacity_factor=self.config.moe_expert_capacity_factor,
                drop_policy=self.config.moe_token_drop_policy,
                pad_to_capacity=self.config.moe_pad_expert_input_to_capacity,
            )

        # Step 4: 计算并打印负载均衡信息
        expert_load = routing_map.sum(dim=0)
        avg_load = expert_load.float().mean().item()
        load_std = expert_load.float().std().item()
        drop_ratio = 1.0 - (expert_load.sum() / expert_load_before_drop.sum()).item() if self.config.moe_expert_capacity_factor else 0.0
        
        if rank == 0:
            print(
                f"[Rank {rank}] Routing Output | "
                f"Expert Load (max/min/avg/std): {expert_load.max().item()}/{expert_load.min().item()}/{avg_load:.2f}/{load_std:.4f} | "
                f"Token Drop Ratio: {drop_ratio:.4f} | Probs (max/min/avg): {probs.max().item():.4f}/{probs.min().item():.4f}/{probs.mean().item():.4f}"
            )

        # Step 5: 训练时计算负载均衡辅助损失
        if self.training and torch.is_grad_enabled():
            self._compute_load_balancing_loss(expert_load)

        # Step 6: Qwen 原生概率归一化
        if self.norm_topk_prob:
            probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        return probs, routing_map

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播逻辑：
        1. 生成路由 logits → 2. Megatron 路由 → 3. 专家计算 → 4. 结果聚合
        """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # Step 1: 生成路由 Logits
        router_logits = self.gate(hidden_states_flat)
        router_logits_reshaped = router_logits.view(sequence_length, batch_size, self.num_experts)

        # Step 2: Megatron 风格路由
        probs, routing_map = self.routing(router_logits_reshaped, padding_mask=None)

        # Step 3: 解析路由结果（Top-K 索引 + 权重）
        topk_probs, selected_experts = torch.topk(probs, self.top_k, dim=-1)  # [num_tokens, top_k]
        num_tokens = probs.shape[0] 
        token_indices = torch.arange(num_tokens, device=probs.device).unsqueeze(1).expand(-1, self.top_k)
        routing_weights = probs[token_indices, selected_experts]  # [num_tokens, top_k]

        # Step 4: 专家计算（仅遍历被命中的专家，提升效率）
        final_hidden_states = torch.zeros_like(hidden_states_flat)
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)  # [num_experts, top_k, num_tokens]
        
        for expert_idx in torch.where(expert_mask.sum(dim=(-1, -2)) > 0)[0]:
            # 获取当前专家的 Token 索引
            top_k_idx, token_idx = torch.where(expert_mask[expert_idx])
            # 专家前向计算
            expert_output = self.experts[expert_idx](hidden_states_flat[token_idx])
            # 加权并聚合结果
            expert_output = expert_output * routing_weights[token_idx, top_k_idx].unsqueeze(-1)
            final_hidden_states.index_add_(0, token_idx, expert_output.to(final_hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(
            batch_size, sequence_length, hidden_dim
        )
        return final_hidden_states, router_logits


class Qwen3MoeDecoderLayer(LlamaDecoderLayer):
    """
    Qwen3 MoE 解码器层：
    1. 重构 Attention 实例化逻辑（兼容多后端）
    2. 对齐 Megatron MoE 损失计算
    3. 完善残差连接和归一化逻辑
    4. 补充张量形状校验
    """
    def __init__(self, config: Qwen3MoeConfig, attention_backend: str = "sdpa"):
        super().__init__(config, attention_backend)
        self.config = config
        self.hidden_size = config.hidden_size
        
        # 1. Attention 后端映射（简洁 + 校验）
        self.attention_backend = attention_backend.lower()
        attn_cls_map = {
            "sdpa": LlamaAttention,
            "usp": LlamaUSPAttention,
            "flex_attention": LlamaFlexAttention,
            "fa": LlamaFlashAttention
        }
        if self.attention_backend not in attn_cls_map:
            raise ValueError(f"Unsupported attention backend: {attention_backend}, supported: {list(attn_cls_map.keys())}")
        self.self_attn = attn_cls_map[self.attention_backend](config=config)

        # 2. MoE 专家层（核心）
        if self.config.routing_type == "sinkhorn":
            self.mlp = Qwen3RoutMoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeSparseMoeBlock(config)

        # 3. 归一化层（对齐 Qwen 官方 + Megatron 精度）
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 4. MoE 损失配置
        self.moe_loss_weight = getattr(config, "moe_loss_weight", 0.01)
        self.save_idx = 0

    def forward(
        self,
        input_emb: torch.Tensor,
        hidden_states: torch.Tensor,
        cache_hidden: List[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        return_router_logits: Optional[bool] = True,
    ) -> Tuple[torch.Tensor, Optional[Any], Optional[torch.Tensor]]:
        """
        前向传播：
        - 完善残差连接
        - 兼容多 Attention 后端输出
        - 计算 MoE 负载均衡损失
        """
        # 输入形状校验
        assert input_emb.shape[-1] == self.hidden_size, f"Input embedding size mismatch: {input_emb.shape[-1]} vs {self.hidden_size}"
        assert hidden_states.shape[-1] == self.hidden_size, f"Hidden states size mismatch: {hidden_states.shape[-1]} vs {self.hidden_size}"

        # 归一化 + 残差准备
        residual = hidden_states
        hidden_states = self.hidden_norm(hidden_states)
        input_emb = self.input_layernorm(input_emb)

        # 拼接输入 embedding 和隐藏状态（保持原有逻辑）
        hidden_states = torch.cat((input_emb, hidden_states), dim=-1)
        assert hidden_states.shape[-1] == 2 * self.hidden_size, f"Concatenated size error: {hidden_states.shape[-1]} vs {2*self.hidden_size}"

        # Self-Attention 层
        attn_outputs = self.self_attn(
            cache_hidden=cache_hidden,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        # 解析 Attention 输出（兼容单/多输出）
        if isinstance(attn_outputs, tuple):
            hidden_states = attn_outputs[0]
            attn_aux_output = attn_outputs[1:] if output_attentions else None
        else:
            hidden_states = attn_outputs
            attn_aux_output = None

        # Attention 残差连接
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.contiguous()

        # MoE MLP 层
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, router_logits = self.mlp(hidden_states)

        # 计算 MoE 负载均衡损失（训练时）
        moe_loss = None
        if self.training and hasattr(self.mlp, "load_balancing_loss") and self.mlp.load_balancing_loss is not None:
            moe_loss = self.mlp.load_balancing_loss * self.moe_loss_weight

        # MLP 残差连接
        hidden_states = residual + hidden_states
        hidden_states = hidden_states.contiguous()
        
        save_path = f"./router/{self.save_idx}.pt"
        self.save_idx += 1
        # 2. 保存张量（转CPU避免GPU张量依赖，detach解耦计算图）
        if 0 == dist.get_rank():
            torch.save(hidden_states.detach().cpu(), save_path)
        if self.save_idx >= 16:
            exit(0)

        return hidden_states



class Qwen3MoERouterForCausalLMEagle3(LlamaForCausalLMEagle3):

    def __init__(self, config, quant_config=None, attention_backend="sdpa") -> None:
        super().__init__(config, quant_config, attention_backend)
        self.config = config
        # Megatron 路由所需配置（可根据需求调整）
                # ========== 核心修正：对齐 Megatron-LM 原生配置命名 ==========
        # 读取配置的兼容处理：同时支持字典/类对象两种 config 格式
        def get_config_val(key, default):
            return getattr(config, key, default)

        self.config.moe_router_pre_softmax = get_config_val("moe_router_pre_softmax", False)
        self.config.moe_router_num_groups = get_config_val("moe_router_num_groups", None)
        self.config.moe_router_group_topk = get_config_val("moe_router_group_topk", None)
        self.config.moe_router_scaling_factor = get_config_val("moe_router_scaling_factor", 1.0)
        self.config.moe_router_score_function = get_config_val("moe_router_score_function", "softmax")
        self.config.moe_expert_capacity_factor = get_config_val("moe_expert_capacity_factor", None)  # 负载限制
        self.config.moe_token_drop_policy = get_config_val("moe_token_drop_policy", "none")  # 超出容量时的策略
        self.config.moe_pad_expert_input_to_capacity = get_config_val("moe_pad_expert_input_to_capacity", False)
        self.config.moe_aux_loss_coeff = get_config_val("moe_aux_loss_coeff", 0.01)  # 辅助损失系数
        # self.config.routing_type = get_config_val("routing_type", "sinkhorn")  # 可选: sinkhorn/aux_loss/seq_aux_loss/none
        self.config.routing_type = get_config_val("routing_type", "none")  # 可选: sinkhorn/aux_loss/seq_aux_loss/none
        self.config.moe_aux_loss_coeff = 0


        self.quant_config = quant_config
        self.vocab_size = config.vocab_size
        self.draft_vocab_size = config.draft_vocab_size
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.midlayer = Qwen3MoeDecoderLayer(config, attention_backend=attention_backend)

        if hasattr(config, "target_hidden_size"):
            self.fc = torch.nn.Linear(
                config.target_hidden_size * 3, config.hidden_size, bias=False
            )
        else:
            self.fc = torch.nn.Linear(
                config.hidden_size * 3, config.hidden_size, bias=False
            )

        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )

        # create vocab buffers
        t2d = torch.ones(self.vocab_size, dtype=torch.bool)
        d2t = torch.zeros(self.draft_vocab_size, dtype=torch.int64)
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)