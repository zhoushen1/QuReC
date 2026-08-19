import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from timm.models.layers import DropPath
from einops import rearrange
import swattention
CUDA_NUM_THREADS = 128

class sw_qkrpb_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, rpb, height, width, kernel_size):
        attn_weight = swattention.qk_rpb_forward(
            query, key, rpb, height, width, kernel_size, CUDA_NUM_THREADS
        )
        ctx.save_for_backward(query, key)
        ctx.height, ctx.width, ctx.kernel_size = height, width, kernel_size
        return attn_weight

    @staticmethod
    def backward(ctx, d_attn_weight):
        query, key = ctx.saved_tensors
        height, width, kernel_size = ctx.height, ctx.width, ctx.kernel_size
        d_query, d_key, d_rpb = swattention.qk_rpb_backward(
            d_attn_weight.contiguous(), query, key, height, width, kernel_size, CUDA_NUM_THREADS
        )
        return d_query, d_key, d_rpb, None, None, None


class sw_av_cuda(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_weight, value, height, width, kernel_size):
        output = swattention.av_forward(
            attn_weight, value, height, width, kernel_size, CUDA_NUM_THREADS
        )
        ctx.save_for_backward(attn_weight, value)
        ctx.height, ctx.width, ctx.kernel_size = height, width, kernel_size
        return output

    @staticmethod
    def backward(ctx, d_output):
        attn_weight, value = ctx.saved_tensors
        height, width, kernel_size = ctx.height, ctx.width, ctx.kernel_size
        d_attn_weight, d_value = swattention.av_backward(
            d_output.contiguous(), attn_weight, value, height, width, kernel_size, CUDA_NUM_THREADS
        )
        return d_attn_weight, d_value, None, None, None


class TX_DWConv(nn.Module):
    def __init__(self, dim=768):
        super(TX_DWConv, self).__init__()
        self.dwconv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TX_ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.fc1 = nn.Linear(in_features, hidden_features * 2)
        self.dwconv = TX_DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x, v = self.fc1(x).chunk(2, dim=-1)
        x = self.act(self.dwconv(x, H, W)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


@torch.no_grad()
def get_relative_position_cpb(query_size, key_size, pretrain_size=None,
                              device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    pretrain_size = pretrain_size or query_size

    axis_qh = torch.arange(query_size[0], dtype=torch.float32, device=device)
    axis_kh = F.adaptive_avg_pool1d(axis_qh.unsqueeze(0), key_size[0]).squeeze(0)

    axis_qw = torch.arange(query_size[1], dtype=torch.float32, device=device)
    axis_kw = F.adaptive_avg_pool1d(axis_qw.unsqueeze(0), key_size[1]).squeeze(0)

    axis_kh, axis_kw = torch.meshgrid(axis_kh, axis_kw, indexing="ij")
    axis_qh, axis_qw = torch.meshgrid(axis_qh, axis_qw, indexing="ij")

    axis_kh = torch.reshape(axis_kh, [-1])
    axis_kw = torch.reshape(axis_kw, [-1])
    axis_qh = torch.reshape(axis_qh, [-1])
    axis_qw = torch.reshape(axis_qw, [-1])

    denom_h = max(pretrain_size[0] - 1, 1)
    denom_w = max(pretrain_size[1] - 1, 1)

    relative_h = (axis_qh[:, None] - axis_kh[None, :]) / denom_h * 8
    relative_w = (axis_qw[:, None] - axis_kw[None, :]) / denom_w * 8
    relative_hw = torch.stack([relative_h, relative_w], dim=-1).view(-1, 2)

    relative_coords_table, idx_map = torch.unique(relative_hw, return_inverse=True, dim=0)

    log_base = torch.log2(torch.tensor(8.0, device=device))
    relative_coords_table = (
        torch.sign(relative_coords_table)
        * torch.log2(torch.abs(relative_coords_table) + 1.0)
        / log_base
    )

    return idx_map, relative_coords_table


@torch.no_grad()
def get_seqlen_scale(input_resolution, window_size, device):
    return F.avg_pool2d(
        torch.ones(1, input_resolution[0], input_resolution[1], device=device) * (window_size ** 2),
        window_size,
        stride=1,
        padding=window_size // 2,
    ).reshape(-1, 1)


def _format_seq_length_scale(seq_length_scale, x):
    if not torch.is_tensor(seq_length_scale):
        seq_length_scale = torch.tensor(seq_length_scale, device=x.device, dtype=x.dtype)
    else:
        seq_length_scale = seq_length_scale.to(device=x.device, dtype=x.dtype)

    if seq_length_scale.dim() == 0:
        seq_length_scale = seq_length_scale.view(1, 1, 1, 1)
    elif seq_length_scale.dim() == 1:
        seq_length_scale = seq_length_scale.view(1, 1, -1, 1)
    elif seq_length_scale.dim() == 2:
        seq_length_scale = seq_length_scale.unsqueeze(0).unsqueeze(0)
    elif seq_length_scale.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported seq_length_scale shape: {seq_length_scale.shape}")

    return seq_length_scale


class DQRM_LGRCM(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, window_size=3, qkv_bias=True,
                 attn_drop=0., proj_drop=0., sr_ratio=1, is_extrapolation=False,
                 global_memory_size=7, gate_eps=1e-6):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.is_extrapolation = is_extrapolation
        self.gate_eps = gate_eps

        if not is_extrapolation and input_resolution is not None:
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W
            self.trained_pool_H = max(1, input_resolution[0] // self.sr_ratio)
            self.trained_pool_W = max(1, input_resolution[1] // self.sr_ratio)
            self.trained_pool_len = self.trained_pool_H * self.trained_pool_W

        assert window_size % 2 == 1, "window size must be odd"
        self.window_size = window_size
        self.local_len = window_size ** 2

        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1)
        )

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02
            )
        )

        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.router_proj = nn.Linear(dim, 512)
        self.routing_logit_scale = nn.Parameter(
            torch.log(torch.tensor(1 / 0.07, dtype=torch.float32))
        )

        self.prompt_proj = nn.Sequential(
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, dim)
        )

        self.last_route_logits = None
        self.last_route_weights = None

        self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
        self.norm = nn.LayerNorm(dim)
        self.act = nn.GELU()

        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

        self.relative_pos_bias_local = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_heads, self.local_len), mean=0, std=0.0004
            )
        )

        self.learnable_tokens_local = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_heads, self.head_dim, self.local_len), mean=0, std=0.02
            )
        )
        self.learnable_bias_local = nn.Parameter(torch.zeros(num_heads, 1, self.local_len))

        if isinstance(global_memory_size, numbers.Integral):
            global_memory_size = (global_memory_size, global_memory_size)
        self.global_memory_size = global_memory_size
        gh, gw = self.global_memory_size

        self.learnable_tokens_global = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(num_heads, self.head_dim, gh, gw), mean=0, std=0.02
            )
        )
        self.learnable_bias_global = nn.Parameter(torch.zeros(num_heads, 1, gh, gw))

        self.local_gate_alpha = nn.Parameter(torch.zeros(num_heads, 1, 1))
        self.global_gate_alpha = nn.Parameter(torch.zeros(num_heads, 1, 1))

    def _get_global_memory(self, pool_H, pool_W):
        tokens_global = F.interpolate(
            self.learnable_tokens_global,
            size=(pool_H, pool_W),
            mode="bilinear",
            align_corners=False
        )

        bias_global = F.interpolate(
            self.learnable_bias_global,
            size=(pool_H, pool_W),
            mode="bilinear",
            align_corners=False
        )

        tokens_global = tokens_global.reshape(self.num_heads, self.head_dim, pool_H * pool_W)
        bias_global = bias_global.reshape(self.num_heads, 1, pool_H * pool_W)
        return tokens_global, bias_global

    def _apply_bounded_memory_gate(self, attn_local, attn_pool, q_norm, pool_H, pool_W):
        mem_local = torch.einsum("bhnd,hdm->bhnm", q_norm, self.learnable_tokens_local) \
                    + self.learnable_bias_local.unsqueeze(0)

        tokens_global, bias_global = self._get_global_memory(pool_H, pool_W)
        mem_pool = torch.einsum("bhnd,hdm->bhnm", q_norm, tokens_global) \
                   + bias_global.unsqueeze(0)

        gate_local = 1.0 + torch.tanh(self.local_gate_alpha).unsqueeze(0) * torch.tanh(mem_local)
        gate_pool = 1.0 + torch.tanh(self.global_gate_alpha).unsqueeze(0) * torch.tanh(mem_pool)

        attn_local = attn_local * gate_local
        attn_pool = attn_pool * gate_pool

        denom = (
            attn_local.sum(dim=-1, keepdim=True) +
            attn_pool.sum(dim=-1, keepdim=True) +
            self.gate_eps
        )
        attn_local = attn_local / denom
        attn_pool = attn_pool / denom

        return attn_local, attn_pool

    def _build_querywise_prompt(self, x, text_prototypes):
        if text_prototypes is None:
            self.last_route_logits = None
            self.last_route_weights = None
            return None

        B, N, _ = x.shape
        text_prototypes = F.normalize(text_prototypes.to(x.device, x.dtype), dim=-1)

        token_feat = self.router_proj(x)
        token_feat = F.normalize(token_feat, dim=-1)

        logit_scale = self.routing_logit_scale.exp().clamp(max=100.0)
        route_logits = torch.matmul(token_feat, text_prototypes.t()) * logit_scale
        route_weights = F.softmax(route_logits, dim=-1)

        self.last_route_logits = route_logits
        self.last_route_weights = route_weights

        token_prompt = torch.matmul(route_weights, text_prototypes)
        token_prompt = self.prompt_proj(token_prompt)
        token_prompt = token_prompt.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        token_prompt = F.normalize(token_prompt, dim=-1)

        return token_prompt

    def forward(self, x, H, W, relative_pos_index, relative_coords_table,
                seq_length_scale, text_prototypes=None):
        B, N, C = x.shape
        pool_H, pool_W = max(1, H // self.sr_ratio), max(1, W // self.sr_ratio)
        pool_len = pool_H * pool_W

        q_norm = F.normalize(
            self.q(x).reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3),
            dim=-1
        )

        total_qe = self.query_embedding.unsqueeze(0)
        token_prompt = self._build_querywise_prompt(x, text_prototypes)
        if token_prompt is not None:
            total_qe = total_qe + token_prompt

        seq_length_scale = _format_seq_length_scale(seq_length_scale, x)

        q_norm_scaled = (
            (q_norm + total_qe)
            * F.softplus(self.temperature).unsqueeze(0)
            * seq_length_scale
        )

        k_local, v_local = self.kv(x).reshape(
            B, N, 2 * self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3).chunk(2, dim=1)

        attn_local = sw_qkrpb_cuda.apply(
            q_norm_scaled.contiguous(),
            F.normalize(k_local, dim=-1).contiguous(),
            self.relative_pos_bias_local,
            H, W, self.window_size
        )

        x_ = x.permute(0, 2, 1).reshape(B, -1, H, W).contiguous()
        x_ = F.adaptive_avg_pool2d(
            self.act(self.sr(x_)), (pool_H, pool_W)
        ).reshape(B, -1, pool_len).permute(0, 2, 1)
        x_ = self.norm(x_)

        kv_pool = self.kv(x_).reshape(
            B, pool_len, 2 * self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        k_pool, v_pool = kv_pool.chunk(2, dim=1)

        if self.is_extrapolation:
            pool_bias = self.cpb_fc2(
                self.cpb_act(self.cpb_fc1(relative_coords_table))
            ).transpose(0, 1)[:, relative_pos_index.view(-1)].view(-1, N, pool_len)
        else:
            pool_bias = self.cpb_fc2(
                self.cpb_act(self.cpb_fc1(relative_coords_table))
            ).transpose(0, 1)[:, relative_pos_index.view(-1)].view(
                -1, self.trained_len, self.trained_pool_len
            )
            pool_bias = pool_bias.reshape(
                -1, self.trained_len, self.trained_pool_H, self.trained_pool_W
            )
            pool_bias = F.interpolate(
                pool_bias, (pool_H, pool_W), mode="bilinear", align_corners=False
            )
            pool_bias = pool_bias.reshape(
                -1, self.trained_len, pool_len
            ).transpose(-1, -2).reshape(
                -1, pool_len, self.trained_H, self.trained_W
            )
            pool_bias = F.interpolate(
                pool_bias, (H, W), mode="bilinear", align_corners=False
            ).reshape(-1, pool_len, N).transpose(-1, -2)

        attn_pool = q_norm_scaled @ F.normalize(k_pool, dim=-1).transpose(-2, -1) + pool_bias

        attn = torch.cat([attn_local, attn_pool], dim=-1).softmax(dim=-1)
        attn = self.attn_drop(attn)
        attn_local, attn_pool = torch.split(attn, [self.local_len, pool_len], dim=-1)

        attn_local, attn_pool = self._apply_bounded_memory_gate(
            attn_local, attn_pool, q_norm, pool_H, pool_W
        )

        x_local = sw_av_cuda.apply(
            attn_local.type_as(v_local), v_local.contiguous(), H, W, self.window_size
        )
        x_pool = attn_pool @ v_pool

        x = (x_local + x_pool).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class QRCB(nn.Module):
    def __init__(self, dim, input_resolution, num_heads=8, qkv_bias=True, attn_drop=0.,
                 proj_drop=0., is_extrapolation=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.is_extrapolation = is_extrapolation

        if not is_extrapolation and input_resolution is not None:
            self.trained_H, self.trained_W = input_resolution
            self.trained_len = self.trained_H * self.trained_W

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.temperature = nn.Parameter(
            torch.log((torch.ones(num_heads, 1, 1) / 0.24).exp() - 1)
        )

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.query_embedding = nn.Parameter(
            nn.init.trunc_normal_(
                torch.empty(self.num_heads, 1, self.head_dim), mean=0, std=0.02
            )
        )

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.router_proj = nn.Linear(dim, 512)
        self.routing_logit_scale = nn.Parameter(
            torch.log(torch.tensor(1 / 0.07, dtype=torch.float32))
        )

        self.prompt_proj = nn.Sequential(
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, dim)
        )

        self.last_route_logits = None
        self.last_route_weights = None

        self.cpb_fc1 = nn.Linear(2, 512, bias=True)
        self.cpb_act = nn.ReLU(inplace=True)
        self.cpb_fc2 = nn.Linear(512, num_heads, bias=True)

    def _build_querywise_prompt(self, x, text_prototypes):
        if text_prototypes is None:
            self.last_route_logits = None
            self.last_route_weights = None
            return None

        B, N, _ = x.shape
        text_prototypes = F.normalize(text_prototypes.to(x.device, x.dtype), dim=-1)

        token_feat = self.router_proj(x)
        token_feat = F.normalize(token_feat, dim=-1)

        logit_scale = self.routing_logit_scale.exp().clamp(max=100.0)
        route_logits = torch.matmul(token_feat, text_prototypes.t()) * logit_scale
        route_weights = F.softmax(route_logits, dim=-1)

        self.last_route_logits = route_logits
        self.last_route_weights = route_weights

        token_prompt = torch.matmul(route_weights, text_prototypes)
        token_prompt = self.prompt_proj(token_prompt)
        token_prompt = token_prompt.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        token_prompt = F.normalize(token_prompt, dim=-1)
        return token_prompt

    def forward(self, x, H, W, relative_pos_index, relative_coords_table,
                seq_length_scale, text_prototypes=None):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, -1, 3 * self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=1)

        if self.is_extrapolation:
            rel_bias = self.cpb_fc2(
                self.cpb_act(self.cpb_fc1(relative_coords_table))
            ).transpose(0, 1)[:, relative_pos_index.view(-1)].view(-1, N, N)
        else:
            rel_bias = self.cpb_fc2(
                self.cpb_act(self.cpb_fc1(relative_coords_table))
            ).transpose(0, 1)[:, relative_pos_index.view(-1)].view(
                -1, self.trained_len, self.trained_len
            )
            rel_bias = rel_bias.reshape(-1, self.trained_len, self.trained_H, self.trained_W)
            rel_bias = F.interpolate(rel_bias, (H, W), mode="bilinear", align_corners=False)
            rel_bias = rel_bias.reshape(-1, self.trained_len, N).transpose(-1, -2).reshape(
                -1, N, self.trained_H, self.trained_W
            )
            rel_bias = F.interpolate(rel_bias, (H, W), mode="bilinear", align_corners=False).reshape(
                -1, N, N
            ).transpose(-1, -2)

        total_qe = self.query_embedding.unsqueeze(0)
        token_prompt = self._build_querywise_prompt(x, text_prototypes)
        if token_prompt is not None:
            total_qe = total_qe + token_prompt

        seq_length_scale = _format_seq_length_scale(seq_length_scale, x)

        attn = (
            (F.normalize(q, dim=-1) + total_qe)
            * F.softplus(self.temperature).unsqueeze(0)
            * seq_length_scale
        ) @ F.normalize(k, dim=-1).transpose(-2, -1) + rel_bias

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class QRCBlock(nn.Module):
    def __init__(self, dim, num_heads, input_resolution, window_size=3, mlp_ratio=4.,
                 qkv_bias=False, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1,
                 is_extrapolation=False):
        super().__init__()
        self.norm1 = norm_layer(dim)

        if sr_ratio == 1:
            self.attn = QRCB(
                dim,
                input_resolution,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                is_extrapolation=is_extrapolation
            )
        else:
            self.attn = DQRM_LGRCM(
                dim,
                input_resolution,
                window_size=window_size,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                attn_drop=attn_drop,
                proj_drop=drop,
                sr_ratio=sr_ratio,
                is_extrapolation=is_extrapolation
            )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = TX_ConvolutionalGLU(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W, relative_pos_index, relative_coords_table,
                seq_length_scale, text_prototypes=None):
        x = x + self.drop_path(
            self.attn(
                self.norm1(x),
                H, W,
                relative_pos_index,
                relative_coords_table,
                seq_length_scale,
                text_prototypes=text_prototypes
            )
        )
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class QRCBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=3, sr_ratio=1):
        super().__init__()
        self.block = QRCBlock(
            dim=dim,
            num_heads=num_heads,
            input_resolution=None,
            window_size=window_size,
            mlp_ratio=4.,
            qkv_bias=True,
            sr_ratio=sr_ratio,
            is_extrapolation=True
        )
        self.dim = dim
        self.sr_ratio = sr_ratio

    def forward(self, x, text_prototypes=None):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)

        sr_ratio = self.sr_ratio
        key_H, key_W = max(1, H // sr_ratio), max(1, W // sr_ratio)

        relative_pos_index, relative_coords_table = get_relative_position_cpb(
            query_size=(H, W),
            key_size=(key_H, key_W),
            pretrain_size=(H, W),
            device=x.device
        )

        if self.sr_ratio > 1:
            local_seq_length = get_seqlen_scale((H, W), self.block.attn.window_size, device=x.device)
            seq_length_scale = torch.log(local_seq_length + key_H * key_W)
        else:
            seq_length_scale = torch.log(torch.as_tensor(H * W, device=x.device, dtype=x.dtype))

        out = self.block(
            x_flat,
            H, W,
            relative_pos_index,
            relative_coords_table,
            seq_length_scale,
            text_prototypes=text_prototypes
        )

        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        return out

def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")

def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == "BiasFree":
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class MDTA_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(MDTA_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3,
            kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        k = rearrange(k, "b (head c) h w -> b head c (h w)", head=self.num_heads)
        v = rearrange(v, "b (head c) h w -> b head c (h w)", head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2)
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.body(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = MDTA_Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)

class Model(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type="WithBias",
                 decoder=True):
        super(Model, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.decoder = decoder

        if self.decoder:
            self.QRCB1 = QRCBlock(dim=dim * 2 ** 3, num_heads=heads[3], sr_ratio=2)
            self.QRCB2 = QRCBlock(dim=dim * 2 ** 2, num_heads=heads[2], sr_ratio=2)
            self.QRCB3 = QRCBlock(dim=dim * 2 ** 1, num_heads=heads[1], sr_ratio=2)

        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])

        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])

        self.down3_4 = Downsample(int(dim * 2 ** 2))

        self.latent = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 3),
                num_heads=heads[3],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[3])
        ])

        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias
        )

        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 2),
                num_heads=heads[2],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[2])
        ])

        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias
        )

        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1),
                num_heads=heads[1],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[1])
        ])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_blocks[0])
        ])

        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 1),
                num_heads=heads[0],
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(
            int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias
        )

    def get_routing_weights(self):
        routing_weights = []
        if self.decoder:
            for branch in [self.QRCB1, self.QRCB2, self.QRCB3]:
                attn = branch.block.attn
                if hasattr(attn, "last_route_weights") and attn.last_route_weights is not None:
                    routing_weights.append(attn.last_route_weights)
        return routing_weights

    def forward(self, inp_img, text_prototypes=None):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)

        if self.decoder:
            latent = self.QRCB1(latent, text_prototypes=text_prototypes)

        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)

        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        if self.decoder:
            out_dec_level3 = self.QRCB2(out_dec_level3, text_prototypes=text_prototypes)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        if self.decoder:
            out_dec_level2 = self.QRCB3(out_dec_level2, text_prototypes=text_prototypes)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
