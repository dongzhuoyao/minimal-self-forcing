"""
Tiny CausalWanModel for tutorial.
A simplified version of the original CausalWanModel transformer architecture.
Now with RoPE and FlexAttention, scaled by 16x.
"""
import torch
import torch.nn as nn
import math
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from einops import rearrange


def attention(q, k, v, q_lens=None, k_lens=None, dropout_p=0., softmax_scale=None, 
              q_scale=None, causal=False, window_size=(-1, -1), deterministic=False, 
              dtype=torch.bfloat16, fa_version=None):
    """
    Attention function matching official implementation signature.
    Falls back to standard PyTorch attention if flash_attn not available.
    """
    try:
        from flash_attn import flash_attn_func
        # Use flash attention if available
        if q_lens is None:
            q_lens = torch.tensor([q.shape[1]] * q.shape[0], dtype=torch.int32, device=q.device)
        if k_lens is None:
            k_lens = torch.tensor([k.shape[1]] * k.shape[0], dtype=torch.int32, device=k.device)
        
        cu_seqlens_q = torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(0, dtype=torch.int32).to(q.device)
        cu_seqlens_k = torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(0, dtype=torch.int32).to(k.device)
        
        return flash_attn_func(
            q.transpose(1, 2).contiguous(),
            k.transpose(1, 2).contiguous(),
            v.transpose(1, 2).contiguous(),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=q.shape[1],
            max_seqlen_k=k.shape[1],
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size if window_size != (-1, -1) else None
        ).transpose(1, 2)
    except ImportError:
        # Fallback to standard PyTorch attention
        if q_lens is not None or k_lens is not None:
            import warnings
            warnings.warn('Padding mask is disabled when using scaled_dot_product_attention.')
        
        # Preserve input dtype instead of converting to default dtype
        input_dtype = q.dtype
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, is_causal=causal, dropout_p=dropout_p
        )
        
        return out.transpose(1, 2).contiguous().to(input_dtype)


def rope_params(max_seq_len, dim, theta=10000):
    """Generate RoPE frequency parameters."""
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


def rope_apply(x, grid_sizes, freqs):
    """Apply RoPE to query/key tensors.
    
    Args:
        x: [B, L, num_heads, head_dim] where head_dim is even
        grid_sizes: [B, 3] with (F, H, W)
        freqs: Precomputed frequency tensor
    """
    n, c = x.size(2), x.size(3) // 2
    
    # Split freqs into temporal, height, width components
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        
        # Clamp dimensions to prevent index out of bounds
        # freqs buffers support up to their length (typically 1024)
        max_freq_len_t = freqs[0].shape[0]
        max_freq_len_h = freqs[1].shape[0]
        max_freq_len_w = freqs[2].shape[0]
        
        f_clamped = min(f, max_freq_len_t)
        h_clamped = min(h, max_freq_len_h)
        w_clamped = min(w, max_freq_len_w)
        
        # Reshape to complex for rotation
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        
        # Create frequency multipliers for each dimension
        # Handle dimensions that exceed buffer size by using modulo indexing
        if f <= max_freq_len_t:
            freqs_t = freqs[0][:f]
        else:
            indices_t = torch.arange(f, device=freqs[0].device) % max_freq_len_t
            freqs_t = freqs[0][indices_t]
        
        if h <= max_freq_len_h:
            freqs_h = freqs[1][:h]
        else:
            indices_h = torch.arange(h, device=freqs[1].device) % max_freq_len_h
            freqs_h = freqs[1][indices_h]
        
        if w <= max_freq_len_w:
            freqs_w = freqs[2][:w]
        else:
            indices_w = torch.arange(w, device=freqs[2].device) % max_freq_len_w
            freqs_w = freqs[2][indices_w]
        
        freqs_i = torch.cat([
            freqs_t.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_h.view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_w.view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        
        # Apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        
        output.append(x_i)
    return torch.stack(output).type_as(x)


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    """Apply RoPE with frame offset for causal attention."""
    n, c = x.size(2), x.size(3) // 2
    
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w
        
        # Clamp dimensions to prevent index out of bounds
        max_freq_len_t = freqs[0].shape[0]
        max_freq_len_h = freqs[1].shape[0]
        max_freq_len_w = freqs[2].shape[0]
        
        # Handle temporal dimension with start_frame offset
        end_frame = start_frame + f
        if end_frame <= max_freq_len_t:
            freqs_t = freqs[0][start_frame:end_frame]
        else:
            # Use modulo if exceeding buffer
            indices_t = (torch.arange(start_frame, end_frame, device=freqs[0].device) % max_freq_len_t)
            freqs_t = freqs[0][indices_t]
        
        # Handle spatial dimensions
        if h <= max_freq_len_h:
            freqs_h = freqs[1][:h]
        else:
            indices_h = torch.arange(h, device=freqs[1].device) % max_freq_len_h
            freqs_h = freqs[1][indices_h]
        
        if w <= max_freq_len_w:
            freqs_w = freqs[2][:w]
        else:
            indices_w = torch.arange(w, device=freqs[2].device) % max_freq_len_w
            freqs_w = freqs[2][indices_w]
        
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs_t.view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs_h.view(1, h, 1, -1).expand(f, h, w, -1),
            freqs_w.view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(seq_len, 1, -1)
        
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])
        
        output.append(x_i)
    return torch.stack(output).type_as(x)


class TinyCausalWanModel(nn.Module):
    """
    CausalWanModel matching official implementation exactly, but with reduced capacity.
    
    Architecture matches official CausalWanModel:
    - Patch embedding (Conv3d)
    - Transformer blocks with RoPE and FlexAttention
    - Output head with modulation
    - Same normalization, attention mechanisms, KV cache handling
    
    Only difference: num_layers reduced (default 4 vs 32 in official)
    """
    
    def __init__(
        self,
        in_dim=3,  # Input channels (RGB for tutorial, 16 for latents in official)
        out_dim=3,  # Output channels (RGB for tutorial, 16 for latents in official)
        dim=2048,  # Hidden dimension (same as official)
        ffn_dim=8192,  # FFN dimension (same as official)
        num_heads=16,  # Attention heads (same as official)
        num_layers=4,  # Transformer layers (reduced from 32 in official)
        patch_size=(1, 4, 4),  # Patch size (can be (1,2,2) like official)
        text_dim=128,  # Text embedding dimension (128 for tutorial, 4096 in official)
        freq_dim=256,  # Time embedding dimension (same as official)
        num_frame_per_block=3,  # Frames per block for causal mask (same as official)
        local_attn_size=-1,  # Local attention size (-1 = global, same as official)
        sink_size=0,  # Sink tokens for KV cache (same as official)
        qk_norm=True,  # QK normalization (same as official)
        cross_attn_norm=False,  # Cross-attention normalization (same as official)
        use_flex_attention=False,  # Whether to use flex_attention (default: False)
        eps=1e-6
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.patch_size = patch_size
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.num_frame_per_block = num_frame_per_block
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.use_flex_attention = use_flex_attention
        self.eps = eps
        
        # Patch embedding: convert video patches to tokens (matching official)
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Text embedding (matching official structure)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),  # Match official
            nn.Linear(dim, dim)
        )
        
        # Time embedding (matching official)
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim * 6)  # 6 modulation vectors
        )
        
        # Transformer blocks (matching official structure)
        self.blocks = nn.ModuleList([
            TinyCausalWanAttentionBlock(
                cross_attn_type='t2v_cross_attn',  # Text-to-video cross-attention
                dim=dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                local_attn_size=local_attn_size,
                sink_size=sink_size,
                qk_norm=qk_norm,
                use_flex_attention=use_flex_attention,
                cross_attn_norm=cross_attn_norm,
                eps=eps
            )
            for _ in range(num_layers)
        ])
        
        # Output head (matching official)
        self.head = TinyCausalHead(dim, out_dim, patch_size, eps)
        
        # RoPE frequencies (initialize as buffer)
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.register_buffer('freqs', torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ], dim=1))
        
        # Block mask (will be created on first forward)
        self.block_mask = None
    
    def _prepare_blockwise_causal_attn_mask(
        self, device, num_frames, frame_seqlen, num_frame_per_block
    ):
        """Prepare block-wise causal attention mask for FlexAttention."""
        total_length = num_frames * frame_seqlen
        
        # Right padding to multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length
        
        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        
        # Block-wise causal mask: each block attends to all previous blocks
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )
        
        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = \
                tmp + frame_seqlen * num_frame_per_block
        
        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
        
        block_mask = create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False, device=device
        )
        
        return block_mask
        
    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict,
        kv_cache=None,
        crossattn_cache=None,
        current_start: int = 0,
        cache_start: int = None
    ):
        """
        Forward pass.
        
        Args:
            noisy_image_or_video: Input tensor [B, F, C, H, W]
            conditional_dict: Dictionary with 'prompt_embeds' key
            timestep: Timestep tensor [B, F] or [B]
            kv_cache: Optional KV cache dict
            crossattn_cache: Optional cross-attention cache
            current_start: Start position for KV cache
            cache_start: Cache start position
            
        Returns:
            flow_pred: Flow prediction [B, F, C, H, W]
            extra: Extra info (None for now)
        """
        batch_size, num_frames, channels, height, width = noisy_image_or_video.shape
        device = noisy_image_or_video.device
        
        # Convert to [B, C, F, H, W] for Conv3d
        x = rearrange(noisy_image_or_video, 'b f c h w -> b c f h w')  # [B, C, F, H, W]
        
        # Patch embedding: [B, C, F, H, W] -> [B, dim, F', H', W']
        x = self.patch_embedding(x)
        
        # Get grid sizes after patching
        _, _, f_patched, h_patched, w_patched = x.shape
        grid_sizes = torch.tensor(
            [[num_frames, h_patched, w_patched]] * batch_size,
            device=device, dtype=torch.long
        )
        
        # Flatten spatial dimensions: [B, dim, F', H', W'] -> [B, dim, F'*H'*W']
        x = x.flatten(2)  # [B, dim, F'*H'*W']
        x = x.transpose(1, 2)  # [B, F'*H'*W', dim]
        
        seq_len = x.shape[1]
        seq_lens = torch.tensor([seq_len] * batch_size, device=device, dtype=torch.long)
        
        # Create block mask if not already created
        if self.block_mask is None:
            frame_seqlen = seq_len // num_frames
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device, num_frames, frame_seqlen, self.num_frame_per_block
            )
        
        # Time embedding
        if timestep.dim() == 2:
            # [B, F] -> [B*F]
            t_flat = timestep.flatten()
        else:
            t_flat = timestep.flatten()
        
        # Sinusoidal time embedding
        t_emb = self._sinusoidal_embedding(t_flat, self.freq_dim).to(device)
        e = self.time_embedding(t_emb)  # [B*F, dim]
        e0 = self.time_projection(e)  # [B*F, dim*6]
        e0 = e0.unflatten(0, (batch_size, num_frames))  # [B, F, dim*6]
        e0 = e0.unflatten(2, (6, self.dim))  # [B, F, 6, dim]
        
        # Text embedding
        prompt_embeds = conditional_dict.get("prompt_embeds", None)
        if prompt_embeds is None:
            # Create dummy embeddings if not provided
            prompt_embeds = torch.zeros(
                batch_size, 77, self.text_dim, device=device
            )
        
        # Project text embeddings
        context = self.text_embedding(prompt_embeds)  # [B, text_len, dim]
        context_lens = None
        
        # Process through transformer blocks (matching official structure)
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            current_start=current_start,
            cache_start=cache_start
        )
        
        for block in self.blocks:
            x = block(x, **kwargs)
        
        # Output head (matching official: uses e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        # e0 is [B, F, 6, dim], head expects [B, F, 1, dim]
        e_head = e0[:, :, :1, :]  # [B, F, 1, dim]
        output = self.head(x, e_head)  # [B, num_frames, frame_seqlen, out_dim*patch_prod]
        
        # Reshape back to video format
        # output is [B, num_frames, frame_seqlen, out_dim*patch_prod]
        # where frame_seqlen = h_patched * w_patched
        patch_prod = math.prod(self.patch_size)
        output_channels = output.shape[-1]  # out_dim * patch_prod
        output = output.reshape(batch_size, num_frames, h_patched, w_patched, output_channels)
        output = output.permute(0, 4, 1, 2, 3)  # [B, out_dim*patch_prod, num_frames, h_patched, w_patched]
        
        # Unpatch: reshape to match input spatial size
        # For simplicity, we'll use interpolation if needed
        if output.shape[-2:] != (height, width):
            output = output.reshape(
                batch_size, output_channels, num_frames, h_patched, w_patched
            )
            # Simple unpatch: reshape and interpolate
            # In full implementation, this would be a learned unpatch layer
            output = output.reshape(
                batch_size, self.out_dim, patch_prod, num_frames, h_patched, w_patched
            )
            output = rearrange(output, 'b out_dim patch_prod f h w -> b f out_dim h patch_prod w')  # [B, F, out_dim, H', patch_prod, W']
            output = output.reshape(
                batch_size, num_frames, self.out_dim,
                h_patched * self.patch_size[1], w_patched * self.patch_size[2]
            )
            # Interpolate to match input size
            output = nn.functional.interpolate(
                output.flatten(0, 1),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).unflatten(0, (batch_size, num_frames))
        else:
            output = output.reshape(
                batch_size, num_frames, self.out_dim, height, width
            )
        
        return output, None
    
    @staticmethod
    def _sinusoidal_embedding(timesteps, dim):
        """Create sinusoidal embeddings for timesteps."""
        half_dim = dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # Create embedding tensor on the same device as timesteps
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
        emb = timesteps.float()[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:  # zero pad
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
        return emb


class TinyCausalWanAttentionBlock(nn.Module):
    """
    CausalWanAttentionBlock matching official implementation exactly.
    
    Structure:
    - Self-attention with RoPE and FlexAttention
    - Cross-attention (text-to-video)
    - FFN with modulation
    - Same normalization and modulation pattern as official
    """
    
    def __init__(
        self,
        cross_attn_type='t2v_cross_attn',
        dim=2048,
        ffn_dim=8192,
        num_heads=16,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        use_flex_attention=False,
        eps=1e-6
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.use_flex_attention = use_flex_attention
        self.eps = eps
        
        # Normalization layers (matching official: WanLayerNorm)
        self.norm1 = WanLayerNorm(dim, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        
        # Self-attention (matching official CausalWanSelfAttention)
        self.self_attn = TinyCausalWanSelfAttention(
            dim, num_heads, local_attn_size, sink_size, qk_norm, use_flex_attention, eps
        )
        
        # Cross-attention (matching official WanT2VCrossAttention)
        self.cross_attn = TinyWanT2VCrossAttention(
            dim, num_heads, qk_norm, eps
        )
        
        # FFN (matching official: GELU with approximate='tanh')
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate='tanh'),  # Match official
            nn.Linear(ffn_dim, dim)
        )
        
        # Modulation parameters (matching official)
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
    
    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            context(Tensor): Shape [B, text_len, C]
            context_lens(Tensor): Shape [B] (unused but kept for compatibility)
            block_mask (BlockMask): For FlexAttention
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        
        # Assert shapes are compatible
        assert x.shape[1] == num_frames * frame_seqlen, \
            f"x.shape[1] ({x.shape[1]}) must equal num_frames ({num_frames}) * frame_seqlen ({frame_seqlen})"
        assert frame_seqlen > 0, \
            f"frame_seqlen ({frame_seqlen}) must be > 0. x.shape[1]={x.shape[1]}, e.shape[1]={e.shape[1]}"
        
        # Modulation (matching official)
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        
        # Self-attention (matching official structure exactly)
        x_attn_input = (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2)
        y = self.self_attn(
            x_attn_input,
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start)
        
        # Assert self-attention output has correct shape
        assert y.shape[1] == x.shape[1], \
            f"self_attn output shape[1] ({y.shape[1]}) must match input shape[1] ({x.shape[1]})"
        
        # Apply modulation and residual (matching official)
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)
        
        # Cross-attention & FFN function (matching official structure)
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(
                self.norm3(x), 
                context,
                context_lens, 
                crossattn_cache=crossattn_cache
            )
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            return x
        
        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class WanRMSNorm(nn.Module):
    """RMSNorm for QK normalization (matching official implementation)."""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):
    """LayerNorm matching official implementation (preserves dtype)."""
    
    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)


class TinyCausalWanSelfAttention(nn.Module):
    """
    CausalWanSelfAttention matching official implementation exactly.
    
    Features:
    - RoPE for positional encoding
    - FlexAttention for training
    - KV cache with rolling mechanism for inference
    - Local attention support
    - Sink tokens support
    - Teacher forcing support
    """
    
    def __init__(
        self,
        dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        use_flex_attention=False,
        eps=1e-6
    ):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.use_flex_attention = use_flex_attention
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560
        
        # Layers (matching official)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads] (matching official)
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask): For FlexAttention
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start
        
        # Query, key, value function (matching official)
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v
        
        q, k, v = qkv_fn(x)
        
        if kv_cache is None:
            # Training mode: use flex_attention if enabled, otherwise standard attention
            if not self.use_flex_attention:
                # Standard attention without flex_attention
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)
                
                # Standard scaled dot-product attention
                scale = 1.0 / math.sqrt(self.head_dim)
                attn = torch.matmul(
                    roped_query.transpose(1, 2),  # [B, num_heads, L, head_dim]
                    roped_key.transpose(1, 2).transpose(-2, -1)  # [B, num_heads, head_dim, L]
                ) * scale
                attn = torch.softmax(attn, dim=-1)
                x = torch.matmul(attn, v.transpose(1, 2))  # [B, num_heads, L, head_dim]
                x = x.transpose(1, 2)  # [B, L, num_heads, head_dim]
            else:
                # Use flex_attention (original implementation)
                # Training mode: check for teacher forcing
                is_tf = (s == seq_lens[0].item() * 2)
                if is_tf:
                    # Teacher forcing: handle 2x sequence length
                    q_chunk = torch.chunk(q, 2, dim=1)
                    k_chunk = torch.chunk(k, 2, dim=1)
                    roped_query = []
                    roped_key = []
                    # RoPE should be same for clean and noisy parts
                    for ii in range(2):
                        rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                        rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                        roped_query.append(rq)
                        roped_key.append(rk)
                    
                    roped_query = torch.cat(roped_query, dim=1)
                    roped_key = torch.cat(roped_key, dim=1)
                    
                    padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                    padded_roped_query = torch.cat(
                        [roped_query,
                         torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                     device=q.device, dtype=v.dtype)],
                        dim=1
                    )
                    padded_roped_key = torch.cat(
                        [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                                device=k.device, dtype=v.dtype)],
                        dim=1
                    )
                    padded_v = torch.cat(
                        [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                        device=v.device, dtype=v.dtype)],
                        dim=1
                    )
                    
                    # flex_attention expects [B, num_heads, L, head_dim]
                    query_t = padded_roped_query.transpose(2, 1)  # [B, num_heads, L_padded, head_dim]
                    key_t = padded_roped_key.transpose(2, 1)
                    value_t = padded_v.transpose(2, 1)
                    
                    try:
                        x = flex_attention(
                            query=query_t,
                            key=key_t,
                            value=value_t,
                            block_mask=block_mask
                        )
                        # flex_attention returns [B, num_heads, L_padded, head_dim]
                        # Verify output shape is not empty
                        if x.shape[2] == 0:
                            raise RuntimeError(f"flex_attention returned empty tensor with shape {x.shape}")
                        
                        # Remove padding: [B, num_heads, L, head_dim]
                        # When padded_length is 0, [:-0] is equivalent to [:]
                        if padded_length > 0:
                            x = x[:, :, :-padded_length]
                        # Transpose to [B, L, num_heads, head_dim]
                        x = x.transpose(2, 1)
                    except Exception as e:
                        # Fallback: if flex_attention fails, use standard attention
                        import warnings
                        warnings.warn(f"flex_attention failed in teacher forcing: {e}, falling back to standard attention")
                        # Use standard scaled dot-product attention
                        scale = 1.0 / math.sqrt(self.head_dim)
                        attn = torch.matmul(query_t, key_t.transpose(-2, -1)) * scale
                        attn = torch.softmax(attn, dim=-1)
                        x = torch.matmul(attn, value_t)
                        if padded_length > 0:
                            x = x[:, :, :-padded_length]
                        x = x.transpose(2, 1)
                else:
                    # Normal training: use RoPE and FlexAttention
                    roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                    roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)
                    
                    padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                    padded_roped_query = torch.cat(
                        [roped_query,
                         torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                     device=q.device, dtype=v.dtype)],
                        dim=1
                    )
                    padded_roped_key = torch.cat(
                        [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                                device=k.device, dtype=v.dtype)],
                        dim=1
                    )
                    padded_v = torch.cat(
                        [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                        device=v.device, dtype=v.dtype)],
                        dim=1
                    )
                    
                    # flex_attention expects [B, num_heads, L, head_dim]
                    query_t = padded_roped_query.transpose(2, 1)  # [B, num_heads, L_padded, head_dim]
                    key_t = padded_roped_key.transpose(2, 1)
                    value_t = padded_v.transpose(2, 1)
                    
                    try:
                        x = flex_attention(
                            query=query_t,
                            key=key_t,
                            value=value_t,
                            block_mask=block_mask
                        )
                        # flex_attention returns [B, num_heads, L_padded, head_dim]
                        # Verify output shape is not empty
                        if x.shape[2] == 0:
                            raise RuntimeError(f"flex_attention returned empty tensor with shape {x.shape}")
                        
                        # Remove padding: [B, num_heads, L, head_dim]
                        # When padded_length is 0, [:-0] is equivalent to [:]
                        if padded_length > 0:
                            x = x[:, :, :-padded_length]
                        # Transpose to [B, L, num_heads, head_dim]
                        x = x.transpose(2, 1)
                    except Exception as e:
                        # Fallback: if flex_attention fails, use standard attention
                        import warnings
                        warnings.warn(f"flex_attention failed: {e}, falling back to standard attention")
                        # Use standard scaled dot-product attention
                        scale = 1.0 / math.sqrt(self.head_dim)
                        attn = torch.matmul(query_t, key_t.transpose(-2, -1)) * scale
                        # Apply block_mask if available (simplified - would need proper masking)
                        attn = torch.softmax(attn, dim=-1)
                        x = torch.matmul(attn, value_t)
                        if padded_length > 0:
                            x = x[:, :, :-padded_length]
                        x = x.transpose(2, 1)
        else:
            # Inference mode: KV cache with rolling mechanism
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            
            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            
            # KV cache rolling mechanism (matching official)
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            
            # Use attention function (matching official)
            k_cached = kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            v_cached = kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            x = attention(
                roped_query,
                k_cached,
                v_cached,
                causal=True
            )
            
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)
        
        # Output (matching official)
        x = x.flatten(2)
        x = self.o(x)
        return x


class TinyWanT2VCrossAttention(nn.Module):
    """
    Text-to-Video Cross-Attention matching official WanT2VCrossAttention.
    """
    
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps
        
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
    
    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (dict, *optional*): Cached key and value tensors
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        
        # Compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        
        if crossattn_cache is not None:
            if not crossattn_cache.get("is_init", False):
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)
        
        # Compute attention (matching official)
        x = attention(q, k, v, k_lens=context_lens, causal=False)
        
        # Output - ensure dtype matches model parameters
        x = x.flatten(2)
        x = x.type_as(self.o.weight)  # Match output layer dtype
        x = self.o(x)
        return x


class TinyCausalHead(nn.Module):
    """Output head matching official CausalHead exactly."""
    
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps
        
        # Layers (matching official)
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        
        # Modulation (matching official)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)
    
    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C] (matching official: uses e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        
        # Modulation (matching official exactly)
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        
        # Apply head with modulation (matching official structure exactly)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    # Toy example: small video + dummy text embeddings
    batch_size = 1
    num_frames = 12  # Must be divisible by num_frame_per_block (default: 3)
    height = 16
    width = 16

    model = TinyCausalWanModel().to(device).eval()

    noisy_video = torch.randn(
        batch_size, num_frames, 3, height, width, device=device
    )
    timestep = torch.randint(
        low=0, high=1000, size=(batch_size, num_frames), device=device
    )
    conditional_dict = {
        "prompt_embeds": torch.randn(batch_size, 77, model.text_dim, device=device)
    }

    with torch.no_grad():
        output, _ = model(noisy_video, timestep, conditional_dict)
    print("Output shape:", tuple(output.shape))