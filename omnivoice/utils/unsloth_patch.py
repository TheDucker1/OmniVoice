import torch
import logging
import triton
import triton.language as tl

logger = logging.getLogger("omnivoice.utils.unsloth_patch")


@triton.jit
def _in_place_rope_kernel(
    Q, Q_batch_stride, Q_head_stride, Q_seq_stride, Q_dim_stride,
    cos, cos_batch_stride, cos_seq_stride, cos_dim_stride,
    sin, sin_batch_stride, sin_seq_stride, sin_dim_stride,
    seq_len, n_heads, head_dim: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // seq_len
    seq_id = pid % seq_len
    
    col_offsets = tl.arange(0, BLOCK_SIZE)
    half_head_dim = head_dim // 2
    mask = col_offsets < half_head_dim

    # Pointer to cos and sin for this row (batch_id, seq_id)
    cos_ptr = cos + batch_id * cos_batch_stride + seq_id * cos_seq_stride
    sin_ptr = sin + batch_id * sin_batch_stride + seq_id * sin_seq_stride
    
    # Load cos and sin
    cos_row = tl.load(cos_ptr + col_offsets * cos_dim_stride, mask=mask, other=0)
    sin_row = tl.load(sin_ptr + col_offsets * sin_dim_stride, mask=mask, other=0)

    if BACKWARD_PASS:
        sin_row = -sin_row

    # Process all heads in-place directly using strides
    for h in range(n_heads):
        q_ptr = Q + batch_id * Q_batch_stride + h * Q_head_stride + seq_id * Q_seq_stride
        
        q0 = tl.load(q_ptr + col_offsets * Q_dim_stride, mask=mask, other=0)
        q1 = tl.load(q_ptr + half_head_dim * Q_dim_stride + col_offsets * Q_dim_stride, mask=mask, other=0)
        
        # Store back in-place
        tl.store(q_ptr + col_offsets * Q_dim_stride, q0 * cos_row - q1 * sin_row, mask=mask)
        tl.store(q_ptr + half_head_dim * Q_dim_stride + col_offsets * Q_dim_stride, q1 * cos_row + q0 * sin_row, mask=mask)


class InPlace_RoPE_Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, cos, sin):
        # Q shape: [batch, n_heads, seq_len, head_dim]
        # cos, sin shape: [batch, seq_len, head_dim] or [1, seq_len, head_dim]
        batch, n_heads, seq_len, head_dim = Q.shape
        
        # Clone to prevent modifying the inputs of other layers
        Q_out = Q.clone()
        
        # Expand cos and sin to match batch size if they are broadcasted
        if cos.shape[0] == 1 and batch > 1:
            cos = cos.expand(batch, -1, -1)
            sin = sin.expand(batch, -1, -1)
            
        BLOCK_SIZE = triton.next_power_of_2(head_dim // 2)
        num_warps = 4
        
        _in_place_rope_kernel[(batch * seq_len,)](
            Q_out, Q_out.stride(0), Q_out.stride(1), Q_out.stride(2), Q_out.stride(3),
            cos, cos.stride(0), cos.stride(1), cos.stride(2),
            sin, sin.stride(0), sin.stride(1), sin.stride(2),
            seq_len, n_heads, head_dim,
            BACKWARD_PASS=False,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        
        ctx.BLOCK_SIZE = BLOCK_SIZE
        ctx.num_warps = num_warps
        ctx.cos = cos
        ctx.sin = sin
        ctx.seq_len = seq_len
        ctx.n_heads = n_heads
        ctx.head_dim = head_dim
        
        return Q_out

    @staticmethod
    def backward(ctx, dY):
        # dY shape: [batch, n_heads, seq_len, head_dim]
        dY_out = dY.clone()
        cos = ctx.cos
        sin = ctx.sin
        
        _in_place_rope_kernel[(dY_out.shape[0] * ctx.seq_len,)](
            dY_out, dY_out.stride(0), dY_out.stride(1), dY_out.stride(2), dY_out.stride(3),
            cos, cos.stride(0), cos.stride(1), cos.stride(2),
            sin, sin.stride(0), sin.stride(1), sin.stride(2),
            ctx.seq_len, ctx.n_heads, ctx.head_dim,
            BACKWARD_PASS=True,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            num_warps=ctx.num_warps,
        )
        
        return dY_out, None, None


def apply_unsloth_patches():
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Skipping Unsloth speedup patches.")
        return
        
    try:
        import triton
        triton_available = True
    except ImportError:
        logger.warning("Triton is not installed. Skipping Unsloth speedup patches.")
        return

    try:
        import transformers.models.qwen3.modeling_qwen3 as qm
        
        # 1. Patch RMSNorm class to use fused Triton RMSNorm kernel
        try:
            from omnivoice.utils.unsloth.kernels.rms_layernorm import fast_rms_layernorm
            def patched_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return fast_rms_layernorm(self, hidden_states, gemma=False)
            qm.Qwen3RMSNorm.forward = patched_rmsnorm_forward
            logger.info("Successfully patched Qwen3RMSNorm with Triton fast_rms_layernorm.")
        except Exception as e:
            logger.warning(f"Failed to patch RMSNorm with Triton kernel: {e}. Falling back to default.")
        
        # 2. Patch apply_rotary_pos_emb to use Triton InPlace_RoPE_Embedding
        try:
            def patched_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                # Squeeze out extra unsqueeze_dim dimensions from cos/sin if present
                if cos.dim() == 4:
                    cos = cos.squeeze(1) if cos.shape[1] == 1 else cos.squeeze(0)
                    sin = sin.squeeze(1) if sin.shape[1] == 1 else sin.squeeze(0)
                
                # Expand cos and sin to match batch size if broadcasted
                if cos.shape[0] == 1 and q.shape[0] > 1:
                    cos = cos.expand(q.shape[0], -1, -1)
                    sin = sin.expand(q.shape[0], -1, -1)
                    
                q_embed = InPlace_RoPE_Embedding.apply(q, cos, sin)
                k_embed = InPlace_RoPE_Embedding.apply(k, cos, sin)
                return q_embed, k_embed
                
            qm.apply_rotary_pos_emb = patched_apply_rotary_pos_emb
            logger.info("Successfully patched apply_rotary_pos_emb with Triton InPlace_RoPE_Embedding.")
        except Exception as e:
            logger.warning(f"Failed to patch apply_rotary_pos_emb with Triton kernel: {e}. Falling back to default.")
            
        logger.info("Successfully applied Unsloth Qwen3 training/inference speedup patches!")
    except Exception as e:
        logger.warning(f"Failed to apply Unsloth speedup patches due to: {e}. Falling back to default PyTorch execution.")
