import torch
import logging

logger = logging.getLogger("omnivoice.utils.unsloth_patch")

def apply_unsloth_patches():
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. Skipping Unsloth speedup patches.")
        return
        
    try:
        import triton
    except ImportError:
        logger.warning("Triton is not installed. Skipping Unsloth speedup patches. Please install triton to enable hardware acceleration.")
        return

    try:
        import transformers.models.qwen3.modeling_qwen3 as qm
        from omnivoice.utils.unsloth.kernels.rms_layernorm import fast_rms_layernorm
        from omnivoice.utils.unsloth.kernels.rope_embedding import fast_rope_embedding
        from omnivoice.utils.unsloth.kernels.swiglu import swiglu_fg_kernel

        def patched_rmsnorm_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            return fast_rms_layernorm(self, hidden_states, gemma=False)

        def patched_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            if cos.dim() == 3:
                cos = cos[0]
                sin = sin[0]
            elif cos.dim() == 4:
                cos = cos[0, 0]
                sin = sin[0, 0]
            return fast_rope_embedding(q, k, cos, sin)

        def patched_mlp_forward(self, x):
            gate = self.gate_proj(x)
            up = self.up_proj(x)
            return self.down_proj(swiglu_fg_kernel(gate, up))

        # 1. Patch RMSNorm class to use fused Triton RMSNorm kernel
        qm.Qwen3RMSNorm.forward = patched_rmsnorm_forward
        
        # 2. Patch apply_rotary_pos_emb to use fast RoPE Triton kernel
        qm.apply_rotary_pos_emb = patched_apply_rotary_pos_emb
        
        # 3. Patch Qwen3MLP to use fused SwiGLU Triton kernel
        qm.Qwen3MLP.forward = patched_mlp_forward
        
        logger.info("Successfully applied Unsloth Qwen3 training/inference speedup patches!")
    except Exception as e:
        logger.warning(f"Failed to apply Unsloth speedup patches due to: {e}. Falling back to default PyTorch execution.")
