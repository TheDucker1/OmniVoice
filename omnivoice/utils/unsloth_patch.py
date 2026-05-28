import torch
import logging

logger = logging.getLogger("omnivoice.utils.unsloth_patch")


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
        
        # 2. Patch apply_rotary_pos_emb to use Triton fast_rope_embedding
        try:
            from omnivoice.utils.unsloth.kernels.rope_embedding import fast_rope_embedding
            def patched_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
                return fast_rope_embedding(q, k, cos, sin, position_ids)
            qm.apply_rotary_pos_emb = patched_apply_rotary_pos_emb
            logger.info("Successfully patched apply_rotary_pos_emb with Triton fast_rope_embedding.")
        except Exception as e:
            logger.warning(f"Failed to patch apply_rotary_pos_emb with Triton kernel: {e}. Falling back to default.")
            
        logger.info("Successfully applied Unsloth Qwen3 training/inference speedup patches!")
    except Exception as e:
        logger.warning(f"Failed to apply Unsloth speedup patches due to: {e}. Falling back to default PyTorch execution.")
