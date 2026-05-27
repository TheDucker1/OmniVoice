import triton
import torch
from contextlib import nullcontext

MAX_FUSED_SIZE: int = 65536
next_power_of_2 = triton.next_power_of_2
DEVICE_COUNT = torch.cuda.device_count() if torch.cuda.is_available() else 0

def calculate_settings(n: int) -> (int, int):
    BLOCK_SIZE: int = next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds "
            f"the maximum CUDA blocksize = {MAX_FUSED_SIZE}."
        )
    num_warps: int = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    return BLOCK_SIZE, num_warps

def torch_gpu_device(device):
    return nullcontext()

def torch_device_stream(device):
    return torch.cuda.current_stream(device)
