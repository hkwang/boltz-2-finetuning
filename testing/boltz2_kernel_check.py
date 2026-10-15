"""Report which cuEquivariance triangle kernels run on the visible GPU (needs CUDA).

Uses Mac1-sized pair tensors (256 tokens) so Boltz's short-sequence fallback
(CUEQ_TRI*_FALLBACK_THRESHOLD=100) does not apply.
"""

import warnings

import torch
from cuequivariance_torch.primitives.triangle import triangle_attention, triangle_multiplicative_update
from torch.profiler import ProfilerActivity, profile

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}, compute capability {torch.cuda.get_device_capability(0)}")
batch, seq, dim, heads, head_dim = 1, 256, 128, 4, 32
bf16 = dict(device=device, dtype=torch.bfloat16)

x = torch.randn(batch, seq, seq, dim, **bf16)
mask = torch.ones(batch, seq, seq, device=device)
weights = dict(
    norm_in_weight=torch.ones(dim, **bf16),
    norm_in_bias=torch.zeros(dim, **bf16),
    p_in_weight=torch.randn(2 * dim, dim, **bf16) * 0.05,
    g_in_weight=torch.randn(2 * dim, dim, **bf16) * 0.05,
    norm_out_weight=torch.ones(dim, **bf16),
    norm_out_bias=torch.zeros(dim, **bf16),
    p_out_weight=torch.randn(dim, dim, **bf16) * 0.05,
    g_out_weight=torch.randn(dim, dim, **bf16) * 0.05,
)
q, k, v = (torch.randn(batch, seq, heads, seq, head_dim, **bf16) for _ in range(3))
bias = torch.randn(batch, 1, heads, seq, seq, **bf16)
attn_mask = torch.ones(batch, seq, 1, 1, seq, device=device, dtype=torch.bool)


def profile_kernels(fn) -> tuple[list[str], bool]:
    """Return the CUDA kernel names launched by fn and whether a fallback warning fired."""
    with warnings.catch_warnings(record=True) as caught, profile(activities=[ProfilerActivity.CUDA]) as prof:
        warnings.simplefilter("always")
        fn()
        torch.cuda.synchronize()
    names = sorted({event.key for event in prof.key_averages() if event.device_type == torch.autograd.DeviceType.CUDA})
    return names, any("FALLING BACK" in str(warning.message) for warning in caught)


checks = {
    "triangle_multiplicative_update": lambda: triangle_multiplicative_update(
        x, direction="outgoing", mask=mask, **weights
    ),
    "triangle_attention": lambda: triangle_attention(q, k, v, bias, mask=attn_mask, scale=head_dim**-0.5),
}
for label, fn in checks.items():
    profile_kernels(fn)  # warm-up
    names, fell_back = profile_kernels(fn)
    print(f"{label}: fallback_warning={fell_back}")
    for name in names:
        print(f"    {name[:140]}")
