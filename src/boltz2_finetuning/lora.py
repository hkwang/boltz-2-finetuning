"""LoRA for the `nn.Linear` layers of a Boltz-2 submodule, without `peft` and without touching the installed `boltz`.

`LoRALinear` wraps a linear layer as `base(x) + scale * B(A(x))`; `B` starts at zero so the wrapped model equals the pretrained one.
`merged_weights` returns the dense weights `W + scale * B @ A` under the original parameter names, which is the format
`experiments/make_finetuned_checkpoint.py` expects.
"""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + self.scale * functional.linear(functional.linear(x, self.lora_a), self.lora_b)

    def merged_weight(self) -> Tensor:
        return self.base.weight + self.scale * self.lora_b @ self.lora_a


def add_lora(model: nn.Module, prefix: str, rank: int, alpha: float) -> int:
    """Wrap every `nn.Linear` whose module name starts with `prefix`; returns the number wrapped. Base weights stay frozen."""
    targets = [(name, module) for name, module in model.named_modules()
               if name.startswith(prefix) and isinstance(module, nn.Linear)]
    for name, module in targets:
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, LoRALinear(module, rank, alpha))
    return len(targets)


def merged_weights(model: nn.Module) -> dict[str, Tensor]:
    return {f"{name}.weight": module.merged_weight().detach().cpu()
            for name, module in model.named_modules() if isinstance(module, LoRALinear)}
