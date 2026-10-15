"""LoRA wrapper: identity at init, merged weights reproduce the adapted forward pass, only adapter parameters train."""

import torch
from boltz2_finetuning.lora import add_lora, merged_weights
from torch import nn


def build_model() -> nn.Module:
    torch.manual_seed(0)
    return nn.ModuleDict({"score": nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4, bias=False)), "other": nn.Linear(6, 3)})


def test_identity_at_init_and_only_adapters_train() -> None:
    model = build_model()
    inputs = torch.randn(5, 6)
    before = model["score"](inputs)
    for parameter in model.parameters():
        parameter.requires_grad = False
    assert add_lora(model, "score", rank=2, alpha=4.0) == 2
    assert torch.equal(model["score"](inputs), before)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {f"score.{index}.{part}" for index in (0, 2) for part in ("lora_a", "lora_b")}
    assert isinstance(model["other"], nn.Linear)


def test_merged_weights_match_adapted_forward() -> None:
    model = build_model()
    original = {name: parameter.clone() for name, parameter in model.named_parameters()}
    add_lora(model, "score", rank=2, alpha=4.0)
    for name, parameter in model.named_parameters():
        if name.endswith("lora_b"):
            parameter.data.normal_()
    merged = merged_weights(model)
    assert set(merged) == {"score.0.weight", "score.2.weight"}
    inputs = torch.randn(5, 6)
    dense = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 4, bias=False))
    dense[0].weight.data, dense[0].bias.data = merged["score.0.weight"], original["score.0.bias"]
    dense[2].weight.data = merged["score.2.weight"]
    assert torch.allclose(dense(inputs), model["score"](inputs), atol=1e-5)
