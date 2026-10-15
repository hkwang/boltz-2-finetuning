"""Affinity loss terms and batch sampler. Run: PYTHONPATH=src python testing/test_affinity_loss.py"""

import torch
import numpy as np
from boltz2_finetuning.affinity import MixedBatchSampler, collate_affinity, focal_loss, huber, pairwise_loss


def test_pairwise_only_within_assay() -> None:
    prediction = torch.tensor([0.0, 1.0, 5.0])
    target = torch.tensor([0.0, 1.0, 0.0])
    weight = torch.ones(3)
    assert pairwise_loss(prediction, target, weight, ["a", "a", "b"], delta=1.0) == 0
    assert pairwise_loss(prediction, target, weight, ["a", "a", "a"], delta=1.0) > 0
    assert pairwise_loss(prediction, target, weight, ["a", "b", "c"], delta=1.0) == 0


def test_pairwise_ignores_a_constant_offset() -> None:
    prediction = torch.tensor([0.5, 1.5, 3.0])
    target = torch.tensor([0.0, 1.0, 2.5])
    offset_loss = pairwise_loss(prediction + 10, target, torch.ones(3), ["a"] * 3, delta=1.0)
    assert offset_loss == 0
    assert huber(prediction, target, 1.0).mean() > 0


def test_focal_loss_downweights_easy_examples_and_weights_positives() -> None:
    easy = focal_loss(torch.tensor([6.0]), torch.tensor([1.0]), 0.25, 2.0)
    hard = focal_loss(torch.tensor([-1.0]), torch.tensor([1.0]), 0.25, 2.0)
    assert hard > 100 * easy
    positive = focal_loss(torch.tensor([0.0]), torch.tensor([1.0]), 0.25, 2.0)
    negative = focal_loss(torch.tensor([0.0]), torch.tensor([0.0]), 0.25, 2.0)
    assert torch.isclose(negative, 3 * positive)


def test_sampler_batches_have_one_assay_and_fixed_composition() -> None:
    kinds = ["ic50"] * 6 + ["hit"] * 3 + ["nonhit"] * 5
    assays = ["a", "a", "a", "b", "b", "b"] + ["h"] * 3 + ["n"] * 5
    sampler = MixedBatchSampler(kinds, assays, n_ic50=3, n_hit=2, n_nonhit=4, steps=20, seed=0, rank=0)
    for batch in sampler:
        assert [kinds[i] for i in batch] == ["ic50"] * 3 + ["hit"] * 2 + ["nonhit"] * 4
        assert len({assays[i] for i in batch[:3]}) == 1


def fake_cached_complex(tokens: int, ligand_tokens: int, seed: int) -> dict:
    generator = np.random.default_rng(seed)
    is_ligand = np.zeros(tokens, dtype=bool)
    is_ligand[-ligand_tokens:] = True
    dense = generator.normal(size=(tokens, tokens, 3)).astype(np.float16)
    dense[np.ix_(~is_ligand, ~is_ligand)] = 0
    return {
        "s_inputs": generator.normal(size=(tokens, 4)).astype(np.float32),
        "z_ligand_rows": dense[is_ligand],
        "z_ligand_cols": dense[:, is_ligand],
        "rep_coords": generator.normal(size=(tokens, 3)).astype(np.float32),
        "is_ligand": is_ligand, "is_receptor": ~is_ligand,
        "weight": 1.0, "binary": 1.0, "log10_ic50_uM": 0.0, "assay_id": "a", "kind": "ic50", "record_id": f"r{seed}",
    }


def test_collate_pads_complexes_of_different_sizes_without_mixing_them() -> None:
    small, large = fake_cached_complex(6, 2, 0), fake_cached_complex(9, 3, 1)
    batch = collate_affinity([small, large])
    assert batch["z"].shape == (2, 9, 9, 3)
    assert batch["pad_mask"].sum(1).tolist() == [6, 9]
    assert torch.equal(batch["z"][0, :6, :6][small["is_ligand"]], torch.from_numpy(small["z_ligand_rows"]).float())
    assert torch.equal(batch["z"][0, :6, :6][:, small["is_ligand"]], torch.from_numpy(small["z_ligand_cols"]).float())
    assert batch["z"][0, 6:].abs().sum() == 0 and batch["z"][0, :, 6:].abs().sum() == 0
    assert torch.equal(batch["z"][1][large["is_ligand"]], torch.from_numpy(large["z_ligand_rows"]).float())
    receptor = torch.from_numpy(large["is_receptor"])
    assert batch["z"][1][receptor][:, receptor].abs().sum() == 0


if __name__ == "__main__":
    for name, function in list(globals().items()):
        if name.startswith("test_"):
            function()
            print("ok", name)
