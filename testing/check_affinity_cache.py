"""Pretrained affinity modules on cached features should give values close to `boltz predict`'s affinity JSON for the same complex.

Run: python testing/check_affinity_cache.py CACHE_DIR BOLTZ_OUT_DIR
The two differ only through the pose: the cache uses the structure stage's top pose, `boltz predict` re-samples poses in its affinity stage.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import torch
from boltz2_finetuning.affinity import AffinityCacheDataset, AffinityHead, collate_affinity

cache_dir, boltz_out = Path(sys.argv[1]), Path(sys.argv[2])
checkpoint = torch.load(Path.home() / ".cache/boltz/boltz2_aff.ckpt", map_location="cpu", weights_only=False)
head = AffinityHead(checkpoint["hyper_parameters"], {}, 1e-5, 10, 0.0)
head.load_pretrained(checkpoint["state_dict"])
head = head.eval().cuda()
labels = pd.DataFrame({"record_id": [path.stem for path in sorted(cache_dir.glob("*.npz"))]})
labels = labels.assign(kind="ic50", assay_id="x", weight=1.0, binary=1, log10_ic50_uM=0.0, censored=False)
for index in range(len(labels)):
    batch = collate_affinity([AffinityCacheDataset(cache_dir, labels)[index]])
    batch = {key: value.cuda() if torch.is_tensor(value) else value for key, value in batch.items()}
    with torch.no_grad():
        out = head.predict_batch(batch)
    record_id = labels.record_id[index]
    reference = json.loads((boltz_out / "predictions" / record_id / f"affinity_{record_id}.json").read_text())
    print(record_id, "cached value %.3f probability %.3f" % (float(out["value"]), float(out["probability"])),
          "| boltz predict value %.3f probability %.3f" % (reference["affinity_pred_value"], reference["affinity_probability_binary"]))
