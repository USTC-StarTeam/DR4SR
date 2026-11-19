from pathlib import Path
import random
import runpy
import sys

import numpy as np
import pytest
import torch
from torch.nn.utils.rnn import pad_sequence


ROOT = Path(__file__).resolve().parents[1]
PRETRAIN = "2.Pretrain_regenerator.py"
INFERENCE = "3.Hybrid_inference.py"


class ReachedGpuBoundary(Exception):
    """The script resolved the dataset and finished its first CPU work."""


@pytest.mark.parametrize("script", [PRETRAIN, INFERENCE])
@pytest.mark.parametrize("collection,alias,sos", [
    ("amazon-toys", "toy", 11925),
    ("yelp", "yelp", 20034),
])
@pytest.mark.parametrize("path_form", ["trailing-slash", "no-trailing-slash", "native-absolute"])
def test_regeneration_uses_root_directory_alias(
        script, collection, alias, sos, path_form, tmp_path, monkeypatch):
    data_dir = tmp_path / "dataset" / collection / alias
    data_dir.mkdir(parents=True)
    relative_path = "./" + data_dir.relative_to(tmp_path).as_posix()
    if path_form == "trailing-slash":
        root_arg = relative_path + "/"
    elif path_form == "no-trailing-slash":
        root_arg = relative_path
    else:
        root_arg = str(data_dir)

    if script == PRETRAIN:
        torch.save([([1, 2], [2])], data_dir / "seq-pat-pair.pth")

        def check_padded_source(sequences, *args, **kwargs):
            source = pad_sequence(sequences, *args, **kwargs)
            assert source.tolist() == [[sos, 1, 2, sos + 1]]
            raise ReachedGpuBoundary

        monkeypatch.setattr(torch.nn.utils.rnn, "pad_sequence", check_padded_source)
    else:
        embeddings = torch.arange(4 * 64, dtype=torch.float32).reshape(4, 64)
        torch.save({"parameters": {"item_embedding.weight": embeddings}},
                   data_dir / "pre-trained_embedding.ckpt")
        original_to = torch.nn.Module.to

        def check_first_cuda_transfer(model, *args, **kwargs):
            if args and args[0] == "cuda":
                torch.testing.assert_close(model.item_embedding.weight[:4], embeddings)
                raise ReachedGpuBoundary
            return original_to(model, *args, **kwargs)

        monkeypatch.setattr(torch.nn.Module, "to", check_first_cuda_transfer)

    # Run each real CLI through dataset resolution without starting GPU work.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "--root_path", root_arg])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", torch.backends.cudnn.deterministic)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[]), torch.autograd.set_detect_anomaly(
                torch.is_anomaly_enabled(), torch.is_anomaly_check_nan_enabled()
        ), pytest.raises(ReachedGpuBoundary):
            runpy.run_path(str(ROOT / script), run_name="__main__")
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
