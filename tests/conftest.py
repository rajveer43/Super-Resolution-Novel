import os

import pytest
import torch

from calonsf.data.synthetic import make_calochallenge
from calonsf.utils.config import load_config

ROOT = os.path.dirname(os.path.dirname(__file__))


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synth")
    make_calochallenge(str(d), 40, 2)
    return d


TINY = ["model.d_model=32", "model.heads=2", "model.enc_layers=1", "model.dec_layers=1", "model.flow.mixer_layers=1",
        "jepa.d_pred=32", "jepa.pred_heads=2", "jepa.pred_layers=1", "data.batch_size=4", "jepa.batch_size=4",
        "device=cpu", "train.max_steps_per_epoch=2", "train.epochs=1", "jepa.epochs=1", "train.log_every=1000"]


@pytest.fixture
def cfg(synth_dir, tmp_path):
    files = f"[{synth_dir}/dataset_2_1.hdf5, {synth_dir}/dataset_2_2.hdf5]"
    torch.manual_seed(0)
    return load_config(os.path.join(ROOT, "configs/experiments/calochallenge_synthetic.yaml"),
                       TINY + [f"data.files={files}", f"out_dir={tmp_path}"])
