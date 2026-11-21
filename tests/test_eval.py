"""CaloChallenge metric implementations on small known inputs."""
import math
import os

import numpy as np
import torch

from calonsf.eval import classifier, correlations, distances, features, manifold, separation

R_EDGES = [0, 4.65, 9.3, 13.95, 18.6, 23.25, 27.9, 32.55, 37.2, 41.85]


def test_features_single_voxel():
    x = np.zeros((2, 45, 16, 9), np.float32)
    x[0, 3, 4, 2] = 10.0  # one voxel in layer 3
    x[1, 3, 0, 0] = 1.0
    x[1, 3, 8, 0] = 1.0   # two voxels at the same r, opposite angles
    f = features.high_level_features(x, np.array([100.0, 10.0]), R_EDGES, threshold=0.5)
    r, a = 0.5 * (R_EDGES[2] + R_EDGES[3]), (4 + 0.5) * 2 * math.pi / 16
    assert np.isclose(f["E_layer"][0, 3], 10) and np.isclose(f["E_tot_over_Einc"][0], 0.1)
    assert np.isclose(f["CoE_r"][0, 3], r) and np.isclose(f["CoE_eta"][0, 3], r * math.cos(a))
    assert np.isclose(f["CoE_phi"][0, 3], r * math.sin(a))
    assert np.isclose(f["width_eta"][0, 3], 0, atol=1e-6)
    assert np.isnan(f["CoE_r"][0, 0]) and f["E_layer"][0, 0] == 0  # empty layer: undefined
    assert np.isclose(f["sparsity"][0, 3], 1 - 1 / 144) and np.isclose(f["sparsity"][0, 0], 1.0)
    # two opposite voxels: centre of energy at the origin, eta-width = |eta|
    eta0 = 0.5 * (R_EDGES[0] + R_EDGES[1]) * math.cos(0.5 * 2 * math.pi / 16)  # alpha bin 0, r bin 0
    assert np.isclose(f["CoE_eta"][1, 3], 0, atol=1e-6) and np.isclose(f["width_eta"][1, 3], eta0, rtol=1e-6)
    m = features.feature_matrix(f)
    assert m.shape == (2, 1 + 45 * 8) and np.isfinite(m).all()


def test_separation_power_definition():
    assert separation.separation_power([1, 2, 3], [1, 2, 3]) == 0
    assert np.isclose(separation.separation_power([1, 0], [0, 1]), 1.0)  # no overlap
    h1, h2 = np.array([0.5, 0.5]), np.array([0.75, 0.25])
    assert np.isclose(separation.separation_power(h1, h2), 0.5 * (0.25 ** 2 / 1.25 + 0.25 ** 2 / 0.75))


def test_separation_pipeline_and_band():
    rng = np.random.default_rng(0)
    N = 400
    x = (rng.random((N, 6, 4, 3)) * (rng.random((N, 6, 4, 3)) > 0.5) * 50).astype(np.float32)
    e = rng.uniform(1e3, 1e6, N)
    f = features.high_level_features(x, e, [0, 1, 2, 3], 1.0)
    ed = separation.make_edges(f, 10)
    h = separation.feature_hists(f, ed)
    assert all(v == 0 for lst in separation.sep_from_hists(h, h).values() for v in lst)
    ved = separation.voxel_edges(x, 10)
    vh = separation.voxel_event_hists(x, ved)
    assert vh.sum() == (x > 0).sum()  # every positive voxel counted once
    band = separation.geant4_band(f, vh, ed, n_pairs=3)
    assert "voxel_energy" in band and np.all(np.array(band["E_layer"]["max"]) >= np.array(band["E_layer"]["min"]))
    shifted = {**f, "CoE_r": f["CoE_r"] + 5.0}  # far outside the reference range: all mass in the edge bin
    s = separation.sep_from_hists(h, separation.feature_hists(shifted, ed))
    assert np.nanmax(s["CoE_r"]) > 0.9


def test_pcc():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(500, 1))
    e = np.concatenate([a, 2 * a + 1, -a, np.ones((500, 1))], 1)  # last layer constant -> undefined, set to 0
    c = correlations.layer_pcc(e)
    assert np.allclose(c[:3, :3], np.sign([[1, 1, -1], [1, 1, -1], [-1, -1, 1]]), atol=1e-9)
    assert c[3, 0] == 0 and c[3, 3] == 1
    assert correlations.pcc_difference(c, c)["max_abs_diff"] == 0


def test_fpd_kpd():
    rng = np.random.default_rng(0)
    a, b, c = (rng.normal(size=(1500, 6)) for _ in range(3))
    assert distances.fpd(a, b) < 0.05 and distances.fpd(a, c + 2.0) > 10
    same, _ = distances.kpd(a, b, batch=500, n_batches=3)
    far, _ = distances.kpd(a, c + 2.0, batch=500, n_batches=3)
    assert abs(same) < 0.05 < far


def test_prdc():
    rng = np.random.default_rng(0)
    r, f = rng.normal(size=(600, 8)).astype(np.float32), rng.normal(size=(600, 8)).astype(np.float32)
    same = manifold.prdc(r, f, k=5)
    far = manifold.prdc(r, f + 20.0, k=5)
    assert same["precision"] > 0.8 and same["recall"] > 0.8 and 0.5 < same["density"] < 1.6 and same["coverage"] > 0.7
    assert far == {"precision": 0.0, "recall": 0.0, "density": 0.0, "coverage": 0.0}
    x = np.zeros((2, 45, 16, 9), np.float32)
    x[0, 0, 0, 0] = 100.0
    mf = manifold.manifold_features(x)
    assert mf[0, 0] == 2.0 and mf[1, 0] == -1.0 and mf.shape == (2, 6480)  # log10(100); floor log10(0.1 MeV) = -1


def test_resnet_shapes():
    net = classifier.ResNet3D()
    assert net(torch.randn(3, 1, 45, 16, 9), torch.randn(3)).shape == (3,)
    # stem + 16 block convs = 17 convolutions, plus the final linear layer = 18 layers
    n = sum(1 for m in net.modules() if isinstance(m, torch.nn.Conv3d) and m.kernel_size != (1, 1, 1))
    assert n == 17


def test_binary_classifier_auc():
    rng = np.random.default_rng(0)
    n = 400
    e = rng.uniform(1e3, 1e6, n).astype(np.float32)
    a = rng.normal(size=(n, 5)).astype(np.float32)

    def mk(h):
        return classifier._Source(None, e, h)

    kw = dict(seeds=2, epochs=8, lr=1e-3, batch=64)
    diff = classifier.run_binary("dnn_high", mk(a), mk(a + 3.0), **kw)
    same = classifier.run_binary("dnn_high", mk(a), mk(rng.normal(size=(n, 5)).astype(np.float32)), **kw)
    assert diff["auc_mean"] > 0.97 and len(diff["aucs"]) == 2
    assert same["auc_mean"] < 0.7


def test_readout_cutoff_zeroes_subthreshold_voxels_only():
    x = np.array([0.0, 0.005, 0.01515, 0.02, 3.0], np.float32)
    y = features.readout_cutoff(x, 0.01515)
    assert y.dtype == x.dtype
    np.testing.assert_array_equal(y, np.array([0.0, 0.0, 0.01515, 0.02, 3.0], np.float32))
    assert features.readout_cutoff(x, 0.0) is x


def test_training_curves_are_written_from_the_csv_logs(tmp_path):
    from calonsf.eval.training_plots import best_epoch, plot_training, read_log
    (tmp_path / "jepa").mkdir()
    (tmp_path / "sr").mkdir()
    (tmp_path / "jepa" / "jepa_log.csv").write_text(
        "epoch,train_loss,val_loss,train_jepa_in,val_jepa_in,train_jepa_cross,val_jepa_cross,train_target_std,val_target_std\n"
        + "\n".join(f"{e},{1/(e+1)},{1.1/(e+1)},0.5,0.5,0.4,0.4,0.85,0.85" for e in range(4)))
    (tmp_path / "sr" / "sr_log.csv").write_text(
        "epoch,train_loss,val_loss,train_fm,val_fm,val_metric\n" + "\n".join(f"{e},{1/(e+1)},{1/(e+1)},1,1,{[.3,.1,.2,.4][e]}" for e in range(4)))
    assert best_epoch(read_log(str(tmp_path / "sr" / "sr_log.csv"))) == 1  # val_metric decides, not the loss
    made = plot_training(str(tmp_path))
    assert len(made) == 2 and all(os.path.exists(p) and os.path.exists(p[:-4] + ".pdf") for p in made)
    assert plot_training(str(tmp_path / "missing"), str(tmp_path / "o")) == []
