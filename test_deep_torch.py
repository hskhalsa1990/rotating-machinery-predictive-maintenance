"""NumPy-helper tests always run. PyTorch tests run only when torch is installed (NOT run by the author)."""
import numpy as np

import deep_torch as dt


def test_prepare_raw_zscore_and_length():
    rng = np.random.default_rng(0)
    x = rng.normal(5, 3, (4, 4096)).astype(np.float32)
    for f in (1, 2, 8):
        z = dt.prepare_raw(x, f)
        assert z.shape == (4, 4096 // f) and np.isfinite(z).all()
    z1 = dt.prepare_raw(x, 1)
    assert np.allclose(z1.mean(1), 0, atol=1e-4) and np.allclose(z1.std(1), 1, atol=1e-3)


def test_antialias_removes_high_frequency():
    t = np.arange(4096) / 12000.0
    x = np.sin(2 * np.pi * 5000 * t)[None, :]                    # 5 kHz would alias after decimating by 4
    a = dt.prepare_raw(x, 4, antialias=True)
    b = dt.prepare_raw(x, 4, antialias=False)
    # z-scoring happens before decimation, so compare the leftover energy
    assert a.std() < 0.2 * b.std()


def test_to_patches_shape_and_content():
    x = np.arange(2 * 100, dtype=float).reshape(2, 100)
    p = dt.to_patches(x, 16)
    assert p.shape == (2, 6, 16) and p[0, 1, 0] == 16 and p[1, 0, 0] == 100


def test_group_val_split_holds_out_whole_groups():
    g = np.repeat(np.arange(10), 5)
    v = dt.group_val_split(g, 0.2, seed=1)
    assert 0 < v.sum() < len(g)
    for gid in np.unique(g):
        assert len(set(v[g == gid])) == 1
    assert (dt.group_val_split(g, 0.2, seed=1) == v).all()


def test_class_weights_balance():
    w = dt.class_weights([0, 0, 0, 1], 3)
    assert w[1] > w[0] and w[2] == 1.0


def test_torch_models_if_available():
    if not dt.HAS_TORCH:
        print("   (torch not installed: PyTorch test skipped, NOT verified)")
        return
    import torch
    x = torch.randn(4, 1, 2048)
    assert dt.CNN1D()(x).shape == (4, 6)
    assert dt.LSTMClassifier()(x[:, :, :512]).shape == (4, 6)
    assert dt.count_parameters(dt.CNN1D()) > 0


def test_torch_training_smoke_if_available():
    if not dt.HAS_TORCH:
        return
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 512)).astype(np.float32)
    y = rng.integers(0, 6, 60)
    g = np.repeat(np.arange(6), 10)
    (acc, f1), hist = dt.train_eval(dt.CNN1D(), X, y, X[:12], y[:12], epochs=2, groups_tr=g, return_history=True)
    assert 0 <= acc <= 1 and len(hist) >= 1
