import numpy as np
import datagen as dg
import features as ft


def test_bearing_frequencies_identity():
    g = dg.BearingGeometry()
    assert abs(g.bpfo(25) + g.bpfi(25) - g.n_balls * 25) < 1e-9


def test_signal_shape_and_determinism():
    a = dg.generate_signal("healthy", 0, 25, 0.5, np.random.default_rng(1))
    b = dg.generate_signal("healthy", 0, 25, 0.5, np.random.default_rng(1))
    assert a.shape == (dg.N_SAMPLES,) and np.array_equal(a, b)


def test_unknown_fault_raises():
    try:
        dg.generate_signal("nope", 0.5, 25, 0.5, np.random.default_rng(0))
    except ValueError:
        return
    assert False


def test_imbalance_raises_1x():
    h = dg.generate_signal("healthy", 0, 25, 0.5, np.random.default_rng(2))
    i = dg.generate_signal("imbalance", 0.8, 25, 0.5, np.random.default_rng(2))
    assert ft.extract_features(i, speed_hz=25)["a1x"] > 3 * ft.extract_features(h, speed_hz=25)["a1x"]


def test_outer_race_envelope_peak():
    h = ft.extract_features(dg.generate_signal("healthy", 0, 25, 0.6, np.random.default_rng(3)), speed_hz=25)
    o = ft.extract_features(dg.generate_signal("bearing_outer", 0.8, 25, 0.6, np.random.default_rng(3)), speed_hz=25)
    assert o["env_snr_bpfo"] > 3 * h["env_snr_bpfo"] and o["kurtosis"] > h["kurtosis"]


def test_dataset_shape_and_labels():
    s, m = dg.generate_dataset(n_machines=2, per_class=2, seed=0)
    assert s.shape == (24, dg.N_SAMPLES) and set(m.fault) == set(dg.FAULTS)
    assert (m[m.fault == "healthy"].severity == 0).all()


def test_run_to_failure_rul():
    s, m = dg.run_to_failure("imbalance", 20, seed=0)
    assert len(s) == 20 and m.true_rul.iloc[0] == 20 and m.true_rul.iloc[-1] == 1
    assert (np.diff(m.severity) > 0).all()
