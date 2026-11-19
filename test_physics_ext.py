import numpy as np

import datagen as dg
import features as ft
import physics_ext as pe


def _f(faults, seed=0, speed=25, load=0.6, **kw):
    return pe.generate_signal_ext(faults, speed, load, np.random.default_rng(seed), **kw)


def test_deterministic_and_shape():
    a, b = _f({"bearing_outer": 0.8}, seed=1), _f({"bearing_outer": 0.8}, seed=1)
    assert a.shape == (dg.N_SAMPLES,) and np.array_equal(a, b)


def test_unknown_fault_raises():
    try:
        _f({"nope": 0.5})
    except ValueError:
        return
    assert False


def test_multifault_shows_both_signatures():
    h = ft.extract_features(_f({}, seed=2), speed_hz=25)
    m = ft.extract_features(_f({"bearing_outer": 0.9, "imbalance": 0.9}, seed=2), speed_hz=25)
    assert m["a1x"] > 3 * h["a1x"] and m["env_snr_bpfo"] > 3 * h["env_snr_bpfo"]


def test_shaft_phase_ramp_reaches_end_speed():
    t, fr, theta = pe.shaft_phase(4096, dg.FS, 15.0, 35.0)
    assert abs(fr[0] - 15.0) < 0.1 and abs(fr[-1] - 35.0) < 1e-6
    inst = np.diff(theta) * dg.FS / (2 * np.pi)
    assert abs(inst[-1] - fr[-1]) < 0.1 and (np.diff(theta) > 0).all()


def test_shock_raises_kurtosis():
    base = ft.extract_features(_f({}, seed=3), speed_hz=25)["kurtosis"]
    hit = ft.extract_features(_f({}, seed=3, shock_amp=6.0), speed_hz=25)["kurtosis"]
    assert hit > base + 1.0


def test_slip_weakens_bearing_envelope_peak_on_average():
    def mean_snr(slip):
        return np.mean([ft.extract_features(_f({"bearing_outer": 0.8}, seed=s, slip=slip), speed_hz=25)["env_snr_bpfo"]
                        for s in range(8)])
    assert mean_snr(0.0) > mean_snr(0.2)


def test_multifault_dataset_labels():
    s, m = pe.generate_multifault_dataset(n_machines=2, per_combo=1, seed=0)
    assert s.shape == (22, dg.N_SAMPLES) and set(m.n_faults) == {0, 1, 2}
    row = m[m.combo == "bearing_outer+imbalance"].iloc[0]
    assert row.y_bearing_outer == 1 and row.y_imbalance == 1 and row.y_gear_tooth == 0
