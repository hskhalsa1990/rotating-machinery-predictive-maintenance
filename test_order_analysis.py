import numpy as np

import datagen as dg
import order_analysis as oa
import physics_ext as pe

MP = dg.MachineParams(0, 1.0, 3000.0, 0.3)


def _ramp(faults, s0=20.0, s1=30.0, seed=1):
    x = pe.generate_signal_ext(faults, s0, 0.7, np.random.default_rng(seed), mp=MP, speed_end=s1)
    _, fr, th = pe.shaft_phase(len(x), dg.FS, s0, s1)
    return x, fr, th


def test_angle_from_speed_matches_shaft_phase():
    _, fr, th = pe.shaft_phase(4096, dg.FS, 20.0, 30.0)
    assert np.allclose(oa.angle_from_speed(fr), th)


def test_resample_gives_integer_style_revolutions():
    x, _, th = _ramp({})
    xa, n_rev = oa.resample_to_angle(x, th, spr=256)
    assert len(xa) == int(np.floor(n_rev * 256)) and 7 < n_rev < 10


def test_resample_rejects_non_increasing_angle():
    try:
        oa.resample_to_angle(np.zeros(10), np.zeros(10))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_pure_order_tone_stays_at_its_order_under_ramp():
    _, fr, th = pe.shaft_phase(4096, dg.FS, 20.0, 30.0)
    x = np.sin(3 * th)
    o, A = oa.order_spectrum(x, th)
    assert abs(o[np.argmax(A)] - 3.0) < 0.2


def test_imbalance_order_track_follows_speed_squared():
    x, _, th = _ramp({"imbalance": 0.8})
    tr = oa.order_track(x, th, orders=(1,))
    a = tr["amp"][1]
    sp = tr["speed_hz"]
    expected = 0.12 + 0.08 * 0.7 + 1.6 * 0.8 * (sp / 25.0) ** 2
    assert np.all(np.diff(sp) > 0) and a[-1] > a[0] and np.max(np.abs(a - expected) / expected) < 0.15


def test_order_envelope_sees_bpfo_order_during_ramp():
    x, _, th = _ramp({"bearing_outer": 0.9})
    xh, _, th = _ramp({})
    b = dg.BearingGeometry().bpfo(1.0)
    snr = []
    for sig in (x, xh):
        oe, E = oa.order_envelope_spectrum(sig, th)
        snr.append(oa.order_amplitude(oe, E, b, 0.2) / np.median(E[(oe > 0.5) & (oe < 20)]))
    assert snr[0] > 3 * snr[1]


def test_cepstrum_finds_sideband_family():
    fr, fm = 25.0, 575.0
    t = np.arange(dg.N_SAMPLES) / dg.FS
    x = (1 + 0.8 * np.cos(2 * np.pi * fr * t)) * np.sin(2 * np.pi * fm * t) + 0.02 * np.random.default_rng(0).normal(size=t.size)
    flat = np.sin(2 * np.pi * fm * t) + 0.02 * np.random.default_rng(0).normal(size=t.size)
    assert oa.cepstral_peak_ratio(x, fr) > 2 * oa.cepstral_peak_ratio(flat, fr)


def test_ramp_experiment_runs_and_order_domain_not_worse():
    r = oa.ramp_experiment(ramps=((15.0, 35.0),), n=60, seed=2)[0]
    assert r["auc_order_domain"] >= r["auc_frequency_domain"] - 0.02 and r["auc_order_domain"] > 0.9
