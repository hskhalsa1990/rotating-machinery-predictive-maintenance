import numpy as np

import advanced_diagnostics as ad
import datagen as dg
import features as ft


def _sig(fault, sev=0.9, seed=0, f_res=3000.0, speed=25.0):
    mp = dg.MachineParams(machine_id=0, gain=1.0, f_res=f_res, noise=0.3)
    return dg.generate_signal(fault, sev, speed, 0.7, np.random.default_rng(seed), mp=mp)


def test_sk_of_gaussian_noise_is_near_zero():
    x = np.random.default_rng(0).normal(size=60000)
    _, sk = ad.spectral_kurtosis(x, nperseg=128)
    assert abs(np.median(sk)) < 0.3


def test_sk_higher_for_bearing_fault_than_healthy():
    _, sk_b = ad.spectral_kurtosis(_sig("bearing_outer"), nperseg=64)
    _, sk_h = ad.spectral_kurtosis(_sig("healthy", 0.0), nperseg=64)
    assert np.max(sk_b[8:]) > np.max(sk_h[8:]) + 0.5


def test_kurtogram_shape_and_best_band_valid():
    f, K = ad.kurtogram(_sig("bearing_outer"))
    assert K.shape == (len(ad.DEFAULT_NPERSEG), len(f))
    lo, hi, sk = ad.best_band(_sig("bearing_outer"))
    assert 500 <= lo < hi <= 0.95 * dg.FS / 2 and sk > 0


def test_best_band_follows_resonance():
    bands = []
    for fres in (2200.0, 3800.0):
        lo, hi, _ = ad.best_band(_sig("bearing_outer", f_res=fres))
        bands.append((lo + hi) / 2)
    assert bands[1] > bands[0]


def test_band_envelope_snr_sees_outer_race():
    snr, band = ad.band_envelope_snr(_sig("bearing_outer"), 25.0)
    snr_h, _ = ad.band_envelope_snr(_sig("healthy", 0.0), 25.0)
    assert snr["bpfo"] > 3 * snr_h["bpfo"] and len(band) == 2


def test_novelty_flags_faults_not_unseen_healthy():
    sigs, meta = dg.generate_dataset(n_machines=16, per_class=6, seed=3)
    X = ft.features_dataframe(sigs, meta.speed_hz.to_numpy())
    tr = (meta.machine_id < 8) & (meta.fault == "healthy")
    det = ad.HealthyNovelty().fit(X[tr], groups=meta.machine_id[tr])
    te = meta.machine_id >= 8
    healthy_fa = det.is_anomaly(X[te & (meta.fault == "healthy")]).mean()
    fault_hit = det.is_anomaly(X[te & (meta.fault != "healthy") & (meta.severity > 0.4)]).mean()
    assert healthy_fa < 0.15 and fault_hit > 0.9


def test_novelty_scores_deterministic_and_threshold_set():
    X = np.abs(np.random.default_rng(0).normal(1, 0.2, size=(60, 4)))
    a = ad.HealthyNovelty().fit(X)
    b = ad.HealthyNovelty().fit(X)
    assert a.threshold_ == b.threshold_ and np.allclose(a.score(X), b.score(X))


def test_kurtogram_beats_fixed_band_when_resonance_outside_it():
    res = ad.resonance_shift_experiment(res_ranges=((1000, 1700),), n=120, seed=3)[0]
    assert res["auc_kurtogram"] > res["auc_fixed_band"] + 0.05 and res["auc_kurtogram"] > 0.9
