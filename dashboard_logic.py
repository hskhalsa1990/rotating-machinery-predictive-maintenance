"""Pure-Python logic behind the Streamlit dashboard (app.py). No Streamlit import here, so it is testable.

ALL SIGNALS ARE SYNTHETIC. The classifier is trained on synthetic data at start-up."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import datagen as dg
import features as ft
import iso10816 as iso
import maintenance as mt
from advanced_diagnostics import HealthyNovelty, best_band

MACHINE_CLASSES = list(iso.THRESHOLDS)


def simulate(fault, severity, speed_hz, load, seed=0, noise_scale=1.0):
    """One synthetic signal; noise_scale multiplies the machine noise level."""
    rng = np.random.default_rng(seed)
    mp = dg.MachineParams(machine_id=0, gain=1.0, f_res=3000.0, noise=0.3 * noise_scale)
    return dg.generate_signal(fault, severity, speed_hz, load, rng, mp=mp)


def analyze(signal, speed_hz, machine_class="medium_rigid"):
    """Features, ISO-style zone and spectra for one signal."""
    feats = ft.extract_features(signal, speed_hz=speed_hz)
    f, a = ft.amplitude_spectrum(signal)
    fe, e = ft.envelope_spectrum(signal)
    geom = dg.BearingGeometry()
    return {"features": feats, "zone": iso.zone(feats["vel_rms"], machine_class),
            "freq": f, "amp": a, "env_freq": fe, "env_amp": e,
            "bpfo": geom.bpfo(speed_hz), "bpfi": geom.bpfi(speed_hz), "mesh": dg.N_TEETH * speed_hz}


def train_model(n_machines=12, per_class=6, seed=0):
    """Small Random forest trained on synthetic data (fast). Returns a fitted sklearn Pipeline."""
    sigs, meta = dg.generate_dataset(n_machines=n_machines, per_class=per_class, seed=seed)
    X = ft.features_dataframe(sigs, meta.speed_hz.to_numpy())
    model = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=1))])
    model.fit(X, meta.fault)
    return model


def diagnose(model, feats):
    """Class probabilities (dict fault -> probability), sorted high to low."""
    import pandas as pd
    X = pd.DataFrame([feats], columns=ft.FEATURE_COLS)
    p = model.predict_proba(X)[0]
    return dict(sorted(zip(model.classes_, map(float, p)), key=lambda kv: -kv[1]))


def cost_comparison(p_fail, rul_median, planned_repair, planned_downtime_h, unplanned_repair, unplanned_downtime_h, cost_per_h):
    """Expected cost of repairing now vs waiting (INR, assumptions typed by the user)."""
    cp = mt.CostParams(planned_repair, planned_downtime_h, unplanned_repair, unplanned_downtime_h, cost_per_h)
    ec = mt.expected_costs(p_fail, rul_median, cp)
    return {**ec, "break_even_p": cp.planned_total / cp.unplanned_total}


def train_novelty(n_machines=12, per_class=8, seed=0):
    """Novelty detector fitted on HEALTHY synthetic signals only (machine-grouped threshold)."""
    sigs, meta = dg.generate_dataset(n_machines=n_machines, per_class=per_class, seed=seed)
    h = (meta.fault == "healthy").to_numpy()
    X = ft.features_dataframe(sigs[h], meta.speed_hz.to_numpy()[h])
    return HealthyNovelty().fit(X, groups=meta.machine_id.to_numpy()[h])


def novelty_report(detector, feats, signal):
    """Score, threshold, flag and kurtogram best band for one signal."""
    import pandas as pd
    X = pd.DataFrame([feats], columns=ft.FEATURE_COLS)
    score = float(detector.score(X)[0])
    lo, hi, sk = best_band(signal)
    return {"score": score, "threshold": float(detector.threshold_), "anomaly": score > detector.threshold_,
            "band_lo": lo, "band_hi": hi, "sk": sk}
