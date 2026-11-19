"""Condition-monitoring features: time domain, spectrum, envelope (bearing) and gear features."""
import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt
from scipy.stats import kurtosis, skew

from datagen import FS, N_TEETH, BearingGeometry

FEATURE_COLS = [
    "rms", "peak", "crest", "kurtosis", "skewness", "shape_factor", "impulse_factor",
    "vel_rms", "a1x", "a2x", "a3x", "a2x_over_1x", "a3x_over_1x",
    "mesh_amp", "sideband_amp", "sideband_ratio",
    "env_snr_bpfo", "env_snr_bpfi", "centroid", "hf_ratio",
]


def velocity_rms_mm_s(x, fs=FS, f_lo=10.0, f_hi=1000.0):
    """Velocity RMS [mm/s] from acceleration [m/s^2] by frequency-domain integration (10-1000 Hz)."""
    x = np.asarray(x, float)
    X = np.fft.rfft(x - x.mean())
    f = np.fft.rfftfreq(len(x), 1 / fs)
    m = (f >= f_lo) & (f <= f_hi)
    V = np.zeros_like(X)
    V[m] = X[m] / (2j * np.pi * f[m])
    v = np.fft.irfft(V, n=len(x))
    return float(np.sqrt(np.mean(v ** 2)) * 1000.0)


def amplitude_spectrum(x, fs=FS):
    """One-sided Hann-windowed amplitude spectrum (peak amplitude of sinusoids)."""
    x = np.asarray(x, float)
    w = np.hanning(len(x))
    X = np.fft.rfft((x - x.mean()) * w)
    return np.fft.rfftfreq(len(x), 1 / fs), 2 * np.abs(X) / w.sum()


def envelope_spectrum(x, fs=FS, band=(1800.0, 5000.0)):
    """Envelope spectrum of the band-passed signal (standard bearing-fault demodulation)."""
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    y = sosfiltfilt(sos, np.asarray(x, float))
    env = np.abs(hilbert(y))
    env = env - env.mean()
    w = np.hanning(len(env))
    E = 2 * np.abs(np.fft.rfft(env * w)) / w.sum()
    return np.fft.rfftfreq(len(env), 1 / fs), E


def _peak(f, a, f0, tol):
    m = (f >= f0 - tol) & (f <= f0 + tol)
    return float(a[m].max()) if m.any() else 0.0


def extract_features(x, fs=FS, speed_hz=25.0, geom=None, n_teeth=N_TEETH):
    """Feature dict for one signal. speed_hz comes from a tachometer (assumed available)."""
    geom = geom or BearingGeometry()
    x = np.asarray(x, float)
    xc = x - x.mean()
    rms = float(np.sqrt(np.mean(xc ** 2)))
    peak = float(np.max(np.abs(xc)))
    mean_abs = float(np.mean(np.abs(xc))) + 1e-12
    f, a = amplitude_spectrum(x, fs)
    df = f[1] - f[0]
    fr = speed_hz
    tol = 0.04 * fr + df
    a1, a2, a3 = (_peak(f, a, k * fr, tol) for k in (1, 2, 3))
    fm = n_teeth * fr
    mesh = _peak(f, a, fm, 2 * tol)
    sb = 0.5 * (_peak(f, a, fm - fr, tol) + _peak(f, a, fm + fr, tol))
    fe, E = envelope_spectrum(x, fs)
    ref = float(np.median(E[(fe >= 10) & (fe <= 500)])) + 1e-12
    snr = {}
    for name, f0 in (("bpfo", geom.bpfo(fr)), ("bpfi", geom.bpfi(fr))):
        te = 0.03 * f0 + df
        snr[name] = (_peak(fe, E, f0, te) + _peak(fe, E, 2 * f0, te)) / (2 * ref)
    p = a ** 2
    return {
        "rms": rms, "peak": peak, "crest": peak / (rms + 1e-12),
        "kurtosis": float(kurtosis(xc, fisher=False)), "skewness": float(skew(xc)),
        "shape_factor": rms / mean_abs, "impulse_factor": peak / mean_abs,
        "vel_rms": velocity_rms_mm_s(x, fs),
        "a1x": a1, "a2x": a2, "a3x": a3,
        "a2x_over_1x": a2 / (a1 + 1e-12), "a3x_over_1x": a3 / (a1 + 1e-12),
        "mesh_amp": mesh, "sideband_amp": sb, "sideband_ratio": sb / (mesh + 1e-12),
        "env_snr_bpfo": snr["bpfo"], "env_snr_bpfi": snr["bpfi"],
        "centroid": float(np.sum(f * p) / np.sum(p)),
        "hf_ratio": float(p[f > 2000].sum() / p.sum()),
    }


def features_dataframe(signals, speeds, fs=FS):
    """Feature table (one row per signal)."""
    rows = [extract_features(s, fs, sp) for s, sp in zip(signals, speeds)]
    return pd.DataFrame(rows, columns=FEATURE_COLS)
