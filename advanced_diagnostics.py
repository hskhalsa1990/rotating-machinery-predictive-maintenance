"""Advanced diagnostics (v1.3): spectral-kurtosis kurtogram with automatic band selection, and a
healthy-only novelty detector (no fault labels needed). ALL DATA IS SYNTHETIC.

Kurtogram: for several STFT window lengths, spectral kurtosis SK(f) = E|X|^4 / (E|X|^2)^2 - 2 is computed
per frequency; impulsive (bearing) energy shows up as a high-SK band. The best band feeds the envelope
spectrum, so the demodulation band no longer has to be fixed in advance (features.py uses 1800-5000 Hz).
"""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt, stft

from datagen import FS, BearingGeometry

DEFAULT_NPERSEG = (16, 32, 64)  # short windows: bearing impacts ring only about 1-3 ms (checked on seed 5, confirmed on other seeds)


def spectral_kurtosis(x, fs=FS, nperseg=128):
    """Spectral kurtosis per frequency bin from an STFT (Gaussian noise -> about 0). Returns (freqs, sk)."""
    x = np.asarray(x, float)
    f, _, Z = stft(x - x.mean(), fs=fs, nperseg=nperseg, noverlap=nperseg * 3 // 4, boundary=None)
    p2 = np.mean(np.abs(Z) ** 2, axis=1)
    p4 = np.mean(np.abs(Z) ** 4, axis=1)
    return f, p4 / (p2 ** 2 + 1e-24) - 2.0


def kurtogram(x, fs=FS, nperseg_list=DEFAULT_NPERSEG):
    """Matrix of SK values (rows = window length, columns = frequency bins of the longest window grid)."""
    f_ref = np.fft.rfftfreq(max(nperseg_list), 1 / fs)
    rows = []
    for n in nperseg_list:
        f, sk = spectral_kurtosis(x, fs, n)
        rows.append(np.interp(f_ref, f, sk))
    return f_ref, np.asarray(rows)


def best_band(x, fs=FS, nperseg_list=DEFAULT_NPERSEG, min_hz=500.0):
    """(f_lo, f_hi, sk_max): band around the SK maximum above min_hz, width = bin spacing of the best window."""
    f, K = kurtogram(x, fs, nperseg_list)
    K = K.copy()
    K[:, f < min_hz] = -np.inf
    K[:, f > 0.95 * fs / 2] = -np.inf
    i, j = np.unravel_index(np.argmax(K), K.shape)
    width = 2.0 * fs / nperseg_list[i]
    lo = max(min_hz, f[j] - width / 2)
    hi = min(0.95 * fs / 2, f[j] + width / 2)
    return float(lo), float(hi), float(K[i, j])


def envelope_spectrum_band(x, band, fs=FS):
    """Envelope spectrum of x band-passed to `band` (Hz)."""
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    env = np.abs(hilbert(sosfiltfilt(sos, np.asarray(x, float))))
    env -= env.mean()
    w = np.hanning(len(env))
    return np.fft.rfftfreq(len(env), 1 / fs), 2 * np.abs(np.fft.rfft(env * w)) / w.sum()


def band_envelope_snr(x, speed_hz, fs=FS, band=None, geom=None):
    """Envelope SNR at BPFO/BPFI (peak over local median) using `band`, or the kurtogram's best band."""
    geom = geom or BearingGeometry()
    if band is None:
        band = best_band(x, fs)[:2]
    fe, E = envelope_spectrum_band(x, band, fs)
    ref = float(np.median(E[(fe >= 10) & (fe <= 500)])) + 1e-12
    df = fe[1] - fe[0]
    out = {}
    for name, f0 in (("bpfo", geom.bpfo(speed_hz)), ("bpfi", geom.bpfi(speed_hz))):
        tol = 0.03 * f0 + df
        pk = [E[(fe >= k * f0 - tol) & (fe <= k * f0 + tol)].max() for k in (1, 2)]
        out[name] = float(np.mean(pk) / ref)
    return out, band


class HealthyNovelty:
    """Robust Mahalanobis novelty detector fitted on HEALTHY feature rows only.

    score = Mahalanobis distance of log-features (median/MAD scaling + shrunk covariance).
    threshold = quantile of OUT-OF-FOLD healthy scores (default 99 %); in-sample scores would be too optimistic
    and cause far more false alarms on new machines.
    """

    def __init__(self, quantile=0.99, shrink=0.3, n_folds=5):
        self.quantile, self.shrink, self.n_folds = quantile, shrink, n_folds

    @staticmethod
    def _t(X):
        return np.log1p(np.abs(np.asarray(X, float)))

    def _fit_core(self, X):
        Z = self._t(X)
        self.med_ = np.median(Z, axis=0)
        self.mad_ = 1.4826 * np.median(np.abs(Z - self.med_), axis=0) + 1e-9
        W = (Z - self.med_) / self.mad_
        C = np.cov(W, rowvar=False)
        C = (1 - self.shrink) * C + self.shrink * np.eye(C.shape[0]) * np.trace(C) / C.shape[0]
        self.prec_ = np.linalg.pinv(C)

    def fit(self, X_healthy, groups=None):
        """groups (e.g. machine ids) keeps all rows of one machine in the same fold for the threshold."""
        X = np.asarray(X_healthy, float)
        n = len(X)
        if groups is None:
            folds = np.arange(n) % self.n_folds
        else:
            _, g = np.unique(np.asarray(groups), return_inverse=True)
            folds = g % self.n_folds
        oof = np.zeros(n)
        for k in range(self.n_folds):
            te = folds == k
            if te.all() or not te.any():
                continue
            self._fit_core(X[~te])
            oof[te] = self.score(X[te])
        self._fit_core(X)
        self.threshold_ = float(np.quantile(oof, self.quantile))
        return self

    def score(self, X):
        W = (self._t(X) - self.med_) / self.mad_
        return np.sqrt(np.einsum("ij,jk,ik->i", W, self.prec_, W))

    def is_anomaly(self, X):
        return self.score(X) > self.threshold_


def resonance_shift_experiment(res_ranges=((2500, 3500), (1000, 1700), (5000, 5600)), n=240, seed=21):
    """AUC of outer-race detection (healthy vs bearing_outer) with the fixed 1800-5000 Hz band vs the kurtogram band,
    for machines whose structural resonance lies in each range. Returns a list of dicts. SYNTHETIC DATA."""
    from sklearn.metrics import roc_auc_score

    import datagen as dg
    import features as ft
    rng = np.random.default_rng(seed)
    out = []
    for lo, hi in res_ranges:
        y, kur, fix = [], [], []
        for k in range(n):
            fault = "bearing_outer" if k % 2 else "healthy"
            mp = dg.MachineParams(k, float(rng.uniform(0.8, 1.2)), float(rng.uniform(lo, hi)), float(rng.uniform(0.22, 0.38)))
            sp = float(rng.uniform(20, 30))
            sev = 0.0 if fault == "healthy" else float(rng.uniform(0.2, 1.0))
            s = dg.generate_signal(fault, sev, sp, float(rng.uniform(0.3, 1.0)), rng, mp=mp)
            y.append(fault != "healthy")
            kur.append(band_envelope_snr(s, sp)[0]["bpfo"])
            fix.append(ft.extract_features(s, speed_hz=sp)["env_snr_bpfo"])
        out.append(dict(resonance_hz=f"{lo}-{hi}", auc_fixed_band=float(roc_auc_score(y, fix)),
                        auc_kurtogram=float(roc_auc_score(y, kur)), n=n))
    return out
