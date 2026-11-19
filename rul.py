"""Health index, degradation fitting and remaining-useful-life (RUL) with bootstrap uncertainty."""
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

HI_COLS = ["vel_rms", "kurtosis", "a2x_over_1x", "sideband_ratio", "env_snr_bpfo", "env_snr_bpfi", "crest", "hf_ratio"]


def health_index(feat: pd.DataFrame, baseline_n: int = 8, cols=HI_COLS, smooth_span: int = 5) -> np.ndarray:
    """Distance of log-features from the machine's own early-life baseline (increases only)."""
    X = np.log1p(feat[list(cols)].clip(lower=0).to_numpy(float))
    base = X[:baseline_n]
    mu, sd = base.mean(0), np.maximum(base.std(0), 0.1)
    z = np.clip((X - mu) / sd, 0, 20)
    hi = np.sqrt(np.mean(z ** 2, axis=1))
    return pd.Series(hi).ewm(span=smooth_span).mean().to_numpy()


def _exp_model(t, c, a, b):
    return c + a * np.exp(b * t)


def _pow_model(t, c, a, p):
    return c + a * np.power(t + 1.0, p)


def fit_degradation(t, y, kind: str = "pow"):
    """Fit a degradation curve. kind 'pow' (c + a*(t+1)^p, default) or 'exp' (c + a*exp(b*t)).
    Falls back to a straight line if the fit fails. In our tests 'pow' had lower bias than 'exp'."""
    t, y = np.asarray(t, float), np.asarray(y, float)
    try:
        if kind == "exp":
            p0 = [float(y[:5].mean()), max(float(y[-1] - y[0]), 1e-3) * 0.2, 0.03]
            popt, _ = curve_fit(_exp_model, t, y, p0=p0, bounds=([-np.inf, 0, 1e-4], [np.inf, np.inf, 0.5]), maxfev=5000)
            return ("exp", popt)
        popt, _ = curve_fit(_pow_model, t, y, p0=[float(y[:5].mean()), 0.01, 2.0],
                            bounds=([-np.inf, 0, 0.8], [np.inf, np.inf, 5.0]), maxfev=8000)
        return ("pow", popt)
    except Exception:
        return ("lin", np.polyfit(t, y, 1))


def predict_curve(model, t):
    kind, p = model
    t = np.asarray(t, float)
    if kind == "exp":
        return _exp_model(t, *p)
    if kind == "pow":
        return _pow_model(t, *p)
    return p[0] * t + p[1]


def time_to_threshold(model, thr, t_now, cap=200.0):
    """Remaining steps until the fitted curve reaches thr (0 if already above, cap if never)."""
    kind, p = model
    if predict_curve(model, t_now) >= thr:
        return 0.0
    if kind == "exp":
        c, a, b = p
        val = (thr - c) / a
        return float(cap) if val <= 0 else float(np.clip(np.log(val) / b - t_now, 0, cap))
    if kind == "pow":
        c, a, pw = p
        val = (thr - c) / a
        return float(cap) if val <= 0 else float(np.clip(val ** (1.0 / pw) - 1.0 - t_now, 0, cap))
    slope, icpt = p
    return float(cap) if slope <= 0 else float(np.clip((thr - icpt) / slope - t_now, 0, cap))


def rul_with_uncertainty(t, y, thr, n_boot: int = 200, cap: float = 200.0, seed: int = 0, kind: str = "pow"):
    """RUL estimate: median and 5-95 % interval from residual bootstrap of the degradation fit.

    NOTE: residual bootstrap ignores autocorrelation and model error, so the intervals are
    usually too narrow. Check coverage on held-out machines (done in the notebook).
    """
    rng = np.random.default_rng(seed)
    t, y = np.asarray(t, float), np.asarray(y, float)
    model = fit_degradation(t, y, kind)
    fit = predict_curve(model, t)
    resid = y - fit
    t_now = t[-1]
    samples = [time_to_threshold(model, thr, t_now, cap)]
    for _ in range(n_boot):
        m = fit_degradation(t, fit + rng.choice(resid, len(resid)), kind)
        samples.append(time_to_threshold(m, thr, t_now, cap))
    s = np.array(samples)
    return {"median": float(np.median(s)), "lo": float(np.percentile(s, 5)),
            "hi": float(np.percentile(s, 95)), "samples": s, "model": model}


def holt_forecast(y, alpha: float = 0.5, beta: float = 0.2, horizon: int = 10):
    """Holt linear-trend exponential smoothing. Returns (fitted, forecast)."""
    y = np.asarray(y, float)
    level, trend = y[0], y[1] - y[0]
    fitted = [y[0]]
    for v in y[1:]:
        fitted.append(level + trend)
        prev = level
        level = alpha * v + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
    return np.array(fitted), np.array([level + (h + 1) * trend for h in range(horizon)])


def calibrate_interval(pred_median, true_rul, alpha: float = 0.1):
    """Split-conformal-style calibration: empirical quantiles of (true - predicted) RUL
    measured on TRAINING machines. Returns (lower_offset, upper_offset)."""
    err = np.asarray(true_rul, float) - np.asarray(pred_median, float)
    return float(np.quantile(err, alpha / 2)), float(np.quantile(err, 1 - alpha / 2))


def apply_calibration(median: float, offsets):
    lo, hi = offsets
    return max(0.0, median + lo), max(0.0, median + hi)
