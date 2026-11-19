"""Sensor-data cleaning: duplicates, flat-lined sensors, outliers (Hampel), missing values."""
import numpy as np
import pandas as pd


def hampel_mask(s: pd.Series, window: int = 7, n_sigmas: float = 3.5) -> pd.Series:
    """True where a point deviates from the rolling median by > n_sigmas * robust sigma."""
    med = s.rolling(window, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(window, center=True, min_periods=1).median()
    sigma = 1.4826 * mad
    floor = 0.02 * (s.abs().median() + 1e-9)
    return ((s - med).abs() > n_sigmas * np.maximum(sigma, floor)) & s.notna()


def flatline_mask(s: pd.Series, min_len: int = 5) -> pd.Series:
    """True for runs of >= min_len identical consecutive values (stuck sensor)."""
    grp = (s != s.shift()).cumsum()
    size = s.groupby(grp).transform("size")
    return (size >= min_len) & s.notna()


def make_messy(df: pd.DataFrame, cols=("vel_rms", "temp_c"), seed: int = 0) -> pd.DataFrame:
    """Inject realistic sensor problems (for demonstration only)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for c in cols:
        out[c] = out[c].astype(float)
        n = len(out)
        out.loc[out.index[rng.random(n) < 0.03], c] = np.nan                    # dropouts
        spikes = out.index[rng.random(n) < 0.01]
        out.loc[spikes, c] = out.loc[spikes, c] * rng.uniform(5, 10, len(spikes))  # spikes
    for mid in out["machine_id"].unique()[::4]:                                  # stuck sensors
        idx = out.index[out["machine_id"] == mid]
        start = int(rng.integers(10, len(idx) - 20))
        out.loc[idx[start:start + 8], cols[0]] = out.loc[idx[start], cols[0]]
    dup = out.sample(frac=0.01, random_state=seed)                               # duplicates
    return pd.concat([out, dup], ignore_index=True)


def clean_readings(df: pd.DataFrame, cols=("vel_rms", "temp_c"), max_gap: int = 3):
    """Full cleaning pipeline. Returns (clean_df, report dict)."""
    rep = {"rows_in": len(df)}
    d = df.drop_duplicates(subset=["machine_id", "day"], keep="first")
    rep["duplicates_removed"] = rep["rows_in"] - len(d)
    d = d.sort_values(["machine_id", "day"]).reset_index(drop=True)
    rep.update({"missing_before": int(d[list(cols)].isna().sum().sum()), "flatline_points": 0, "outliers": 0})
    parts = []
    for _, g in d.groupby("machine_id"):
        g = g.copy()
        for c in cols:
            s = g[c].astype(float)
            fl = flatline_mask(s) if c == cols[0] else pd.Series(False, index=s.index)
            rep["flatline_points"] += int(fl.sum())
            s = s.mask(fl)
            ol = hampel_mask(s)
            rep["outliers"] += int(ol.sum())
            s = s.mask(ol).interpolate(limit=max_gap, limit_direction="both").clip(lower=0)
            g[c] = s
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    rep["missing_after"] = int(out[list(cols)].isna().sum().sum())
    return out, rep
