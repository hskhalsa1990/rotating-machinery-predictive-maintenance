import numpy as np
import features as ft


def _sine(a=1.0, f=25.0, fs=12000, n=4096):
    return a * np.sin(2 * np.pi * f * np.arange(n) / fs)


def test_rms_of_sine():
    assert abs(ft.extract_features(_sine(2.0), speed_hz=25)["rms"] - 2 / np.sqrt(2)) < 0.02


def test_velocity_rms_of_sine():
    expected = 1.0 / (2 * np.pi * 25) / np.sqrt(2) * 1000
    assert abs(ft.velocity_rms_mm_s(_sine(1.0, 25.0 * 4096 / 12000 * 12000 / 4096)) - expected) / expected < 0.05


def test_crest_and_kurtosis():
    f = ft.extract_features(_sine(f=12000 * 8 / 4096), speed_hz=25)  # whole number of cycles
    assert abs(f["crest"] - np.sqrt(2)) < 0.05
    g = ft.extract_features(np.random.default_rng(0).normal(size=4096), speed_hz=25)
    assert abs(g["kurtosis"] - 3) < 0.4


def test_feature_columns_complete():
    f = ft.extract_features(np.random.default_rng(1).normal(size=4096), speed_hz=25)
    assert set(f) == set(ft.FEATURE_COLS)
