"""Synthetic maintenance-log text (template based) for NLP fault classification."""
import re

import numpy as np
import pandas as pd

from datagen import FAULTS

TEMPLATES = {
    "healthy": ["routine greasing done, vibration normal", "checked motor, no abnormal noise, readings within limit",
                "scheduled inspection ok, bearing temperature normal", "cleaned filters, all parameters fine",
                "belt tension checked, no issue found", "monthly round complete, machine running smooth"],
    "imbalance": ["vibration high at 1x, dust buildup on fan blades", "rotor imbalance suspected, trim weight added",
                  "heavy vibration after impeller cleaning, balancing required", "1x peak high, uneven mass on rotor",
                  "fan wobbling at running speed, blade deposit removed and balanced"],
    "misalignment": ["coupling misalignment found, 2x vibration high", "shaft alignment out of tolerance, laser alignment done",
                     "axial vibration high, coupling gap uneven", "motor and pump shaft offset, re-aligned and shims added",
                     "foot soft, angular misalignment at coupling"],
    "bearing_outer": ["outer race defect noticed, bearing replaced", "rumbling noise from DE bearing, outer race spalling",
                      "envelope spectrum shows BPFO peaks, brg changed", "bearing noise high, pitting on outer ring",
                      "grinding sound from drive end bearing, outer race damage"],
    "bearing_inner": ["inner race fault found on NDE bearing", "BPFI peaks with sidebands, inner ring spalling",
                      "bearing squeal, inner race crack seen, bearing replaced", "high frequency vibration, inner race wear",
                      "inner ring defect suspected from envelope analysis"],
    "gear_tooth": ["gearbox whining noise, worn gear tooth", "gear mesh sidebands increased, tooth pitting seen",
                   "oil has metal particles, gear tooth damage", "gearbox noise high, broken tooth found on pinion",
                   "mesh frequency vibration rising, gear wear confirmed"],
}
SHARED = ["vibration high", "checked on night shift", "please monitor", "temperature slightly up",
          "operator reported abnormal sound", "work order raised", "awaaz zyada", "machine garam hai"]
ABBREV = {"vibration": "vib", "bearing": "brg", "coupling": "cplg", "replaced": "repl"}


def make_log(fault: str, rng) -> str:
    text = str(rng.choice(TEMPLATES[fault]))
    if rng.random() < 0.5:
        text += ", " + str(rng.choice(SHARED))
    if rng.random() < 0.3:
        for k, v in ABBREV.items():
            text = text.replace(k, v)
    if rng.random() < 0.15:  # typo: drop a random character
        i = int(rng.integers(0, len(text)))
        text = text[:i] + text[i + 1:]
    return text


def generate_logs(n: int = 1500, seed: int = 0, label_noise: float = 0.04) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = rng.choice(FAULTS, size=n)
    texts = [make_log(f, rng) for f in labels]
    noisy = [str(rng.choice(FAULTS)) if rng.random() < label_noise else f for f in labels]
    return pd.DataFrame({"text": texts, "fault": noisy})


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", t.lower())).strip()
