"""Extended physics for the synthetic vibration generator (ALL DATA SYNTHETIC).

Adds to datagen.py (which stays unchanged, so earlier results do not move):
- several faults at the same time (multi-fault superposition)
- shaft speed that ramps (run-up / coast-down) or ripples; harmonics and impact trains
  follow the instantaneous shaft ANGLE, so order lines smear in a physically sensible way
- random slip (cage slip / jitter) of the bearing impacts, expressed as a fraction of the period
- isolated shock transients (a hit that rings the structural resonance once)

Simplified physics only. Not validated against real machines.
"""
import numpy as np
import pandas as pd

from datagen import FAULTS, FS, N_SAMPLES, N_TEETH, BearingGeometry, MachineParams, random_machine

FAULT_NAMES = FAULTS[1:]          # the five real faults; "healthy" means no entry

PAIR_COMBOS = [("bearing_outer", "imbalance"), ("bearing_inner", "misalignment"),
               ("gear_tooth", "imbalance"), ("bearing_outer", "gear_tooth"),
               ("imbalance", "misalignment")]


def shaft_phase(n, fs, speed_hz, speed_end=None, ripple=0.0, ripple_hz=1.5, phase0=0.0):
    """Return (time, instantaneous speed [Hz], shaft angle [rad]).

    speed_end ramps the speed linearly over the record; ripple adds a sinusoidal speed fluctuation."""
    t = np.arange(n) / fs
    end = speed_hz if speed_end is None else speed_end
    fr = speed_hz + (end - speed_hz) * t / max(t[-1], 1e-12)
    if ripple:
        fr = fr * (1 + ripple * np.sin(2 * np.pi * ripple_hz * t + phase0))
    theta = 2 * np.pi * np.cumsum(fr) / fs
    return t, fr, theta


def _impacts(n, fs, t, theta, order, f_res, amp, rng, slip, am_with_shaft=False, damping=900.0):
    """Impacts every 1/order shaft turns (angle domain), jittered by `slip` (fraction of a period)."""
    x = np.zeros(n)
    cycles = theta * order / (2 * np.pi)
    ck = np.arange(np.ceil(cycles[0]) + rng.uniform(0, 1), cycles[-1], 1.0)
    ck = ck + rng.normal(0, slip, ck.size)
    ck = ck[(ck >= cycles[0]) & (ck <= cycles[-1])]
    te = np.interp(ck, cycles, t)
    m = int(0.005 * fs)
    tt = np.arange(m) / fs
    h = np.exp(-damping * tt) * np.sin(2 * np.pi * f_res * tt)
    for tk in te:
        i = int(tk * fs)
        if i < 0 or i >= n:
            continue
        a = amp * (1 + 0.25 * rng.standard_normal())
        if am_with_shaft:
            a *= 1 + 0.5 * np.cos(np.interp(tk, t, theta))
        end = min(n, i + m)
        x[i:end] += a * h[: end - i]
    return x


def generate_signal_ext(faults, speed_hz, load, rng, fs=FS, n=N_SAMPLES, mp=None, geom=None,
                        speed_end=None, ripple=0.0, slip=0.01, shock_amp=0.0, n_shocks=1):
    """One synthetic acceleration signal [m/s^2] with any combination of faults.

    faults: dict {fault_name: severity in [0, 1.2]}; empty dict = healthy machine."""
    for f in faults:
        if f not in FAULT_NAMES:
            raise ValueError(f"unknown fault: {f}")
    mp = mp or MachineParams()
    geom = geom or BearingGeometry()
    ph = rng.uniform(0, 2 * np.pi, 6)
    t, fr, theta = shaft_phase(n, fs, speed_hz, speed_end, ripple, phase0=ph[5])
    sev = {k: float(np.clip(v, 0, 1.2)) for k, v in faults.items()}
    a1 = 0.12 + 0.08 * load
    a2, a3 = 0.03, 0.015
    mesh_a, side_m = 0.05, 0.05
    s = sev.get("imbalance", 0.0)
    a1 = a1 + 1.6 * s * (fr / 25.0) ** 2                      # centrifugal force ~ speed^2 (time varying)
    s = sev.get("misalignment", 0.0)
    a1, a2, a3 = a1 + 0.2 * s, a2 + 1.0 * s, a3 + 0.45 * s
    s = sev.get("gear_tooth", 0.0)
    mesh_a += 0.35 * s
    side_m += 1.5 * s
    x = a1 * np.sin(theta + ph[0]) + a2 * np.sin(2 * theta + ph[1]) + a3 * np.sin(3 * theta + ph[2])
    x += mesh_a * (1 + side_m * np.cos(theta + ph[3])) * np.sin(N_TEETH * theta + ph[4])
    if "bearing_outer" in sev:
        x += _impacts(n, fs, t, theta, geom.bpfo(1.0), mp.f_res, 3.0 * sev["bearing_outer"] * (0.5 + load), rng, slip)
    if "bearing_inner" in sev:
        x += _impacts(n, fs, t, theta, geom.bpfi(1.0), mp.f_res, 2.6 * sev["bearing_inner"] * (0.5 + load), rng, slip,
                      am_with_shaft=True)
    if shock_amp > 0:                                          # isolated hits (not periodic)
        pos = rng.uniform(0.1, 0.9, n_shocks)
        m = int(0.008 * fs)
        tt = np.arange(m) / fs
        h = np.exp(-600.0 * tt) * np.sin(2 * np.pi * mp.f_res * tt)
        for p in pos:
            i = int(p * n)
            end = min(n, i + m)
            x[i:end] += shock_amp * h[: end - i]
    x = x * mp.gain
    x = x + rng.normal(0, mp.noise * (0.25 + 0.15 * load) / 0.3, n)
    return x.astype(np.float32)


def generate_multifault_dataset(n_machines=20, per_combo=3, seed=11, sev_range=(0.3, 1.0)):
    """Labelled dataset with healthy, 5 single faults and 5 fault pairs (11 combinations).

    Returns (signals [N, n], meta DataFrame with multi-hot columns y_<fault>, n_faults, combo)."""
    rng = np.random.default_rng(seed)
    combos = [()] + [(f,) for f in FAULT_NAMES] + PAIR_COMBOS
    sigs, rows = [], []
    for m in range(n_machines):
        mp = random_machine(rng, m)
        for combo in combos:
            for _ in range(per_combo):
                faults = {f: float(rng.uniform(*sev_range)) for f in combo}
                speed = float(rng.uniform(20, 30))
                load = float(rng.uniform(0.3, 1.0))
                sigs.append(generate_signal_ext(faults, speed, load, rng, mp=mp))
                row = dict(sample_id=len(rows), machine_id=m, combo="+".join(combo) or "healthy",
                           n_faults=len(combo), speed_hz=speed, load=load)
                row.update({f"y_{f}": int(f in combo) for f in FAULT_NAMES})
                rows.append(row)
    return np.asarray(sigs, dtype=np.float32), pd.DataFrame(rows)
