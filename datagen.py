"""Physics-based synthetic vibration generator.

ALL DATA PRODUCED HERE IS SYNTHETIC. Signals are simulated accelerations in m/s^2.
Fault physics (simplified): shaft harmonics (imbalance, misalignment), bearing
defect frequencies (BPFO/BPFI) exciting a structural resonance, and gear-mesh
tones with shaft-speed sidebands.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

FAULTS = ["healthy", "imbalance", "misalignment", "bearing_outer", "bearing_inner", "gear_tooth"]
FS = 12000          # sampling rate [Hz]
N_SAMPLES = 4096    # samples per signal (about 0.34 s)
N_TEETH = 23        # teeth on the monitored gear (generic value)


@dataclass(frozen=True)
class BearingGeometry:
    """Generic deep-groove bearing geometry (illustrative numbers, mm)."""
    n_balls: int = 9
    d_ball: float = 7.94
    d_pitch: float = 39.04
    contact_angle_deg: float = 0.0

    def _r(self) -> float:
        return self.d_ball / self.d_pitch * np.cos(np.radians(self.contact_angle_deg))

    def bpfo(self, fr: float) -> float:
        """Ball pass frequency, outer race [Hz] for shaft speed fr [Hz]."""
        return self.n_balls / 2 * fr * (1 - self._r())

    def bpfi(self, fr: float) -> float:
        """Ball pass frequency, inner race [Hz]."""
        return self.n_balls / 2 * fr * (1 + self._r())

    def bsf(self, fr: float) -> float:
        """Ball spin frequency [Hz]."""
        return self.d_pitch / (2 * self.d_ball) * fr * (1 - self._r() ** 2)

    def ftf(self, fr: float) -> float:
        """Fundamental train (cage) frequency [Hz]."""
        return fr / 2 * (1 - self._r())


@dataclass(frozen=True)
class MachineParams:
    """Per-machine nuisance parameters (sensor gain, structural resonance, noise)."""
    machine_id: int = 0
    gain: float = 1.0
    f_res: float = 3000.0
    noise: float = 0.3


def random_machine(rng, machine_id: int) -> MachineParams:
    return MachineParams(machine_id, float(rng.uniform(0.8, 1.2)),
                         float(rng.uniform(2500, 3500)), float(rng.uniform(0.22, 0.38)))


def _impulse_train(n, fs, f_fault, f_res, amp, rng, am_freq=None, slip=0.01, damping=900.0):
    """Periodic impacts at f_fault, each ringing a decaying resonance at f_res."""
    x = np.zeros(n)
    period = fs / f_fault
    m = int(0.005 * fs)
    tt = np.arange(m) / fs
    h = np.exp(-damping * tt) * np.sin(2 * np.pi * f_res * tt)
    pos = np.arange(rng.uniform(0, period), n, period)
    pos = pos + rng.normal(0, slip * period, pos.size)  # random slip
    for p in pos:
        i = int(p)
        if i < 0 or i >= n:
            continue
        a = amp * (1 + 0.25 * rng.standard_normal())
        if am_freq:
            a *= 1 + 0.5 * np.cos(2 * np.pi * am_freq * i / fs)
        end = min(n, i + m)
        x[i:end] += a * h[: end - i]
    return x


def generate_signal(fault, severity, speed_hz, load, rng, fs=FS, n=N_SAMPLES, mp=None, geom=None):
    """One synthetic acceleration signal [m/s^2]. severity in [0, 1]."""
    if fault not in FAULTS:
        raise ValueError(f"unknown fault: {fault}")
    mp = mp or MachineParams()
    geom = geom or BearingGeometry()
    sev = float(np.clip(severity, 0, 1.2))
    t = np.arange(n) / fs
    fr = speed_hz
    ph = rng.uniform(0, 2 * np.pi, 6)
    a1 = 0.12 + 0.08 * load
    a2, a3 = 0.03, 0.015
    mesh_a, side_m = 0.05, 0.05
    if fault == "imbalance":          # centrifugal force grows with speed^2
        a1 += 1.6 * sev * (fr / 25.0) ** 2
    elif fault == "misalignment":     # strong 2x, some 3x
        a1 += 0.2 * sev
        a2 += 1.0 * sev
        a3 += 0.45 * sev
    elif fault == "gear_tooth":
        mesh_a += 0.35 * sev
        side_m += 1.5 * sev
    x = (a1 * np.sin(2 * np.pi * fr * t + ph[0]) + a2 * np.sin(2 * np.pi * 2 * fr * t + ph[1])
         + a3 * np.sin(2 * np.pi * 3 * fr * t + ph[2]))
    fm = N_TEETH * fr
    x += mesh_a * (1 + side_m * np.cos(2 * np.pi * fr * t + ph[3])) * np.sin(2 * np.pi * fm * t + ph[4])
    if fault == "bearing_outer":
        x += _impulse_train(n, fs, geom.bpfo(fr), mp.f_res, 3.0 * sev * (0.5 + load), rng)
    elif fault == "bearing_inner":
        x += _impulse_train(n, fs, geom.bpfi(fr), mp.f_res, 2.6 * sev * (0.5 + load), rng, am_freq=fr)
    x = x * mp.gain
    x += rng.normal(0, mp.noise * (0.25 + 0.15 * load) / 0.3, n)
    return x.astype(np.float32)


def generate_dataset(n_machines=30, per_class=8, seed=42, sev_range=(0.15, 1.0)):
    """Labelled dataset. Returns (signals [N, n], meta DataFrame)."""
    rng = np.random.default_rng(seed)
    sigs, rows = [], []
    for m in range(n_machines):
        mp = random_machine(rng, m)
        for fault in FAULTS:
            for _ in range(per_class):
                sev = 0.0 if fault == "healthy" else float(rng.uniform(*sev_range))
                speed = float(rng.uniform(20, 30))
                load = float(rng.uniform(0.3, 1.0))
                sigs.append(generate_signal(fault, sev, speed, load, rng, mp=mp))
                rows.append(dict(sample_id=len(rows), machine_id=m, fault=fault,
                                 label=FAULTS.index(fault), severity=sev, speed_hz=speed,
                                 load=load, gain=mp.gain, f_res=mp.f_res))
    return np.asarray(sigs, dtype=np.float32), pd.DataFrame(rows)


def run_to_failure(fault, n_steps, seed, machine_id=0, fs=FS, n=N_SAMPLES, start_sev=0.12, power=None):
    """Simulate one machine degrading until severity reaches 1.0 at step n_steps.

    Returns (signals [n_steps, n], meta). true_rul = n_steps - t (in steps).
    """
    rng = np.random.default_rng(seed)
    mp = random_machine(rng, machine_id)
    p = power if power else float(rng.uniform(1.5, 3.0))
    steps = np.arange(n_steps)
    sev = start_sev + (1 - start_sev) * (steps / n_steps) ** p
    sigs, rows = [], []
    for t, s in zip(steps, sev):
        speed = float(rng.uniform(22, 28))
        load = float(rng.uniform(0.4, 0.9))
        sigs.append(generate_signal(fault, s, speed, load, rng, fs, n, mp=mp))
        rows.append(dict(machine_id=machine_id, fault=fault, t=int(t), severity=float(s),
                         true_rul=int(n_steps - t), speed_hz=speed, load=load))
    return np.asarray(sigs, dtype=np.float32), pd.DataFrame(rows)
