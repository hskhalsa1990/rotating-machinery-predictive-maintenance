"""Order tracking and cepstrum for machines whose speed changes (v1.4). ALL DATA IS SYNTHETIC.

Order analysis: the signal is resampled at equal SHAFT-ANGLE steps using the shaft angle from a tachometer
(assumed available; here taken from the simulator), so faults tied to the shaft (1x, 2x, gear mesh, BPFO as an
"order" of about 3.6) stay at a fixed order while a speed ramp smears them in the ordinary frequency spectrum.
Cepstrum: log-spectrum transformed back; families of harmonics/sidebands spaced by the shaft speed show up as
one peak at quefrency 1/fr.
"""
import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

from datagen import FS, N_TEETH, BearingGeometry


def angle_from_speed(speed_hz, fs=FS):
    """Shaft angle [rad] from an instantaneous speed array [Hz] (what a tachometer pulse train gives)."""
    return 2 * np.pi * np.cumsum(np.asarray(speed_hz, float)) / fs


def resample_to_angle(x, theta, spr=512, fs=FS, antialias=True):
    """Resample x at uniform shaft-angle steps of 2*pi/spr. Returns (x_angle, n_revolutions).
    A low-pass (0.45 * spr * slowest speed) is applied first so that nothing aliases in the angle domain."""
    x = np.asarray(x, float)
    theta = np.asarray(theta, float)
    if np.any(np.diff(theta) <= 0):
        raise ValueError("shaft angle must increase")
    if antialias:
        fr_min = np.min(np.diff(theta)) * fs / (2 * np.pi)
        fc = min(0.45 * fs, 0.45 * spr * fr_min)
        if fc < 0.45 * fs:
            x = sosfiltfilt(butter(6, fc, btype="lowpass", fs=fs, output="sos"), x)
    n_rev = (theta[-1] - theta[0]) / (2 * np.pi)
    m = int(np.floor(n_rev * spr))
    grid = theta[0] + np.arange(m) * (2 * np.pi / spr)
    return np.interp(grid, theta, x), m / spr


def order_spectrum(x, theta, spr=512, fs=FS):
    """(orders, amplitude) one-sided Hann-windowed amplitude spectrum in the angle domain."""
    xa, n_rev = resample_to_angle(x, theta, spr, fs)
    xa = xa - xa.mean()
    w = np.hanning(len(xa))
    A = 2 * np.abs(np.fft.rfft(xa * w)) / w.sum()
    return np.arange(len(A)) / n_rev, A


def order_envelope_spectrum(x, theta, band=(1800.0, 5000.0), spr=64, fs=FS):
    """Order-domain envelope spectrum (bearing faults appear at BPFO/BPFI orders, independent of speed)."""
    y = sosfiltfilt(butter(4, band, btype="bandpass", fs=fs, output="sos"), np.asarray(x, float))
    env = np.abs(hilbert(y))
    ea, n_rev = resample_to_angle(env, theta, spr, fs)
    ea = ea - ea.mean()
    w = np.hanning(len(ea))
    E = 2 * np.abs(np.fft.rfft(ea * w)) / w.sum()
    return np.arange(len(E)) / n_rev, E


def order_amplitude(orders, amp, order, tol=0.25):
    """Peak amplitude within +-tol orders of `order`."""
    m = (orders >= order - tol) & (orders <= order + tol)
    return float(amp[m].max()) if m.any() else 0.0


def order_track(x, theta, orders=(1, 2, 3, N_TEETH), revs_per_block=2.0, spr=512, fs=FS):
    """Amplitude of selected orders over time (order tracking by block FFT in the angle domain).

    Returns dict with t_mid [s], speed [Hz] per block and amp {order: array}."""
    xa, n_rev = resample_to_angle(x, theta, spr, fs)
    blk = int(round(revs_per_block * spr))
    step = blk // 2
    w = np.hanning(blk)
    t = np.arange(len(theta)) / fs
    th_grid = theta[0] + np.arange(len(xa)) * (2 * np.pi / spr)
    t_of_angle = np.interp(th_grid, theta, t)
    starts = np.arange(0, len(xa) - blk + 1, step)
    amp = {o: [] for o in orders}
    t_mid, speed = [], []
    for s in starts:
        seg = xa[s:s + blk]
        A = 2 * np.abs(np.fft.rfft((seg - seg.mean()) * w)) / w.sum()
        ords = np.arange(len(A)) / revs_per_block
        for o in orders:
            amp[o].append(order_amplitude(ords, A, o, tol=0.6 / revs_per_block + 0.1))
        t0, t1 = t_of_angle[s], t_of_angle[s + blk - 1]
        t_mid.append(0.5 * (t0 + t1))
        speed.append(revs_per_block / (t1 - t0))
    return {"t_mid": np.array(t_mid), "speed_hz": np.array(speed), "amp": {o: np.array(v) for o, v in amp.items()}}


def real_cepstrum(x, fs=FS, f_max=2500.0, f_min=0.0):
    """(quefrency [s], cepstrum) of the log amplitude spectrum between f_min and f_max (Hann window)."""
    x = np.asarray(x, float)
    w = np.hanning(len(x))
    X = np.abs(np.fft.rfft((x - x.mean()) * w))
    f = np.fft.rfftfreq(len(x), 1 / fs)
    L = np.log(X[(f <= f_max) & (f >= f_min)] + 1e-12)
    L = L - L.mean()
    c = np.fft.irfft(L)
    df = f[1] - f[0]
    q = np.arange(len(c)) / (len(c) * df)
    h = len(c) // 2
    return q[:h], c[:h]


def cepstral_peak_ratio(x, speed_hz, fs=FS, f_max=2500.0, tol=0.08, mesh_band=True, n_teeth=N_TEETH):
    """Cepstrum value near quefrency 1/speed (shaft-speed spaced sidebands) over the median |cepstrum| in 5-100 ms.
    mesh_band=True (default) uses only the spectrum around the gear-mesh frequency (+-15 %); the full-band version is
    dominated by noise floor and is much weaker. Absolute value: sidebands can give a negative cepstral peak."""
    if mesh_band:
        fm = n_teeth * speed_hz
        q, c = real_cepstrum(x, fs, 1.15 * fm, 0.85 * fm)
    else:
        q, c = real_cepstrum(x, fs, f_max)
    q0 = 1.0 / speed_hz
    m = (q >= q0 * (1 - tol)) & (q <= q0 * (1 + tol))
    ref = np.median(np.abs(c[(q >= 0.005) & (q <= 0.1)])) + 1e-12
    return float(np.abs(c[m]).max() / ref) if m.any() else 0.0


def ramp_experiment(ramps=((25.0, 25.0), (20.0, 30.0), (15.0, 35.0)), n=160, seed=5, fault="bearing_outer"):
    """AUC (fault vs healthy) using the envelope SNR at the fault frequency: ordinary frequency-domain envelope
    spectrum at the MEAN speed vs order-domain envelope spectrum, for different speed ramps. SYNTHETIC DATA."""
    from sklearn.metrics import roc_auc_score

    import datagen as dg
    import features as ft
    import physics_ext as pe
    geom = BearingGeometry()
    rng = np.random.default_rng(seed)
    key = {"bearing_outer": geom.bpfo, "bearing_inner": geom.bpfi}[fault]
    order = key(1.0)
    out = []
    for s0, s1 in ramps:
        y, fq, od = [], [], []
        for k in range(n):
            faulty = k % 2 == 1
            mp = dg.random_machine(rng, k)
            x = pe.generate_signal_ext({fault: float(rng.uniform(0.3, 1.0))} if faulty else {}, s0, float(rng.uniform(0.3, 1.0)),
                                       rng, mp=mp, speed_end=s1)
            _, _, th = pe.shaft_phase(len(x), dg.FS, s0, s1)
            fe, E = ft.envelope_spectrum(x)
            f0 = key(0.5 * (s0 + s1))
            tol = 0.03 * f0 + (fe[1] - fe[0])
            ref = np.median(E[(fe >= 10) & (fe <= 500)]) + 1e-12
            fq.append(np.mean([E[(fe >= h * f0 - tol) & (fe <= h * f0 + tol)].max() for h in (1, 2)]) / ref)
            oe, Eo = order_envelope_spectrum(x, th)
            refo = np.median(Eo[(oe >= 0.5) & (oe <= 30)]) + 1e-12
            od.append(np.mean([order_amplitude(oe, Eo, h * order, 0.12) for h in (1, 2)]) / refo)
            y.append(faulty)
        out.append(dict(ramp=f"{s0:g}-{s1:g} Hz", auc_frequency_domain=float(roc_auc_score(y, fq)),
                        auc_order_domain=float(roc_auc_score(y, od)), n=n))
    return out


def cepstrum_experiment(n_machines=20, per_class=6, seed=7):
    """AUC (gear_tooth vs healthy) of the cepstral peak at 1/shaft-speed vs the sideband ratio feature. SYNTHETIC DATA."""
    from sklearn.metrics import roc_auc_score

    import datagen as dg
    import features as ft
    sigs, meta = dg.generate_dataset(n_machines=n_machines, per_class=per_class, seed=seed)
    ids = meta.index[meta.fault.isin(["healthy", "gear_tooth"])]
    y = (meta.fault[ids] == "gear_tooth").astype(int).to_numpy()
    cep = [cepstral_peak_ratio(sigs[i], meta.speed_hz[i], mesh_band=False) for i in ids]
    cep_mb = [cepstral_peak_ratio(sigs[i], meta.speed_hz[i]) for i in ids]
    sb = [ft.extract_features(sigs[i], speed_hz=meta.speed_hz[i])["sideband_ratio"] for i in ids]
    sev = meta.severity[ids].to_numpy()
    low = (y == 0) | (sev < 0.4)
    return dict(auc_cepstrum=float(roc_auc_score(y, cep)), auc_cepstrum_mesh_band=float(roc_auc_score(y, cep_mb)),
                auc_sideband_ratio=float(roc_auc_score(y, sb)),
                auc_cepstrum_weak=float(roc_auc_score(y[low], np.array(cep_mb)[low])),
                auc_sideband_weak=float(roc_auc_score(y[low], np.array(sb)[low])), n=len(y))
