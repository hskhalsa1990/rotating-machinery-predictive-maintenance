import numpy as np

import dashboard_logic as dl


def test_simulate_deterministic_and_noise_scale():
    a = dl.simulate("healthy", 0, 25, 0.5, seed=1)
    assert np.array_equal(a, dl.simulate("healthy", 0, 25, 0.5, seed=1))
    assert dl.simulate("healthy", 0, 25, 0.5, seed=1, noise_scale=3).std() > a.std()


def test_analyze_zone_and_peaks():
    s = dl.simulate("imbalance", 1.0, 25, 0.8, seed=2)
    r = dl.analyze(s, 25)
    assert r["zone"] in "CD" and r["bpfo"] > 0 and len(r["freq"]) == len(r["amp"])
    h = dl.analyze(dl.simulate("healthy", 0, 25, 0.5, seed=2), 25)
    assert h["zone"] in "AB"


def test_model_diagnoses_clear_faults():
    m = dl.train_model(n_machines=8, per_class=4, seed=0)
    hits = 0
    for fault in ("imbalance", "misalignment", "gear_tooth"):
        s = dl.simulate(fault, 0.9, 25, 0.7, seed=5)
        p = dl.diagnose(m, dl.analyze(s, 25)["features"])
        assert abs(sum(p.values()) - 1) < 1e-6
        hits += next(iter(p)) == fault
    assert hits >= 2


def test_cost_comparison_break_even():
    o = dl.cost_comparison(0.0, 30, 35000, 8, 250000, 48, 6000)
    assert o["wait"] < o["repair_now"] and 0 < o["break_even_p"] < 1
    o2 = dl.cost_comparison(1.0, 30, 35000, 8, 250000, 48, 6000)
    assert o2["wait"] > o2["repair_now"]


def test_novelty_report_flags_fault_not_healthy():
    det = dl.train_novelty(n_machines=10, per_class=8, seed=1)
    out = {}
    for fault, sev in (("healthy", 0.0), ("bearing_outer", 0.9)):
        sig = dl.simulate(fault, sev, 25.0, 0.7, seed=5)
        out[fault] = dl.novelty_report(det, dl.analyze(sig, 25.0)["features"], sig)
    assert out["bearing_outer"]["anomaly"] and out["bearing_outer"]["score"] > out["healthy"]["score"]
    assert out["bearing_outer"]["band_lo"] < out["bearing_outer"]["band_hi"]
