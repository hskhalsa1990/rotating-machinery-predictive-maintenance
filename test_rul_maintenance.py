import numpy as np
import rul
import maintenance as mt
import db
import textlogs


def test_rul_on_clean_curve():
    t = np.arange(60)
    y = 1 + 0.002 * (t + 1) ** 2.2
    thr = 1 + 0.002 * 100 ** 2.2
    r = rul.rul_with_uncertainty(t, y, thr, n_boot=20)
    assert abs(r["median"] - (99 - 59)) < 5 and r["lo"] <= r["median"] <= r["hi"]


def test_holt_follows_trend():
    _, fc = rul.holt_forecast(np.arange(20, dtype=float), horizon=3)
    assert abs(fc[0] - 20) < 1.5 and fc[2] > fc[0]


def test_calibration_offsets_order():
    lo, hi = rul.calibrate_interval([10, 20, 30, 40], [12, 18, 35, 38])
    assert lo <= hi


def test_costs_and_probability():
    cp = mt.CostParams()
    assert cp.unplanned_total > cp.planned_total
    assert mt.failure_probability([1, 2, 30], 5) == 2 / 3
    c = mt.expected_costs(0.9, 50, cp)
    assert c["repair_now"] < c["wait"]


def test_plan_action_rules():
    soon = {"samples": np.array([2.0, 3.0, 4.0]), "lo": 2, "median": 3}
    far = {"samples": np.array([90.0, 100.0]), "lo": 90, "median": 95}
    assert mt.plan_action("D", far) == "STOP_AND_INSPECT_NOW"
    assert mt.plan_action("C", soon) == "URGENT_REPAIR"
    assert mt.plan_action("B", far) == "MONITOR"


def test_rl_matches_value_iteration_on_visited_states():
    Q, _ = mt.q_learning(seed=0)
    Qv = mt.value_iteration()
    assert (Q.argmax(1)[:4] == Qv.argmax(1)[:4]).all()
    run_to_fail = mt.evaluate_policy([0] * 6)[0]
    assert mt.evaluate_policy(Qv.argmax(1))[0] > run_to_fail


def test_sql_window_queries_run():
    conn = db.create_demo_database(":memory:")
    assert len(db.run_query(conn, "latest_rank")) == 24
    assert {"vel_7d_avg", "delta_1d"} <= set(db.run_query(conn, "rolling_trend").columns)


def test_logs():
    d = textlogs.generate_logs(50, seed=1)
    assert list(d.columns) == ["text", "fault"] and len(d) == 50
