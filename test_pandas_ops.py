"""Checks that the pandas merge/rolling logic agrees with the SQL window functions."""
import numpy as np
import pandas as pd

import db


def test_pandas_rolling_matches_sql_window():
    conn = db.create_demo_database(":memory:")
    readings = pd.read_sql("SELECT * FROM readings", conn).sort_values(["machine_id", "day"])
    roll = db.run_query(conn, "rolling_trend")
    readings["m7"] = readings.groupby("machine_id").vel_rms.transform(lambda s: s.rolling(7, min_periods=1).mean())
    both = readings.merge(roll[["machine_id", "day", "vel_7d_avg"]], on=["machine_id", "day"], validate="one_to_one")
    assert np.allclose(both.m7, both.vel_7d_avg)


def test_merge_keeps_row_count():
    conn = db.create_demo_database(":memory:")
    r = pd.read_sql("SELECT * FROM readings", conn)
    m = pd.read_sql("SELECT * FROM machines", conn)
    assert len(r.merge(m, on="machine_id", validate="many_to_one")) == len(r)
