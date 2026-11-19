import numpy as np
import pandas as pd
import iso10816 as iso
import cleaning as cl


def test_zone_boundaries():
    assert [iso.zone(v) for v in (0.5, 1.4, 2.8, 4.5, 9)] == ["A", "B", "C", "D", "D"]


def test_alarm_needs_persistence():
    out = iso.AlarmLogic(confirm_n=3).run([1, 1, 5, 1, 1, 1])  # single spike
    assert (out.state == "NORMAL").all()


def test_alarm_escalates_and_clears():
    out = iso.AlarmLogic(confirm_n=3, clear_n=3).run([1, 3.5, 3.5, 3.5, 1, 1, 1])
    assert out.state.iloc[3] == "ALERT" and out.state.iloc[-1] == "NORMAL"


def test_hampel_flags_spike():
    s = pd.Series([1.0, 1.1, 0.9, 1.0, 50.0, 1.0, 1.1, 0.9, 1.0])
    m = cl.hampel_mask(s)
    assert m.iloc[4] and m.sum() == 1


def test_flatline():
    s = pd.Series([1, 2, 3, 3, 3, 3, 3, 4, 5.0])
    assert cl.flatline_mask(s, 5).sum() == 5


def test_clean_pipeline_fills_and_dedups():
    df = pd.DataFrame({"machine_id": 1, "day": list(range(10)) + [3],
                       "vel_rms": [1, 1.1, np.nan, 1.0, 1.2, 1.1, 1.0, 1.1, 1.2, 1.0, 1.0],
                       "temp_c": 60.0})
    out, rep = cl.clean_readings(df)
    assert rep["duplicates_removed"] == 1 and out.vel_rms.isna().sum() == 0 and len(out) == 10
