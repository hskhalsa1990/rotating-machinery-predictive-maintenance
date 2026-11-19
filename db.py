"""SQLite layer: plants, machines, daily condition readings, maintenance logs (ALL SYNTHETIC).

Readings are summary-level (daily velocity RMS, temperature) from a simple trend model,
NOT derived from the waveform generator. Plant names are fictional; city coordinates
are approximate public geography.
"""
import os
import sqlite3

import numpy as np
import pandas as pd

import textlogs
from datagen import FAULTS

PLANTS = [  # plant_id, name, city, lat, lon  (fictional plants, approximate city coordinates)
    (1, "Plant-LDH", "Ludhiana", 30.9010, 75.8573), (2, "Plant-JAL", "Jalandhar", 31.3260, 75.5762),
    (3, "Plant-ASR", "Amritsar", 31.6340, 74.8723), (4, "Plant-PTA", "Patiala", 30.3398, 76.3869),
    (5, "Plant-BTI", "Bathinda", 30.2110, 74.9455), (6, "Plant-MOH", "Mohali", 30.7046, 76.7179),
]
MACHINE_TYPES = ["motor", "pump", "gearbox", "fan", "compressor"]
N_DAYS = 120

SCHEMA = """
CREATE TABLE plants (plant_id INTEGER PRIMARY KEY, plant_name TEXT, city TEXT, lat REAL, lon REAL);
CREATE TABLE machines (machine_id INTEGER PRIMARY KEY, plant_id INTEGER, machine_type TEXT,
                       machine_class TEXT, true_fault TEXT, FOREIGN KEY(plant_id) REFERENCES plants(plant_id));
CREATE TABLE readings (machine_id INTEGER, day INTEGER, read_date TEXT, vel_rms REAL, temp_c REAL,
                       PRIMARY KEY(machine_id, day));
CREATE TABLE maintenance_logs (log_id INTEGER PRIMARY KEY, machine_id INTEGER, log_date TEXT,
                               log_text TEXT, fault_label TEXT, cost_inr REAL, downtime_h REAL);
"""

QUERIES = {
    "plant_summary": """
SELECT p.plant_name, COUNT(DISTINCT m.machine_id) AS n_machines,
       ROUND(AVG(r.vel_rms), 2) AS avg_vel, ROUND(MAX(r.vel_rms), 2) AS max_vel
FROM plants p JOIN machines m USING (plant_id) JOIN readings r USING (machine_id)
GROUP BY p.plant_name ORDER BY avg_vel DESC;""",
    "rolling_trend": """
SELECT machine_id, day, vel_rms,
       AVG(vel_rms) OVER (PARTITION BY machine_id ORDER BY day
                          ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS vel_7d_avg,
       vel_rms - LAG(vel_rms, 1) OVER (PARTITION BY machine_id ORDER BY day) AS delta_1d
FROM readings;""",
    "latest_rank": """
WITH latest AS (SELECT machine_id, MAX(day) AS d FROM readings GROUP BY machine_id),
cur AS (SELECT r.machine_id, r.vel_rms, r.temp_c FROM readings r
        JOIN latest l ON r.machine_id = l.machine_id AND r.day = l.d)
SELECT m.plant_id, c.machine_id, m.machine_type, ROUND(c.vel_rms, 2) AS vel_rms,
       RANK() OVER (PARTITION BY m.plant_id ORDER BY c.vel_rms DESC) AS rank_in_plant,
       NTILE(4) OVER (ORDER BY c.vel_rms) AS risk_quartile
FROM cur c JOIN machines m USING (machine_id) ORDER BY m.plant_id, rank_in_plant;""",
    "first_zone_c": """
SELECT machine_id, MIN(day) AS first_day_zone_C FROM readings
WHERE vel_rms >= 2.8 GROUP BY machine_id ORDER BY first_day_zone_C;""",
    "cumulative_cost": """
SELECT p.plant_name, l.log_date, l.cost_inr,
       SUM(l.cost_inr) OVER (PARTITION BY p.plant_id ORDER BY l.log_date, l.log_id) AS cum_cost_inr
FROM maintenance_logs l JOIN machines m USING (machine_id) JOIN plants p USING (plant_id);""",
}


def create_demo_database(path: str = "pdm_demo.db", seed: int = 7, machines_per_plant: int = 4):
    if path != ":memory:" and os.path.exists(path):
        os.remove(path)
    rng = np.random.default_rng(seed)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany("INSERT INTO plants VALUES (?,?,?,?,?)", PLANTS)
    mrows, rrows, lrows = [], [], []
    mid, lid = 0, 0
    days = np.arange(N_DAYS)
    dates = pd.date_range("2026-01-01", periods=N_DAYS).strftime("%Y-%m-%d")
    for pid, *_ in PLANTS:
        for _ in range(machines_per_plant):
            mid += 1
            mtype = MACHINE_TYPES[(mid - 1) % len(MACHINE_TYPES)]
            faulty = rng.random() < 0.45
            fault = str(rng.choice(FAULTS[1:])) if faulty else "healthy"
            mrows.append((mid, pid, mtype, "medium_rigid", fault))
            base = rng.uniform(0.7, 1.3)
            d0 = int(rng.integers(30, 90))
            growth = rng.uniform(0.04, 0.12) if faulty else 0.0
            vel = base + growth * np.clip(days - d0, 0, None) ** 1.15 * 0.25 + rng.normal(0, 0.06, N_DAYS)
            temp = 52 + 6 * vel + rng.normal(0, 1.2, N_DAYS)
            for d in days:
                rrows.append((mid, int(d), dates[d], float(max(vel[d], 0.05)), float(temp[d])))
            lid += 1
            lrows.append((lid, mid, dates[int(rng.integers(5, 25))], textlogs.make_log("healthy", rng),
                          "healthy", 2_000.0, 0.5))
            if faulty:
                lid += 1
                lrows.append((lid, mid, dates[min(d0 + 25, N_DAYS - 1)], textlogs.make_log(fault, rng), fault,
                              float(rng.uniform(25_000, 60_000)), float(rng.uniform(6, 14))))
    conn.executemany("INSERT INTO machines VALUES (?,?,?,?,?)", mrows)
    conn.executemany("INSERT INTO readings VALUES (?,?,?,?,?)", rrows)
    conn.executemany("INSERT INTO maintenance_logs VALUES (?,?,?,?,?,?,?)", lrows)
    conn.commit()
    return conn


def run_query(conn, name: str) -> pd.DataFrame:
    return pd.read_sql(QUERIES[name], conn)
