# Changelog

## 1.5.0 - 2026-11-19
- order_analysis.py: angle-domain resampling (needs tachometer angle), order spectrum, order-domain envelope, order tracking, real cepstrum, gear-mesh-band cepstral peak ratio, ramp and cepstrum experiments
- Notebook section 16d, charts 40-41; 8 new tests (62 in total)
- Findings: order domain gives only a small gain (AUC 0.984 -> 1.000 at a 15-35 Hz ramp); mesh-band cepstrum 0.867 vs sideband feature 0.695 for gear faults
- Not covered: tachometer errors/noise, speed estimated from vibration only, real data

## 1.4.0 - 2026-11-19
- Streamlit dashboard: new tab "Anomaly (healthy-only)" (dashboard_logic.train_novelty / novelty_report); app.py only compile-checked, not run in Streamlit
- advanced_diagnostics.resonance_shift_experiment, notebook cell 16c-2, chart 39: kurtogram AUC 0.985 / 0.999 vs fixed band 0.850 / 0.809 when resonance is outside 1800-5000 Hz
- 54 tests in total

## 1.3.0 - 2026-11-19
- advanced_diagnostics.py: spectral kurtosis / kurtogram, automatic best band, band envelope SNR, HealthyNovelty (robust Mahalanobis, out-of-fold threshold, machine-grouped)
- Notebook section 16c, charts 37-38; 7 new tests (52 in total)
- Honest finding: kurtogram band is not better than the fixed band on this simulator (AUC 0.987 vs 0.999 outer race)

## 1.2.0 - 2026-11-19
- physics_ext.py: speed ramps/ripple (angle-domain harmonics), multi-fault superposition, bearing slip, shock transients, multi-fault dataset; notebook section 16b; charts 34-36
- deep_torch.py rewritten: anti-aliased decimation, patch-based bidirectional LSTM, dropout, class weights, group-wise early stopping, AdamW + cosine schedule, gradient clipping; NumPy helpers tested; torch part still NOT run
- Streamlit dashboard: app.py + dashboard_logic.py (logic tested; app smoke-tested with a stub only)
- Tests: 45 in total (new: physics_ext, deep_torch helpers, dashboard_logic)

## 1.1.0 - 2026-11-19
- Added notebook section 3b: pandas merge / merge_asof / pivot_table / resample / rolling, with a check against the SQL window average
- New chart 33, new test file test_pandas_ops.py (27 tests in total)

## 1.0.0 - 2026-11-19
- Notebook with 17 sections executed end to end (custom runner), 32 PNG charts
- Added sections: maintenance cost, reinforcement learning, NLP, geospatial, AI ethics, dashboard
- Added README with Mermaid diagrams and chart gallery, SUMMARY.md, this changelog

## 0.2.0
- Physics-based generator, features, ISO zones, SQL, cleaning, ML, explainability, time series, RUL

## Not verified
- PyTorch 1D-CNN / LSTM, SHAP, pytest, Jupyter kernel run, real data
