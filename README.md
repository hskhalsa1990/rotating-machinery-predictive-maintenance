# Rotating Machinery Predictive Maintenance Suite

Fault detection and remaining useful life (RUL) for motors, pumps, gearboxes and bearings from vibration signals.

**Author:** Harpreet Singh (GitHub: [hskhalsa1990](https://github.com/hskhalsa1990)) | **License:** MIT

> **Data note: all data in this project is synthetic.** Signals come from a physics-based simulator; plants, readings, costs and maintenance logs are invented. Results show that the pipeline works on simulated data. They do **not** show performance on real machines.

## Quick start

```bash
pip install -r requirements.txt
python simple_test_runner.py        # or: pytest -q
pip install streamlit               # optional, for the dashboard
streamlit run app.py                # interactive dashboard (see limitations)
python run_all.py                   # runs the whole notebook source (about 2-5 min), writes the .ipynb and all PNG charts
jupyter notebook Rotating_Machinery_PdM_Suite.ipynb
```

All files live in one flat folder; charts are saved to the current folder.

## What is inside

- Order tracking and cepstrum (`order_analysis.py`): angle-domain resampling with tachometer angle, order spectrum, order-domain envelope, order tracking during run-ups, gear-mesh-band cepstrum
- Dashboard tab "Anomaly (healthy-only)" (logic tested, tab not run in real Streamlit)
- Advanced diagnostics (`advanced_diagnostics.py`): spectral-kurtosis kurtogram with automatic envelope band, and a novelty detector fitted on healthy data only
- Extended physics (`physics_ext.py`): speed run-up with order lines, several faults at once, bearing slip, shock transients, multi-label fault classification
- Interactive Streamlit dashboard (`app.py`, logic in `dashboard_logic.py`)
- Upgraded PyTorch 1D-CNN / LSTM code (patch-based LSTM, early stopping by machine, class weights) with tests; **not run** (no torch)
- Physics-based data generator: imbalance (1x), misalignment (2x/3x), bearing outer/inner race (BPFO/BPFI impacts ringing a resonance), gear tooth (mesh sidebands), noise, load and speed variation, per-machine sensor gain
- ISO 10816-style zones A/B/C/D and alarm logic (illustrative limits, check ISO 20816)
- SQLite database with window-function queries, a pandas section (`merge`, `merge_asof`, `pivot_table`, `resample`, `rolling`, checked against the SQL window result), data cleaning, 20 engineered features
- Random forest, logistic regression, gradient boosting, MLPs, validated on **unseen machines** (GroupKFold)
- Health index, RUL with calibrated uncertainty, Holt forecast, cost-based maintenance plan
- Reinforcement learning (Q-learning vs exact value iteration), NLP on maintenance logs, geospatial risk map, slice evaluation and model card style limitations
- 41 charts including several matplotlib 3D figures

## Diagrams

### Pipeline

```mermaid
flowchart LR
    A[Physics simulator] --> B[Cleaning]
    B --> C[Feature engineering]
    C --> D[Fault classifier]
    C --> E[Health index]
    E --> F[RUL with uncertainty]
    D --> G[ISO zones and alarms]
    F --> H[Maintenance plan and cost]
    G --> H
    H --> I[Dashboard]
```

### Module map

```mermaid
flowchart TD
    N[notebook_source.py / notebook .ipynb] --> datagen
    N --> cleaning
    N --> db
    N --> features
    N --> iso10816
    N --> rul
    N --> maintenance
    N --> textlogs
    N --> deep_torch
    N --> plotting
    N --> physics_ext
    N --> advanced_diagnostics
    N --> order_analysis
    order_analysis --> datagen
    tests --> order_analysis
    advanced_diagnostics --> datagen
    tests --> advanced_diagnostics
    physics_ext --> datagen
    app --> dashboard_logic
    dashboard_logic --> datagen
    dashboard_logic --> features
    dashboard_logic --> iso10816
    dashboard_logic --> maintenance
    textlogs --> datagen
    features --> datagen
    run_all --> make_notebook
    tests[test_*.py] --> datagen
    tests --> features
    tests --> iso10816
    tests --> cleaning
    tests --> rul
    tests --> maintenance
    tests --> physics_ext
    tests --> deep_torch
    tests --> dashboard_logic
```

### Data flow

```mermaid
flowchart LR
    S[Synthetic signals 12 kHz] --> F[Feature table]
    S --> SP[Spectrograms]
    DB[(SQLite: plants, machines, readings, logs)] --> Q[SQL window queries]
    DB --> NL[Log text for NLP]
    DB --> GE[Geospatial risk]
    F --> ML[ML models]
    SP --> CV[Image MLP]
    F --> HI[Health index]
    HI --> RUL[RUL and cost]
```

### Maintenance decision flowchart

```mermaid
flowchart TD
    S[New reading] --> Z{Zone D?}
    Z -- yes --> STOP[STOP AND INSPECT NOW]
    Z -- no --> U{P failure within lead time >= 20 percent?}
    U -- yes --> URG[URGENT REPAIR]
    U -- no --> L{Lower RUL bound within lead time + window?}
    L -- yes --> PLAN[PLAN NEXT WINDOW]
    L -- no --> MON[MONITOR]
```

## Results (honest summary, all on synthetic data)

| Model | Input | Accuracy (5-fold, unseen machines) | Status |
|---|---|---|---|
| Logistic regression | 20 engineered features | 0.980 +/- 0.008 | executed |
| Random forest | 20 engineered features | 0.984 +/- 0.011 | executed |
| Hist. gradient boosting | 20 engineered features | 0.983 +/- 0.014 | executed |
| MLP on log-spectrum | 128-bin spectrum | 0.709 +/- 0.025 | executed |
| MLP on spectrogram image | 32x15 image | 0.607 +/- 0.026 | executed |
| 1D-CNN (PyTorch) | raw signal | - | **NOT RUN** (torch not available) |
| LSTM (PyTorch) | raw signal | - | **NOT RUN** (torch not available) |

- Random forest: missed-fault rate 1.6 %, false-alarm rate 1.7 %
- RUL on 10 test machines (30 predictions): MAE 29.3 days; raw 90 % interval covered only 13 %, calibrated interval covered 93 % (very small sample)
- RUL for gear-tooth machines is poor (MAE 71 days); the health index is not a good fit for them
- NLP on synthetic log text: accuracy 0.979 (template text, optimistic)
- RL: Q-learning policy matches exact value iteration on visited states
- Cost numbers (INR) are assumptions, not plant data
- Extended physics: multi-label model trained with fault pairs gets 0.970 exact-match on pair signals (unseen machines); trained on single faults only it gets 0.520 on pairs, so it does not generalise to unseen fault combinations
- Bearing slip: envelope SNR at BPFO for severity 1.0 falls from 17.8 (no slip) to 2.7 (slip 0.25)
- Order analysis, outer-race detection AUC (160 simulated signals per ramp, tachometer angle assumed known): frequency-domain 1.000 / 0.998 / 0.984 vs order-domain 1.000 / 1.000 / 1.000 for constant speed, 20-30 Hz and 15-35 Hz ramps. The gain is small because this detection task is easy; order tracking of imbalance follows the simulated speed-squared law within 15 %
- Cepstrum for gear tooth vs healthy (240 signals): full-band 0.666, gear-mesh-band 0.867, existing sideband-ratio feature 0.695 (on the weakest faults the sideband ratio scores 0.30, i.e. worse than chance, mesh-band cepstrum 0.68). A first version without absolute value gave 0.44 because the cepstral peak can be negative
- Novelty detector (healthy data only, machines 0-14 for training, 15 unseen machines for testing): false-alarm rate on unseen healthy signals 2.5 %; flags 93-100 % of faulty signals per class and 92.5 % of the weakest faults (severity below 0.4)
- Kurtogram band vs fixed 1800-5000 Hz band for bearing detection (AUC, unseen signals): outer 0.987 vs 0.999, inner 0.992 vs 0.999. **The kurtogram is not better when the resonance lies inside the fixed band.** When the resonance is moved outside it (240 signals per case, synthetic), AUC fixed vs kurtogram is 0.850 vs 0.985 (resonance 1000-1700 Hz) and 0.809 vs 0.999 (5000-5600 Hz); inside the band 1.000 vs 0.996. The first version with long STFT windows scored only 0.61 AUC; short windows (16-64 samples) were chosen after trying options on one seed and checked on two others


## Limitations

- Everything is simulated; real machines have effects the simulator does not model
- The simple MLPs on spectrum/spectrogram are weak; this does not say anything about CNNs, which were not run
- The 1D-CNN and LSTM code in `deep_torch.py` is **unverified** (torch was not installed in the build environment)
- SHAP step is optional and was skipped
- Train and test come from the same simulator, so scores are optimistic
- ISO limits are illustrative; run-to-failure lives and costs are assumptions
- RUL uses 25 simulated machines; the interval statistics are based on very few predictions
- Tests were run with the included simple runner (62 passed); pytest itself was not available during the build
- The notebook was executed by a custom runner (`run_all.py`), not a Jupyter kernel
- `app.py` (Streamlit) was only smoke-tested against a stub of Streamlit, never in real Streamlit or a browser; the logic in `dashboard_logic.py` is tested
- In `deep_torch.py` the NumPy helpers are tested; the PyTorch classes and training loop are still unrun
- Extended physics is a simplified model; the multi-fault scores are on the same simulator and optimistic
- Not for safety-critical use

## Files

| File | Purpose |
|---|---|
| `Rotating_Machinery_PdM_Suite.ipynb` | main notebook with outputs (22 sections) |
| `notebook_source.py` | notebook source in percent format |
| `make_notebook.py`, `run_all.py` | build and execute the notebook |
| `datagen.py` | physics-based signal generator |
| `cleaning.py` | missing values, outliers, flat-lines, duplicates |
| `db.py` | SQLite layer |
| `features.py` | time, frequency and envelope features |
| `iso10816.py` | severity zones and alarm logic |
| `rul.py` | health index, RUL, uncertainty, Holt forecast |
| `maintenance.py` | cost model, decision rule, Q-learning |
| `textlogs.py` | synthetic maintenance-log text |
| `deep_torch.py` | optional PyTorch 1D-CNN / LSTM; NumPy helpers tested, torch part unverified |
| `physics_ext.py` | extended physics: ramps, multi-fault, slip, shocks |
| `order_analysis.py` | angle resampling, order spectrum / tracking, order envelope, cepstrum |
| `advanced_diagnostics.py` | kurtogram, best-band envelope SNR, healthy-only novelty detector |
| `dashboard_logic.py`, `app.py` | logic and Streamlit dashboard (app not verified in real Streamlit) |
| `plotting.py` | PNG saving helper |
| `test_*.py`, `simple_test_runner.py` | tests (62) |
| `model_comparison.csv` | model table |
| `01_...png` to `41_...png` | charts |
| `CHANGELOG.md`, `SUMMARY.md`, `LICENSE`, `requirements.txt`, `.gitignore` | project files |

## Chart gallery

### Signals, spectra and 3D waterfall

![01_signals_by_fault](01_signals_by_fault.png)

![02_spectra_by_fault](02_spectra_by_fault.png)

![03_envelope_spectrum](03_envelope_spectrum.png)

![04_waterfall_3d](04_waterfall_3d.png)


### SQL, pandas, cleaning and features

![05_sql_window_trends](05_sql_window_trends.png)

![33_pandas_merge_pivot_resample_rolling](33_pandas_merge_pivot_resample_rolling.png)

![06_cleaning_before_after](06_cleaning_before_after.png)

![07_feature_correlation](07_feature_correlation.png)

![08_features_3d_scatter](08_features_3d_scatter.png)


### ISO zones and machine learning

![09_iso_zones](09_iso_zones.png)

![10_tree_depth_curve](10_tree_depth_curve.png)

![11_confusion_matrix_rf](11_confusion_matrix_rf.png)

![12_accuracy_by_severity](12_accuracy_by_severity.png)

![13_noise_robustness](13_noise_robustness.png)

![14_permutation_importance](14_permutation_importance.png)


### Computer vision (spectrograms) and model comparison

![15_spectrogram_gallery](15_spectrogram_gallery.png)

![16_spectrogram_3d_surface](16_spectrogram_3d_surface.png)

![17_model_comparison](17_model_comparison.png)


### Time series, RUL and alarms

![18_health_trend_rul](18_health_trend_rul.png)

![19_degradation_3d](19_degradation_3d.png)

![20_rul_pred_vs_true](20_rul_pred_vs_true.png)

![21_holt_forecast](21_holt_forecast.png)

![22_alarm_states](22_alarm_states.png)


### Maintenance cost and reinforcement learning

![23_cost_comparison](23_cost_comparison.png)

![24_cost_3d_surface](24_cost_3d_surface.png)

![25_rl_learning_and_policy](25_rl_learning_and_policy.png)

![26_rl_q_values_3d](26_rl_q_values_3d.png)


### NLP, geospatial and ethics

![27_nlp_confusion](27_nlp_confusion.png)

![28_geo_plants_map](28_geo_plants_map.png)

![29_geo_risk_map](29_geo_risk_map.png)

![30_geo_risk_3d](30_geo_risk_3d.png)

![31_slice_evaluation](31_slice_evaluation.png)


### Extended physics

![34_ext_physics_signals](34_ext_physics_signals.png)

![35_ext_slip_3d](35_ext_slip_3d.png)

![36_ext_multifault_results](36_ext_multifault_results.png)

### Advanced diagnostics

![37_kurtogram_best_band](37_kurtogram_best_band.png)

![38_novelty_detector](38_novelty_detector.png)

![39_kurtogram_resonance_shift](39_kurtogram_resonance_shift.png)

### Order tracking and cepstrum

![40_order_tracking](40_order_tracking.png)

![41_cepstrum_gear](41_cepstrum_gear.png)

### Dashboard

![32_dashboard](32_dashboard.png)


## Author

Harpreet Singh, GitHub: hskhalsa1990
