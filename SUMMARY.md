# SUMMARY (for my records)

**Repository name:** rotating-machinery-predictive-maintenance
**Description:** Synthetic-data predictive maintenance suite: bearing/gear/imbalance fault detection and RUL with uncertainty from vibration signals (Python, scikit-learn, SQLite).
**Topics:** predictive-maintenance, vibration-analysis, condition-monitoring, remaining-useful-life, machine-learning, time-series, fault-detection, mechanical-engineering, jupyter-notebook, synthetic-data

## What was done
- Notebook: 22 sections (incl. 3b pandas, 16b extended physics, 16c advanced diagnostics), executed via custom runner, 41 PNG charts, 62 tests passed (own runner)
- Upgrades v1.5: order tracking + cepstrum (synthetic; tacho angle assumed known)
- Upgrades v1.4: dashboard anomaly tab; kurtogram proven useful for unknown resonance (synthetic)
- Upgrades v1.3: kurtogram + healthy-only novelty detector (2.5 % false alarms on unseen healthy)
- Upgrades v1.2: extended physics, rewritten deep_torch.py, Streamlit dashboard
- Python, Pandas, cleaning, SQL window functions, visualization (several 3D), features, ML, explainability, time series, MLP spectrogram "vision", NLP, geospatial, ethics, RL

## Key numbers (synthetic data)
- Random forest 0.984 accuracy on unseen machines; missed-fault 1.6 %
- RUL MAE 29.3 days, calibrated 90 % interval coverage 93 % (n = 30)

## NOT verified
- Streamlit app in real Streamlit/browser (only stub smoke test); PyTorch classes/training loop
- PyTorch 1D-CNN / LSTM (torch missing), SHAP, pytest, Jupyter kernel, real data
- ISO limits and costs are illustrative

## Upload (Android)
Unzip, upload all files from the one folder to the new repo (no subfolders).
