"""Interactive dashboard (Streamlit). ALL DATA IS SYNTHETIC.

Run:  pip install streamlit   then   streamlit run app.py
STATUS: written and smoke-tested only against a stub of Streamlit (see CHANGELOG); the real Streamlit
was not available to the author, so the look and behaviour in a browser are NOT verified."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

import dashboard_logic as dl
import datagen as dg
import iso10816 as iso

st.set_page_config(page_title="Rotating Machinery PdM (synthetic)", layout="wide")
st.title("Rotating Machinery Predictive Maintenance Suite")
st.caption("All signals are simulated. Results do not describe real machines. Author: Harpreet Singh (hskhalsa1990)")


@st.cache_resource(show_spinner="Training a small model on synthetic data...")
def get_model():
    return dl.train_model()


with st.sidebar:
    st.header("Simulated machine")
    fault = st.selectbox("Fault", dg.FAULTS, index=3)
    severity = st.slider("Severity", 0.0, 1.0, 0.7, 0.05)
    speed = st.slider("Shaft speed [Hz]", 15.0, 40.0, 25.0, 0.5)
    load = st.slider("Load", 0.3, 1.0, 0.7, 0.05)
    noise = st.slider("Extra sensor noise (x)", 0.5, 4.0, 1.0, 0.1)
    machine_class = st.selectbox("ISO machine class", dl.MACHINE_CLASSES, index=1)
    seed = st.number_input("Random seed", 0, 9999, 0)

sig = dl.simulate(fault, 0.0 if fault == "healthy" else severity, speed, load, int(seed), noise)
res = dl.analyze(sig, speed, machine_class)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Signal and spectra", "Diagnosis", "ISO zone", "Cost logic", "Anomaly (healthy-only)"])

@st.cache_resource(show_spinner="Fitting novelty detector on healthy synthetic signals...")
def get_novelty():
    return dl.train_novelty()


with tab1:
    t = [i / dg.FS for i in range(len(sig))]
    fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
    ax[0].plot(t[:1200], sig[:1200], lw=0.7)
    ax[0].set_title("Acceleration (first 0.1 s)")
    ax[0].set_xlabel("s")
    ax[1].plot(res["freq"], res["amp"], lw=0.7)
    ax[1].set_xlim(0, 1500)
    ax[1].set_title("Amplitude spectrum")
    ax[1].set_xlabel("Hz")
    ax[2].plot(res["env_freq"], res["env_amp"], lw=0.7)
    ax[2].set_xlim(0, 300)
    for f0, nm, c in ((res["bpfo"], "BPFO", "r"), (res["bpfi"], "BPFI", "g")):
        ax[2].axvline(f0, color=c, ls="--", lw=0.8, label=nm)
    ax[2].legend()
    ax[2].set_title("Envelope spectrum")
    ax[2].set_xlabel("Hz")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with tab2:
    probs = dl.diagnose(get_model(), res["features"])
    top = next(iter(probs))
    st.metric("Most likely class (model trained on synthetic data)", top, f"{probs[top]:.0%}")
    st.bar_chart(probs)
    st.caption("Probabilities are not calibrated and only valid inside this simulator.")

with tab3:
    v = res["features"]["vel_rms"]
    z = res["zone"]
    st.metric("Velocity RMS [mm/s]", f"{v:.2f}", f"zone {z}: {iso.ZONE_MEANING[z]}")
    st.write("Limits (illustrative, not the current standard):", iso.THRESHOLDS[machine_class])
    st.info("Early bearing and gear faults can stay in zone A/B while the envelope features already show them.")

with tab4:
    c1, c2 = st.columns(2)
    p_fail = c1.slider("Failure probability before the next window", 0.0, 1.0, 0.25, 0.01)
    rul = c1.number_input("Median RUL [days]", 0, 365, 30)
    pr = c2.number_input("Planned repair [INR]", 0, 2_000_000, 35_000, 1000)
    pd_ = c2.number_input("Planned downtime [h]", 0, 500, 8)
    ur = c2.number_input("Unplanned repair [INR]", 0, 5_000_000, 250_000, 1000)
    ud = c2.number_input("Unplanned downtime [h]", 0, 1000, 48)
    ch = c2.number_input("Downtime cost per hour [INR]", 0, 100_000, 6000, 100)
    out = dl.cost_comparison(p_fail, rul, pr, pd_, ur, ud, ch)
    st.metric("Repair now", f"{out['repair_now']:,.0f} INR")
    st.metric("Wait", f"{out['wait']:,.0f} INR")
    st.write(f"Break-even failure probability: {out['break_even_p']:.2f}. All costs are your own assumptions.")

with tab5:
    nr = dl.novelty_report(get_novelty(), res["features"], sig)
    st.metric("Novelty score", f"{nr['score']:.1f}", "ANOMALY" if nr["anomaly"] else "normal", delta_color="inverse")
    st.write(f"Threshold {nr['threshold']:.1f} (99 % quantile of out-of-fold healthy scores). Kurtogram best band: {nr['band_lo']:.0f}-{nr['band_hi']:.0f} Hz.")
    st.caption("Fitted on healthy synthetic signals only; needs no fault labels. About 2.5 % false alarms on unseen healthy machines in the notebook test. Not verified in real Streamlit.")
