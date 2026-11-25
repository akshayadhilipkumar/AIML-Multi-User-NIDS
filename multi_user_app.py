from pathlib import Path
from collections import deque

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

API_URL = "https://aiml-multi-user-nids-1.onrender.com"
ROOT = Path(__file__).resolve().parent

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #080d17; color: #e5ebf5; }
[data-testid="stSidebar"] { background: #0b1220; border-right: 1px solid #1d293b; }
[data-testid="stMetric"] { background: linear-gradient(145deg,#111b2b,#0d1522); border: 1px solid #1e2b3e; border-radius: 13px; padding: 17px 19px; }
[data-testid="stMetricLabel"] { color: #9aa9bd; }
[data-testid="stMetricValue"] { color: #f3f6fb; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-color: #1c293b; }
h1,h2,h3 { letter-spacing: -.02em; }
.eyebrow { color:#57a6ff; font-size:12px; text-transform:uppercase; letter-spacing:.12em; font-weight:700; }
.muted { color:#91a0b5; }
button[kind="primary"] { background:#2868db; }
</style>
""", unsafe_allow_html=True)



@st.cache_resource
def load_artifacts():
    """Load models. Missing files are downloaded from the Hugging Face repo.

    If the HF repo is private, add HF_TOKEN in Streamlit Cloud -> Settings -> Secrets.
    """
    from huggingface_hub import hf_hub_download

    hf_repo = "akshayaaa06/nids-random-forest-model"
    try:
        hf_token = st.secrets.get("HF_TOKEN")
    except Exception:
        hf_token = None

    paths = {}
    for name in ("random_forest_model.pkl", "isolation_forest_model.pkl", "preprocessor.pkl"):
        local = ROOT / name
        if not local.exists():
            try:
                local = Path(
                    hf_hub_download(
                        repo_id=hf_repo,
                        filename=name,
                        token=hf_token,
                        local_dir=str(ROOT),
                    )
                )
            except Exception as exc:
                raise FileNotFoundError(
                    f"{name} not found locally and could not be downloaded from Hugging Face "
                    f"({type(exc).__name__}). If the repo is private, set HF_TOKEN in Streamlit secrets."
                ) from exc
        paths[name] = local

    return (
        joblib.load(paths["random_forest_model.pkl"]),
        joblib.load(paths["isolation_forest_model.pkl"]),
        joblib.load(paths["preprocessor.pkl"]),
    )


def prepare_features(frame, preprocessor):
    """Separate known ground-truth columns before transforming model features."""
    data = frame.copy()
    by_normalized = {str(c).strip().lower(): c for c in data.columns}
    label_col = by_normalized.get("label")
    attack_col = by_normalized.get("attack_cat")
    actual = data[label_col].copy() if label_col is not None else None
    data = data.drop(columns=[c for c in (label_col, attack_col) if c is not None], errors="ignore")

    expected = getattr(preprocessor, "feature_names_in_", None)
    if expected is not None:
        expected = list(expected)
        leaked = [c for c in expected if str(c).strip().lower() in {"label", "attack_cat"}]
        if leaked:
            raise ValueError("The saved preprocessor expects target/label columns as features (" + ", ".join(map(str, leaked)) + "). Retrain it without labels to prevent leakage.")
        missing = [c for c in expected if c not in data.columns]
        if missing:
            raise ValueError("Missing model feature columns: " + ", ".join(map(str, missing[:12])) + (" …" if len(missing) > 12 else ""))
        data = data.loc[:, expected]
    elif hasattr(preprocessor, "n_features_in_") and data.shape[1] != preprocessor.n_features_in_:
        raise ValueError(f"Preprocessor expects {preprocessor.n_features_in_} feature columns; found {data.shape[1]} after label removal.")
    return data, actual


def binary_label(value):
    val = str(value).strip().lower()
    if val in {"0", "0.0", "normal", "benign", "false"}:
        return "Normal"
    if val in {"1", "1.0", "attack", "malicious", "true"}:
        return "Attack"
    return None


def section_header(title, subtitle=None):
    st.markdown(f'<div class="eyebrow">Network security monitor</div><h1 style="margin:4px 0 0">{title}</h1>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<p class="muted" style="margin-top:4px">{subtitle}</p>', unsafe_allow_html=True)


try:
    rf_model, iso_model, preprocessor = load_artifacts()
    model_ready = True
except Exception as exc:
    model_ready = False
    load_error = str(exc)

def api_headers():
    return {
        "Authorization": f"Bearer {st.session_state.token}"
    }
def session_expired():
    st.session_state.token = None
    st.warning("Your session expired. Please log in again.")
    st.rerun()


def fetch_my_flows():
    try:
        response = requests.get(
            f"{API_URL}/my-flows",
            headers=api_headers(),
            timeout=60
        )
        if response.status_code == 401:
            session_expired()
        response.raise_for_status()
        return response.json()

    except requests.RequestException as exc:
        st.error(f"Unable to fetch network flows: {exc}")
        return []


def fetch_admin_flows():
    try:
        response = requests.get(
            f"{API_URL}/admin/flows",
            headers=api_headers(),
            params={"limit": 5000},
            timeout=60
        )
        if response.status_code == 401:
            session_expired()
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        st.error(f"Unable to fetch lab flows: {exc}")
        return []


@st.cache_data(ttl=60, show_spinner=False)
def fetch_is_admin(token):
    try:
        response = requests.get(
            f"{API_URL}/me",
            headers={"Authorization": f"Bearer {token}"},
            timeout=60
        )
        return bool(response.ok and response.json().get("is_admin"))
    except requests.RequestException:
        return False


def fetch_my_devices():
    try:
        response = requests.get(
            f"{API_URL}/devices",
            headers=api_headers(),
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return []


# LOGIN AND REGISTRATION

if "token" not in st.session_state:
    st.session_state.token = None

if not st.session_state.token:

    st.title("🛡️ NIDS Network Security")

    mode = st.radio(
        "Choose an option",
        ["Login", "Create Account"],
        horizontal=True
    )

    if mode == "Create Account":

        st.subheader("Create Your Account")

        with st.form("register_form"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm = st.text_input("Confirm Password", type="password")

            submitted = st.form_submit_button("Register")

        if submitted:
            if not username or not email or not password:
                st.error("Please fill all fields.")
            elif password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    response = requests.post(
                        f"{API_URL}/register",
                        json={
                            "username": username,
                            "email": email,
                            "password": password
                        },
                        timeout=10
                    )

                    if response.status_code in (200, 201):
                        st.success("Account created successfully!")

                        login_response = requests.post(
                            f"{API_URL}/login",
                            json={
                                "email": email,
                                "password": password
                            },
                            timeout=10
                        )

                        if login_response.status_code == 200:
                            st.session_state.token = login_response.json()["access_token"]
                            st.rerun()
                        else:
                            st.warning("Account created. Please login manually.")
                    else:
                        st.error(response.text)

                except requests.RequestException:
                    st.error("Backend is not reachable.")

    else:

        st.subheader("Welcome Back")

        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")

        if submitted:
            try:
                response = requests.post(
                    f"{API_URL}/login",
                    json={"email": email, "password": password},
                    timeout=10
                )

                if response.status_code == 200:
                    st.session_state.token = response.json()["access_token"]
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

            except requests.RequestException:
                st.error("Backend is not reachable.")

    st.stop()




with st.sidebar:
    st.markdown("## 🛡️ NIDS")
    st.caption("Authenticated network security monitor")
    if st.button("Logout"):
        st.session_state.token = None
        st.rerun()

    st.divider()
    nav_pages = [
            "User Monitor",
            "Dashboard",
            "Alerts",
            "Traffic Monitor",
            "Analysis",
            "Reports",
            "Zeek Flow Demo",
            "Live Monitoring",
    ]
    if fetch_is_admin(st.session_state.token):
        nav_pages.insert(0, "Lab Overview")
    page = st.radio("Navigation", nav_pages, key="main_navigation")
    st.divider()
    st.markdown("**Data source**")
    st.caption("User flows, uploaded CSV, saved Zeek logs, or live Zeek capture")
    uploaded = st.file_uploader(
        "Upload traffic CSV",
        type=["csv"],
        key="traffic_csv_upload",
        help="Used by Dashboard, Alerts, Traffic Monitor, Analysis, and Reports.",
    )


if page == "Lab Overview":
    section_header(
        "Lab overview",
        "Flows from every authorized device registered on this system (admin only)",
    )

    lab_a, lab_b = st.columns(2)
    with lab_a:
        lab_auto = st.toggle("Auto-refresh every 10s", value=True, key="lab_auto")
    with lab_b:
        st.button("Refresh now", key="lab_refresh")
    if lab_auto:
        st_autorefresh(interval=10000, key="lab_refresh_timer")

    lab_flows = fetch_admin_flows()
    if not lab_flows:
        st.info("No flows yet from any device.")
        st.stop()

    lab_df = pd.DataFrame(lab_flows)
    lab_df["prediction"] = lab_df["prediction"].fillna("BASELINE").astype(str).str.upper()
    lab_df["owner / device"] = lab_df["username"].astype(str) + " / " + lab_df["device_name"].astype(str)

    options = sorted(lab_df["owner / device"].unique())
    picked = st.multiselect("Devices", options, default=options, key="lab_devices")
    lab_df = lab_df[lab_df["owner / device"].isin(picked)]
    lab_unusual = lab_df["prediction"].str.contains("UNUSUAL", na=False)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Flows", f"{len(lab_df):,}")
    k2.metric("Unusual flows", f"{int(lab_unusual.sum()):,}")
    k3.metric("Devices reporting", lab_df["owner / device"].nunique())
    k4.metric("Users", lab_df["username"].nunique())
    st.warning("UNUSUAL is an AI anomaly flag, not confirmation of a cyberattack.")

    st.subheader("Per-device summary")
    lab_summary = (
        lab_df.assign(unusual=lab_unusual)
        .groupby("owner / device", as_index=False)
        .agg(flows=("prediction", "size"), unusual=("unusual", "sum"), last_seen=("created_at", "max"))
    )
    st.dataframe(lab_summary, use_container_width=True, hide_index=True)

    st.subheader("Flows per device")
    fig = px.bar(lab_summary, x="owner / device", y=["flows", "unusual"], barmode="group")
    fig.update_layout(template="plotly_dark", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent unusual flows (all devices)")
    lab_flagged = lab_df[lab_unusual]
    if lab_flagged.empty:
        st.info("No unusual flows in the selected devices.")
    else:
        st.dataframe(lab_flagged.head(100), use_container_width=True, hide_index=True)

    st.subheader("Recent flows (all devices)")
    st.dataframe(lab_df.head(100), use_container_width=True, hide_index=True)
    st.stop()


if page == "User Monitor":
    section_header(
        "User-specific network monitor",
        "Authenticated device and flow records stored by the NIDS backend",
    )
    try:
        devices_response = requests.get(
            f"{API_URL}/devices", headers=api_headers(), timeout=10
        )
        devices_response.raise_for_status()
        devices = devices_response.json()
    except requests.RequestException as exc:
        st.error(f"Could not load your devices: {exc}")
        st.stop()

    if not devices:
        st.warning("No registered devices found for this account.")
        st.subheader("Register your device")
        with st.form("device_register_form"):
            device_name = st.text_input("Device name", "My Laptop")
            hostname = st.text_input("Hostname", "localhost")
            operating_system = st.text_input("Operating system", "Windows 11")
            register_device = st.form_submit_button("Register device")
        if register_device:
            try:
                response = requests.post(
                    f"{API_URL}/devices/register",
                    headers=api_headers(),
                    json={
                        "device_name": device_name,
                        "hostname": hostname,
                        "operating_system": operating_system,
                    },
                    timeout=10,
                )
                if response.status_code in (200, 201):
                    st.success("Device registered successfully.")
                    st.rerun()
                else:
                    st.error(response.text)
            except requests.RequestException as exc:
                st.error(f"Could not register device: {exc}")
        st.stop()

    device_options = {
        f"{item['device_name']} (ID: {item['id']})": item["id"]
        for item in devices
    }
    selected_device_label = st.selectbox(
        "Select your device", list(device_options.keys())
    )
    selected_device_id = device_options[selected_device_label]

    try:
        flows_response = requests.get(
            f"{API_URL}/my-flows", headers=api_headers(), timeout=15
        )
        flows_response.raise_for_status()
        flows = flows_response.json()
    except requests.RequestException as exc:
        st.error(
            f"Could not load flow records: {exc}. "
            "Check that the backend is running and the /my-flows endpoint exists."
        )
        st.stop()

    user_df = pd.DataFrame(flows)
    if user_df.empty:
        user_df = pd.DataFrame(
            columns=[
                "device_id", "prediction", "anomaly_score",
                "source_ip", "destination_ip", "protocol", "created_at",
            ]
        )
    user_df = user_df[user_df["device_id"] == selected_device_id].copy()
    total_flows = len(user_df)
    unusual_flows = int((user_df["prediction"] == "UNUSUAL").sum())
    baseline_flows = int((user_df["prediction"] == "BASELINE").sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Total flows", total_flows)
    m2.metric("Unusual flows", unusual_flows)
    m3.metric("Baseline flows", baseline_flows)
    st.warning(
        "UNUSUAL is an AI anomaly flag, not confirmation of a cyberattack."
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        st.subheader("Traffic classification")
        if total_flows:
            chart_df = pd.DataFrame({
                "Prediction": ["BASELINE", "UNUSUAL"],
                "Count": [baseline_flows, unusual_flows],
            })
            fig = px.pie(
                chart_df, names="Prediction", values="Count", hole=0.55
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No flow records are available for this device yet.")
    with chart_right:
        st.subheader("Anomaly score distribution")
        if total_flows and "anomaly_score" in user_df.columns:
            fig = px.histogram(
                user_df, x="anomaly_score", color="prediction", nbins=25
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No anomaly scores are available yet.")

    st.subheader("Recent unusual flows")
    if total_flows:
        alerts_df = user_df[user_df["prediction"] == "UNUSUAL"]
        if "id" in alerts_df.columns:
            alerts_df = alerts_df.sort_values("id", ascending=False)
        st.dataframe(alerts_df.head(50), use_container_width=True, hide_index=True)
        st.subheader("Recent network traffic")
        recent_df = user_df.sort_values("id", ascending=False) if "id" in user_df.columns else user_df
        st.dataframe(recent_df.head(100), use_container_width=True, hide_index=True)
    else:
        st.info("Your device is registered, but no flow records have arrived yet.")
    if st.button("Refresh user flow data"):
        st.rerun()
    st.stop()

st.markdown(
    '<div class="eyebrow">Network security analysis prototype</div>',
    unsafe_allow_html=True,
)

if page == "Zeek Flow Demo":
    st.caption(
        "Analyze a saved Zeek flow TSV. This is a saved-capture demo, not a continuous live feed."
    )
elif page == "Live Monitoring":
    st.caption(
        "Shows flows sent by authorized Zeek sensors. Each sensor only reports traffic visible on its own machine/interface."
    )
else:
    st.caption("Predictions are generated from the uploaded traffic CSV.")

if page == "Zeek Flow Demo":
    section_header(
        "Zeek flow anomaly demo",
        "Score flows from a saved capture using the separate Zeek-trained model",
    )
    zeek_model_path = ROOT / "zeek_anomaly_model.pkl"
    if not zeek_model_path.is_file():
        st.error(
            "zeek_anomaly_model.pkl is missing. Place the Zeek model beside app.py."
        )
        st.stop()

    try:
        zeek_artifact = joblib.load(zeek_model_path)
        zeek_detector = zeek_artifact["pipeline"]
        zeek_features = list(zeek_artifact["features"])
    except Exception as exc:
        st.error(f"Could not load the Zeek anomaly model: {exc}")
        st.info("This usually means the model was saved with a different scikit-learn version. Use the same version used for training, or retrain the Zeek model in this app environment.")
        st.stop()

    zeek_file = st.file_uploader(
        "Upload Zeek flow TSV",
        type=["tsv", "txt"],
        key="zeek_flow_upload",
    )
    st.caption(
        "Use a TSV exported with the same Zeek feature columns. This file contains flow metrics, not packet contents."
    )
    if zeek_file is None:
        st.info("Upload a Zeek TSV to see which flows this experimental model considers unusual.")
        st.stop()

    zeek_columns = [
        "duration",
        "orig_pkts",
        "resp_pkts",
        "orig_bytes",
        "resp_bytes",
        "orig_ip_bytes",
        "resp_ip_bytes",
    ]
    try:
        zeek_data = pd.read_csv(
            zeek_file,
            sep="\\t",
            header=None,
            names=zeek_columns,
            na_values=["-"],
        )
        for column in zeek_columns:
            zeek_data[column] = pd.to_numeric(zeek_data[column], errors="coerce")

        missing_features = [
            column for column in zeek_features if column not in zeek_data.columns
        ]
        if missing_features:
            raise ValueError(
                "TSV is missing model feature columns: "
                + ", ".join(missing_features)
            )

        usable = zeek_data[zeek_features].replace([np.inf, -np.inf], np.nan)
        predictions = zeek_detector.predict(usable)
    except Exception as exc:
        st.error(f"Could not score this Zeek TSV: {exc}")
        st.caption(
            "Check that the file was exported with the same feature names and order used to train the Zeek model."
        )
        st.stop()

    zeek_results = zeek_data.copy()
    zeek_results["Anomaly status"] = [
        "Unusual flow" if value == -1 else "Baseline-like"
        for value in predictions
    ]
    zeek_results["Model score"] = zeek_detector.decision_function(usable)
    unusual = int((predictions == -1).sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Flows analyzed", f"{len(zeek_results):,}")
    k2.metric("Unusual flows", f"{unusual:,}")
    k3.metric("Baseline-like flows", f"{len(zeek_results) - unusual:,}")

    st.warning(
        "‘Unusual flow’ is an anomaly flag, not a confirmed attack. This page scores a saved capture, not live traffic."
    )
    st.subheader("Flagged flows")
    flagged_zeek = zeek_results[
        zeek_results["Anomaly status"] == "Unusual flow"
    ].sort_values("Model score")
    # HTML rendering avoids the PyArrow dataframe dependency on affected Windows systems.
    st.markdown(
        flagged_zeek.head(100).to_html(
            index=False, escape=True, classes="nids-table"
        ),
        unsafe_allow_html=True,
    )
    zeek_csv = zeek_results.to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Download Zeek flow analysis (CSV)",
        data=zeek_csv,
        file_name="zeek_flow_anomaly_results.csv",
        mime="text/csv",
    )
    st.stop()

if page == "Live Monitoring":
    section_header(
        "Live Network Monitoring",
        "Flows sent by authorized Zeek sensors and scored by the NIDS backend",
    )

    ctl_a, ctl_b = st.columns([1, 1])
    with ctl_a:
        auto_refresh = st.toggle("Auto-refresh every 10s", value=True)
    with ctl_b:
        st.button("Refresh now")
    if auto_refresh:
        st_autorefresh(interval=10000, key="live_monitor_refresh")

    devices = fetch_my_devices()
    device_names = {d["id"]: d.get("device_name", f"Device {d['id']}") for d in devices}
    flows = fetch_my_flows()

    if not flows:
        st.info(
            "No flow records yet. Start Zeek and zeek_sender.py on an authorized device, "
            "then wait a few seconds."
        )
        st.stop()

    live_data = pd.DataFrame(flows)
    if "prediction" not in live_data.columns:
        live_data["prediction"] = "BASELINE"
    live_data["prediction"] = live_data["prediction"].fillna("BASELINE").astype(str).str.upper()

    if "device_id" in live_data.columns:
        live_data["device"] = live_data["device_id"].map(device_names).fillna(
            live_data["device_id"].astype(str)
        )
    else:
        live_data["device"] = "unknown"

    all_devices = sorted(live_data["device"].unique())
    chosen = st.multiselect("Devices", all_devices, default=all_devices)
    live_data = live_data[live_data["device"].isin(chosen)]
    if "id" in live_data.columns:
        live_data = live_data.sort_values("id", ascending=False)

    unusual_mask = live_data["prediction"].str.contains("UNUSUAL", na=False)
    total_flows = len(live_data)
    unusual_count = int(unusual_mask.sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Flows analyzed", f"{total_flows:,}")
    c2.metric("Unusual flows", f"{unusual_count:,}")
    c3.metric("Baseline flows", f"{total_flows - unusual_count:,}")
    c4.metric("Active devices", live_data["device"].nunique())

    st.warning("UNUSUAL is an AI anomaly flag, not confirmation of a cyberattack.")

    st.subheader("Per-device summary")
    summary = (
        live_data.assign(unusual=unusual_mask)
        .groupby("device", as_index=False)
        .agg(flows=("prediction", "size"), unusual=("unusual", "sum"))
    )
    if "created_at" in live_data.columns:
        last_seen = live_data.groupby("device")["created_at"].max().rename("last_seen")
        summary = summary.merge(last_seen, on="device", how="left")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.subheader("Recent network flows")
    st.dataframe(live_data.head(100), use_container_width=True, hide_index=True)

    st.subheader("Unusual flows")
    unusual_data = live_data[unusual_mask]
    if unusual_data.empty:
        st.info("No unusual flows detected in these records.")
    else:
        if "anomaly_score" in unusual_data.columns:
            unusual_data = unusual_data.sort_values("anomaly_score", ascending=False)
        st.dataframe(unusual_data.head(100), use_container_width=True, hide_index=True)

    st.download_button(
        "Download live flow analysis (CSV)",
        data=live_data.to_csv(index=False).encode("utf-8-sig"),
        file_name="zeek_live_flow_analysis.csv",
        mime="text/csv",
    )
    st.stop()

# The Zeek pages use their own model. The Random Forest and Isolation Forest
# artifacts are required only for the CSV-analysis pages below.
if not model_ready:
    st.error(f"Could not load CSV model artifacts: {load_error}")
    st.info("The CSV models may have been saved with a different scikit-learn version. Use the original training environment or retrain the models; do not patch pickle internals.")
    st.info(
        "Place random_forest_model.pkl, isolation_forest_model.pkl, and preprocessor.pkl beside app.py."
    )
    st.stop()

if uploaded is None:
    section_header("Network Intrusion Detection System", "Upload a traffic CSV to inspect predictions, anomalies, and trusted evaluation metrics.")
    st.info("Start by uploading a CSV with the same feature columns used to train the models.")
    st.stop()

try:
    raw = pd.read_csv(uploaded)
    if raw.empty:
        st.warning("This CSV has no data rows.")
        st.stop()
    features, actual = prepare_features(raw, preprocessor)
    transformed = preprocessor.transform(features)
    rf_pred = rf_model.predict(transformed)
    iso_pred = iso_model.predict(transformed)
except Exception as exc:
    st.error(f"Could not analyze this CSV: {exc}")
    st.caption("Check that it contains the model's feature columns and compatible values.")
    st.stop()

results = raw.copy()
results["Random Forest prediction"] = ["Attack" if str(v).strip().lower() in {"1", "1.0", "attack", "true"} else "Normal" for v in rf_pred]
results["Isolation Forest"] = ["Anomaly" if v == -1 else "Normal" for v in iso_pred]
results["Row"] = range(1, len(results) + 1)
if actual is not None:
    results["Trusted label"] = actual.map(binary_label)

with st.sidebar:
    st.divider()
    st.markdown("**Filter results**")
    traffic_types = st.multiselect("RF classification", ["Normal", "Attack"], default=["Normal", "Attack"])
    anomaly_types = st.multiselect("Anomaly flag", ["Normal", "Anomaly"], default=["Normal", "Anomaly"])
    max_rows = st.slider("Rows in table", 1, min(1000, len(results)), min(100, len(results)), step=10 if len(results) >= 10 else 1)

filtered = results[results["Random Forest prediction"].isin(traffic_types) & results["Isolation Forest"].isin(anomaly_types)]
total = len(results)
attack_n = int((results["Random Forest prediction"] == "Attack").sum())
anomaly_n = int((results["Isolation Forest"] == "Anomaly").sum())
normal_n = total - attack_n

if page == "Dashboard":
    section_header("Security overview", f"Analysis of {total:,} uploaded traffic records")
    a,b,c,d = st.columns(4)
    a.metric("Traffic records", f"{total:,}")
    b.metric("Predicted normal", f"{normal_n:,}")
    c.metric("Predicted attacks", f"{attack_n:,}")
    d.metric("Flagged anomalies", f"{anomaly_n:,}")
    left, right = st.columns([1.55, 1])
    trend = results.groupby("Row", as_index=False).agg(Attacks=("Random Forest prediction", lambda s: int((s == "Attack").sum())), Anomalies=("Isolation Forest", lambda s: int((s == "Anomaly").sum())))
    with left:
        st.subheader("Traffic overview")
        st.caption("Counts by row order in the uploaded file")
        if total > 1:
            trend["Window"] = ((trend["Row"] - 1) // max(1, total // 50) + 1)
            grouped = trend.groupby("Window", as_index=False)[["Attacks", "Anomalies"]].sum()
            long = grouped.melt("Window", var_name="Signal", value_name="Records")
            fig = px.line(long, x="Window", y="Records", color="Signal", color_discrete_map={"Attacks":"#ff5c70", "Anomalies":"#f5a524"}, markers=True)
            fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=15,b=0), legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("At least two records are needed to show a trend.")
    with right:
        st.subheader("Attack and anomaly distribution")
        dist = pd.DataFrame({"Category":["Normal", "Attack", "Anomaly"], "Records":[normal_n, attack_n, anomaly_n]})
        fig = px.pie(dist, names="Category", values="Records", hole=.62, color="Category", color_discrete_map={"Normal":"#3478f6", "Attack":"#f05263", "Anomaly":"#f5a524"})
        fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=5,b=0), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    st.subheader("Recent flagged records")
    flagged = filtered[(filtered["Random Forest prediction"] == "Attack") | (filtered["Isolation Forest"] == "Anomaly")]
    st.dataframe(flagged.tail(8).iloc[::-1], use_container_width=True, hide_index=True)

elif page == "Alerts":
    section_header("Security alerts", "Rows flagged by either model in the uploaded CSV")
    flagged = filtered[(filtered["Random Forest prediction"] == "Attack") | (filtered["Isolation Forest"] == "Anomaly")]
    st.metric("Flagged rows in current filter", f"{len(flagged):,}")
    st.dataframe(flagged.iloc[::-1].head(max_rows), use_container_width=True, hide_index=True)

elif page == "Traffic Monitor":
    section_header("Traffic monitor", "Explore model predictions in CSV row order")
    st.caption("This view analyzes the uploaded file; it does not receive live packets.")
    st.dataframe(filtered.head(max_rows), use_container_width=True, hide_index=True)

elif page == "Analysis":
    section_header("Model analysis", "Prediction breakdown and evaluation against trusted labels")
    left,right = st.columns(2)
    with left:
        st.subheader("Random Forest")
        st.bar_chart(results["Random Forest prediction"].value_counts())
    with right:
        st.subheader("Isolation Forest")
        st.bar_chart(results["Isolation Forest"].value_counts())
    if actual is not None:
        valid = results["Trusted label"].notna()
        if valid.any():
            truth = results.loc[valid, "Trusted label"]
            predicted = results.loc[valid, "Random Forest prediction"]
            report = classification_report(truth, predicted, labels=["Normal", "Attack"], output_dict=True, zero_division=0)
            m1,m2,m3 = st.columns(3)
            m1.metric("Accuracy", f"{accuracy_score(truth, predicted):.1%}")
            m2.metric("Attack precision", f"{report['Attack']['precision']:.1%}")
            m3.metric("Attack recall", f"{report['Attack']['recall']:.1%}")
            st.caption("Scores are shown only for recognized values in the CSV's label column. Trust them only when labels are ground truth not used as model inputs.")
            st.write("Confusion matrix · rows actual, columns predicted")
            st.dataframe(pd.DataFrame(confusion_matrix(truth,predicted,labels=["Normal","Attack"]), index=["Actual Normal","Actual Attack"], columns=["Predicted Normal","Predicted Attack"]), use_container_width=True)
        else:
            st.warning("A label column exists, but no values match recognized Normal/Attack labels; evaluation is skipped.")
    else:
        st.info("No label column found. Evaluation metrics need trusted ground-truth labels; predictions still work.")
    st.caption("Isolation Forest: -1 means anomaly and 1 means normal. An anomaly flag is not proof of an attack.")

else:
    section_header("Reports", "Export predictions for the currently selected filters")
    st.metric("Rows included in report", f"{len(filtered):,}")
    st.dataframe(filtered.head(25), use_container_width=True, hide_index=True)
    report_csv = filtered.drop(columns=["Row"], errors="ignore").to_csv(index=False).encode("utf-8-sig")
    st.download_button("Download filtered report (CSV)", data=report_csv, file_name="nids_security_report.csv", mime="text/csv", type="primary")

st.divider()
st.caption("Random Forest and Isolation Forest predictions are model outputs. This dashboard is CSV-based, and does not report fabricated blocked attacks, CPU usage, or active connections.")
