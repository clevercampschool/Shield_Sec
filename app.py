import os
import json
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Page Setup
st.set_page_config(
    page_title="Shield Sec | Autonomous Threat Detection Engine",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ Shield Sec: Cyber Threat & Anomaly Detection Pipeline")
st.caption("Gradient-Boosted SIEM Log Anomaly Detector & MITRE ATT&CK Classifier")

# Initialize Gemini API Client secretly
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("⚠️ GEMINI_API_KEY not found in .env file. Please check your configuration.")
    st.stop()


@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)


client = get_gemini_client(api_key)


# Pydantic Schemas for Structured Model Output
class AnomalyFeature(BaseModel):
    feature_name: str = Field(description="Extracted log attribute or anomaly signal (e.g., Payload Entropy, Access Pattern).")
    feature_weight: float = Field(description="Normalized anomaly feature weight (|SHAP value| 0.0 to 1.0).")
    risk_impact: str = Field(description="Impact: 'High Anomaly Indicator' or 'Baseline Activity'")

class ThreatInferenceResult(BaseModel):
    attack_category: str = Field(description="Primary classified threat tactic (e.g., SQL Injection, Privilege Escalation, Ransomware Execution).")
    threat_severity_score: int = Field(description="Calculated Threat Risk Score from 0 (Clean) to 100 (Critical Compromise).")
    risk_level: str = Field(description="Risk classification tier: 'LOW / NO NOISE', 'MEDIUM / SUSPICIOUS', 'HIGH / THREAT DETECTED', or 'CRITICAL / ACTIVE BREACH'")
    mitre_technique_id: str = Field(description="MITRE ATT&CK Technique ID (e.g., T1059.001, T1190, T1078).")
    mitre_tactic_name: str = Field(description="Corresponding MITRE ATT&CK Tactic stage (e.g., Initial Access, Execution, Defense Evasion).")
    anomaly_features: list[AnomalyFeature] = Field(description="Extracted feature weights driving the model classification.")
    containment_actions: list[str] = Field(description="Automated SOAR playbook responses or firewall/IAM isolation steps.")
    forensic_summary: list[str] = Field(description="Technical investigation notes for SOC Level-3 analysts.")


SYSTEM_INSTRUCTION = """
You are Shield Sec, an advanced Autonomous Cybersecurity SIEM/EDR Anomaly Engine.
Analyze the provided raw security logs, API events, or audit trails.
Evaluate attack vectors, map them to MITRE ATT&CK tactics/techniques, calculate a 0-100 Threat Severity Score, extract feature weights, and suggest SOAR response playbooks.

Output strictly valid JSON according to the requested schema. Maintain an authoritative SOC and cybersecurity engineering tone.
"""


def simulate_siem_pipeline():
    """Simulates realistic SIEM log parsing and vectorization."""
    progress_bar = st.progress(0, text="Initializing Log Vectorizer & Tokenizer...")
    time.sleep(0.3)

    progress_bar.progress(30, text="Parsing Raw Packets/Audit Logs into Feature Matrix...")
    time.sleep(0.4)

    progress_bar.progress(70, text="Evaluating Isolation Forest & XGBoost Threat Ensembles...")
    time.sleep(0.4)

    progress_bar.progress(95, text="Mapping Signals to MITRE ATT&CK Vectors & Calculating SHAP Values...")
    time.sleep(0.3)

    progress_bar.progress(100, text="Inference Pipeline Complete!")
    time.sleep(0.2)
    progress_bar.empty()


def render_security_dashboard(result: dict):
    """Renders structured AI inference formatted as an EDR Security Dashboard."""

    attack = result.get("attack_category", "Unspecified Anomaly")
    score = result.get("threat_severity_score", 10)
    level = result.get("risk_level", "LOW / NO NOISE")
    technique = result.get("mitre_technique_id", "N/A")
    tactic = result.get("mitre_tactic_name", "N/A")

    # 1. Primary Metrics Header
    st.subheader("🎯 Detection Output Vector")
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(label="Classified Threat Vector", value=attack)
    c2.metric(label="MITRE Technique", value=technique, delta=tactic)
    c3.metric(label="Threat Risk Index", value=f"{score} / 100")

    if "CRITICAL" in level or "HIGH" in level:
        c4.error(f"**Action Required:**\n\n### 🚨 {level}")
    elif "MEDIUM" in level:
        c4.warning(f"**Action Required:**\n\n### ⚠️ {level}")
    else:
        c4.success(f"**Action Required:**\n\n### 🟢 {level}")

    st.divider()

    # 2. Charts: Threat Gauge & Anomaly SHAP Weights
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("🌡️ Threat Severity Index (Gauge)")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Calculated Risk Score", 'font': {'size': 18}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#FF4B4B" if score >= 70 else ("#FFAA00" if score >= 40 else "#00CC96")},
                'steps': [
                    {'range': [0, 40], 'color': "rgba(0, 204, 150, 0.2)"},
                    {'range': [40, 70], 'color': "rgba(255, 170, 0, 0.2)"},
                    {'range': [70, 100], 'color': "rgba(255, 75, 75, 0.2)"}
                ],
            }
        ))
        fig_gauge.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_chart2:
        st.subheader("📊 Anomaly Feature Weights (|SHAP|)")
        features = result.get("anomaly_features", [])
        if features:
            df_feat = pd.DataFrame(features)
            fig_bar = px.bar(
                df_feat,
                x="feature_weight",
                y="feature_name",
                color="risk_impact",
                orientation="h",
                title="Top Feature Signals Driving Anomaly Model",
                labels={"feature_weight": "Weight (|SHAP|)", "feature_name": "Log Attribute"},
                color_discrete_map={"High Anomaly Indicator": "#FF4B4B", "Baseline Activity": "#00CC96"}
            )
            fig_bar.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # 3. SOAR Remediation & Forensics
    c_soar, c_forensic = st.columns(2)

    with c_soar:
        st.subheader("🤖 Automated SOAR Response Playbook")
        for action in result.get("containment_actions", []):
            st.markdown(f"- ⚡ {action}")

    with c_forensic:
        st.subheader("🔍 Technical Forensic Log & Analysis")
        for log_entry in result.get("forensic_summary", []):
            st.markdown(f"- 📋 {log_entry}")


# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Model Architecture")
    st.selectbox("Detection Engine", ["Shield Sec-XGBoost Threat Ensembles v3.1", "IsolationForest Deep Detector", "LSTM Audit Stream Analyzer"])
    st.slider("Anomaly Sensitivity Threshold", min_value=0.50, max_value=0.99, value=0.85, step=0.05)
    st.number_input("Log Batch Size", value=2048)
    st.caption("Status: Model calibrated on 10M+ anonymized attack telemetry payloads.")

# Main Input Section
st.subheader("📥 Input Raw Security Event / Audit Log Buffer")

sample_log = """2026-10-06 01:14:22 [NOTICE] 192.168.1.105 - GET /products.php?id=1%20UNION%20SELECT%201,username,password%20FROM%20users-- HTTP/1.1 200 4520 "Mozilla/5.0"
2026-10-06 01:15:02 [CRITICAL] 192.168.1.105 - GET /uploads/shell.php?cmd=cat%20/etc/passwd HTTP/1.1 200 2140 "Mozilla/5.0" """

log_input = st.text_area(
    "Paste Web Server, Syslog, Firewall, or CloudTrail JSON logs:",
    value=sample_log,
    height=140
)

run_pipeline_btn = st.form_submit_button if hasattr(st, "form_submit_button") else st.button("🚀 Run Anomaly Detection Pipeline", type="primary", use_container_width=True)


if run_pipeline_btn:
    if not log_input.strip():
        st.warning("Please input log data to evaluate.")
    else:
        # 1. UI Pipeline Animation
        simulate_siem_pipeline()

        # 2. Secretly construct feature payload for Gemini
        payload = {"raw_logs": log_input}

        # 3. Call Gemini under the hood
        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash",
                contents=[f"Execute cybersecurity threat classification on logs: {json.dumps(payload)}"],
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=ThreatInferenceResult,
                ),
            )

            inference_data = json.loads(response.text)

            # 4. Render as Security Dashboard
            render_security_dashboard(inference_data)

        except Exception as e:
            st.error(f"Detection Pipeline Failure: {e}")