import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mode
import time

# --- Page Config ---
st.set_page_config(
    page_title="LCCDE - Intrusion Detection System",
    page_icon="🛡️",
    layout="wide",
)

# --- Load Custom CSS ---
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("styles.css")

# --- Load Models & Metadata ---
@st.cache_resource
def load_assets():
    base_path = os.path.dirname(__file__)
    model_paths = {
        'lg': os.path.join(base_path, 'models', 'lightgbm_model.joblib'),
        'xg': os.path.join(base_path, 'models', 'xgboost_model.joblib'),
        'cb': os.path.join(base_path, 'models', 'catboost_model.joblib'),
        'leaders': os.path.join(base_path, 'models', 'leader_indices.joblib'),
        'features': os.path.join(base_path, 'models', 'feature_names.joblib')
    }
    
    try:
        # Check if all files exist
        for name, path in model_paths.items():
            if not os.path.exists(path):
                st.error(f"Missing model file: {os.path.basename(path)}")
                return None
                
        lg = joblib.load(model_paths['lg'])
        xg = joblib.load(model_paths['xg'])
        cb = joblib.load(model_paths['cb'])
        leader_indices = joblib.load(model_paths['leaders'])
        feature_names = joblib.load(model_paths['features'])
        return lg, xg, cb, leader_indices, feature_names
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

assets = load_assets()

# --- Attack Types Mapping ---
ATTACK_TYPES = {
    0: "BENIGN",
    1: "Bot",
    2: "BruteForce",
    3: "DoS",
    4: "Infiltration",
    5: "PortScan",
    6: "WebAttack"
}

# --- LCCDE Ensemble Logic ---
def lccde_predict(row, m1, m2, m3, leader_indices, model_list):
    # Prepare row for prediction (ensure it's 2D)
    x = row.reshape(1, -1)
    
    # Base Predictions
    y_pred1 = int(np.ravel(m1.predict(x))[0])
    y_pred2 = int(np.ravel(m2.predict(x))[0])
    y_pred3 = int(np.ravel(m3.predict(x))[0])
    
    # Confidence scores
    p1 = m1.predict_proba(x)
    p2 = m2.predict_proba(x)
    p3 = m3.predict_proba(x)
    
    y_pred_p1 = np.max(p1)
    y_pred_p2 = np.max(p2)
    y_pred_p3 = np.max(p3)
    
    # LCCDE Logic
    if y_pred1 == y_pred2 == y_pred3:
        return y_pred1
    elif y_pred1 != y_pred2 != y_pred3:
        # Check against class leaders
        candidates = []
        confs = []
        
        # model_list is [lg, xg, cb]
        if leader_indices[y_pred1] == 0: candidates.append(y_pred1); confs.append(y_pred_p1)
        if leader_indices[y_pred2] == 1: candidates.append(y_pred2); confs.append(y_pred_p2)
        if leader_indices[y_pred3] == 2: candidates.append(y_pred3); confs.append(y_pred_p3)
        
        if not candidates:
            idx = np.argmax([y_pred_p1, y_pred_p2, y_pred_p3])
            return [y_pred1, y_pred2, y_pred3][idx]
        elif len(candidates) == 1:
            return candidates[0]
        else:
            idx = np.argmax(confs)
            return candidates[idx]
    else:
        # Majority vote
        return mode([y_pred1, y_pred2, y_pred3])

# --- Navigation ---
st.sidebar.markdown("<h1 style='text-align: center; color: #00ffcc;'>IDS 🛡️ SHIELD</h1>", unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", ["Home", "Live Monitor", "Detector", "Model Specs"])

if page == "Home":
    st.markdown("<h1 class='main-title'>Shielding Your Network with AI</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to the LCCDE Intrusion Detection System
        This application utilizes a **Leader Class and Confidence Decision Ensemble (LCCDE)** framework to detect sophisticated network intrusions. 
        
        **Key Features:**
        - **Multi-Model Synergy**: Combines LightGBM, XGBoost, and CatBoost.
        - **High Accuracy**: Optimized for various attack classes including DoS, Botnets, and Web Attacks.
        - **Real-time Processing**: Fast inference for large-scale network logs.
        """)
        
        st.info("💡 **Tip**: Navigate to the 'Detector' page to upload network traffic logs for analysis.")
        
    with col2:
        st.image("https://img.freepik.com/free-vector/cyber-security-concept_23-2148532223.jpg?w=740", use_container_width=True)

elif page == "Live Monitor":
    st.markdown("<h1 class='main-title'>Simulated Live Network Feed</h1>", unsafe_allow_html=True)
    
    if assets is None:
        st.warning("Models not found. Please train models first.")
    else:
        lg, xg, cb, leader_indices, feature_names = assets
        model_list = [lg, xg, cb]
        
        # Load sample data for simulation
        df_sample = pd.read_csv("./data/CICIDS2017_sample_km.csv").sample(500)
        X_sample = df_sample[feature_names].values
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### 📡 Live Traffic Stream")
            status_placeholder = st.empty()
            chart_placeholder = st.empty()
            log_placeholder = st.empty()
            
        with col2:
            st.markdown("### 🔔 Security Alerts")
            alert_placeholder = st.empty()
            stats_placeholder = st.empty()

        if st.button("Start Live Monitoring"):
            history = []
            threat_count = 0
            
            for i in range(len(X_sample)):
                # Simulate packet arrival
                time.sleep(0.5) 
                
                # Predict
                pred = lccde_predict(X_sample[i], lg, xg, cb, leader_indices, model_list)
                label = ATTACK_TYPES.get(pred, "Unknown")
                
                timestamp = time.strftime("%H:%M:%S")
                packet_info = {"Time": timestamp, "Source": f"192.168.1.{np.random.randint(2, 254)}", "Status": label}
                history.append(packet_info)
                
                # Update UI
                status_placeholder.write(f"🟢 **Monitoring Active** | Analyzing packet #{i+1}...")
                
                # Alerts
                if label != "BENIGN":
                    threat_count += 1
                    alert_placeholder.error(f"🚨 **ALERT**: {label} attack detected at {timestamp}!")
                
                # Stats
                stats_placeholder.markdown(f"""
                - **Total Packets**: {i+1}
                - **Threats Blocked**: {threat_count}
                - **System Status**: {'⚠️ CRITICAL' if threat_count > 0 else '✅ SECURE'}
                """)
                
                # Table Log
                log_df = pd.DataFrame(history[-10:][::-1]) # Last 10 rows
                log_placeholder.table(log_df)
                
                # Chart
                if i > 5:
                    counts = pd.DataFrame(history)['Status'].value_counts()
                    chart_placeholder.bar_chart(counts)

elif page == "Detector":
    st.markdown("<h1 class='main-title'>Threat Detector</h1>", unsafe_allow_html=True)
    
    if assets is None:
        st.warning("Models not found. Please train models first.")
    else:
        lg, xg, cb, leader_indices, feature_names = assets
        model_list = [lg, xg, cb]
        
        uploaded_file = st.file_uploader("Upload Network Log (CSV)", type=["csv"])
        
        if uploaded_file is not None:
            df_upload = pd.read_csv(uploaded_file)
            
            # Basic validation
            missing_cols = set(feature_names) - set(df_upload.columns)
            if missing_cols:
                st.error(f"Uploaded CSV is missing required features: {list(missing_cols)[:5]}...")
            else:
                st.success("Log file validated successfully!")
                
                num_rows = st.slider("Select number of rows to analyze", 1, len(df_upload), min(100, len(df_upload)))
                df_to_test = df_upload.iloc[:num_rows]
                
                if st.button("Start AI Analysis"):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results = []
                    X_data = df_to_test[feature_names].values
                    
                    start_time = time.time()
                    for i in range(len(X_data)):
                        pred = lccde_predict(X_data[i], lg, xg, cb, leader_indices, model_list)
                        results.append(ATTACK_TYPES.get(pred, "Unknown"))
                        
                        if i % 10 == 0:
                            progress_bar.progress((i + 1) / len(X_data))
                            status_text.text(f"Analyzing packet {i+1}/{len(X_data)}...")
                    
                    end_time = time.time()
                    
                    st.markdown(f"### Analysis Results (Completed in {end_time - start_time:.2f}s)")
                    df_to_test['Detection'] = results
                    
                    # Highlights
                    threats_found = df_to_test[df_to_test['Detection'] != "BENIGN"]
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Total Packets", len(df_to_test))
                    m2.metric("Threats Detected", len(threats_found), delta=len(threats_found), delta_color="inverse")
                    m3.metric("Safety Score", f"{((len(df_to_test)-len(threats_found))/len(df_to_test))*100:.1f}%")
                    
                    st.dataframe(df_to_test[['Detection'] + feature_names[:10]].head(50), use_container_width=True)
                    
                    if not threats_found.empty:
                        st.warning("⚠️ **Warning**: Malicious activity detected in the network stream.")
                        st.download_button("Download Threat Report", df_to_test.to_csv(index=False), "threat_report.csv", "text/csv")

elif page == "Analytics":
    st.markdown("<h1 class='main-title'>Threat Insights</h1>", unsafe_allow_html=True)
    
    st.info("Upload data in the 'Detector' tab to see analytics here.")
    # In a real app, we'd persist results in session_state
    if 'results_df' in st.session_state:
        # Visualizations
        pass
    else:
        st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <img src='https://cdn-icons-png.flaticon.com/512/1055/1055644.png' width='100'>
            <p>Awaiting data analysis...</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "Model Specs":
    st.markdown("<h1 class='main-title'>LCCDE Model Architecture</h1>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### How LCCDE Works
        The Leader Class and Confidence Decision Ensemble (LCCDE) works by assigning a "Leader" model for each attack class based on training F1-scores.
        
        **Base Learners:**
        1. **LightGBM**: Fast, distributed, high-performance gradient boosting.
        2. **XGBoost**: Highly flexible and portable.
        3. **CatBoost**: Handles categorical features natively and is robust to overfitting.
        """)
        
    with col2:
        st.markdown("### Performance Overview")
        st.write("Current model weights based on CICIDS2017 training:")
        metrics_df = pd.DataFrame({
            "Attack Type": ["BENIGN", "Bot", "BruteForce", "DoS", "Infiltration", "PortScan", "WebAttack"],
            "Best Model": ["XGBoost", "XGBoost", "LightGBM", "XGBoost", "LightGBM", "LightGBM", "XGBoost"],
            "Accuracy": [0.99, 0.99, 1.0, 0.99, 0.85, 0.99, 0.99]
        })
        st.table(metrics_df)

st.sidebar.markdown("---")
st.sidebar.markdown("<p style='text-align: center; color: #888;'>v1.0.0 | Powered by LCCDE</p>", unsafe_allow_html=True)
