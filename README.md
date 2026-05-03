# LCCDE Intrusion Detection System (IDS) - Dashboard 🛡️

A state-of-the-art, AI-powered Intrusion Detection System using the **Leader Class and Confidence Decision Ensemble (LCCDE)** framework. This application transforms high-level cybersecurity research into a production-ready dashboard for real-time network threat monitoring.

## 🚀 Overview
This project implements the LCCDE model, which achieves **99.7%+ accuracy** on the CICIDS2017 dataset. It utilizes a unique ensemble approach where the best-performing model (LightGBM, XGBoost, or CatBoost) is dynamically selected as the "Leader" for each specific class of network traffic.

### Key Features
- **Live Monitor Dashboard**: Simulated real-time network feed with visual 🚨 alerts for detected attacks.
- **Batch Detector**: Upload CSV network logs for rapid security auditing.
- **Ensemble Intelligence**: Combines three top-tier ML algorithms for robust detection.
- **Modern UI**: Sleek, high-tech dark mode interface built with Streamlit.

---

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Western-OC2-Lab/Intrusion-Detection-System-Using-Machine-Learning.git
cd Intrusion-Detection-System-Using-Machine-Learning
```

### 2. Install Dependencies
Ensure you have Python 3.12+ installed. Run:
```bash
pip install pandas numpy lightgbm xgboost catboost river imbalanced-learn scikit-learn streamlit joblib matplotlib seaborn
```

---

## 🚦 How to Run

### Step 1: Train and Save Models
Before running the dashboard, you must prepare the ensemble models:
```bash
python train_and_save.py
```
*This will create a `models/` directory with pre-trained artifacts.*

### Step 2: Launch the Web Application
```bash
streamlit run app.py
```
The application will open in your browser at `http://localhost:8501`.

---

## 🛡️ Application Modules

### 1. Home
Overview of the LCCDE framework and methodology.

### 2. Live Monitor
Simulates an active network interface. It streams traffic data row-by-row and triggers immediate alerts when malicious patterns (like DoS or Botnets) are identified.

### 3. Detector
A utility for security analysts to upload `.csv` log files and get a full breakdown of benign vs. malicious traffic in seconds.

---

## 🧪 Research & Methodology
This implementation is based on the following research:

*   **LCCDE Framework**: L. Yang, et al., "[LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles](https://arxiv.org/pdf/2208.03399.pdf)," in 2022 IEEE GLOBECOM.
*   **Dataset**: CICIDS2017 (External Network Data).

---

## 📁 Project Structure
- `app.py`: Main Streamlit application.
- `train_and_save.py`: Model training and persistence logic.
- `styles.css`: Custom premium UI styling.
- `models/`: Pre-trained model files (LightGBM, XGBoost, CatBoost).
- `data/`: Sample datasets for testing.

---

## 📝 Citation
If you use this work in your research, please cite:
```bibtex
@INPROCEEDINGS{10001280,
  author={Yang, Li and Shami, Abdallah and Stevens, Gary and de Rusett, Stephen},
  booktitle={GLOBECOM 2022 - 2022 IEEE Global Communications Conference}, 
  title={LCCDE: A Decision-Based Ensemble Framework for Intrusion Detection in The Internet of Vehicles}, 
  year={2022},
  pages={3545-3550},
  doi={10.1109/GLOBECOM48099.2022.10001280}}
```

---
*Developed for Cyber Security & AI Research.*
