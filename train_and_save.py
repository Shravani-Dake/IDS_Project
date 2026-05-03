import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import lightgbm as lgb
import catboost as cbt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# Set working directory to the project root
project_root = r"d:\Cyber Security\Intrusion-Detection-System-Using-Machine-Learning"
os.chdir(project_root)

def train_and_save():
    print("Loading dataset...")
    df = pd.read_csv("./data/CICIDS2017_sample_km.csv")
    
    X = df.drop(['Label'], axis=1)
    y = df['Label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

    print("Applying SMOTE...")
    smote = SMOTE(sampling_strategy={2: 1000, 4: 1000}, random_state=0)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    print("Training LightGBM...")
    lg = lgb.LGBMClassifier()
    lg.fit(X_train, y_train)
    lg_f1 = f1_score(y_test, lg.predict(X_test), average=None)

    print("Training XGBoost...")
    xg = xgb.XGBClassifier()
    xg.fit(X_train.values, y_train)
    xg_f1 = f1_score(y_test, xg.predict(X_test.values), average=None)

    print("Training CatBoost...")
    cb = cbt.CatBoostClassifier(verbose=0, boosting_type='Plain')
    cb.fit(X_train, y_train)
    cb_f1 = f1_score(y_test, np.ravel(cb.predict(X_test)), average=None)

    # Determine leader models for LCCDE
    leader_models_indices = []
    for i in range(len(lg_f1)):
        scores = [lg_f1[i], xg_f1[i], cb_f1[i]]
        leader_models_indices.append(np.argmax(scores)) # 0: LightGBM, 1: XGBoost, 2: CatBoost

    print("Saving models...")
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(lg, 'models/lightgbm_model.joblib')
    joblib.dump(xg, 'models/xgboost_model.joblib')
    joblib.dump(cb, 'models/catboost_model.joblib')
    joblib.dump(leader_models_indices, 'models/leader_indices.joblib')
    
    # Save feature names for validation during upload
    joblib.dump(X.columns.tolist(), 'models/feature_names.joblib')

    print("All models and metadata saved to 'models/' directory.")

if __name__ == "__main__":
    train_and_save()
