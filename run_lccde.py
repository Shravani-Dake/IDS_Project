import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score, precision_score, recall_score, f1_score
import lightgbm as lgb
import catboost as cbt
import xgboost as xgb
import time
from river import stream
from statistics import mode
from imblearn.over_sampling import SMOTE
import os

# Set working directory to the project root
os.chdir(r"d:\Cyber Security\Intrusion-Detection-System-Using-Machine-Learning")

# Read dataset
print("Loading dataset...")
df = pd.read_csv("./data/CICIDS2017_sample_km.csv")
print("Value counts:")
print(df.Label.value_counts())

# Split
X = df.drop(['Label'],axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X,y, train_size = 0.8, test_size = 0.2, random_state = 0)

# SMOTE
print("Before SMOTE:")
print(pd.Series(y_train).value_counts())
smote=SMOTE(sampling_strategy={2:1000,4:1000}) 
X_train, y_train = smote.fit_resample(X_train, y_train)
print("After SMOTE:")
print(pd.Series(y_train).value_counts())

# LightGBM
print("Training LightGBM...")
lg = lgb.LGBMClassifier()
lg.fit(X_train, y_train)
y_pred_lg = lg.predict(X_test)
lg_f1 = f1_score(y_test, y_pred_lg, average=None)
print("LightGBM F1:", lg_f1)

# XGBoost
print("Training XGBoost...")
xg = xgb.XGBClassifier()
xg.fit(X_train.values, y_train)
y_pred_xg = xg.predict(X_test.values)
xg_f1 = f1_score(y_test, y_pred_xg, average=None)
print("XGBoost F1:", xg_f1)

# CatBoost
print("Training CatBoost...")
cb = cbt.CatBoostClassifier(verbose=0, boosting_type='Plain')
cb.fit(X_train, y_train)
y_pred_cb = cb.predict(X_test)
cb_f1 = f1_score(y_test, y_pred_cb, average=None)
print("CatBoost F1:", cb_f1)

# LCCDE
model=[]
for i in range(len(lg_f1)):
    if max(lg_f1[i],xg_f1[i],cb_f1[i]) == lg_f1[i]:
        model.append(lg)
    elif max(lg_f1[i],xg_f1[i],cb_f1[i]) == xg_f1[i]:
        model.append(xg)
    else:
        model.append(cb)

def LCCDE(X_test, y_test, m1, m2, m3):
    yt = []
    yp = []
    count = 0
    total = len(X_test)
    for xi, yi in stream.iter_pandas(X_test, y_test):
        xi2=np.array(list(xi.values()))
        y_pred1 = int(np.ravel(m1.predict(xi2.reshape(1, -1)))[0])
        y_pred2 = int(np.ravel(m2.predict(xi2.reshape(1, -1)))[0])
        y_pred3 = int(np.ravel(m3.predict(xi2.reshape(1, -1)))[0])

        p1 = m1.predict_proba(xi2.reshape(1, -1))
        p2 = m2.predict_proba(xi2.reshape(1, -1))
        p3 = m3.predict_proba(xi2.reshape(1, -1))

        y_pred_p1 = np.max(p1)
        y_pred_p2 = np.max(p2)
        y_pred_p3 = np.max(p3)

        if y_pred1 == y_pred2 == y_pred3:
            y_pred = y_pred1
        elif y_pred1 != y_pred2 != y_pred3:
            l = []
            pred_l = []
            pro_l = []
            if model[y_pred1]==m1:
                l.append(m1); pred_l.append(y_pred1); pro_l.append(y_pred_p1)
            if model[y_pred2]==m2:
                l.append(m2); pred_l.append(y_pred2); pro_l.append(y_pred_p2)
            if model[y_pred3]==m3:
                l.append(m3); pred_l.append(y_pred3); pro_l.append(y_pred_p3)

            if len(l)==0:
                pro_l=[y_pred_p1,y_pred_p2,y_pred_p3]
                max_p = max(pro_l)
                if max_p == y_pred_p1: y_pred = y_pred1
                elif max_p == y_pred_p2: y_pred = y_pred2
                else: y_pred = y_pred3
            elif len(l)==1:
                y_pred=pred_l[0]
            else:
                max_p = max(pro_l)
                if max_p == y_pred_p1: y_pred = y_pred1
                elif max_p == y_pred_p2: y_pred = y_pred2
                else: y_pred = y_pred3
        else:
            n = mode([y_pred1,y_pred2,y_pred3])
            y_pred = int(np.ravel(model[n].predict(xi2.reshape(1, -1)))[0])

        yt.append(yi)
        yp.append(y_pred)
        count += 1
        if count % 500 == 0:
            print(f"Processed {count}/{total} samples...")

    return yt, yp

print("Running LCCDE ensemble inference...")
start_time = time.time()
yt, yp = LCCDE(X_test, y_test, m1 = lg, m2 = xg, m3 = cb)
end_time = time.time()

print(f"Inference completed in {end_time - start_time:.2f} seconds.")
print("Accuracy of LCCDE: "+ str(accuracy_score(yt, yp)))
print("Classification Report:")
print(classification_report(yt, yp))
