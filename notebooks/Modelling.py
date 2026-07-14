"""
Giorni 5-6 - Modellazione churn
Continua dalla EDA: split -> feature engineering -> training -> metriche -> pickle
"""
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report, precision_recall_curve
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier
import shap
import matplotlib.pyplot as plt

INPUT_PATH = "C:\\airflow-project\\data\\raw\\dataset.csv"  # dataset già pulito da EDA (num_cols numerici, cat_cols object)

# ============================================
# 0. LOAD + TRAIN/TEST SPLIT (prima di imputazione/encoding)
# ============================================
df = pd.read_csv(INPUT_PATH, sep=';', decimal=',', low_memory=False)

num_cols = [
    'rev_Mean', 'mou_Mean', 'totmrc_Mean', 'da_Mean', 'ovrmou_Mean',
    'ovrrev_Mean', 'vceovr_Mean', 'datovr_Mean', 'roam_Mean', 'change_mou',
    'change_rev', 'drop_vce_Mean', 'drop_dat_Mean', 'blck_vce_Mean',
    'blck_dat_Mean', 'unan_vce_Mean', 'unan_dat_Mean', 'plcd_vce_Mean',
    'plcd_dat_Mean', 'recv_vce_Mean', 'recv_sms_Mean', 'comp_vce_Mean',
    'comp_dat_Mean', 'custcare_Mean', 'ccrndmou_Mean', 'cc_mou_Mean',
    'inonemin_Mean', 'threeway_Mean', 'mou_cvce_Mean', 'mou_cdat_Mean',
    'mou_rvce_Mean', 'owylis_vce_Mean', 'mouowylisv_Mean',
    'iwylis_vce_Mean', 'mouiwylisv_Mean', 'peak_vce_Mean', 'peak_dat_Mean',
    'mou_peav_Mean', 'mou_pead_Mean', 'opk_vce_Mean', 'opk_dat_Mean',
    'mou_opkv_Mean', 'mou_opkd_Mean', 'drop_blk_Mean', 'attempt_Mean',
    'complete_Mean', 'callfwdv_Mean', 'callwait_Mean', 'months', 'uniqsubs',
    'actvsubs', 'totcalls', 'totmou', 'totrev', 'adjrev', 'adjmou',
    'adjqty', 'avgrev', 'avgmou', 'avgqty', 'avg3mou', 'avg3qty', 'avg3rev',
    'avg6mou', 'avg6qty', 'avg6rev', 'hnd_price', 'phones', 'models',
    'truck', 'rv', 'lor', 'adults', 'income', 'forgntvl', 'eqpdays'
]
cat_cols = [
    'new_cell', 'crclscod', 'asl_flag', 'prizm_social_one', 'area',
    'dualband', 'refurb_new', 'hnd_webcap', 'ownrent', 'dwlltype',
    'marital', 'infobase', 'ethnic', 'kid0_2', 'kid3_5', 'kid6_10',
    'kid11_15', 'kid16_17', 'creditcd'
]
missing_pct = df.isna().mean().sort_values(ascending=False)*100

# 3a. Drop colonne con troppi missing (>35%)
cols_to_drop = missing_pct[missing_pct > 35].index.tolist()
print("Drop:", cols_to_drop)
df = df.drop(columns=cols_to_drop)

X = df.drop(columns=['churn', 'Customer_ID'])
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================
# 1. LOG TRANSFORM variabili skewed (utile per LogReg, ininfluente per XGBoost)
# ============================================
skewed_vars = ['totrev', 'mou_Mean']
for col in skewed_vars:
    X_train[f'{col}_log'] = np.log1p(X_train[col].clip(lower=0))
    X_test[f'{col}_log'] = np.log1p(X_test[col].clip(lower=0))

# ============================================
# 2. IMPUTAZIONE (fit su train, applica su test)
# ============================================
num_cols_final = X_train.select_dtypes(include='number').columns
cat_cols_final = X_train.select_dtypes(include='object').columns
medians = X_train[num_cols_final].median()
X_train[num_cols_final] = X_train[num_cols_final].fillna(medians)
X_test[num_cols_final] = X_test[num_cols_final].fillna(medians)
X_train[cat_cols_final] = X_train[cat_cols_final].fillna('Missing')
X_test[cat_cols_final] = X_test[cat_cols_final].fillna('Missing')

# ============================================
# 3. TARGET ENCODING alta cardinalità (fit su train)
# ============================================
high_card_cols = [c for c in ['crclscod', 'area', 'ethnic'] if c in cat_cols_final]

MIN_OBS = 30
rare_categories = {}  # salvato per replicare la stessa logica in inference

for col in high_card_cols:
    counts = X_train[col].value_counts()
    rare = counts[counts < MIN_OBS].index
    rare_categories[col] = rare
    X_train[col] = X_train[col].where(~X_train[col].isin(rare), 'Other')
    X_test[col] = X_test[col].where(~X_test[col].isin(rare), 'Other')
    print(f"{col}: {len(rare)} categorie raggruppate in 'Other' (soglia n<{MIN_OBS})")

global_mean = y_train.mean()
target_maps = {}
for col in high_card_cols:
    target_mean = y_train.groupby(X_train[col]).mean()
    target_maps[col] = target_mean
    X_train[f'{col}_target_enc'] = X_train[col].map(target_mean)
    X_test[f'{col}_target_enc'] = X_test[col].map(target_mean).fillna(global_mean)

# ============================================
# 4. ONE-HOT bassa cardinalità
# ============================================
low_card_cols = [c for c in cat_cols_final if c not in high_card_cols]
X_train = pd.get_dummies(X_train, columns=low_card_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=low_card_cols, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
X_train = X_train.drop(columns=high_card_cols)
X_test = X_test.drop(columns=high_card_cols)

feature_columns = X_train.columns.tolist()

# ============================================
# 5. MODELLI
# Scelta: XGBoost come modello principale.
#   - Multicollinearità confermata in EDA -> i modelli tree-based non ne risentono
#     (split greedy su singole feature, non serve invertire matrici come in LogReg)
#   - EDA mostra relazioni prevalentemente non lineari (bassa corr. lineare,
#     alta importance in RF) -> XGBoost cattura interazioni e non linearità
#   - Class balance ~50/50 -> non serve class_weight/resampling
# LogReg mantenuta come baseline interpretabile (richiede scaling, sensibile a multicollinearità).
# ============================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_scaled, y_train)

# --- VECCHIO: parametri fissi, nessun tuning ---
# xgb = XGBClassifier(
#     n_estimators=300, max_depth=5, learning_rate=0.05,
#     subsample=0.8, colsample_bytree=0.8,
#     eval_metric='auc', random_state=42, n_jobs=-1
# )
# xgb.fit(X_train, y_train)

# --- NUOVO: RandomizedSearchCV, ottimizza su AUC con CV stratificata ---
# Perché: i parametri fissi sopra erano una scelta ragionevole ma arbitraria;
# la ricerca esplora lo spazio iperparametri e riduce il rischio di over/underfitting.
# scoring='roc_auc' perché coerente con la metrica threshold-indipendente scelta prima;
# n_iter basso (20) per tenere i tempi contenuti, aumentabile se si ha più tempo.
xgb_param_dist = {
    'n_estimators': [200, 300, 400, 500],
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.03, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
xgb_search = RandomizedSearchCV(
    XGBClassifier(eval_metric='auc', random_state=42, n_jobs=-1),
    param_distributions=xgb_param_dist,
    n_iter=20, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1, verbose=1
)
xgb_search.fit(X_train, y_train)
xgb = xgb_search.best_estimator_
print(f"\nMigliori iperparametri XGBoost: {xgb_search.best_params_}")
print(f"AUC media in CV: {xgb_search.best_score_:.4f}")

# ============================================
# 6. METRICHE
# Business case: mancare un cliente che sta per fare churn (falso negativo) costa
# più di un falso allarme (retention cost < costo di riacquisizione cliente).
#   - RECALL: priorità, minimizza i churner non intercettati
#   - AUC: valuta il ranking indipendentemente dalla soglia (utile per definire
#     soglie diverse per campagne con budget limitato)
#   - F1: controllo che il recall non sia ottenuto sacrificando troppa precision
#     (costo di campagne su falsi positivi)
# ============================================
def evaluate(model, X_ev, y_ev, name, proba_input=None):
    proba = model.predict_proba(proba_input if proba_input is not None else X_ev)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"\n--- {name} ---")
    print(f"AUC:       {roc_auc_score(y_ev, proba):.4f}")
    print(f"Recall:    {recall_score(y_ev, pred):.4f}")
    print(f"Precision: {precision_score(y_ev, pred):.4f}")
    print(f"F1:        {f1_score(y_ev, pred):.4f}")
    print(confusion_matrix(y_ev, pred))
    return proba

evaluate(logreg, X_test_scaled, y_test, "Logistic Regression")
xgb_proba = evaluate(xgb, X_test, y_test, "XGBoost (soglia 0.5)")


# ============================================
# 7. FEATURE IMPORTANCE (modello finale: XGBoost)
# ============================================
importances = pd.Series(xgb.feature_importances_, index=feature_columns).sort_values(ascending=False)
print("\nTop 15 feature importance (XGBoost):")
print(importances.head(15))

# ============================================
# 8. SHAP - interpretabilità globale e locale (per presentazione business)
# TreeExplainer: esatto e veloce per modelli tree-based (no approssimazione kernel)
# Uso X_test: spiega il comportamento del modello su dati mai visti in training
# ============================================
explainer = shap.TreeExplainer(xgb)
shap_values = explainer(X_test)

# 8a. Summary plot globale: direzione + magnitudine dell'effetto per feature
# (sostituisce/arricchisce il ranking di importanza: qui si vede anche se
# valori alti/bassi della feature spingono verso churn o no)
plt.figure()
shap.summary_plot(shap_values, X_test, max_display=15, show=False)
plt.tight_layout()
plt.savefig('shap_summary_global.png', dpi=150)
plt.close()

# 8b. Bar plot: mean(|SHAP|) per feature, versione più leggibile per slide
plt.figure()
shap.summary_plot(shap_values, X_test, plot_type='bar', max_display=15, show=False)
plt.tight_layout()
plt.savefig('shap_importance_bar.png', dpi=150)
plt.close()

# 8c. Spiegazione locale: waterfall per un singolo cliente ad alto rischio
# Utile per la presentazione: "perché il modello prevede churn per QUESTO cliente"
high_risk_idx = int(np.argmax(xgb_proba))
plt.figure()
shap.plots.waterfall(shap_values[high_risk_idx], max_display=12, show=False)
plt.tight_layout()
plt.savefig('shap_waterfall_high_risk_customer.png', dpi=150)
plt.close()

print(f"\nSHAP salvati: shap_summary_global.png, shap_importance_bar.png, "
      f"shap_waterfall_high_risk_customer.png (cliente idx {high_risk_idx}, "
      f"proba churn={xgb_proba[high_risk_idx]:.3f})")

# ============================================
# 9. SALVATAGGIO MODELLO + ARTEFATTI DI PREPROCESSING
# Salvo anche medians/target_maps/feature_columns: necessari per replicare
# esattamente il preprocessing in inference, senza fit-again su nuovi dati.
# ============================================
artifact = {
    'model': xgb,
    'feature_columns': feature_columns,
    'medians': medians,
    'target_maps': target_maps,
    'global_mean': global_mean,
    'high_card_cols': high_card_cols,
    'low_card_cols': low_card_cols,
    'skewed_vars': skewed_vars,
    'decision_threshold': 0.5,
}
with open('churn_xgb_model.pkl', 'wb') as f:
    pickle.dump(artifact, f)

print("\nModello salvato in churn_xgb_model.pkl")