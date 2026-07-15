"""
Test locale feature engineering - churn model
Basato su data_preparation.py + training.py, ma senza dipendenze Airflow/MLflow.
Usa BEST_PARAMS già trovati (niente RandomizedSearchCV ad ogni run, per iterare veloce).

Uso: cambia i flag USE_STEP_1/2/3 per testare le feature una alla volta.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

# ============================================
# CONFIG - modifica il path e i flag qui
# ============================================
INPUT_PATH = "C:\\airflow-project\\data\\raw\\dataset.csv"  # aggiusta se serve

USE_STEP_1_INTERACTIONS = False   # eqpdays/months/hnd_price interactions
USE_STEP_2_TRENDS = False         # avg3 vs avg6 trend ratios
USE_STEP_3_MISSING_FLAGS = False  # flag was_missing per colonne droppate

BEST_PARAMS = {
    'subsample': 0.9,
    'n_estimators': 400,
    'min_child_weight': 15, #10 
    'max_depth': 5, #4
    'learning_rate': 0.03,
    'colsample_bytree': 0.6,
}

NUM_COLS = [
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
CAT_COLS = [
    'new_cell', 'crclscod', 'asl_flag', 'prizm_social_one', 'area',
    'dualband', 'refurb_new', 'hnd_webcap', 'ownrent', 'dwlltype',
    'marital', 'infobase', 'ethnic', 'kid0_2', 'kid3_5', 'kid6_10',
    'kid11_15', 'kid16_17', 'creditcd'
]
SKEWED_VARS = ['totrev', 'mou_Mean']
HIGH_CARD_COLS = ['crclscod', 'area', 'ethnic']
MIN_OBS = 30
MISSING_THRESHOLD = 35
FLAG_MISSING_COLS = ['numbcars', 'dwllsize', 'HHstatin']


def load_data(input_path):
    df = pd.read_csv(input_path, sep=';', decimal=',', low_memory=False)

    # Step 3: crea flag di missingness PRIMA di droppare le colonne
    if USE_STEP_3_MISSING_FLAGS:
        for col in FLAG_MISSING_COLS:
            if col in df.columns:
                df[f'{col}_was_missing'] = df[col].isna().astype('int8')

    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    # non droppare i flag appena creati
    cols_to_drop = [c for c in cols_to_drop if not c.endswith('_was_missing')]
    print("Drop (missing > 35%):", cols_to_drop)
    df = df.drop(columns=cols_to_drop)
    return df


def prepare_data(df):
    X = df.drop(columns=['churn', 'Customer_ID'])
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 1. Log transform skewed vars
    for col in SKEWED_VARS:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        X_train[f'{col}_log'] = np.log1p(X_train[col].clip(lower=0))
        X_test[f'{col}_log'] = np.log1p(X_test[col].clip(lower=0))

    # Step 1: interazioni dispositivo/contratto
    if USE_STEP_1_INTERACTIONS:
        for X in (X_train, X_test):
            X['eqpdays_per_month'] = X['eqpdays'] / X['months'].replace(0, 1)
            X['price_per_month'] = X['hnd_price'] / X['months'].replace(0, 1)
            X['eqpdays_x_price'] = X['eqpdays'] * X['hnd_price']

    # Step 2: trend recente (accelerazione uso/spesa)
    if USE_STEP_2_TRENDS:
        for X in (X_train, X_test):
            X['mou_trend_3v6'] = X['avg3mou'] / X['avg6mou'].replace(0, np.nan)
            X['rev_trend_3v6'] = X['avg3rev'] / X['avg6rev'].replace(0, np.nan)

    # 2. Imputazione (fit su train)
    num_cols_final = X_train.select_dtypes(include='number').columns
    cat_cols_final = X_train.select_dtypes(include='object').columns
    medians = X_train[num_cols_final].median()
    X_train[num_cols_final] = X_train[num_cols_final].fillna(medians)
    X_test[num_cols_final] = X_test[num_cols_final].fillna(medians)
    X_train[cat_cols_final] = X_train[cat_cols_final].fillna('Missing')
    X_test[cat_cols_final] = X_test[cat_cols_final].fillna('Missing')

    # 3. Target encoding alta cardinalità (fit su train)
    high_card_cols = [c for c in HIGH_CARD_COLS if c in cat_cols_final]
    for col in high_card_cols:
        counts = X_train[col].value_counts()
        rare = counts[counts < MIN_OBS].index
        X_train[col] = X_train[col].where(~X_train[col].isin(rare), 'Other')
        X_test[col] = X_test[col].where(~X_test[col].isin(rare), 'Other')

    global_mean = y_train.mean()
    for col in high_card_cols:
        target_mean = y_train.groupby(X_train[col]).mean()
        X_train[f'{col}_target_enc'] = X_train[col].map(target_mean)
        X_test[f'{col}_target_enc'] = X_test[col].map(target_mean).fillna(global_mean)

    # 4. One-hot bassa cardinalità
    low_card_cols = [c for c in cat_cols_final if c not in high_card_cols]
    X_train = pd.get_dummies(X_train, columns=low_card_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=low_card_cols, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.drop(columns=high_card_cols)
    X_test = X_test.drop(columns=high_card_cols)

    return X_train, X_test, y_train, y_test


def train_and_evaluate(X_train, X_test, y_train, y_test):
    #xgb = XGBClassifier(**BEST_PARAMS, eval_metric='auc', random_state=42, n_jobs=-1)
    #xgb.fit(X_train, y_train)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    xgb = XGBClassifier(
        **BEST_PARAMS,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )
    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    print(f"\nNumero di alberi effettivi usati: {xgb.best_iteration}")

    proba = xgb.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("\n--- Risultati ---")
    print(f"Feature totali: {X_train.shape[1]}")
    print(f"AUC:       {roc_auc_score(y_test, proba):.4f}")
    print(f"Recall:    {recall_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}")
    print(f"F1:        {f1_score(y_test, pred):.4f}")
    

    importances = pd.Series(xgb.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    print("\nTop 15 feature importance:")
    print(importances.head(15))
    print("AUC train:", roc_auc_score(y_train, xgb.predict_proba(X_train)[:,1]))
    print("AUC test:", roc_auc_score(y_test, xgb.predict_proba(X_test)[:,1]))
    return xgb


if __name__ == '__main__':
    print(f"Config: STEP1={USE_STEP_1_INTERACTIONS}, STEP2={USE_STEP_2_TRENDS}, STEP3={USE_STEP_3_MISSING_FLAGS}")
    df = load_data(INPUT_PATH)
    X_train, X_test, y_train, y_test = prepare_data(df)
    train_and_evaluate(X_train, X_test, y_train, y_test)
