"""
Test locale: encoding nativo delle categoriche in XGBoost (enable_categorical=True)
vs approccio attuale (target encoding + one-hot manuale).

XGBoost >= 1.6 gestisce nativamente le colonne 'category' di pandas, trovando
split ottimali sulle categorie durante la costruzione degli alberi, invece di
doverle trasformare a monte in numeri. Puo' catturare meglio relazioni non
lineari tra categorie che l'encoding manuale (soprattutto il one-hot) appiattisce.

Uso: python test_native_categorical.py
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score

# ============================================
# CONFIG
# ============================================
INPUT_PATH = "C:\\airflow-project\\data\\raw\\dataset.csv"  # aggiusta se serve

BEST_PARAMS = {
    'subsample': 0.9,
    'n_estimators': 400,
    'min_child_weight': 5,
    'max_depth': 7,
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


def load_data(input_path):
    df = pd.read_csv(input_path, sep=';', decimal=',', low_memory=False)
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    print("Drop (missing > 35%):", cols_to_drop)
    return df.drop(columns=cols_to_drop)


def split_and_log(df):
    X = df.drop(columns=['churn', 'Customer_ID'])
    y = df['churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    for col in SKEWED_VARS:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        X_train[f'{col}_log'] = np.log1p(X_train[col].clip(lower=0))
        X_test[f'{col}_log'] = np.log1p(X_test[col].clip(lower=0))
    return X_train, X_test, y_train, y_test


# ============================================
# APPROCCIO A (attuale): target encoding alta cardinalita + one-hot bassa cardinalita
# ============================================
def prepare_manual_encoding(X_train, X_test, y_train, y_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cols_final = X_train.select_dtypes(include='number').columns
    cat_cols_final = X_train.select_dtypes(include='object').columns
    medians = X_train[num_cols_final].median()
    X_train[num_cols_final] = X_train[num_cols_final].fillna(medians)
    X_test[num_cols_final] = X_test[num_cols_final].fillna(medians)
    X_train[cat_cols_final] = X_train[cat_cols_final].fillna('Missing')
    X_test[cat_cols_final] = X_test[cat_cols_final].fillna('Missing')

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

    low_card_cols = [c for c in cat_cols_final if c not in high_card_cols]
    X_train = pd.get_dummies(X_train, columns=low_card_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=low_card_cols, drop_first=True)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.drop(columns=high_card_cols)
    X_test = X_test.drop(columns=high_card_cols)

    return X_train, X_test


# ============================================
# APPROCCIO B (nuovo): encoding nativo XGBoost (enable_categorical=True)
# Le colonne categoriche restano come dtype 'category', niente one-hot/target enc.
# Solo imputazione numerica; per le categoriche i NaN sono gestiti nativamente
# da XGBoost come categoria mancante (nessun bisogno di riempirle con 'Missing').
# ============================================
def prepare_native_categorical(X_train, X_test):
    X_train = X_train.copy()
    X_test = X_test.copy()

    num_cols_final = X_train.select_dtypes(include='number').columns
    cat_cols_final = X_train.select_dtypes(include='object').columns

    medians = X_train[num_cols_final].median()
    X_train[num_cols_final] = X_train[num_cols_final].fillna(medians)
    X_test[num_cols_final] = X_test[num_cols_final].fillna(medians)

    # Le categoriche restano dtype 'category'. Le categorie di test devono
    # essere un sottoinsieme di quelle di train (fit-only-on-train), altrimenti
    # XGBoost non le riconosce correttamente in inference.
    for col in cat_cols_final:
        X_train[col] = X_train[col].astype('category')
        train_categories = X_train[col].cat.categories
        X_test[col] = pd.Categorical(X_test[col], categories=train_categories)

    return X_train, X_test


def evaluate(model, X, y, name):
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= 0.5).astype(int)
    print(f"\n--- {name} ---")
    print(f"Feature totali: {X.shape[1]}")
    print(f"AUC:       {roc_auc_score(y, proba):.4f}")
    print(f"Recall:    {recall_score(y, pred):.4f}")
    print(f"Precision: {precision_score(y, pred):.4f}")
    print(f"F1:        {f1_score(y, pred):.4f}")
    return roc_auc_score(y, proba)


if __name__ == '__main__':
    df = load_data(INPUT_PATH)
    X_train, X_test, y_train, y_test = split_and_log(df)

    # --- Approccio A: encoding manuale attuale ---
    Xa_train, Xa_test = prepare_manual_encoding(X_train, X_test, y_train, y_test)
    model_a = XGBClassifier(**BEST_PARAMS, eval_metric='auc', random_state=42, n_jobs=-1)
    model_a.fit(Xa_train, y_train)
    auc_a = evaluate(model_a, Xa_test, y_test, "Approccio A - encoding manuale (attuale)")

    # --- Approccio B: encoding nativo XGBoost ---
    Xb_train, Xb_test = prepare_native_categorical(X_train, X_test)
    model_b = XGBClassifier(
        **BEST_PARAMS, eval_metric='auc', random_state=42, n_jobs=-1,
        enable_categorical=True, tree_method='hist'  # richiesto per il supporto categoriche
    )
    model_b.fit(Xb_train, y_train)
    auc_b = evaluate(model_b, Xb_test, y_test, "Approccio B - encoding nativo XGBoost")

    # --- Confronto ---
    print("\n=== CONFRONTO ===")
    print(f"AUC encoding manuale (attuale): {auc_a:.4f}")
    print(f"AUC encoding nativo XGBoost:     {auc_b:.4f}")
    print(f"Differenza:                      {auc_b - auc_a:+.4f}")

    if auc_b - auc_a > 0.005:
        print("\n-> L'encoding nativo migliora significativamente, valuta di adottarlo in produzione")
    elif auc_b - auc_a < -0.005:
        print("\n-> L'encoding nativo peggiora, meglio mantenere l'approccio manuale attuale")
    else:
        print("\n-> Differenza marginale, ma l'encoding nativo semplifica il codice (niente target/one-hot)")