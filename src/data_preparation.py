"""
Task 1 - data_preparation
Legge il dataset raw, fa split, imputazione, target encoding, one-hot.
Salva X_train/X_test/y_train/y_test + artifact di preprocessing su disco condiviso.
Nessuna dipendenza da Airflow: funzione pura, testabile standalone.
"""
from process_logger import log_event
from datetime import datetime, timezone
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import os 

RAW_DATA_PATH = os.environ.get("RAW_DATA_PATH", "/opt/airflow/data/raw/dataset.csv") 
OUTPUT_DIR = Path(os.environ.get("PROCESSED_DATA_DIR", "/opt/airflow/data/processed"))
 
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
MISSING_THRESHOLD = 35  # % oltre il quale una colonna viene droppata


class SchemaValidationError(Exception):
    """Sollevato quando il dataset in input non rispetta lo schema minimo atteso."""
    pass 


def validate_input_schema(df: pd.DataFrame):
    """Verifica che le colonne critiche siano presenti prima di processare.
    Non blocca su NUM_COLS/CAT_COLS opzionali (già filtrate con 'if c in df.columns'
    piu' avanti), ma su quelle senza le quali la pipeline non puo' funzionare."""
    required_cols = {'churn', 'Customer_ID'}.union(SKEWED_VARS)
    missing = required_cols - set(df.columns)
    if missing:
        raise SchemaValidationError(
            f"Colonne obbligatorie mancanti nel dataset: {sorted(missing)}"
        )

    # churn deve essere binario (0/1)
    # solleva SchemaValidationError con messaggio esplicito
    valid_churn_values = set(df['churn'].dropna().unique()) - {0, 1}
    if valid_churn_values:
        raise SchemaValidationError(
            f"Colonna 'churn' contiene valori non validi (attesi 0/1): {valid_churn_values}"
        )


def prepare_data(input_path: str = RAW_DATA_PATH, output_dir: Path = OUTPUT_DIR, dag_run_id: str = None):
    start = datetime.now(timezone.utc)
    log_event(dag_run_id, 'data_preparation', 'started', started_at=start)
 
    # Lettura a Chunk, con downcast dei dtype, per ridurre uso di memoria

    CHUNK_SIZE = 2000
    chunks = []
    for chunk in pd.read_csv(
        input_path, sep=';', decimal=',', low_memory=False, chunksize=CHUNK_SIZE
    ):
        for col in chunk.select_dtypes(include=['float64']).columns:
            chunk[col] = chunk[col].astype('float32')
        for col in chunk.select_dtypes(include=['int64']).columns:
            chunk[col] = chunk[col].astype('int32')
        for col in chunk.select_dtypes(include=['object']).columns:
            chunk[col] = chunk[col].astype('category')
        chunks.append(chunk)

    # scelta per contenere il picco di RAM nell'ambiente Docker
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    import gc
    gc.collect()
    
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].cat.remove_unused_categories()

    validate_input_schema(df)

    # Drop colonne con troppi missing (>35%)
    missing_pct = df.isna().mean().sort_values(ascending=False) * 100
    cols_to_drop = missing_pct[missing_pct > MISSING_THRESHOLD].index.tolist()
    print("Drop (missing > 35%):", cols_to_drop)
    df = df.drop(columns=cols_to_drop)
 
    num_cols = [c for c in NUM_COLS if c in df.columns]
    cat_cols = [c for c in CAT_COLS if c in df.columns]
 
    # split X|y e Train/Test split stratificato
    X = df.drop(columns=['churn', 'Customer_ID'])
    y = df['churn']
    del df
    import gc
    gc.collect()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    del X,y
    gc.collect()

    # Log transform skewed vars
    for col in SKEWED_VARS:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')
        X_train[f'{col}_log'] = np.log1p(X_train[col].clip(lower=0))
        X_test[f'{col}_log'] = np.log1p(X_test[col].clip(lower=0))

    # Drop delle colonne originali skewed: la versione log le sostituisce.
    # Senza questo drop, originale e log finivano entrambe tra le feature
    # (correlazione ~1 tra loro, doppio conteggio nella SHAP/feature importance).
    X_train = X_train.drop(columns=SKEWED_VARS)
    X_test = X_test.drop(columns=SKEWED_VARS)
 
    # Imputazione (fit su train)
    num_cols_final = X_train.select_dtypes(include='number').columns
    cat_cols_final = X_train.select_dtypes(include=['object', 'category']).columns
    # Aggiungi 'Missing' come categoria valida prima del fillna
    for col in cat_cols_final:
        if pd.api.types.is_categorical_dtype(X_train[col]):
            X_train[col] = X_train[col].cat.add_categories(['Missing'])
        if pd.api.types.is_categorical_dtype(X_test[col]):
            X_test[col] = X_test[col].cat.add_categories(['Missing'])
    
    medians = X_train[num_cols_final].median()
    X_train[num_cols_final] = X_train[num_cols_final].fillna(medians)
    X_test[num_cols_final] = X_test[num_cols_final].fillna(medians)
    X_train[cat_cols_final] = X_train[cat_cols_final].fillna('Missing')
    X_test[cat_cols_final] = X_test[cat_cols_final].fillna('Missing')
 
    # Target encoding alta cardinalità (fit su train)
    high_card_cols = [c for c in HIGH_CARD_COLS if c in cat_cols_final]
    rare_categories = {}
    for col in high_card_cols:
        if pd.api.types.is_categorical_dtype(X_train[col]):
            X_train[col] = X_train[col].cat.add_categories(['Other'])
        if pd.api.types.is_categorical_dtype(X_test[col]):
            X_test[col] = X_test[col].cat.add_categories(['Other'])
        counts = X_train[col].value_counts()
        rare = counts[counts < MIN_OBS].index
        rare_categories[col] = rare
        X_train[col] = X_train[col].where(~X_train[col].isin(rare), 'Other')
        X_test[col] = X_test[col].where(~X_test[col].isin(rare), 'Other')
 
    global_mean = y_train.mean()
    target_maps = {}
    for col in high_card_cols:
        target_mean = y_train.groupby(X_train[col]).mean()
        target_maps[col] = target_mean
        X_train[f'{col}_target_enc'] = X_train[col].map(target_mean)
        X_test[f'{col}_target_enc'] = X_test[col].map(target_mean).fillna(global_mean)
 
    # One-hot bassa cardinalità
    low_card_cols = [c for c in cat_cols_final if c not in high_card_cols]
    X_train = pd.get_dummies(X_train, columns=low_card_cols, drop_first=True, dtype='int8')
    X_test = pd.get_dummies(X_test, columns=low_card_cols, drop_first=True, dtype='int8')
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    X_train = X_train.drop(columns=high_card_cols)
    X_test = X_test.drop(columns=high_card_cols)
 
    feature_columns = X_train.columns.tolist()
 
    # Salvataggio dati pronti per il training (parquet: preserva i dtype
    # esattamente, niente round-trip di precisione come con il CSV)
    X_train.to_parquet(output_dir / 'X_train.parquet')
    X_test.to_parquet(output_dir / 'X_test.parquet')
    y_train.to_frame('churn').to_parquet(output_dir / 'y_train.parquet')
    y_test.to_frame('churn').to_parquet(output_dir / 'y_test.parquet')
 
    # Salvataggio artifact di preprocessing (servirà anche per l'inference finale)
    preprocessing_artifact = {
        'feature_columns': feature_columns,
        'medians': medians,
        'target_maps': target_maps,
        'global_mean': global_mean,
        'rare_categories': rare_categories,
        'high_card_cols': high_card_cols,
        'low_card_cols': low_card_cols,
        'skewed_vars': SKEWED_VARS,
        'dropped_missing_cols': cols_to_drop,
    }
    with open(output_dir / 'preprocessing_artifact.pkl', 'wb') as f:
        pickle.dump(preprocessing_artifact, f)
 
    print(f"Dati salvati in {output_dir}")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    
    log_event(dag_run_id, 'data_preparation', 'completed', started_at=start,
              details={'X_train_shape': list(X_train.shape), 'X_test_shape': list(X_test.shape)})

    return str(output_dir)
 
 
if __name__ == '__main__':
    prepare_data()