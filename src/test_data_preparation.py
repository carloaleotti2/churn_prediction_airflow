"""
Test automatico per data_preparation.py
Esegue: pytest test_data_preparation.py -v
"""
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# aggiunge la cartella con i moduli sorgente al path (adatta se serve)
sys.path.insert(0, str(Path(__file__).parent))

from data_preparation import prepare_data, validate_input_schema, SchemaValidationError


@pytest.fixture
def fake_dataset(tmp_path):
    """Crea un CSV finto con lo stesso formato del dataset reale
    (separatore ';', decimale ',', colonne minime necessarie)."""
    df = pd.DataFrame({
        'Customer_ID': range(1, 21),
        'churn': [0, 1] * 10,
        'eqpdays': [100, None, 300, 250, 400, 90, 120, None, 500, 60,
                    200, 310, None, 80, 150, 220, 330, 40, 190, 275],
        'hnd_price': [50, 100, None, 200, 80, 120, 60, 90, 300, 40,
                      70, 150, 110, 95, 130, 210, 60, 85, 175, 205],
        'crclscod': ['A', 'B', 'A', 'C', 'B', 'A', None, 'C', 'B', 'A',
                     'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C', 'A', 'B'],
        'ethnic': ['X', 'Y', None, 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z',
                   'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X', 'Y', 'Z', 'X'],
        # richieste da SKEWED_VARS in data_preparation.py, non opzionali
        'totrev': [500, 300, 700, 200, 900, 150, 400, 600, 250, 800,
                   350, 450, 550, 650, 750, 850, 950, 100, 300, 500],
        'mou_Mean': [50, 30, 70, 20, 90, 15, 40, 60, 25, 80,
                     35, 45, 55, 65, 75, 85, 95, 10, 30, 50],
    })
    csv_path = tmp_path / "fake_dataset.csv"
    df.to_csv(csv_path, sep=';', decimal=',', index=False)
    return csv_path


@patch('data_preparation.log_event')  # non tenta connessione reale a Postgres
def test_prepare_data_no_missing_values(mock_log, fake_dataset, tmp_path):
    output_dir = tmp_path / "processed"
    output_dir.mkdir()

    prepare_data(input_path=str(fake_dataset), output_dir=output_dir)

    X_train = pd.read_parquet(output_dir / 'X_train.parquet')
    X_test = pd.read_parquet(output_dir / 'X_test.parquet')

    assert X_train.isna().sum().sum() == 0
    assert X_test.isna().sum().sum() == 0


@patch('data_preparation.log_event')
def test_prepare_data_target_not_in_features(mock_log, fake_dataset, tmp_path):
    output_dir = tmp_path / "processed"
    output_dir.mkdir()

    prepare_data(input_path=str(fake_dataset), output_dir=output_dir)

    X_train = pd.read_parquet(output_dir / 'X_train.parquet')
    assert 'churn' not in X_train.columns
    assert 'Customer_ID' not in X_train.columns


@patch('data_preparation.log_event')
def test_prepare_data_output_files_exist(mock_log, fake_dataset, tmp_path):
    output_dir = tmp_path / "processed"
    output_dir.mkdir()

    prepare_data(input_path=str(fake_dataset), output_dir=output_dir)

    for fname in ['X_train.parquet', 'X_test.parquet', 'y_train.parquet',
                  'y_test.parquet', 'preprocessing_artifact.pkl']:
        assert (output_dir / fname).exists()


@patch('data_preparation.log_event')
def test_rare_categories_grouped_as_other(mock_log, fake_dataset, tmp_path):
    """crclscod ha categorie con <30 osservazioni (MIN_OBS):
    devono finire in 'Other' prima del target encoding."""
    output_dir = tmp_path / "processed"
    output_dir.mkdir()

    prepare_data(input_path=str(fake_dataset), output_dir=output_dir)

    import pickle
    with open(output_dir / 'preprocessing_artifact.pkl', 'rb') as f:
        artifact = pickle.load(f)

    assert 'crclscod' in artifact['rare_categories']
    assert len(artifact['rare_categories']['crclscod']) > 0


def test_validate_input_schema_raises_on_missing_required_column():
    """Senza 'churn', la pipeline non puo' funzionare: deve fallire subito
    con un errore chiaro, non con un KeyError generico piu' avanti."""
    df = pd.DataFrame({
        'Customer_ID': [1, 2, 3],
        'totrev': [100, 200, 300],
        'mou_Mean': [10, 20, 30],
    })
    with pytest.raises(SchemaValidationError, match="churn"):
        validate_input_schema(df)


def test_validate_input_schema_raises_on_invalid_churn_values():
    """churn deve essere 0/1: valori diversi (es. 'yes'/'no') indicano
    un formato di input sbagliato."""
    df = pd.DataFrame({
        'Customer_ID': [1, 2, 3],
        'churn': ['yes', 'no', 'yes'],
        'totrev': [100, 200, 300],
        'mou_Mean': [10, 20, 30],
    })
    with pytest.raises(SchemaValidationError, match="valori non validi"):
        validate_input_schema(df)


def test_validate_input_schema_passes_on_valid_minimal_input():
    """Con le sole colonne obbligatorie e churn 0/1, non deve sollevare
    alcun errore, anche se il dataset ha solo poche colonne."""
    df = pd.DataFrame({
        'Customer_ID': [1, 2, 3],
        'churn': [0, 1, 0],
        'totrev': [100, 200, 300],
        'mou_Mean': [10, 20, 30],
    })
    validate_input_schema(df)  # non deve sollevare eccezioni