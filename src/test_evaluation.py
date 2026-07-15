"""
Test automatico per evaluation.py
Esegue: pytest test_evaluation.py -v
"""
import json
import pickle
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from evaluation import evaluate_model


class FakeModel:
    """Modello finto: niente XGBoost/training reale, restituisce
    probabilita' fisse note in anticipo, cosi' le metriche attese
    si possono calcolare a mano nel test."""

    def predict_proba(self, X):
        n = len(X)
        # meta' campioni con proba alta (>=0.5), meta' bassa: cosi' pred
        # e recall/precision non sono banalmente 0 o 1
        proba_churn = np.where(np.arange(n) % 2 == 0, 0.8, 0.2)
        return np.column_stack([1 - proba_churn, proba_churn])


@pytest.fixture
def fake_processed_dir(tmp_path):
    """Crea X_test/y_test finti coerenti con FakeModel:
    indici pari -> churn=1 (proba alta), dispari -> churn=0 (proba bassa),
    cosi' il modello finto azzecca tutto e le metriche sono note (=1.0)."""
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    n = 10
    X_test = pd.DataFrame({'feature_1': range(n), 'feature_2': range(n, 2 * n)})
    y_test = pd.DataFrame({'churn': [1 if i % 2 == 0 else 0 for i in range(n)]})

    X_test.to_parquet(processed_dir / 'X_test.parquet')
    y_test.to_parquet(processed_dir / 'y_test.parquet')
    return processed_dir


@pytest.fixture
def fake_model_dir(tmp_path):
    """Salva il FakeModel come pickle, stesso formato di training.py
    (dict con chiave 'model' e 'decision_threshold')."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    artifact = {
        'model': FakeModel(),
        'decision_threshold': 0.5,
    }
    with open(model_dir / 'churn_xgb_model.pkl', 'wb') as f:
        pickle.dump(artifact, f)
    return model_dir


@patch('evaluation.log_event')
@patch('evaluation.mlflow')
def test_evaluate_model_metrics_are_perfect_with_fake_model(
    mock_mlflow, mock_log, fake_processed_dir, fake_model_dir, tmp_path
):
    """Con FakeModel + dataset costruito ad hoc, il modello azzecca
    tutte le predizioni: AUC, recall, precision, f1 devono essere 1.0."""
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    metrics_dir = tmp_path / "metrics"

    metrics = evaluate_model(
        mlflow_run_id='fake_run_id',
        processed_dir=fake_processed_dir,
        model_dir=fake_model_dir,
        metrics_dir=metrics_dir,
    )

    assert metrics['auc'] == 1.0
    assert metrics['recall'] == 1.0
    assert metrics['precision'] == 1.0
    assert metrics['f1'] == 1.0


@patch('evaluation.log_event')
@patch('evaluation.mlflow')
def test_evaluate_model_metrics_in_valid_range(
    mock_mlflow, mock_log, fake_processed_dir, fake_model_dir, tmp_path
):
    """Controllo generico anti-regressione: qualunque sia il modello,
    le metriche devono stare nel range [0, 1] e la confusion matrix
    deve sommare al numero di campioni di test."""
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    metrics_dir = tmp_path / "metrics"

    metrics = evaluate_model(
        mlflow_run_id='fake_run_id',
        processed_dir=fake_processed_dir,
        model_dir=fake_model_dir,
        metrics_dir=metrics_dir,
    )

    for key in ['auc', 'recall', 'precision', 'f1']:
        assert 0.0 <= metrics[key] <= 1.0

    cm = metrics['confusion_matrix']
    total = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
    assert total == metrics['n_test_samples']


@patch('evaluation.log_event')
@patch('evaluation.mlflow')
def test_evaluate_model_writes_json_file(
    mock_mlflow, mock_log, fake_processed_dir, fake_model_dir, tmp_path
):
    """Verifica che il file di log delle metriche venga effettivamente
    scritto su disco e sia JSON valido con le chiavi attese."""
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    metrics_dir = tmp_path / "metrics"

    evaluate_model(
        mlflow_run_id='fake_run_id',
        processed_dir=fake_processed_dir,
        model_dir=fake_model_dir,
        metrics_dir=metrics_dir,
    )

    json_files = list(metrics_dir.glob('metrics_*.json'))
    assert len(json_files) == 1

    with open(json_files[0]) as f:
        saved_metrics = json.load(f)

    for key in ['timestamp', 'auc', 'recall', 'precision', 'f1',
                'threshold', 'confusion_matrix', 'n_test_samples']:
        assert key in saved_metrics


@patch('evaluation.log_event')
@patch('evaluation.mlflow')
def test_evaluate_model_logs_to_mlflow_same_run(
    mock_mlflow, mock_log, fake_processed_dir, fake_model_dir, tmp_path
):
    """Verifica che venga riaperto lo STESSO run MLflow del training
    (run_id passato esplicitamente), non uno nuovo."""
    mock_mlflow.start_run.return_value.__enter__ = MagicMock()
    mock_mlflow.start_run.return_value.__exit__ = MagicMock(return_value=False)

    metrics_dir = tmp_path / "metrics"

    evaluate_model(
        mlflow_run_id='fake_run_id_123',
        processed_dir=fake_processed_dir,
        model_dir=fake_model_dir,
        metrics_dir=metrics_dir,
    )

    mock_mlflow.start_run.assert_called_once_with(run_id='fake_run_id_123')
    mock_mlflow.start_run.return_value.__enter__.assert_called_once()