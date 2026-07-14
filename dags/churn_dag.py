"""
DAG - Churn Pipeline
Orchestrazione dei 3 task: data_preparation -> training -> evaluation.
La logica vera sta in src/, qui c'è solo la definizione del flusso.
"""
import sys
from datetime import datetime

from airflow.sdk import dag, task

# src/ è montata come volume separato (/opt/airflow/src), va aggiunta al path
# per poter importare gli script come moduli
sys.path.insert(0, '/opt/airflow/src')


@dag(
    dag_id='churn_pipeline',
    schedule=None,          # esecuzione manuale/trigger, non periodica
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['churn'],
)
def churn_pipeline():

    @task
    def data_preparation_task():
        from data_preparation import prepare_data
        return prepare_data()

    @task
    def training_task(processed_dir):
        from training import train_model
        return train_model()

    @task
    def evaluation_task(training_results: dict):
        from evaluation import evaluate_model
        return evaluate_model(mlflow_run_id=training_results['mlflow_run_id'])

    processed_dir = data_preparation_task()
    training_results = training_task(processed_dir)
    evaluation_task(training_results)


churn_pipeline()