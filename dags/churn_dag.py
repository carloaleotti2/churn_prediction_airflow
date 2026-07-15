"""
DAG - Churn Pipeline
Orchestrazione dei 3 task: data_preparation -> training -> evaluation.
La logica vera sta in src/, qui c'è solo la definizione del flusso.
"""
import sys
from datetime import datetime, timedelta

from airflow.sdk import dag, task, get_current_context

sys.path.insert(0, '/opt/airflow/src')

default_args = {
    'retries': 2,
    'retry_delay': timedelta(minutes=2),
}


@dag(
    dag_id='churn_pipeline',
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['churn'],
    default_args=default_args,
)
def churn_pipeline():

    @task
    def data_preparation_task():
        from data_preparation import prepare_data
        from process_logger import log_event
        context = get_current_context()
        dag_run_id = context['dag_run'].run_id
        try:
            return prepare_data(dag_run_id=dag_run_id)
        except Exception as e:
            log_event(dag_run_id, 'data_preparation', 'failed', error_message=str(e))
            raise

    @task
    def training_task(processed_dir):
        from training import train_model
        from process_logger import log_event
        context = get_current_context()
        dag_run_id = context['dag_run'].run_id
        try:
            return train_model(dag_run_id=dag_run_id)
        except Exception as e:
            log_event(dag_run_id, 'training', 'failed', error_message=str(e))
            raise

    @task
    def evaluation_task(training_results: dict):
        from evaluation import evaluate_model
        from process_logger import log_event
        context = get_current_context()
        dag_run_id = context['dag_run'].run_id
        try:
            return evaluate_model(mlflow_run_id=training_results['mlflow_run_id'], dag_run_id=dag_run_id)
        except Exception as e:
            log_event(dag_run_id, 'evaluation', 'failed', error_message=str(e))
            raise

    processed_dir = data_preparation_task()
    training_results = training_task(processed_dir)
    evaluation_task(training_results)


churn_pipeline()