"""
Modulo condiviso per storicizzare gli eventi della pipeline su Postgres
(database process_tracking, separato da quello di Airflow e MLflow).
"""
import psycopg2
from datetime import datetime, timezone
import json

DB_CONN = "postgresql://airflow:airflow@postgres/process_tracking"


def log_event(dag_run_id, task_name, status, details=None, started_at=None, error_message=None):
    conn = psycopg2.connect(DB_CONN)
    try:
        cur = conn.cursor()
        details_to_save = dict(details) if details else {}
        if error_message:
            details_to_save['error'] = error_message
        cur.execute(
            """INSERT INTO pipeline_runs (dag_run_id, task_name, status, started_at, finished_at, details)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (
                dag_run_id,
                task_name,
                status,
                started_at,
                datetime.now(timezone.utc) if status in ('completed', 'failed') else None,
                json.dumps(details_to_save) if details_to_save else None,
            )
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()