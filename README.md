# Churn Prediction Pipeline — SDG Technical Test

Pipeline end-to-end per la previsione del churn clienti in ambito Telco, orchestrata con Apache Airflow su Docker. Include preparazione dati, training del modello, valutazione automatica e tracking degli esperimenti con MLflow.

Repository: [github.com/carloaleotti2/churn_prediction_airflow](https://github.com/carloaleotti2/churn_prediction_airflow)

## Indice

- [Architettura](#architettura)
- [Stack tecnologico](#stack-tecnologico)
- [Struttura del progetto](#struttura-del-progetto)
- [Setup e avvio](#setup-e-avvio)
- [Esecuzione della pipeline](#esecuzione-della-pipeline)
- [Test automatici](#test-automatici)
- [Configurazione](#configurazione)
- [Risultati principali](#risultati-principali)
- [Limiti noti e possibili estensioni](#limiti-noti-e-possibili-estensioni)

## Architettura

La pipeline è un DAG Airflow (`churn_pipeline`) composto da 3 task sequenziali:

```
data_preparation_task → training_task → evaluation_task
```

1. **data_preparation** — legge il dataset raw, valida lo schema di input, esegue split train/test, imputazione, target encoding e one-hot encoding. Salva i dati processati in formato parquet e l'artifact di preprocessing in pickle.
2. **training** — carica i dati processati, allena un modello XGBoost con iperparametri ottimizzati (via RandomizedSearchCV in fase di sviluppo, congelati per i run di produzione), traccia parametri e metriche su MLflow, salva il modello finale in pickle.
3. **evaluation** — carica il modello allenato, lo valuta sul hold-out set, calcola le metriche (AUC, recall, precision, F1), le logga sullo stesso run MLflow del training e salva un report JSON.

Ogni task registra inizio/fine/eventuali errori su un database Postgres dedicato (`process_tracking`), separato dal database interno di Airflow. In caso di fallimento, ogni task effettua fino a 2 retry automatici prima di segnalare l'errore definitivo.

## Stack tecnologico

- **Orchestrazione**: Apache Airflow 3.3.0 (LocalExecutor)
- **Containerizzazione**: Docker + Docker Compose
- **Modeling**: XGBoost, scikit-learn (baseline Logistic Regression)
- **Experiment tracking**: MLflow
- **Storage processo**: PostgreSQL
- **Testing**: pytest

## Struttura del progetto

```
churn_prediction_airflow/
├── config/
│   └── airflow.cfg                       # configurazione Airflow
├── dags/
│   └── churn_dag.py                      # definizione del DAG
├── src/
│   ├── data_preparation.py               # Task 1
│   ├── training.py                       # Task 2
│   ├── evaluation.py                     # Task 3
│   ├── process_logger.py                 # logging eventi su Postgres
│   ├── test_data_preparation.py          # test automatici Task 1
│   └── test_evaluation.py                # test automatici Task 3
├── notebooks/
|   ├── data_descriptions.csv
│   ├── EDA_FirstPart.ipynb               # analisi esplorativa
│   ├── EDA_SecondPart.ipynb              # analisi esplorativa (continuazione)
│   ├── Modelling.py                      # script di sviluppo modello (precursore di src/training.py)
│   ├── Test_native_categorical.py        # confronto: encoding nativo XGBoost vs manuale
│   ├── Test_hyperparameter_tuning.py     # confronto: tuning esteso vs iperparametri in produzione
│   └── Test_feature_engineering.py       # confronto: feature aggiuntive (interazioni, trend, missingness)
├── docker-compose.yaml
├── Dockerfile
├── requirements.txt
└── .gitignore
```

**Nota:** le cartelle `data/`, `models/`, `logs/`, `mlflow_artifacts/` vengono create automaticamente all'avvio dei container (montate come volumi in `docker-compose.yaml`) e non sono tracciate su Git. Il dataset raw (`dataset.csv`) va posizionato manualmente in `data/raw/` prima del primo run — vedi [Setup e avvio](#setup-e-avvio).

I file dentro `notebooks/` documentano il percorso sperimentale seguito per arrivare alla configurazione finale in `src/` (confronto encoding, tuning, feature engineering): non fanno parte della pipeline di produzione, ma motivano le scelte fatte.

## Setup e avvio

**Prerequisiti**: Docker Desktop (con WSL2 su Windows), almeno 4GB di RAM disponibili per Docker.

1. Clona il repository:
   ```bash
   git clone https://github.com/carloaleotti2/churn_prediction_airflow.git
   cd churn_prediction_airflow
   ```
2. Crea un file `.env` nella root del progetto con:
   ```
   AIRFLOW_UID=50000
   FERNET_KEY=<una fernet key valida, generabile con python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())">
   ```
3. Crea la cartella `data/raw/` e posiziona al suo interno il dataset (`dataset.csv`):
   ```bash
   mkdir -p data/raw
   # copia qui il tuo dataset.csv
   ```
4. Build e inizializzazione:
   ```bash
   docker-compose build
   docker-compose up airflow-init
   ```
5. Avvio dei servizi:
   ```bash
   docker-compose up -d
   ```
6. Attendi che tutti i container risultino `healthy`:
   ```bash
   docker-compose ps
   ```
7. Accedi alla UI Airflow su [http://localhost:8080](http://localhost:8080) (utente/password: `airflow`/`airflow`).
8. MLflow UI disponibile su [http://localhost:5000](http://localhost:5000).

## Esecuzione della pipeline

1. Nella UI Airflow, cerca il DAG `churn_pipeline`.
2. Attivalo (toggle in alto a sinistra) e lancia un run manuale (▶).
3. Segui l'esecuzione dei 3 task dalla vista Grid/Graph.
4. Al termine:
   - Il modello allenato è in `models/churn_xgb_model.pkl`
   - Le metriche di valutazione sono in `logs/metrics/metrics_<timestamp>.json`
   - Parametri e metriche sono consultabili su MLflow (esperimento `churn_prediction_v4`)

## Test automatici

I test coprono `data_preparation.py` ed `evaluation.py` con dataset finti generati ad hoc (non richiedono Airflow o Postgres in esecuzione).

```bash
cd src/
pip install pytest pyarrow psycopg2-binary   # se non già presenti nell'ambiente
pytest test_data_preparation.py -v
pytest test_evaluation.py -v
```

Cosa verificano:
- Assenza di valori mancanti residui dopo l'imputazione
- Esclusione corretta di target e identificativo cliente dalle feature
- Corretto raggruppamento delle categorie rare in `'Other'`
- Validazione dello schema di input (colonne obbligatorie, valori validi di `churn`)
- Correttezza del calcolo delle metriche di valutazione e del formato di output
- Corretto riutilizzo del run MLflow tra training ed evaluation

## Configurazione

Le seguenti variabili sono configurabili via ambiente (default già impostati per l'esecuzione in Docker):

| Variabile | Descrizione | Default |
|---|---|---|
| `RAW_DATA_PATH` | Path del dataset di input | `/opt/airflow/data/raw/dataset.csv` |
| `PROCESSED_DATA_DIR` | Directory dati processati | `/opt/airflow/data/processed` |
| `MODEL_DIR` | Directory modello salvato | `/opt/airflow/models` |
| `METRICS_DIR` | Directory report metriche | `/opt/airflow/logs/metrics` |
| `MLFLOW_TRACKING_URI` | Endpoint MLflow | `http://mlflow:5000` |

## Risultati principali

- **Modello**: XGBoost (AUC ≈ 0.69 su hold-out), scelto rispetto a Logistic Regression per la capacità di gestire multicollinearità e relazioni non lineari, entrambe evidenziate in fase di EDA.
- **Fattori di rischio churn principali** (analisi SHAP + EDA):
  1. Anzianità del dispositivo (`eqpdays`) — fattore più predittivo
  2. Durata del contratto (`months`) — clienti nei primi mesi più a rischio
  3. Variazioni recenti di utilizzo (`change_mou`, `change_rev`) — più informative del livello assoluto di utilizzo
  4. Prezzo del dispositivo (`hnd_price`) — dispositivi economici associati a maggior rischio
- **Sperimentazioni condotte** (dettagli in `notebooks/Test_*.py`): encoding nativo XGBoost, tuning iperparametri esteso, feature engineering aggiuntivo (interazioni, trend, flag di missingness) — nessuna ha portato miglioramenti significativi rispetto alla configurazione in produzione, indicando che il limite attuale è nella disponibilità di segnali nel dataset più che nella modellazione.
- **Raccomandazioni business**: tradotte in azioni concrete di retention (campagne di upgrade dispositivo, onboarding rinforzato nei primi mesi, monitoraggio proattivo delle variazioni di utilizzo) — dettaglio completo nella presentazione.

## Limiti noti e possibili estensioni

- Test automatici assenti per `training.py`
- Nessun versionamento del modello (ogni retraining sovrascrive `churn_xgb_model.pkl`)
- Nessuna notifica automatica in caso di fallimento definitivo del DAG (dopo i retry)
- Monitoring (Prometheus/Grafana) non implementato per vincoli di risorse hardware locali
- AUC coerente con la difficoltà intrinseca del problema: mancano nel dataset segnali rilevanti come offerte concorrenti, qualità di rete e interazioni con il servizio clienti
