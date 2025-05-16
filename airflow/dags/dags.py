import os
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from fonctions import download_and_upload_images_from_github
from randomforest import train_random_forest_model
from dotenv import load_dotenv

# Charger les variables d'environnement depuis .env
load_dotenv(dotenv_path="/sources/.env")
GITHUB_PATH_DANDELION = os.getenv("GITHUB_PATH_DANDELION")
GITHUB_PATH_GRASS = os.getenv("GITHUB_PATH_GRASS")

# Configuration du DAG
default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "start_date": datetime(2025, 4, 10),
    "retries": 1,
}

dag = DAG(
    dag_id="pipeline_randomforest",
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
)

# Tâche 1 : Télécharger et envoyer les images "dandelion"
download_dandelion_task = PythonOperator(
    task_id="download_dandelion_images",
    python_callable=download_and_upload_images_from_github,
    op_args=[GITHUB_PATH_DANDELION, "dandelion"],
    dag=dag,
)

# Tâche 2 : Télécharger et envoyer les images "grass"
download_grass_task = PythonOperator(
    task_id="download_grass_images",
    python_callable=download_and_upload_images_from_github,
    op_args=[GITHUB_PATH_GRASS, "grass"],
    dag=dag,
)

# Tâche 3 : Entraînement du modèle RandomForest
train_model_task = PythonOperator(
    task_id="train_random_forest_model",
    python_callable=train_random_forest_model,
    op_args=[["dandelion", "grass"]],
    dag=dag,
)

# Définir l'ordre d'exécution
download_dandelion_task >> download_grass_task >> train_model_task
