import os
import requests
import psycopg2
from io import BytesIO
from minio import Minio
from minio.error import S3Error
from dotenv import load_dotenv

# Chemin vers le fichier .env selon le contexte d'exécution
env_path = "/sources/.env" if os.getenv("DOCKER_ENV") else ".env"
load_dotenv(dotenv_path=env_path)

# Variables d'environnement
GITHUB_REPO = os.getenv("GITHUB_REPO")
GITHUB_BRANCH = os.getenv("GITHUB_BRANCH")
MINIO_HOST = os.getenv("MINIO_HOST_DAG")
MINIO_ACCESS_KEY = os.getenv("MINIO_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_PASS")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")
PG_HOST = os.getenv("DB_HOST_DAG")
PG_PORT = os.getenv("DB_PORT")
PG_DATABASE = os.getenv("DB_NAME")
PG_USER = os.getenv("DB_USER")
PG_PASSWORD = os.getenv("DB_PASS")

# Client MinIO
client = Minio(
    MINIO_HOST, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False
)


def list_github_files(path):
    url = f"https://api.github.com/repos/{GITHUB_REPO}/contents/{path}?ref={GITHUB_BRANCH}"
    resp = requests.get(url)
    if resp.status_code == 200:
        return [
            f["name"]
            for f in resp.json()
            if f["name"].lower().endswith((".jpg", ".jpeg", ".png"))
        ]
    print(f"Erreur GitHub ({resp.status_code}) lors de l'accès à {url}")
    return []


def create_bucket_if_not_exists(bucket_name):
    try:
        if not client.bucket_exists(bucket_name):
            client.make_bucket(bucket_name)
            print(f"Bucket {bucket_name} créé.")
        else:
            print(f"Bucket {bucket_name} déjà présent.")
    except S3Error as err:
        print(f"Erreur création bucket : {err}")


def image_exists_in_minio(folder, image_name):
    try:
        client.stat_object(MINIO_BUCKET, f"{folder}/{image_name}")
        return True
    except S3Error as err:
        return False if err.code == "NoSuchKey" else print(err) or False


def create_table_if_not_exists(table_name):
    try:
        with psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    f"""
                    CREATE TABLE IF NOT EXISTS {table_name} (
                        id SERIAL PRIMARY KEY,
                        url_source VARCHAR(255) NOT NULL,
                        url_s3 VARCHAR(255) NOT NULL,
                        label VARCHAR(100) NOT NULL
                    );
                """
                )
                conn.commit()
        print("Table 'plants_data' prête.")
    except Exception as err:
        print(f"Erreur PostgreSQL table : {err}")


def insert_metadata_into_postgresql(url_source, url_s3, label):
    try:
        with psycopg2.connect(
            host=PG_HOST,
            port=PG_PORT,
            dbname=PG_DATABASE,
            user=PG_USER,
            password=PG_PASSWORD,
        ) as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO plants_data (url_source, url_s3, label)
                    VALUES (%s, %s, %s)
                    """,
                    (url_source, url_s3, label),
                )
                conn.commit()
        print(f"Métadonnées insérées pour : {url_source}")
    except Exception as err:
        print(f"Erreur insertion PostgreSQL : {err}")


def download_and_upload_images_from_github(path, label):
    print(f"Traitement des images depuis : {path}")
    folderMinio = os.path.basename(path)
    create_bucket_if_not_exists(MINIO_BUCKET)
    create_table_if_not_exists("plants_data")

    image_files = list_github_files(path)
    if not image_files:
        print(f"Aucune image trouvée dans {path}.")
        return

    for image in image_files:
        url = f"https://raw.githubusercontent.com/{GITHUB_REPO}/{GITHUB_BRANCH}/{path}/{image}"
        if image_exists_in_minio(folderMinio, image):
            print(f"Image déjà présente : {image}")
            continue

        resp = requests.get(url)
        if resp.status_code == 200:
            print(f"Image téléchargée : {image}")
            client.put_object(
                MINIO_BUCKET,
                f"{folderMinio}/{image}",
                BytesIO(resp.content),
                len(resp.content),
            )
            print(f"Image {image} ajoutée à Minio dans {folderMinio}.")
            url_s3 = f"http://{MINIO_HOST}/{MINIO_BUCKET}/{folderMinio}/{image}"
            insert_metadata_into_postgresql(url, url_s3, label)
        else:
            print(f"Échec téléchargement {image} ({resp.status_code})")
