import os
import pytest
import psycopg2
from minio import Minio
from dotenv import load_dotenv
import fonctions  # <- nouvelle importation directe

# Charger les variables d'environnement
load_dotenv(dotenv_path=".env")

# Variables d’environnement
MINIO_HOST = os.getenv("MINIO_HOST")
MINIO_ACCESS_KEY = os.getenv("MINIO_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_PASS")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

DB_CONN = {
    "dbname": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS"),
    "host": os.getenv("DB_HOST"),
    "port": int(os.getenv("DB_PORT")),
}


# === Fixtures ===


@pytest.fixture
def postgres_connection():
    conn = psycopg2.connect(**DB_CONN)
    yield conn
    conn.close()


@pytest.fixture
def minio_client():
    return Minio(
        endpoint=MINIO_HOST,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


# === Tests ===


def test_postgres_connection(postgres_connection):
    with postgres_connection.cursor() as cursor:
        cursor.execute("SELECT 1;")
        result = cursor.fetchone()
        assert result and result[0] == 1


def test_minio_connection(minio_client):
    try:
        buckets = minio_client.list_buckets()
        assert buckets is not None
    except Exception as e:
        pytest.fail(f"Connexion MinIO échouée : {e}")


def test_list_github_files(mocker):
    mocker.patch("requests.get").return_value.status_code = 200
    mocker.patch("requests.get").return_value.json.return_value = [
        {"name": "image1.jpg"},
        {"name": "image2.png"},
        {"name": "README.md"},
    ]
    files = fonctions.list_github_files("dandelion")
    assert files == ["image1.jpg", "image2.png"]


def test_create_bucket_if_not_exists(mocker):
    mock_client = mocker.patch("fonctions.client")
    mock_client.bucket_exists.return_value = False
    fonctions.create_bucket_if_not_exists("test-bucket")
    mock_client.make_bucket.assert_called_once_with("test-bucket")


def test_image_exists_in_minio(mocker):
    mock_client = mocker.patch("fonctions.client")
    mock_client.stat_object.return_value = True
    assert fonctions.image_exists_in_minio("folder", "image.jpg") is True


def test_create_table_if_not_exists(mocker):
    mock_connect = mocker.patch("psycopg2.connect")
    cursor = mock_connect.return_value.cursor.return_value
    fonctions.create_table_if_not_exists("test_table")

    expected_query = """
        CREATE TABLE IF NOT EXISTS test_table (
            id SERIAL PRIMARY KEY,
            url_source VARCHAR(255) NOT NULL,
            url_s3 VARCHAR(255) NOT NULL,
            label VARCHAR(100) NOT NULL
        );
    """.strip()

    actual_query = cursor.execute.call_args[0][0].strip()
    assert actual_query == expected_query
    mock_connect.return_value.commit.assert_called_once()


def test_insert_metadata_into_postgresql(mocker):
    mock_connect = mocker.patch("psycopg2.connect")
    cursor = mock_connect.return_value.cursor.return_value

    fonctions.insert_metadata_into_postgresql(
        "http://source.com/img.jpg", "http://minio.com/img.jpg", "dandelion"
    )

    expected_query = """
        INSERT INTO plants_data (url_source, url_s3, label)
        VALUES (%s, %s, %s)
    """.strip()

    actual_query = cursor.execute.call_args[0][0].strip()
    params = cursor.execute.call_args[0][1]

    assert actual_query == expected_query
    assert params == (
        "http://source.com/img.jpg",
        "http://minio.com/img.jpg",
        "dandelion",
    )

    mock_connect.return_value.commit.assert_called_once()
