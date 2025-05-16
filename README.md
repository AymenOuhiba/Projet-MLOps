# Projet MLOps : Classification d'Images (Dandelion vs Grass)

## 👥 Réalisé par
**Aymen Ouhiba** et **Nohaila Garra**

---

## 🎯 Objectifs du Projet
Ce projet met en place un pipeline MLOps complet visant à :

- Extraire, nettoyer et stocker des images depuis GitHub vers MinIO et PostgreSQL.
- Entraîner un modèle de classification (Dandelion vs Grass) à l’aide d’un Random Forest.
- Exposer les prédictions via une interface web Streamlit.
- Automatiser l’ensemble avec Apache Airflow.
- Implémenter un pipeline CI/CD simple avec GitHub Actions.
- Conteneuriser les composants avec Docker.

---

## 📁 Structure du Répertoire

```
projet-mlops
│
├── dags.py                  # DAG Airflow
├── fonctions.py             # Fonctions d'importation et stockage
├── randomforest.py          # Entraînement du modèle Random Forest
├── app.py                   # Interface utilisateur Streamlit
├── test_airflow.py          # Tests unitaires
├── docker-compose.yaml      # Stack de services
├── Dockerfile               # Image pour Airflow
├── Dockerfile.streamlit     # Image pour Streamlit
├── requirements.txt         # Dépendances Python
└── .github/workflows/
    └── ci-cd.yml            # Pipeline CI/CD GitHub Actions
```

---

## 🧰 Technologies Utilisées

| Composant               | Technologie             |
|-------------------------|-------------------------|
| Orchestration           | Apache Airflow          |
| Entraînement ML         | Random Forest (sklearn) |
| Stockage                | PostgreSQL + MinIO      |
| Interface               | Streamlit               |
| CI/CD                   | GitHub Actions          |
| Conteneurisation        | Docker & Compose        |
| Tests                   | Pytest                  |

---

## ⚙️ Fonctionnalités du Pipeline

### 🔹 1. Collecte & Stockage des Données
- Récupération des images depuis un dépôt GitHub.
- Stockage des images dans MinIO (format S3).
- Insertion des métadonnées (label, URLs) dans PostgreSQL.

### 🔹 2. Entraînement du Modèle
- Modèle RandomForestClassifier (`scikit-learn`).
- Sauvegarde automatique du modèle dans MinIO avec nom formaté.
- Affichage des métriques dans les logs et dans l’interface.

### 🔹 3. Interface Streamlit
- Upload manuel d’une image.
- Prédiction immédiate et affichage du label.
- Affichage de toutes les métriques (accuracy, f1-score, recall, etc.).

### 🔹 4. Apache Airflow
- Un DAG gère :
  - le téléchargement des images
  - l’entraînement du modèle

### 🔹 5. CI/CD avec GitHub Actions
- Exécution automatique des tests avec Pytest.
- Vérification de l’environnement Docker.
- Build automatique du projet.

---

## 🚀 Lancer le Projet

### 🔧 1. Build de l’image Airflow
```bash
docker build -t custom-airflow:latest .
```

### ▶️ 2. Lancer l’ensemble des services
```bash
docker-compose up -d
```

### 🌐 3. Interfaces disponibles
- Airflow : http://localhost:8080
- MinIO (interface web)  : http://localhost:8900
- Streamlit : http://localhost:8501

### 🧪 4. Tests unitaires
```bash
PYTHONPATH=$(pwd) pytest test_airflow.py
```

### 📊 5. Exemple de tâches Airflow
- `imagesDandelion` : Télécharge les images de pissenlits.
- `imagesGrass` : Télécharge les images d’herbe.
- `createModel` : Entraîne et sauvegarde le modèle.

---

## 🔧 Points d'amélioration
- Ajouter une API de type FastAPI pour exposer le modèle en production.
- Intégrer MLflow pour le suivi des expériences.
- Déployer sur Kubernetes pour la scalabilité.
- Mettre en place un monitoring avancé avec Prometheus/Grafana.

