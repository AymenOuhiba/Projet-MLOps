import streamlit as st
import os
from dotenv import load_dotenv
from minio import Minio
from io import BytesIO
import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# === Chargement des variables d'environnement ===
env_path = "/sources/.env" if os.getenv("DOCKER_ENV") else ".env"
load_dotenv(dotenv_path=env_path)

# === Paramètres MinIO depuis .env ===
MINIO_HOST = os.getenv("MINIO_HOST_DAG")
MINIO_ACCESS_KEY = os.getenv("MINIO_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_PASS")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

# === Connexion à MinIO ===
client = Minio(
    MINIO_HOST,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

# === Interface Streamlit ===
st.set_page_config(page_title="Classification - Dandelion vs Grass", layout="centered")
st.title("🌼 Modèle de classification - Dandelion vs Grass")


# === Dataset personnalisé pour relire les images ===
class MinIODataset(Dataset):
    def __init__(self, minio_client, bucket_name, folders, transform=None):
        self.minio_client = minio_client
        self.bucket_name = bucket_name
        self.folders = folders
        self.transform = transform
        self.image_paths = []
        self.labels = []
        for label, folder in enumerate(folders):
            objects = minio_client.list_objects(
                bucket_name, prefix=folder + "/", recursive=True
            )
            for obj in objects:
                self.image_paths.append(obj.object_name)
                self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        obj = self.minio_client.get_object(self.bucket_name, image_path)
        img = Image.open(obj).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# === Charger le dernier modèle depuis MinIO ===
def load_latest_model():
    model_folder = "model"
    objects = list(
        client.list_objects(MINIO_BUCKET, prefix=f"{model_folder}/", recursive=True)
    )
    model_files = [
        obj.object_name for obj in objects if obj.object_name.endswith(".pkl")
    ]
    if not model_files:
        return None, None
    latest_model_file = sorted(model_files)[-1]
    try:
        accuracy_str = latest_model_file.split("_acc_")[-1].replace("%.pkl", "")
        accuracy = float(accuracy_str)
    except Exception:
        accuracy = None
    response = client.get_object(MINIO_BUCKET, latest_model_file)
    buffer = BytesIO(response.read())
    model = joblib.load(buffer)
    return model, accuracy


# === Charger les données de test ===
def load_test_data():
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = MinIODataset(client, MINIO_BUCKET, ["dandelion", "grass"], transform)
    test_size = int(0.1 * len(dataset))
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - test_size - val_size
    _, _, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    X, y = [], []
    for images, labels in test_loader:
        X.append(images.view(images.size(0), -1).numpy())
        y.extend(labels.numpy())

    X = np.concatenate(X, axis=0)
    y = np.array(y)
    return X, y


# === Charger modèle et afficher infos ===
model, accuracy = load_latest_model()

if model is None:
    st.warning("⚠️ Aucun modèle trouvé dans MinIO.")
else:
    st.success("✅ Modèle chargé avec succès depuis MinIO.")

    if accuracy:
        st.info(f"🎯 **Accuracy enregistrée** : **{accuracy:.2f}%**")

    # === Évaluation sur les vraies données de test ===
    with st.spinner("📊 Évaluation du modèle sur le test set..."):
        X_test, y_test = load_test_data()
        y_pred = model.predict(X_test)
        acc = np.mean(y_pred == y_test)
        st.subheader("📈 Évaluation complète du modèle :")
        st.text(f"✅ Accuracy sur test set : {acc * 100:.2f}%")
        st.text("📋 Classification Report :")
        st.text(
            classification_report(y_test, y_pred, target_names=["Dandelion", "Grass"])
        )
        st.text("🔢 Matrice de Confusion :")
        st.text(confusion_matrix(y_test, y_pred))

    # === Upload d'une image à prédire ===
    uploaded_image = st.file_uploader(
        "📤 Uploadez une image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Image chargée", use_column_width=True)

        transform = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()]
        )
        img_tensor = transform(image).view(1, -1).numpy()
        pred = model.predict(img_tensor)[0]
        label = "🌼 Dandelion" if pred == 0 else "🌱 Grass"
        st.markdown(f"### 🧠 Le modèle prédit : **{label}**")
