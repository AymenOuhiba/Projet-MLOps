import os
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, random_split
from minio import Minio
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from dotenv import load_dotenv
from datetime import datetime
from io import BytesIO
import joblib

# Charger les variables d'environnement
env_path = "/sources/.env" if os.getenv("DOCKER_ENV") else ".env"
load_dotenv(dotenv_path=env_path)
MINIO_HOST = os.getenv("MINIO_HOST_DAG")
MINIO_ACCESS_KEY = os.getenv("MINIO_USER")
MINIO_SECRET_KEY = os.getenv("MINIO_PASS")
MINIO_BUCKET = os.getenv("MINIO_BUCKET")

# Connexion à MinIO
minio_client = Minio(
    MINIO_HOST, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY, secure=False
)


# Dataset personnalisé pour MinIO
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


# Fonction d'entraînement RandomForest
def train_random_forest_model(folders=["dandelion", "grass"]):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    dataset = MinIODataset(minio_client, MINIO_BUCKET, folders, transform=transform)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )

    def extract_features(dataloader):
        features, labels = [], []
        for images, lbls in dataloader:
            imgs_np = images.view(images.size(0), -1).numpy()
            features.extend(imgs_np)
            labels.extend(lbls.numpy())
        return np.array(features), np.array(labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    X_train, y_train = extract_features(train_loader)
    X_val, y_val = extract_features(val_loader)
    X_test, y_test = extract_features(test_loader)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred_val = model.predict(X_val)
    print("Validation report:")
    print(classification_report(y_val, y_pred_val))
    print("Confusion Matrix (Validation):")
    print(confusion_matrix(y_val, y_pred_val))

    y_pred_test = model.predict(X_test)
    print("Test report:")
    print(classification_report(y_test, y_pred_test))
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_pred_test))

    accuracy = np.mean(y_pred_test == y_test)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    # Sauvegarde du modèle dans MinIO
    model_folder = "model"
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_filename = f"rf_model_{date_str}_acc_{accuracy * 100:.2f}%.pkl"
    objects = minio_client.list_objects(
        MINIO_BUCKET, prefix=model_folder + "/", recursive=True
    )
    empty_file = BytesIO()
    if not any(obj.object_name.startswith(model_folder) for obj in objects):
        minio_client.put_object(
            MINIO_BUCKET,
            f"{model_folder}/.keep",
            empty_file,
            len(empty_file.getvalue()),
        )
    buffer = BytesIO()
    joblib.dump(model, buffer)
    buffer.seek(0)
    minio_client.put_object(
        MINIO_BUCKET,
        f"{model_folder}/{model_filename}",
        buffer,
        buffer.getbuffer().nbytes,
    )
    print(f"Modèle sauvegardé dans MinIO : {model_folder}/{model_filename}")
