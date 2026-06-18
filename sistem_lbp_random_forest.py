"""
Sistem Identifikasi Jenis Penyakit Kulit
Metode: Ekstraksi Fitur Local Binary Pattern (LBP) + Random Forest
Split data: 80% training dan 20% testing
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from skimage.feature import local_binary_pattern
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATASET_CANDIDATES = [Path("dataset"), Path("finalpenyakitkulit")]
DATASET_DIR = next((path for path in DATASET_CANDIDATES if path.exists()), DATASET_CANDIDATES[-1])
OUTPUT_DIR = Path("output_lbp_random_forest")
MODEL_DIR = OUTPUT_DIR / "model"
RESULT_DIR = OUTPUT_DIR / "hasil_pengujian"
FEATURE_DIR = OUTPUT_DIR / "fitur"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
RESIZE_TO = (224, 224)
LBP_POINTS = 8
LBP_RADIUS = 1
LBP_METHOD = "uniform"
N_ESTIMATORS = 300


def prepare_output_dirs() -> None:
    for directory in [OUTPUT_DIR, MODEL_DIR, RESULT_DIR, FEATURE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def collect_dataset(dataset_dir: Path = DATASET_DIR) -> pd.DataFrame:
    """Tahap pengumpulan dataset dari folder kelas."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Folder dataset tidak ditemukan: {dataset_dir.resolve()}")

    records = []
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    for class_dir in class_dirs:
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                records.append({"path": str(image_path), "label": class_dir.name})

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError(f"Tidak ada gambar yang ditemukan di {dataset_dir.resolve()}")
    return df


def preprocess_image(image_path: str, size: tuple[int, int] = RESIZE_TO) -> np.ndarray:
    """Tahap preprocessing: baca citra, resize, grayscale, dan reduksi noise."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Gambar gagal dibaca: {image_path}")

    image = cv2.resize(image, size)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    return gray


def extract_lbp_features(gray_image: np.ndarray) -> np.ndarray:
    """Tahap ekstraksi fitur Local Binary Pattern (LBP)."""
    lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_POINTS + 2 if LBP_METHOD == "uniform" else 2 ** LBP_POINTS
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float32")
    hist /= hist.sum() + 1e-7
    return hist


def build_feature_dataset(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    features = []
    labels = []
    paths = []
    failed = []

    for row in df.itertuples(index=False):
        try:
            gray = preprocess_image(row.path)
            lbp_feature = extract_lbp_features(gray)
            features.append(lbp_feature)
            labels.append(row.label)
            paths.append(row.path)
        except Exception as exc:
            failed.append({"path": row.path, "label": row.label, "error": str(exc)})

    if not features:
        raise ValueError("Ekstraksi fitur gagal untuk semua gambar.")

    if failed:
        pd.DataFrame(failed).to_csv(RESULT_DIR / "gambar_gagal_dibaca.csv", index=False)

    feature_names = [f"lbp_bin_{i}" for i in range(len(features[0]))]
    feature_df = pd.DataFrame(features, columns=feature_names)
    feature_df.insert(0, "path", paths)
    feature_df["label"] = labels
    feature_df.to_csv(FEATURE_DIR / "fitur_lbp.csv", index=False)

    return np.asarray(features), np.asarray(labels), feature_names, feature_df


def plot_dataset_distribution(df: pd.DataFrame) -> None:
    counts = df["label"].value_counts().sort_index()
    plt.figure(figsize=(9, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="Set2")
    plt.title("Distribusi Dataset per Kelas")
    plt.xlabel("Jenis penyakit kulit")
    plt.ylabel("Jumlah citra")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "distribusi_dataset.png", dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str]) -> None:
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix - LBP + Random Forest")
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()


def train_and_evaluate() -> dict:
    prepare_output_dirs()

    print("=" * 70)
    print("PEMODELAN SISTEM")
    print("Input citra digital -> preprocessing -> ekstraksi fitur LBP -> Random Forest -> hasil identifikasi")
    print("=" * 70)

    print("\n[1] PENGUMPULAN DATASET")
    df = collect_dataset(DATASET_DIR)
    class_names = sorted(df["label"].unique())
    distribution = df["label"].value_counts().sort_index()
    print(distribution.to_string())
    print(f"Total citra: {len(df)}")
    plot_dataset_distribution(df)

    print("\n[2] PREPROCESSING DATA DAN [3] EKSTRAKSI FITUR LBP")
    X, y, feature_names, feature_df = build_feature_dataset(df)
    print(f"Jumlah data berhasil diproses: {len(X)}")
    print(f"Jumlah fitur LBP per citra: {X.shape[1]}")
    print(f"File fitur disimpan: {FEATURE_DIR / 'fitur_lbp.csv'}")

    print("\n[4] PEMBAGIAN DATA TRAINING DAN TESTING")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        train_size=0.8,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"Training: {len(X_train)} data (80%)")
    print(f"Testing : {len(X_test)} data (20%)")

    print("\n[5] KLASIFIKASI MENGGUNAKAN RANDOM FOREST DAN TRAINING MODEL")
    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(
        {
            "model": model,
            "class_names": class_names,
            "feature_names": feature_names,
            "resize_to": RESIZE_TO,
            "lbp_points": LBP_POINTS,
            "lbp_radius": LBP_RADIUS,
            "lbp_method": LBP_METHOD,
        },
        MODEL_DIR / "model_lbp_random_forest.joblib",
    )
    print(f"Model disimpan: {MODEL_DIR / 'model_lbp_random_forest.joblib'}")

    print("\n[6] PENGUJIAN MODEL")
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "train_accuracy": float(accuracy_score(y_train, y_train_pred)),
        "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
        "precision_weighted": float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0)),
        "total_data": int(len(X)),
        "training_data": int(len(X_train)),
        "testing_data": int(len(X_test)),
        "classes": class_names,
    }

    report_dict = classification_report(y_test, y_test_pred, target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_test, y_test_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_test_pred, labels=class_names)

    pd.DataFrame(report_dict).transpose().to_csv(RESULT_DIR / "classification_report.csv")
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(RESULT_DIR / "confusion_matrix.csv")
    (RESULT_DIR / "classification_report.txt").write_text(report_text, encoding="utf-8")
    (RESULT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(cm, class_names)

    print(f"Akurasi training: {metrics['train_accuracy']:.4f}")
    print(f"Akurasi testing : {metrics['test_accuracy']:.4f}")
    print(f"Precision       : {metrics['precision_weighted']:.4f}")
    print(f"Recall          : {metrics['recall_weighted']:.4f}")
    print(f"F1-score        : {metrics['f1_weighted']:.4f}")
    print("\nClassification Report:")
    print(report_text)
    print(f"Hasil pengujian disimpan di: {RESULT_DIR}")

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
