from __future__ import annotations

import base64
import cgi
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

import cv2
import joblib
import numpy as np
from skimage.feature import local_binary_pattern


ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "output_lbp_random_forest" / "model" / "model_lbp_random_forest.joblib"
INDEX_PATH = APP_DIR / "index.html"

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_IMAGE_MIME_TYPES = {"image/jpeg", "image/png"}
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

CLASS_INFO = {
    "non_skin": {
        "name": "Bukan Citra Kulit",
        "description": (
            "Gambar yang diunggah tidak terdeteksi sebagai citra kulit, sehingga sistem tidak "
            "melanjutkan identifikasi penyakit kulit."
        ),
    },
    "acne": {
        "name": "Acne",
        "description": "Kemungkinan citra menunjukkan pola tekstur yang mirip dengan kelas acne pada dataset.",
    },
    "alopecia": {
        "name": "Alopecia",
        "description": "Kemungkinan citra menunjukkan pola area kulit/rambut yang mirip dengan kelas alopecia pada dataset.",
    },
    "atopic": {
        "name": "Atopic Dermatitis",
        "description": "Kemungkinan citra menunjukkan pola tekstur yang mirip dengan kelas atopic dermatitis pada dataset.",
    },
    "normal": {
        "name": "Normal",
        "description": "Kemungkinan citra menunjukkan pola kulit yang mirip dengan kelas normal pada dataset.",
    },
    "psoriasis": {
        "name": "Psoriasis",
        "description": "Kemungkinan citra menunjukkan pola tekstur yang mirip dengan kelas psoriasis pada dataset.",
    },
    "vitiligo": {
        "name": "Vitiligo",
        "description": "Kemungkinan citra menunjukkan pola pigmentasi yang mirip dengan kelas vitiligo pada dataset.",
    },
}

MIN_SKIN_RATIO = 0.10
MIN_CENTER_SKIN_RATIO = 0.08


def assess_skin_image(image: np.ndarray) -> dict:
    """Estimate whether an image contains enough skin-colored pixels for classification."""
    height, width = image.shape[:2]
    scale = min(1.0, 512 / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)))

    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)
    h_channel, s_channel, v_channel = cv2.split(hsv)

    ycrcb_skin = (
        (y_channel > 35)
        & (cr_channel >= 133)
        & (cr_channel <= 180)
        & (cb_channel >= 75)
        & (cb_channel <= 135)
    )
    hsv_skin = (
        (((h_channel <= 25) | (h_channel >= 160)))
        & (s_channel >= 20)
        & (s_channel <= 190)
        & (v_channel >= 45)
    )
    skin_mask = ycrcb_skin & hsv_skin

    skin_ratio = float(np.count_nonzero(skin_mask) / skin_mask.size)

    mask_height, mask_width = skin_mask.shape
    y_start = mask_height // 4
    y_end = mask_height - y_start
    x_start = mask_width // 4
    x_end = mask_width - x_start
    center_mask = skin_mask[y_start:y_end, x_start:x_end]
    center_skin_ratio = float(np.count_nonzero(center_mask) / center_mask.size)

    is_skin = skin_ratio >= MIN_SKIN_RATIO and center_skin_ratio >= MIN_CENTER_SKIN_RATIO
    score = min(1.0, max(skin_ratio / MIN_SKIN_RATIO, center_skin_ratio / MIN_CENTER_SKIN_RATIO) / 2)

    return {
        "is_skin": bool(is_skin),
        "skin_ratio": skin_ratio,
        "center_skin_ratio": center_skin_ratio,
        "score": score,
    }


def load_model_bundle() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan di {MODEL_PATH}. Jalankan sistem_lbp_random_forest.py terlebih dahulu."
        )
    return joblib.load(MODEL_PATH)


MODEL_BUNDLE = load_model_bundle()
MODEL = MODEL_BUNDLE["model"]
RESIZE_TO = tuple(MODEL_BUNDLE.get("resize_to", (224, 224)))
LBP_POINTS = int(MODEL_BUNDLE.get("lbp_points", 8))
LBP_RADIUS = int(MODEL_BUNDLE.get("lbp_radius", 1))
LBP_METHOD = MODEL_BUNDLE.get("lbp_method", "uniform")


def image_to_data_url(image: np.ndarray, ext: str = ".png") -> str:
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Gagal mengubah citra menjadi preview.")
    encoded = base64.b64encode(buffer).decode("ascii")
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def decode_uploaded_image(file_bytes: bytes) -> np.ndarray:
    image_array = np.frombuffer(file_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("File tidak dapat dibaca sebagai citra. Gunakan JPG, JPEG, atau PNG.")
    return image


def validate_uploaded_file(file_item: cgi.FieldStorage) -> None:
    filename = Path(file_item.filename or "").name
    extension = Path(filename).suffix.lower()
    content_type = (getattr(file_item, "type", "") or "").split(";")[0].strip().lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS or (content_type and content_type not in ALLOWED_IMAGE_MIME_TYPES):
        raise ValueError("File harus berupa foto dengan format JPG, JPEG, atau PNG. File PDF tidak diperbolehkan.")


def validate_image_bytes(file_bytes: bytes) -> None:
    is_jpeg = file_bytes.startswith(b"\xff\xd8\xff")
    is_png = file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
    if not is_jpeg and not is_png:
        raise ValueError("File harus berupa foto asli dengan format JPG, JPEG, atau PNG. File PDF tidak diperbolehkan.")


def preprocess_image(image: np.ndarray) -> np.ndarray:
    resized = cv2.resize(image, RESIZE_TO)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def extract_lbp(gray_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method=LBP_METHOD)
    n_bins = LBP_POINTS + 2 if LBP_METHOD == "uniform" else 2**LBP_POINTS
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_bins + 1), range=(0, n_bins))
    hist = hist.astype("float32")
    hist /= hist.sum() + 1e-7
    return lbp, hist


def predict_image(file_bytes: bytes) -> dict:
    original = decode_uploaded_image(file_bytes)
    gray = preprocess_image(original)
    lbp, hist = extract_lbp(gray)
    skin_check = assess_skin_image(original)

    lbp_preview = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    stages = {
        "original_size": {"width": int(original.shape[1]), "height": int(original.shape[0])},
        "processed_size": {"width": int(RESIZE_TO[0]), "height": int(RESIZE_TO[1])},
        "preprocessed_image": image_to_data_url(gray),
        "lbp_image": image_to_data_url(lbp_preview),
    }
    model_info = {
        "method": "Local Binary Pattern (LBP) + Random Forest",
        "lbp_points": LBP_POINTS,
        "lbp_radius": LBP_RADIUS,
        "lbp_method": LBP_METHOD,
    }

    if not skin_check["is_skin"]:
        return {
            "prediction": {
                "label": "non_skin",
                "name": CLASS_INFO["non_skin"]["name"],
                "confidence": 1.0 - skin_check["score"],
                "description": CLASS_INFO["non_skin"]["description"],
                "is_skin_image": False,
            },
            "probabilities": [],
            "histogram": [float(value) for value in hist],
            "stages": stages,
            "model": model_info,
            "skin_check": skin_check,
        }

    probabilities = MODEL.predict_proba([hist])[0]
    classes = list(MODEL.classes_)
    predicted_index = int(np.argmax(probabilities))
    predicted_label = classes[predicted_index]
    confidence = float(probabilities[predicted_index])

    probability_rows = [
        {
            "label": label,
            "name": CLASS_INFO.get(label, {}).get("name", label.title()),
            "probability": float(prob),
        }
        for label, prob in sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
    ]

    return {
        "prediction": {
            "label": predicted_label,
            "name": CLASS_INFO.get(predicted_label, {}).get("name", predicted_label.title()),
            "confidence": confidence,
            "description": CLASS_INFO.get(predicted_label, {}).get("description", ""),
            "is_skin_image": True,
        },
        "probabilities": probability_rows,
        "histogram": [float(value) for value in hist],
        "stages": stages,
        "model": model_info,
        "skin_check": skin_check,
    }


class SkinDiseaseHandler(BaseHTTPRequestHandler):
    server_version = "SkinLBPWeb/1.0"

    def do_GET(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path in {"/", "/index.html"}:
            self.send_file(INDEX_PATH, "text/html; charset=utf-8")
            return
        if parsed_path.path == "/api/health":
            self.send_json({"status": "ok", "model_loaded": MODEL_PATH.exists()})
            return
        self.send_error(404, "Halaman tidak ditemukan.")

    def do_POST(self) -> None:
        parsed_path = urlparse(self.path)
        if parsed_path.path != "/api/predict":
            self.send_error(404, "Endpoint tidak ditemukan.")
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            if content_length <= 0:
                raise ValueError("Tidak ada file yang dikirim.")
            if content_length > MAX_UPLOAD_BYTES:
                raise ValueError("Ukuran file terlalu besar. Maksimal 8 MB.")

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type"),
                    "CONTENT_LENGTH": str(content_length),
                },
            )

            file_item = form["image"] if "image" in form else None
            if file_item is None or not getattr(file_item, "file", None):
                raise ValueError("Field file 'image' tidak ditemukan.")

            validate_uploaded_file(file_item)
            file_bytes = file_item.file.read()
            validate_image_bytes(file_bytes)
            result = predict_image(file_bytes)
            self.send_json(result)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)

    def send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self.send_error(404, "File aplikasi tidak ditemukan.")
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload: dict, status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args) -> None:
        print("%s - %s" % (self.address_string(), format % args))


def run(host: str = "127.0.0.1", port: int = 8000) -> None:
    server = ThreadingHTTPServer((host, port), SkinDiseaseHandler)
    print(f"Aplikasi web berjalan di http://{host}:{port}")
    print("Tekan Ctrl+C untuk menghentikan server.")
    server.serve_forever()


if __name__ == "__main__":
    run()
