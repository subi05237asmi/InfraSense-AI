
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO


# Project folders
BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "road_analysis_results"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REPO = "dronefreak/rdd2022-yolov8s"
MODEL_FILE = "best.pt"

_model = None


def load_model():
    """Load the pretrained road-damage model once."""
    global _model

    if _model is None:
        weights = hf_hub_download(
            repo_id=MODEL_REPO,
            filename=MODEL_FILE,
        )
        _model = YOLO(weights)

    return _model


def analyze_road_photo(photo_path):
    """Detect road cracks and potholes and save an annotated image."""
    photo_path = Path(photo_path)

    if not photo_path.is_absolute():
        photo_path = BASE_DIR / photo_path

    if not photo_path.is_file():
        raise FileNotFoundError(
            f"Road photo not found: {photo_path}"
        )

    model = load_model()

    results = model.predict(
        source=str(photo_path),
        device="cpu",
        imgsz=640,
        conf=0.10,
        verbose=False,
        save=False,
    )

    result = results[0]
    regions = []

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        class_id = int(box.cls[0].item())
        confidence = float(box.conf[0].item())
        class_name = model.names[class_id]

        regions.append({
            "x1": int(x1),
            "y1": int(y1),
            "x2": int(x2),
            "y2": int(y2),
            "class_id": class_id,
            "class_name": class_name,
            "label": class_name,
            "confidence": round(confidence, 4),
        })

    # Draw actual YOLO predictions, not OpenCV dark-region contours.
    annotated_image = result.plot()

    output_path = (
        ANALYSIS_DIR / f"{photo_path.stem}_analysis.jpg"
    )

    if not cv2.imwrite(str(output_path), annotated_image):
        raise OSError(
            f"Could not save analysis image: {output_path}"
        )

    return {
        "photo": photo_path.name,
        "candidate_count": len(regions),
        "regions": regions,
        "result_image": str(output_path),
        "method": "YOLOv8 pretrained road-damage detection",
        "is_confirmed_damage": False,
    }

