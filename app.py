import re
import json
import sqlite3
import cv2

from datetime import datetime
from pathlib import Path

from flask import (
    Flask,
    request,
    redirect,
    url_for,
    render_template_string,
    send_from_directory,
)
from werkzeug.utils import secure_filename

from road_detector import analyze_road_photo


# ============================================================
# 1. CONFIGURATION
# ============================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_DIR = BASE_DIR / "uploads"
DB_PATH = BASE_DIR / "infrasense.db"
ANALYSIS_DIR = BASE_DIR / "road_analysis_results"
OPENCV_ANALYSIS_DIR = BASE_DIR / "opencv_analysis_results"
ANALYSIS_DATA_DIR = BASE_DIR / "analysis_results_data"

for folder in (
    UPLOAD_DIR,
    ANALYSIS_DIR,
    OPENCV_ANALYSIS_DIR,
    ANALYSIS_DATA_DIR,
):
    folder.mkdir(parents=True, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}

CATEGORIES = [
    "Damaged road",
    "Water leakage",
    "Overflowing waste bin",
    "Broken public facility",
]

PRIORITIES = ["High", "Medium", "Normal"]
STATUSES = ["Reported", "In Progress", "Resolved"]


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def normalize_location(location):
    location = (location or "").strip().casefold()
    return re.sub(r"\s+", " ", location)


def get_priority(category):
    priorities = {
        "Damaged road": "High",
        "Water leakage": "Medium",
        "Overflowing waste bin": "Medium",
        "Broken public facility": "Normal",
    }
    return priorities.get(category, "Normal")


def connect_db():
    return sqlite3.connect(str(DB_PATH))


def analysis_filename(photo_filename):
    stem = Path(secure_filename(photo_filename)).stem
    return f"{stem}_analysis.jpg"


def opencv_analysis_filename(photo_filename):
    stem = Path(secure_filename(photo_filename)).stem
    return f"{stem}_opencv.jpg"


def analysis_data_path(photo_filename):
    stem = Path(secure_filename(photo_filename)).stem
    return ANALYSIS_DATA_DIR / f"{stem}_results.json"


def save_analysis_data(photo_filename, data):
    """Save AI results separately from the reports database."""
    path = analysis_data_path(photo_filename)
    temporary_path = path.with_suffix(".tmp")

    temporary_path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )

    temporary_path.replace(path)


def load_analysis_data(photo_filename):
    if not photo_filename:
        return {}

    path = analysis_data_path(photo_filename)

    if not path.is_file():
        return {}

    try:
        return json.loads(
            path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        app.logger.exception(
            "Could not load analysis data for %s",
            photo_filename,
        )
        return {}


def init_db():
    """Create the reports table only if it doesn't already exist."""
    with connect_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                location TEXT NOT NULL,
                category TEXT NOT NULL,
                photo TEXT NOT NULL,
                status TEXT DEFAULT 'Reported',
                created_at TEXT
            )
        """)


# ============================================================
# 3. OPENCV ANALYSIS
# ============================================================

def analyze_with_opencv(photo_path):
    """
    Highlight potential image features using OpenCV.
    Candidate regions are not confirmed road damage.
    """
    photo_path = Path(photo_path)

    if not photo_path.is_absolute():
        photo_path = BASE_DIR / photo_path

    if not photo_path.is_file():
        raise FileNotFoundError(
            f"Photo not found: {photo_path}"
        )

    image = cv2.imread(str(photo_path))

    if image is None:
        raise ValueError(
            f"Could not read image: {photo_path}"
        )

    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(height, width))

    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    blackhat_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (17, 17),
    )

    dark_features = cv2.morphologyEx(
        blurred,
        cv2.MORPH_BLACKHAT,
        blackhat_kernel,
    )

    _, mask = cv2.threshold(
        dark_features,
        18,
        255,
        cv2.THRESH_BINARY,
    )

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=2,
    )

    open_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3),
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        open_kernel,
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    annotated = image.copy()
    candidates = []
    image_area = image.shape[0] * image.shape[1]

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        if area < 500 or w < 15 or h < 8:
            continue

        if area > image_area * 0.25:
            continue

        candidates.append((x, y, w, h))

    for index, (x, y, w, h) in enumerate(
        candidates,
        start=1,
    ):
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 165, 255),
            2,
        )

        cv2.putText(
            annotated,
            f"OpenCV candidate {index}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )

    output_path = (
        OPENCV_ANALYSIS_DIR
        / opencv_analysis_filename(photo_path.name)
    )

    if not cv2.imwrite(str(output_path), annotated):
        raise OSError(
            f"Could not save OpenCV result: {output_path}"
        )

    return {
        "candidate_count": len(candidates),
        "result_image": str(output_path),
    }


# ============================================================
# 4. RESPONSIVE DASHBOARD UI
# ============================================================

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>InfraSense AI | Infrastructure Dashboard</title>

<style>
:root {
    --navy: #102b43;
    --navy-light: #193f5d;
    --blue: #087eae;
    --blue-dark: #075985;
    --blue-soft: #e0f2fe;
    --page: #f1f5f9;
    --surface: #ffffff;
    --border: #e2e8f0;
    --text: #172033;
    --muted: #64748b;
    --green: #15803d;
    --red: #b91c1c;
    --orange: #c2410c;
    --shadow: 0 4px 16px rgba(15, 23, 42, 0.045);
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: var(--page);
    color: var(--text);
    font-family: Inter, -apple-system, BlinkMacSystemFont,
                 "Segoe UI", Arial, sans-serif;
    font-size: 14px;
    line-height: 1.5;
}

a {
    color: var(--blue);
}

header {
    background: var(--navy);
    color: white;
    padding: 19px 5%;
}

.header-inner {
    width: 100%;
    max-width: 1500px;
    margin: auto;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 15px;
}

.brand {
    font-size: 25px;
    font-weight: 800;
    letter-spacing: -0.6px;
}

.brand span {
    color: #38bdf8;
}

.tagline {
    color: #cbd5e1;
    font-size: 12px;
    margin-top: 3px;
}

.header-badge {
    background: var(--navy-light);
    padding: 8px 13px;
    border-radius: 30px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.35px;
}

.container {
    width: 100%;
    max-width: 1500px;
    padding: 25px 24px 18px;
    margin: auto;
}

.welcome {
    margin-bottom: 20px;
}

.welcome h1 {
    font-size: clamp(23px, 3vw, 30px);
    line-height: 1.2;
    letter-spacing: -0.6px;
    margin: 0 0 8px;
}

.muted {
    color: var(--muted);
    line-height: 1.65;
}

.notice {
    border-radius: 10px;
    padding: 13px 15px;
    margin: 0 0 14px;
    font-size: 13px;
    overflow-wrap: anywhere;
}

.success {
    background: #dcfce7;
    border: 1px solid #bbf7d0;
    color: #166534;
}

.error {
    background: #fee2e2;
    border: 1px solid #fecaca;
    color: #991b1b;
}

.alert {
    background: #fff7ed;
    border: 1px solid #fed7aa;
    color: #9a3412;
    border-radius: 11px;
    padding: 14px 16px;
    margin-bottom: 18px;
    font-size: 13px;
}

.alert strong {
    display: block;
    margin-bottom: 3px;
}

.stats {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 15px;
    margin: 20px 0;
}

.card {
    min-width: 0;
    background: var(--surface);
    border: 1px solid rgba(226, 232, 240, 0.8);
    border-radius: 14px;
    padding: 20px;
    box-shadow: var(--shadow);
}

.stat-label {
    color: var(--muted);
    font-size: 12px;
    font-weight: 600;
}

.stat-number {
    font-size: 32px;
    font-weight: 800;
    letter-spacing: -1px;
    line-height: 1.2;
    margin: 10px 0 7px;
}

.stat-note {
    color: var(--muted);
    font-size: 11px;
}

.blue { color: #2563eb; }
.orange { color: var(--orange); }
.green { color: var(--green); }
.red { color: var(--red); }

.main-layout {
    display: grid;
    grid-template-columns: minmax(250px, 310px) minmax(0, 1fr);
    align-items: start;
    gap: 20px;
}

.section-heading {
    font-size: 18px;
    font-weight: 750;
    letter-spacing: -0.25px;
    margin: 0 0 5px;
}

.section-description {
    color: var(--muted);
    font-size: 12px;
    line-height: 1.65;
    margin-bottom: 18px;
}

label {
    display: block;
    font-weight: 700;
    font-size: 12px;
    margin: 15px 0 7px;
}

input,
select {
    width: 100%;
    min-width: 0;
    padding: 11px 12px;
    border: 1px solid #cbd5e1;
    border-radius: 9px;
    background: #fff;
    color: var(--text);
    font: inherit;
    font-size: 13px;
}

input:focus,
select:focus {
    outline: 2px solid #bae6fd;
    border-color: var(--blue);
}

input[type="file"] {
    padding: 9px;
    font-size: 11px;
}

.small-text {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.6;
    overflow-wrap: anywhere;
}

button,
.button-link {
    display: inline-flex;
    justify-content: center;
    align-items: center;
    gap: 7px;
    border: 0;
    border-radius: 9px;
    background: var(--blue);
    color: #fff;
    font-size: 12px;
    font-weight: 700;
    padding: 11px 14px;
    text-decoration: none;
    text-align: center;
    cursor: pointer;
    transition: background 0.15s ease, transform 0.15s ease;
}

button:hover,
.button-link:hover {
    background: var(--blue-dark);
}

button:active,
.button-link:active {
    transform: translateY(1px);
}

.form-button {
    width: 100%;
    margin-top: 20px;
}

.filters {
    background: #f8fafc;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 14px;
    margin-bottom: 18px;
}

.filter-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.2fr)
                           minmax(0, 1fr)
                           minmax(0, 1fr);
    gap: 12px;
}

.filter-grid label {
    margin-top: 0;
}

.filter-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 13px;
}

.filter-actions > * {
    flex: 1;
}

.report-summary {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px;
    margin: 0 0 13px;
}

.result-count {
    font-size: 12px;
    color: var(--muted);
}

.report-list {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: 13px;
}

.report-card {
    min-width: 0;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px;
    background: white;
    transition: border-color 0.15s ease,
                box-shadow 0.15s ease;
}

.report-card:hover {
    border-color: #cbd5e1;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
}

.report-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 12px;
    padding-bottom: 13px;
    border-bottom: 1px solid #eef2f7;
}

.report-title {
    font-size: 14px;
    font-weight: 800;
    line-height: 1.4;
    overflow-wrap: anywhere;
}

.report-date {
    margin-top: 4px;
    color: var(--muted);
    font-size: 11px;
}

.badges {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
}

.badge {
    display: inline-flex;
    align-items: center;
    border-radius: 30px;
    padding: 5px 9px;
    font-size: 10px;
    line-height: 1.2;
    font-weight: 750;
    white-space: nowrap;
}

.priority-high {
    color: #b91c1c;
    background: #fee2e2;
}

.priority-medium {
    color: #92400e;
    background: #fef3c7;
}

.priority-normal {
    color: #166534;
    background: #dcfce7;
}

.status-reported {
    color: #1d4ed8;
    background: #dbeafe;
}

.status-progress {
    color: #92400e;
    background: #fef3c7;
}

.status-resolved {
    color: #166534;
    background: #dcfce7;
}

.repeat-yes {
    color: #9a3412;
    background: #ffedd5;
}

.repeat-no {
    color: #64748b;
    background: #f1f5f9;
}

.report-body {
    display: grid;
    grid-template-columns: minmax(0, 1fr) minmax(0, 1.3fr);
    gap: 16px;
    padding-top: 14px;
}

.info-block {
    min-width: 0;
}

.block-title {
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.45px;
    color: #64748b;
    margin-bottom: 9px;
}

.category-name {
    font-size: 13px;
    font-weight: 650;
    overflow-wrap: anywhere;
}

.analysis-links {
    display: flex;
    flex-wrap: wrap;
    gap: 7px;
}

.analysis-link {
    display: inline-flex;
    align-items: center;
    border: 1px solid #dbeafe;
    border-radius: 8px;
    padding: 7px 9px;
    background: #f8fbff;
    color: #0369a1;
    font-size: 11px;
    font-weight: 750;
    text-decoration: none;
    overflow-wrap: anywhere;
}

.analysis-link:hover {
    background: #e0f2fe;
    border-color: #bae6fd;
}

.ai-panel {
    min-width: 0;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    background: #f8fafc;
    padding: 12px;
}

.ai-panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 11px;
}

.ai-panel-title {
    font-size: 12px;
    font-weight: 800;
}

.ai-method {
    font-size: 10px;
    color: var(--muted);
}

.ai-methods {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 10px;
}

.ai-method-card {
    min-width: 0;
    padding: 11px;
    border-radius: 9px;
    background: white;
    border: 1px solid #e2e8f0;
}

.ai-method-name {
    font-size: 11px;
    font-weight: 800;
    margin-bottom: 5px;
}

.ai-count {
    font-size: 22px;
    line-height: 1.25;
    font-weight: 850;
    letter-spacing: -0.5px;
    margin-bottom: 3px;
}

.ai-count-label {
    font-size: 10px;
    color: var(--muted);
}

.detection-list {
    display: grid;
    gap: 8px;
    margin-top: 11px;
}

.detection-item {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 8px;
    align-items: center;
}

.detection-name {
    min-width: 0;
    font-size: 11px;
    font-weight: 650;
    overflow-wrap: anywhere;
}

.confidence {
    font-size: 11px;
    font-weight: 800;
    color: #1d4ed8;
    white-space: nowrap;
}

.confidence-track {
    height: 5px;
    border-radius: 20px;
    background: #e2e8f0;
    overflow: hidden;
    margin-top: 5px;
}

.confidence-fill {
    height: 100%;
    border-radius: inherit;
    background: #3b82f6;
}

.no-detections {
    color: var(--muted);
    font-size: 11px;
    line-height: 1.6;
    margin-top: 8px;
}

.ai-note {
    color: var(--muted);
    font-size: 10px;
    line-height: 1.5;
    margin-top: 10px;
}

.status-update {
    display: flex;
    align-items: center;
    gap: 9px;
    padding-top: 13px;
    margin-top: 13px;
    border-top: 1px solid #eef2f7;
}

.status-update label {
    margin: 0;
    white-space: nowrap;
}

.status-update select {
    flex: 1;
    max-width: 230px;
    padding: 9px 10px;
}

.status-update button {
    padding: 9px 13px;
}

.empty {
    padding: 35px 15px;
    border: 1px dashed #cbd5e1;
    border-radius: 12px;
    color: var(--muted);
    text-align: center;
    line-height: 1.7;
    background: #f8fafc;
}

.footer {
    text-align: center;
    color: var(--muted);
    font-size: 11px;
    padding: 26px 10px 8px;
}

@media (min-width: 1200px) {
    .report-list {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .report-body {
        grid-template-columns: minmax(0, 1fr);
    }

    .ai-methods {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 1050px) {
    .main-layout {
        grid-template-columns: minmax(0, 1fr);
    }

    .report-list {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }

    .report-body {
        grid-template-columns: minmax(0, 1fr);
    }
}

@media (max-width: 720px) {
    header {
        padding: 17px 16px;
    }

    .container {
        padding: 20px 14px 12px;
    }

    .stats {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
    }

    .card {
        padding: 16px;
    }

    .stat-number {
        font-size: 27px;
    }

    .filter-grid {
        grid-template-columns: minmax(0, 1fr);
    }

    .report-list {
        grid-template-columns: minmax(0, 1fr);
    }

    .ai-methods {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}

@media (max-width: 420px) {
    .header-inner {
        align-items: flex-start;
        flex-direction: column;
    }

    .header-badge {
        font-size: 9px;
    }

    .stats {
        gap: 8px;
    }

    .card {
        padding: 13px;
        border-radius: 11px;
    }

    .stat-label {
        font-size: 11px;
    }

    .stat-number {
        font-size: 25px;
    }

    .stat-note {
        font-size: 10px;
    }

    .report-top {
        flex-direction: column;
    }

    .ai-methods {
        grid-template-columns: minmax(0, 1fr);
    }

    .status-update {
        align-items: stretch;
        flex-direction: column;
    }

    .status-update select {
        max-width: none;
    }

    .status-update button {
        width: 100%;
    }

    .filter-actions {
        flex-direction: column;
    }

    .filter-actions > * {
        width: 100%;
    }
}
</style>
</head>

<body>

<header>
    <div class="header-inner">
        <div>
            <div class="brand">InfraSense <span>AI</span></div>
            <div class="tagline">
                Smart Infrastructure Monitoring and Reporting
            </div>
        </div>

        <div class="header-badge">
            URBAN INTELLIGENCE DASHBOARD
        </div>
    </div>
</header>

<main class="container">

    <section class="welcome">
        <h1>Infrastructure Dashboard</h1>
        <div class="muted">
            Monitor reports, identify recurring problems, and track
            infrastructure issues until they are resolved.
        </div>
    </section>

    {% if message %}
    <div class="notice success">{{ message }}</div>
    {% endif %}

    {% if error %}
    <div class="notice error">{{ error }}</div>
    {% endif %}

    {% if recurring_count > 0 %}
    <div class="alert">
        <strong>Recurring issue alert</strong>
        {{ recurring_count }} report(s) belong to locations and
        categories with multiple reports. Review them to identify
        repeated problems.
    </div>
    {% endif %}

    <!-- DASHBOARD STATISTICS -->

    <section class="stats">

        <div class="card">
            <div class="stat-label">Total Reports</div>
            <div class="stat-number blue">{{ total }}</div>
            <div class="stat-note">All submitted reports</div>
        </div>

        <div class="card">
            <div class="stat-label">Pending Reports</div>
            <div class="stat-number orange">{{ pending }}</div>
            <div class="stat-note">Reported or in progress</div>
        </div>

        <div class="card">
            <div class="stat-label">Resolved Reports</div>
            <div class="stat-number green">{{ resolved }}</div>
            <div class="stat-note">Marked as resolved</div>
        </div>

        <div class="card">
            <div class="stat-label">High-Priority Issues</div>
            <div class="stat-number red">{{ high_priority_count }}</div>
            <div class="stat-note">
                Unresolved damaged-road reports
            </div>
        </div>

    </section>

    <section class="main-layout">

        <!-- REPORT SUBMISSION FORM -->

        <div class="card">
            <h2 class="section-heading">Report an Issue</h2>

            <div class="section-description">
                Submit an infrastructure issue and attach a photo.
            </div>

            <form action="{{ url_for('submit_report') }}"
                  method="POST"
                  enctype="multipart/form-data">

                <label for="location">Location</label>

                <input
                    type="text"
                    id="location"
                    name="location"
                    placeholder="e.g. Main Road, Jagtial"
                    maxlength="200"
                    required
                >

                <label for="category">Issue Category</label>

                <select id="category" name="category" required>
                    <option value="">Select an issue</option>

                    {% for category in categories %}
                    <option value="{{ category }}">
                        {{ category }}
                    </option>
                    {% endfor %}
                </select>

                <label for="photo">Upload Photo</label>

                <input
                    type="file"
                    id="photo"
                    name="photo"
                    accept=".jpg,.jpeg,.png,.webp"
                    required
                >

                <div class="small-text">
                    JPG, JPEG, PNG or WEBP. Maximum size: 5 MB.
                </div>

                <button class="form-button" type="submit">
                    Submit Infrastructure Report
                </button>

            </form>
        </div>

        <!-- REPORTS AND FILTERS -->

        <div class="card">

            <h2 class="section-heading">Infrastructure Reports</h2>

            <div class="section-description">
                Search reports, inspect AI detections, and update
                the status of infrastructure issues.
            </div>

            <form action="{{ url_for('home') }}"
                  method="GET"
                  class="filters">

                <div class="filter-grid">

                    <div>
                        <label for="search">Search</label>

                        <input
                            type="text"
                            id="search"
                            name="search"
                            placeholder="Location or category"
                            value="{{ search_query }}"
                        >
                    </div>

                    <div>
                        <label for="category_filter">Category</label>

                        <select id="category_filter" name="category">
                            <option value="">All categories</option>

                            {% for category in categories %}
                            <option
                                value="{{ category }}"
                                {% if selected_category == category %}
                                selected
                                {% endif %}
                            >
                                {{ category }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                    <div>
                        <label for="priority_filter">Priority</label>

                        <select id="priority_filter" name="priority">
                            <option value="">All priorities</option>

                            {% for priority in priorities %}
                            <option
                                value="{{ priority }}"
                                {% if selected_priority == priority %}
                                selected
                                {% endif %}
                            >
                                {{ priority }}
                            </option>
                            {% endfor %}
                        </select>
                    </div>

                </div>

                <div class="filter-actions">
                    <button type="submit">Search &amp; Filter</button>

                    <a class="button-link"
                       href="{{ url_for('home') }}">
                        Clear Filters
                    </a>
                </div>

            </form>

            <div class="report-summary">
                <div class="result-count">
                    Showing {{ filtered_total }} matching report(s).
                </div>
            </div>

            <!-- RESPONSIVE REPORT CARDS -->

            {% if reports %}

            <div class="report-list">

                {% for r in reports %}

                <article class="report-card">

                    <div class="report-top">

                        <div>
                            <div class="report-title">
                                #{{ r[0] }} — {{ r[1] }}
                            </div>

                            <div class="report-date">
                                {{ r[4] or "Date unavailable" }}
                            </div>
                        </div>

                        <div class="badges">

                            <span class="badge
                                {{ 'priority-high' if r[7] == 'High'
                                   else 'priority-medium' if r[7] == 'Medium'
                                   else 'priority-normal' }}">
                                {{ r[7] }} Priority
                            </span>

                            <span class="badge
                                {{ 'status-resolved' if r[3] == 'Resolved'
                                   else 'status-progress' if r[3] == 'In Progress'
                                   else 'status-reported' }}">
                                {{ r[3] }}
                            </span>

                            <span class="badge
                                {{ 'repeat-yes' if r[6] else 'repeat-no' }}">
                                {{ 'Possible Repeat' if r[6] else 'No Match' }}
                            </span>

                        </div>

                    </div>

                    <div class="report-body">

                        <!-- CATEGORY AND LINKS -->

                        <div class="info-block">

                            <div class="block-title">Issue Category</div>

                            <div class="category-name">
                                {{ r[2] }}
                            </div>

                            <div class="block-title"
                                 style="margin-top:16px;">
                                Photo &amp; Analysis
                            </div>

                            {% if r[5] %}

                            <div class="analysis-links">

                                <a class="analysis-link"
                                   href="{{ url_for('uploaded_file', filename=r[5]) }}"
                                   target="_blank"
                                   rel="noopener noreferrer">
                                    View Photo
                                </a>

                                {% if r[8] %}
                                <a class="analysis-link"
                                   href="{{ url_for('analysis_file', filename=r[5]) }}"
                                   target="_blank"
                                   rel="noopener noreferrer">
                                    YOLO Image
                                </a>
                                {% endif %}

                                {% if r[9] %}
                                <a class="analysis-link"
                                   href="{{ url_for('opencv_analysis_file', filename=r[5]) }}"
                                   target="_blank"
                                   rel="noopener noreferrer">
                                    OpenCV Image
                                </a>
                                {% endif %}

                            </div>

                            {% else %}

                            <div class="small-text">No photo uploaded.</div>

                            {% endif %}

                        </div>

                        <!-- AI RESULTS -->

                        <div class="info-block">

                            <div class="ai-panel">

                                <div class="ai-panel-header">
                                    <div class="ai-panel-title">
                                        AI Analysis Results
                                    </div>

                                    <div class="ai-method">
                                        {% if r[2] == 'Damaged road' %}
                                        Road damage analysis
                                        {% else %}
                                        Not applicable
                                        {% endif %}
                                    </div>
                                </div>

                                {% if r[10] %}

                                <div class="ai-methods">

                                    <!-- YOLO RESULTS -->

                                    <div class="ai-method-card">

                                        <div class="ai-method-name">
                                            YOLO Detection
                                        </div>

                                        <div class="ai-count blue">
                                            {{ r[10].get('yolo_count', 0) }}
                                        </div>

                                        <div class="ai-count-label">
                                            detected region(s)
                                        </div>

                                        {% if r[10].get('yolo_regions') %}

                                        <div class="detection-list">

                                            {% for item in r[10].get('yolo_regions', []) %}

                                            <div>
                                                <div class="detection-item">
                                                    <div class="detection-name">
                                                        {{ item.get('class_name', 'Unknown')
                                                           | replace('_', ' ')
                                                           | title }}
                                                    </div>

                                                    <div class="confidence">
                                                        {{ "%.1f"|format(
                                                            item.get('confidence', 0) * 100
                                                        ) }}%
                                                    </div>
                                                </div>

                                                <div class="confidence-track">
                                                    <div
                                                        class="confidence-fill"
                                                        style="width: {{ [
                                                            100,
                                                            [
                                                                0,
                                                                item.get('confidence', 0) * 100
                                                            ] | max
                                                        ] | min }}%;">
                                                    </div>
                                                </div>
                                            </div>

                                            {% endfor %}

                                        </div>

                                        {% else %}

                                        <div class="no-detections">
                                            No damage detected by YOLO
                                            in this image.
                                        </div>

                                        {% endif %}

                                    </div>

                                    <!-- OPENCV RESULTS -->

                                    <div class="ai-method-card">

                                        <div class="ai-method-name">
                                            OpenCV Analysis
                                        </div>

                                        {% if r[10].get('opencv_count') is not none %}

                                        <div class="ai-count orange">
                                            {{ r[10].get('opencv_count') }}
                                        </div>

                                        <div class="ai-count-label">
                                            candidate region(s)
                                        </div>

                                        {% else %}

                                        <div class="ai-count orange">
                                            —
                                        </div>

                                        <div class="ai-count-label">
                                            Results unavailable
                                        </div>

                                        {% endif %}

                                        <div class="ai-note">
                                            Image-processing candidates
                                            are not confirmed damage.
                                        </div>

                                    </div>

                                </div>

                                <div class="ai-note">
                                    YOLO confidence is a model score,
                                    not a guarantee of accuracy.
                                </div>

                                {% else %}

                                <div class="no-detections">
                                    {% if r[2] == 'Damaged road' %}
                                    No saved AI results for this report.
                                    New road reports will save results
                                    when analysis runs successfully.
                                    {% else %}
                                    AI road-damage analysis is only
                                    run for damaged-road reports.
                                    {% endif %}
                                </div>

                                {% endif %}

                            </div>

                        </div>

                    </div>

                    <!-- STATUS UPDATE -->

                    <form class="status-update"
                          action="{{ url_for('update_status', report_id=r[0]) }}"
                          method="POST">

                        <label for="status-{{ r[0] }}">
                            Update status
                        </label>

                        <select
                            id="status-{{ r[0] }}"
                            name="status"
                            required
                        >
                            {% for status in statuses %}
                            <option
                                value="{{ status }}"
                                {% if r[3] == status %}selected{% endif %}
                            >
                                {{ status }}
                            </option>
                            {% endfor %}
                        </select>

                        <button type="submit">Save Status</button>

                    </form>

                </article>

                {% endfor %}

            </div>

            {% else %}

            <div class="empty">
                <strong>No matching reports found.</strong>
                <br>
                Try another search or change your filters.
            </div>

            {% endif %}

        </div>

    </section>

    <footer class="footer">
        InfraSense AI · Smart Infrastructure Monitoring Prototype
    </footer>

</main>
</body>
</html>
"""


# ============================================================
# 5. DASHBOARD, SEARCH AND FILTERS
# ============================================================

@app.route("/")
def home():
    with connect_db() as conn:
        all_reports = conn.execute("""
            SELECT id, location, category, status, created_at, photo
            FROM reports
            ORDER BY id DESC
        """).fetchall()

    # Count reports for recurring issue detection.
    counts = {}

    for report in all_reports:
        key = (
            normalize_location(report[1]),
            normalize_location(report[2]),
        )

        counts[key] = counts.get(key, 0) + 1

    display_reports = []

    for report in all_reports:
        key = (
            normalize_location(report[1]),
            normalize_location(report[2]),
        )

        is_recurring = counts[key] > 1
        priority = get_priority(report[2])

        analysis_name = (
            analysis_filename(report[5])
            if report[5]
            else ""
        )

        has_analysis = bool(
            analysis_name
            and (ANALYSIS_DIR / analysis_name).is_file()
        )

        has_opencv_analysis = bool(
            report[5] and report[2] == "Damaged road"
        )

        saved_results = load_analysis_data(report[5])

        display_reports.append(
            (
                *report,
                is_recurring,
                priority,
                has_analysis,
                has_opencv_analysis,
                saved_results,
            )
        )

    search_query = request.args.get("search", "").strip()
    selected_category = request.args.get("category", "")
    selected_priority = request.args.get("priority", "")

    if selected_category not in CATEGORIES:
        selected_category = ""

    if selected_priority not in PRIORITIES:
        selected_priority = ""

    filtered_reports = []

    for report in display_reports:
        location = report[1]
        category = report[2]
        priority = report[7]

        if search_query:
            query = search_query.casefold()

            if (
                query not in location.casefold()
                and query not in category.casefold()
            ):
                continue

        if selected_category and category != selected_category:
            continue

        if selected_priority and priority != selected_priority:
            continue

        filtered_reports.append(report)

    total = len(all_reports)

    pending = sum(
        1 for report in all_reports
        if report[3] != "Resolved"
    )

    resolved = sum(
        1 for report in all_reports
        if report[3] == "Resolved"
    )

    recurring_count = sum(
        1 for report in display_reports
        if report[6]
    )

    high_priority_count = sum(
        1 for report in all_reports
        if (
            get_priority(report[2]) == "High"
            and report[3] != "Resolved"
        )
    )

    return render_template_string(
        PAGE,
        reports=filtered_reports,
        total=total,
        pending=pending,
        resolved=resolved,
        recurring_count=recurring_count,
        high_priority_count=high_priority_count,
        filtered_total=len(filtered_reports),
        categories=CATEGORIES,
        priorities=PRIORITIES,
        statuses=STATUSES,
        search_query=search_query,
        selected_category=selected_category,
        selected_priority=selected_priority,
        message=request.args.get("message", ""),
        error=request.args.get("error", ""),
    )


# ============================================================
# 6. UPDATE REPORT STATUS
# ============================================================

@app.route("/update-status/<int:report_id>", methods=["POST"])
def update_status(report_id):
    new_status = request.form.get("status", "")

    if new_status not in STATUSES:
        return redirect(url_for(
            "home",
            error="Invalid status selected.",
        ))

    try:
        with connect_db() as conn:
            cursor = conn.execute("""
                UPDATE reports
                SET status = ?
                WHERE id = ?
            """, (new_status, report_id))

            if cursor.rowcount == 0:
                return redirect(url_for(
                    "home",
                    error="Report not found.",
                ))

    except sqlite3.Error:
        app.logger.exception(
            "Could not update report status."
        )

        return redirect(url_for(
            "home",
            error="Status update failed. Please try again.",
        ))

    return redirect(url_for(
        "home",
        message=(
            f"Report #{report_id} status updated "
            f"to {new_status}."
        ),
    ))


# ============================================================
# 7. SERVE ORIGINAL PHOTOS AND ANALYSIS IMAGES
# ============================================================

@app.route("/uploads/<filename>")
def uploaded_file(filename):
    safe_name = secure_filename(filename)

    return send_from_directory(
        str(UPLOAD_DIR),
        safe_name,
    )


@app.route("/analysis/<filename>")
def analysis_file(filename):
    safe_name = secure_filename(filename)
    result_name = analysis_filename(safe_name)

    return send_from_directory(
        str(ANALYSIS_DIR),
        result_name,
    )


@app.route("/opencv-analysis/<filename>")
def opencv_analysis_file(filename):
    safe_name = secure_filename(filename)
    photo_path = UPLOAD_DIR / safe_name

    if not photo_path.is_file():
        return "Photo not found.", 404

    result_name = opencv_analysis_filename(safe_name)
    result_path = OPENCV_ANALYSIS_DIR / result_name

    # Generate the OpenCV image if an older report doesn't have one.
    if not result_path.is_file():
        try:
            analyze_with_opencv(photo_path)

        except Exception:
            app.logger.exception(
                "OpenCV analysis failed for %s",
                safe_name,
            )

            return (
                "OpenCV analysis failed. "
                "Check the Flask terminal for details.",
                500,
            )

    return send_from_directory(
        str(OPENCV_ANALYSIS_DIR),
        result_name,
    )


# ============================================================
# 8. SUBMIT NEW REPORTS
# ============================================================

@app.route("/submit", methods=["POST"])
def submit_report():
    location = request.form.get("location", "").strip()
    category = request.form.get("category", "")
    photo = request.files.get("photo")

    if not location or category not in CATEGORIES:
        return redirect(url_for(
            "home",
            error="Enter a location and select a valid category.",
        ))

    if len(location) > 200:
        return redirect(url_for(
            "home",
            error="Location must be 200 characters or fewer.",
        ))

    if not photo or not photo.filename:
        return redirect(url_for(
            "home",
            error="Please select an image.",
        ))

    safe_name = secure_filename(photo.filename)
    extension = Path(safe_name).suffix.lower()

    if not safe_name or extension not in ALLOWED_EXTENSIONS:
        return redirect(url_for(
            "home",
            error="Upload a JPG, JPEG, PNG or WEBP image.",
        ))

    unique_name = (
        datetime.now().strftime("%Y%m%d%H%M%S%f")
        + extension
    )

    photo_path = UPLOAD_DIR / unique_name

    try:
        photo.save(str(photo_path))

        with connect_db() as conn:
            conn.execute("""
                INSERT INTO reports
                (location, category, photo, status, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                location,
                category,
                unique_name,
                "Reported",
                datetime.now().strftime("%Y-%m-%d %H:%M"),
            ))

    except (OSError, sqlite3.Error):
        app.logger.exception(
            "Failed to save infrastructure report."
        )

        return redirect(url_for(
            "home",
            error=(
                "The report could not be saved. "
                "Please try again."
            ),
        ))

    analysis_message = ""

    if category == "Damaged road":

        saved_results = {
            "yolo_count": 0,
            "yolo_regions": [],
            "opencv_count": None,
        }

        # Run YOLO independently.
        try:
            result = analyze_road_photo(photo_path)
            detections = result.get("regions", [])

            saved_results["yolo_count"] = len(detections)

            saved_results["yolo_regions"] = [
                {
                    "class_name": item.get(
                        "class_name",
                        "Unknown",
                    ),
                    "confidence": float(
                        item.get("confidence", 0)
                    ),
                }
                for item in detections
            ]

            if detections:
                labels = ", ".join(
                    f"{item['class_name']} "
                    f"({item['confidence']:.0%})"
                    for item in detections
                )

                analysis_message = (
                    f" YOLO detected {len(detections)} region(s): "
                    f"{labels}."
                )

            else:
                analysis_message = (
                    " YOLO completed but detected no road damage."
                )

        except Exception:
            app.logger.exception(
                "YOLO analysis failed for %s",
                unique_name,
            )

            analysis_message = (
                " YOLO analysis failed; the report was saved."
            )

        # Run OpenCV independently.
        try:
            cv_result = analyze_with_opencv(photo_path)

            saved_results["opencv_count"] = (
                cv_result["candidate_count"]
            )

            analysis_message += (
                f" OpenCV found "
                f"{cv_result['candidate_count']} candidate region(s)."
            )

        except Exception:
            app.logger.exception(
                "OpenCV analysis failed for %s",
                unique_name,
            )

            analysis_message += (
                " OpenCV analysis failed; check the terminal."
            )

        # Persist results without modifying existing report records.
        try:
            save_analysis_data(
                unique_name,
                saved_results,
            )

        except (OSError, TypeError, ValueError):
            app.logger.exception(
                "Could not save analysis results for %s",
                unique_name,
            )

            analysis_message += (
                " Results could not be saved for dashboard display."
            )

    return redirect(url_for(
        "home",
        message=(
            "Report submitted successfully!"
            + analysis_message
        ),
    ))


# ============================================================
# 9. HANDLE LARGE IMAGE UPLOADS
# ============================================================

@app.errorhandler(413)
def file_too_large(error):
    return (
        "Image is too large. Please choose an image under 5 MB.",
        413,
    )


# ============================================================
# 10. START APPLICATION
# ============================================================

if __name__ == "__main__":
    init_db()
    app.run(debug=True)