
import cv2
from pathlib import Path

# InfraSense AI paths
project_folder = Path(__file__).resolve().parent
uploads_folder = project_folder / "uploads"
output_folder = project_folder / "opencv_analysis_results"

output_folder.mkdir(parents=True, exist_ok=True)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def analyze_road_photo(photo_path):
    image = cv2.imread(str(photo_path))

    if image is None:
        print("ERROR: Could not read image.")
        return

    # Resize large images for faster processing.
    height, width = image.shape[:2]
    scale = min(1.0, 1200 / max(height, width))

    if scale < 1.0:
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale))
        )

    # Convert to grayscale and reduce noise.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Highlight dark features.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (17, 17)
    )

    dark_features = cv2.morphologyEx(
        blurred, cv2.MORPH_BLACKHAT, kernel
    )

    _, mask = cv2.threshold(
        dark_features, 18, 255, cv2.THRESH_BINARY
    )

    # Connect nearby features.
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (3, 3)
    )

    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, small_kernel, iterations=2
    )

    # Find possible candidate regions.
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
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

    # Draw orange boxes around candidates.
    for index, (x, y, w, h) in enumerate(candidates, start=1):
        cv2.rectangle(
            annotated,
            (x, y),
            (x + w, y + h),
            (0, 165, 255),
            2
        )

        cv2.putText(
            annotated,
            f"OpenCV candidate {index}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 165, 255),
            2,
            cv2.LINE_AA
        )

    # Save separately from YOLO results.
    output_path = output_folder / f"{photo_path.stem}_opencv.jpg"

    if not cv2.imwrite(str(output_path), annotated):
        print("ERROR: Could not save result.")
        return

    # Short terminal output.
    print("\n--- OpenCV Result ---")
    print("Photo:", photo_path.name)
    print("Candidates:", len(candidates))
    print("Saved:", output_path)


def main():
    print("InfraSense AI - OpenCV Test")

    if not uploads_folder.exists():
        print("ERROR: uploads folder not found.")
        return

    photos = sorted(
        [
            file for file in uploads_folder.iterdir()
            if file.is_file()
            and file.suffix.lower() in EXTENSIONS
        ],
        key=lambda file: file.name.lower()
    )

    if not photos:
        print("No photos found in uploads.")
        return

    for index, photo in enumerate(photos, start=1):
        print(f"{index}. {photo.name}")

    choice = input("Enter photo number: ").strip()

    if not choice.isdigit():
        print("Invalid number.")
        return

    number = int(choice)

    if not 1 <= number <= len(photos):
        print("Photo number not found.")
        return

    analyze_road_photo(photos[number - 1])

    print("Note: Candidates are not confirmed road damage.")


if __name__ == "__main__":
    main()

