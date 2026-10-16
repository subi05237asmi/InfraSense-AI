
import cv2
from pathlib import Path

# InfraSense AI project folder
project_folder = Path(__file__).resolve().parent
uploads_folder = project_folder / "uploads"

print("InfraSense AI - Road Damage Prototype")
print("OpenCV version:", cv2.__version__)

# Find uploaded road photos
extensions = {".jpg", ".jpeg", ".png", ".webp"}

photos = [
    file for file in uploads_folder.iterdir()
    if file.is_file() and file.suffix.lower() in extensions
] if uploads_folder.exists() else []

if not photos:
    print("No photos found in uploads.")
    print("Upload a road photo first.")
    raise SystemExit

print("\nAvailable photos:")
for index, photo in enumerate(photos, start=1):
    print(f"{index}. {photo.name}")

choice = input("\nEnter photo number: ").strip()

if not choice.isdigit() or not 1 <= int(choice) <= len(photos):
    print("Invalid photo number.")
    raise SystemExit

photo_path = photos[int(choice) - 1]
image = cv2.imread(str(photo_path))

if image is None:
    print("Could not open this image.")
    raise SystemExit

# Resize large images for easier processing
height, width = image.shape[:2]
scale = min(1.0, 1000 / max(height, width))

if scale < 1.0:
    image = cv2.resize(
        image, None, fx=scale, fy=scale,
        interpolation=cv2.INTER_AREA
    )

# Prepare the image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Find unusually dark regions
_, mask = cv2.threshold(
    blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Remove small isolated noise
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

# Find candidate regions
contours, _ = cv2.findContours(
    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

result = image.copy()
image_area = image.shape[0] * image.shape[1]
candidate_count = 0

for contour in contours:
    area = cv2.contourArea(contour)

    # Ignore tiny spots and extremely large regions
    if area < image_area * 0.002 or area > image_area * 0.15:
        continue

    x, y, w, h = cv2.boundingRect(contour)

    # Ignore very thin regions
    if min(w, h) == 0 or max(w, h) / min(w, h) > 4:
        continue

    candidate_count += 1

    cv2.rectangle(
        result, (x, y), (x + w, y + h), (0, 0, 255), 2
    )
    cv2.putText(
        result, "Possible dark region", (x, max(20, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2
    )

# Save the result without changing uploaded photos or the database
output_path = project_folder / "road_detection_result.jpg"

if not cv2.imwrite(str(output_path), result):
    print("Could not save the result image.")
    raise SystemExit

print("\nAnalysis completed.")
print("Photo:", photo_path.name)
print("Candidate regions:", candidate_count)
print("Result saved to:", output_path)

# Display the images
cv2.imshow("Original Road Photo", image)
cv2.imshow("Possible Dark Regions", result)
cv2.imshow("Image Processing Mask", mask)

print("\nPress any key while an image window is selected to close it.")
cv2.waitKey(0)
cv2.destroyAllWindows()
