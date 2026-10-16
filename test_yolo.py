
from pathlib import Path
from road_detector import analyze_road_photo

uploads_dir = Path(__file__).resolve().parent / "uploads"

photos = []
for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp"):
    photos.extend(uploads_dir.glob(pattern))

if not photos:
    print("No photos found in the uploads folder.")
else:
    print("Available photos:")
    for i, photo in enumerate(photos, start=1):
        print(f"{i}. {photo.name}")

    choice = input("Enter photo number to test: ").strip()

    try:
        photo = photos[int(choice) - 1]
        result = analyze_road_photo(photo)

        print("\nYOLO ANALYSIS RESULT")
        print("Method:", result["method"])
        print("Detections:", result["candidate_count"])

        for region in result["regions"]:
            print(
                region["class_name"],
                "confidence:",
                region["confidence"],
            )

        print("Annotated image:", result["result_image"])

    except (ValueError, IndexError):
        print("Invalid photo number.")
    except Exception as exc:
        print(f"Analysis failed: {type(exc).__name__}: {exc}")
