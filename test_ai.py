
from pathlib import Path

from PIL import Image
from transformers import pipeline


def main():
    print("=" * 45)
    print("       INFRASENSE AI - IMAGE TEST")
    print("=" * 45)

    # Find the uploads folder beside this Python file.
    project_folder = Path(__file__).resolve().parent
    uploads_folder = project_folder / "uploads"

    if not uploads_folder.exists():
        print("\nERROR: The uploads folder was not found.")
        print("Expected folder:", uploads_folder)
        return

    # Find supported image files.
    supported_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    photos = [
        file
        for file in uploads_folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in supported_extensions
    ]

    if not photos:
        print("\nNo photos were found in the uploads folder.")
        print("Open your website and submit a report with a photo.")
        return

    print("\nAvailable photos:")

    for index, photo in enumerate(photos, start=1):
        print(f"{index}. {photo.name}")

    choice = input("\nEnter the photo number to analyze: ").strip()

    if not choice.isdigit():
        print("\nERROR: Please enter a number.")
        return

    photo_number = int(choice)

    if photo_number < 1 or photo_number > len(photos):
        print("\nERROR: That photo number is not in the list.")
        return

    selected_photo = photos[photo_number - 1]

    # Check whether the chosen file can be opened as an image.
    try:
        with Image.open(selected_photo) as img:
            img.verify()
    except Exception as error:
        print("\nERROR: This photo could not be opened.")
        print("Details:", error)
        return

    print("\nSelected photo:", selected_photo.name)
    print("\nLoading the image-analysis model...")
    print("The first run may download model files.")

    try:
        analyzer = pipeline(
            task="zero-shot-image-classification",
            model="openai/clip-vit-base-patch32"
        )
    except Exception as error:
        print("\nERROR: The AI model could not be loaded.")
        print("Check your internet connection and installation.")
        print("Details:", error)
        return

    labels = [
        "a road with a pothole",
        "a road with cracks",
        "a normal road without visible damage",
        "water leaking from a pipe",
        "an overflowing garbage bin",
        "a broken public facility"
    ]

    print("\nAnalyzing the photo. Please wait...")

    try:
        results = analyzer(
            str(selected_photo),
            candidate_labels=labels
        )
    except Exception as error:
        print("\nERROR: The image could not be analyzed.")
        print("Details:", error)
        return

    print("\n" + "=" * 45)
    print("         ANALYSIS RESULTS")
    print("=" * 45)

    for result in results:
        label = result["label"]
        score = result["score"] * 100
        print(f"{label}: {score:.2f}%")

    print("\nAnalysis completed successfully.")
    print("These are model predictions, not confirmed findings.")
    print("Please check the photo manually.")
    print("=" * 45)


if __name__ == "__main__":
    main()

