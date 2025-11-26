import pandas as pd
import os
import glob
import re
import shutil
import face_recognition
from PIL import Image

def output_matched_face(input_f, validated_f, output_f):
    # Skip unsupported files (e.g., .svg, .gif)
    try:
        # Load and encode the validated CEO face
        validated_image = face_recognition.load_image_file(validated_f)
        validated_encodings = face_recognition.face_encodings(validated_image)
        if not validated_encodings:
            raise Exception(f"No face found in the validated image: {validated_f}")
        validated_encoding = validated_encodings[0]

        # Load and encode all faces in the input photo
        image_to_search = face_recognition.load_image_file(input_f)
        face_locations = face_recognition.face_locations(image_to_search)
        face_encodings = face_recognition.face_encodings(image_to_search, face_locations)
        image_height, image_width, _ = image_to_search.shape

        def adjust_face_location(top, right, bottom, left, image_height, image_width, margin=0.2):
            width = right - left
            height = bottom - top
            new_top = max(0, top - int(height * margin))
            new_right = min(image_width, right + int(width * margin))
            new_bottom = min(image_height, bottom + int(height * margin))
            new_left = max(0, left - int(width * margin))
            return new_top, new_right, new_bottom, new_left

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            match = face_recognition.compare_faces([validated_encoding], face_encoding)
            if match[0]:
                # If one face is detected, use the whole image; otherwise crop the face
                if len(face_locations) == 1:
                    top, right, bottom, left = 0, image_width, image_height, 0
                else:
                    top, right, bottom, left = adjust_face_location(top, right, bottom, left, image_height, image_width)

                pil_image = Image.fromarray(image_to_search)
                face_image = pil_image.crop((left, top, right, bottom))

                os.makedirs(os.path.dirname(output_f), exist_ok=True)
                face_image.save(output_f)
                print(f"Matched: {input_f} -> {output_f}")
                return True
        print(f"No match for: {input_f}")
        return False

    except Exception as e:
        # Handle errors for unsupported files gracefully
        print(f"Skipping unsupported file: {input_f}. Error: {str(e)}")
        return False


# === MAIN SCRIPT STARTS HERE ===

# Base directory and CSV paths
# -----------------------------
BASE_DIR = os.getcwd()
csv_path = os.path.join(BASE_DIR, "ceo_data.csv")
ceo_df = pd.read_csv(csv_path)

for _, row in ceo_df.iterrows():
    ceo = row["CEO"]

    # Load verified CEO photo
    validated_files = glob.glob(f"verified photos/{ceo}.*")
    if not validated_files:
        print(f"Missing verified photo for {ceo}")
        continue
    validated_photo = validated_files[0]

    # Find all yearly photos for the CEO
    photo_files = glob.glob(f"pictures/{ceo}/*/*")
    photo_files = [f for f in photo_files if os.path.isfile(f) and not f.endswith('.csv')]

    # Group photos by year
    photos_by_year = {}
    for photo_path in photo_files:
        parts = photo_path.split(os.sep)
        if len(parts) < 3:
            continue
        year = parts[-2]
        if not re.match(r'^\d{4}$', year):
            continue
        photos_by_year.setdefault(year, []).append(photo_path)

    # === Process each year for this CEO ===
    all_years = sorted(photos_by_year.keys())
    for year in all_years:
        # Define the output path for the matched photo for this year
        output_path = os.path.join("matched_pictures", ceo, year)

        # If the folder already contains a photo, skip this year and move to the next CEO
        if os.path.exists(output_path) and any(os.path.isfile(os.path.join(output_path, f)) for f in os.listdir(output_path)):
            print(f"Skipping year {year} for {ceo} as it already has a matched photo.")
            continue

        photo_list = photos_by_year[year]
        matched = False

        for photo_path in photo_list:
            output_photo_path = os.path.join(output_path, os.path.basename(photo_path))
            if output_matched_face(photo_path, validated_photo, output_photo_path):
                matched = True
                break  # Stop after first match in the year

        if not matched:
            # Fallback: copy the verified photo if no match is found
            fallback_path = os.path.join(output_path, f"{ceo}_verified.jpg")
            os.makedirs(output_path, exist_ok=True)
            shutil.copy(validated_photo, fallback_path)
            print(f"No match found for {ceo} in {year} — copied verified photo instead.")

    # Move to the next CEO after all years processed

from PIL import Image
import os

# Max allowed dimensions (fit-to-page): 8.5 x 11 inches at 300 DPI
max_width = 2550
max_height = 3300

matched_dir = "matched_pictures"  # Change if needed

for root, dirs, files in os.walk(matched_dir):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            image_path = os.path.join(root, file)
            try:
                with Image.open(image_path) as img:
                    width, height = img.size

                    # Only resize if the image is larger than the max dimensions
                    if width > max_width or height > max_height:
                        img.thumbnail((max_width, max_height), Image.LANCZOS)
                        img.save(image_path)
                        print(f"Resized: {image_path}")
                    else:
                        print(f"Already fits: {image_path}")
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
