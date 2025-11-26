import pandas as pd
import numpy as np
from deepface import DeepFace
import tqdm
import os

ceo_data = pd.read_csv('ceo_data.csv')

# Load your CSV with CEO and year data
ceo_df = pd.read_csv("ceo_data.csv") # update path if needed
BASE_DIR = os.getcwd()

# Base directory where images are stored
base_dir = os.path.join(BASE_DIR, "matched_pictures")

# List to store all analysis results
all_data = []

# Loop through each row in the DataFrame
for index, row in tqdm.tqdm(ceo_df.iterrows()):
    ceo = row['CEO']
    year = str(row['Year'])
    ticker = row['Ticker']

    folder_path = os.path.join(base_dir, ceo, year)
    print(folder_path)
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue

    # Try to find the first valid image file
    image_file = None
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_file = os.path.join(folder_path, file)
            break

    if not image_file:
        print(f"No image found in {folder_path}")
        continue

    try:
        result = DeepFace.analyze(img_path=image_file,
                                  actions=['age', 'gender', 'race', 'emotion'],
                                  detector_backend='opencv')
        data = result[0] if isinstance(result, list) else result

        flat_data = {
            'Ticker': ticker,
            'CEO': ceo,
            'Year': year,
            'Age': data['age'],
            'dominant_gender': data['dominant_gender'],
            'dominant_race': data['dominant_race'],
            'dominant_emotion': data['dominant_emotion'],
            **data['gender'],
            **data['race'],
            **data['emotion']
        }

        all_data.append(flat_data)
    except Exception as e:
        print(f"Error analyzing {image_file}: {e}")

# Create final results DataFrame
df = pd.DataFrame(all_data)
df.to_csv("ceo_face_analysis.csv", index=False)
print(df)
