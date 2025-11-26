import os
import re
import time
import urllib.parse
import mimetypes
import requests
import pandas as pd
import tqdm
import traceback

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains
from webdriver_manager.chrome import ChromeDriverManager

# -----------------------------
# Base directory and CSV paths
# -----------------------------
BASE_DIR = os.getcwd()
csv_path = os.path.join(BASE_DIR, "ceo_data.csv")

# -----------------------------
# Load CSV
# -----------------------------
df = pd.read_csv(csv_path)  # add names=['Company','Ticker','Year','CEO'] if no headers
print("DataFrame head:", flush=True)
print(df.head(), flush=True)

# -----------------------------
# Generate Google Image Search URLs
# -----------------------------
output_dir = os.path.join(BASE_DIR, "ceo_data_output")
os.makedirs(output_dir, exist_ok=True)

base_url = "https://www.google.com/search?q={query}&tbm=isch&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"

search_urls = []
current_year = 2019  # adjust if needed

for _, row in df.iterrows():
    name = row["CEO"]
    firm = row["Company"]
    start_year = row["Year"]

    for year in range(start_year, current_year + 1):
        query = urllib.parse.quote(f"{name} {firm}")
        url = base_url.format(
            query=query,
            start_date=f"1/1/{year}",
            end_date=f"12/31/{year}"
        )
        search_urls.append((name, year, url))

output_filename = "ceo_image_search_urls.csv"
search_df = pd.DataFrame(search_urls, columns=["Name", "Year", "Search_URL"])
search_df.to_csv(os.path.join(output_dir, output_filename), index=False)
print(f"Search URLs saved to {os.path.join(output_dir, output_filename)}", flush=True)
print(search_df.head(), flush=True)

# -----------------------------
# Google Image Scraper Class
# -----------------------------
class GoogleImageScraper:
    def __init__(self, headless=False):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument("--start-maximized")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-notifications")

        self.driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=chrome_options
        )

    def hover_batch(self, elements, batch_size=5, delay=0.5):
        actions = ActionChains(self.driver)
        for i in range(0, len(elements), batch_size):
            batch = elements[i:i + batch_size]
            for el in batch:
                actions.move_to_element(el)
            actions.perform()
            time.sleep(delay)

    def scrape_images(self, name, year, search_url, max_links=30):
        save_dir = os.path.join(BASE_DIR, 'pictures', name, str(year))
        os.makedirs(save_dir, exist_ok=True)

        try:
            self.driver.get(search_url)
            time.sleep(2)
            print(search_url, flush=True)
            print(name, year, flush=True)

            search_div = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.ID, "search"))
            )
            time.sleep(2)

            print('Hovering images...', flush=True)
            image_divs = search_div.find_elements(By.CSS_SELECTOR, 'div[jsname="qQjpJ"]')[:max_links]
            self.hover_batch(image_divs, batch_size=5, delay=0.5)
            print('Hover done', flush=True)

            elements = WebDriverWait(search_div, 10).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'div[jsname="qQjpJ"] h3 a'))
            )
            print(f"Found {len(elements)} potential image elements", flush=True)
            time.sleep(2)

            image_urls = []
            for i, element in enumerate(elements):
                href = element.get_attribute('href')
                if href:
                    match_img = re.search(r'imgurl=([^&]+)', href)
                    match_story = re.search(r'imgrefurl=([^&]+)', href)
                    if match_img and match_story:
                        image_url = urllib.parse.unquote(match_img.group(1))
                        story_url = urllib.parse.unquote(match_story.group(1))
                        print(f'element {i} has url {image_url}', flush=True)
                        image_urls.append([image_url, story_url])
                        if len(image_urls) >= max_links:
                            break

            # Save image URLs to CSV
            image_urls_df = pd.DataFrame(image_urls, columns=['image_url', 'story_url'])
            image_urls_df.to_csv(os.path.join(save_dir, 'image_urls.csv'), index=False)

        except Exception as e:
            print(f"Error processing {name} for year {year}: {e}", flush=True)
            traceback.print_exc()

    def close(self):
        self.driver.quit()

# -----------------------------
# Run Scraper
# -----------------------------
search_df = pd.read_csv(os.path.join(output_dir, "ceo_image_search_urls.csv"))

scraper = GoogleImageScraper(headless=False)  # headless=True if desired
for i, (_, row) in enumerate(tqdm.tqdm(search_df.iterrows(), total=len(search_df), desc="Scraping")):

    save_path = os.path.join(BASE_DIR, 'pictures', row["Name"], str(row["Year"]))
    try:
        files = os.listdir(save_path)
    except FileNotFoundError:
        files = []

    if any(f.lower().endswith(('png','jpg','jpeg')) for f in files):
        print('Skipping ', row['Name'], row['Year'], flush=True)
        continue

    scraper.scrape_images(
        name=row["Name"],
        year=row["Year"],
        search_url=row["Search_URL"],
        max_links=4
    )

scraper.close()

# -----------------------------
# Step 3: Download Images from URLs
# -----------------------------
def download_images(image_urls, save_dir, limit=10):
    def get_file_extension(content_type):
        extension = mimetypes.guess_extension(content_type)
        if extension:
            return extension
        # fallback
        extension_map = {
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/webp': '.webp'
        }
        return extension_map.get(content_type, '.jpg')

    image_urls = image_urls[:limit]
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/91.0.4472.124'
    }

    print(f"Downloading {len(image_urls)} images", flush=True)
    for i, url in enumerate(image_urls):
        try:
            img_response = requests.get(url, headers=headers, timeout=30)
            if img_response.status_code == 200:
                content_type = img_response.headers.get('content-type', 'image/jpeg')
                extension = get_file_extension(content_type)
                filename = f'pic{i+1}{extension}'
                filepath = os.path.join(save_dir, filename)
                with open(filepath, 'wb') as f:
                    f.write(img_response.content)
                print(f"Downloaded {filename}", flush=True)
        except Exception as e:
            print(f"Error downloading image {i}: {e}", flush=True)

# Download images for all scraped image URLs
for _, row in tqdm.tqdm(search_df.iterrows(), total=len(search_df), desc="Downloading images"):
    name = row['Name']
    year = str(row["Year"])
    save_dir = os.path.join(BASE_DIR, 'pictures', name, year)

    # Skip if images already exist
    if os.path.exists(save_dir):
        image_files = [f for f in os.listdir(save_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
        if image_files:
            print(f"Skipping {name} ({year}) - images already exist", flush=True)
            continue

    csv_path = os.path.join(save_dir, 'image_urls.csv')

    # Skip if the CSV file does not exist
    if not os.path.exists(csv_path):
        print(f"Skipping {name} ({year}) - image_urls.csv not found", flush=True)
        continue

    image_urls_df = pd.read_csv(csv_path)
    image_urls = image_urls_df['image_url'].tolist()
    download_images(image_urls, save_dir, limit=15)