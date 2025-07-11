import os
import requests
import csv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from pystac_client import Client
import planetary_computer

# === CONFIG ===
OUTPUT_DIR = "../data"
CSV_PATH = "tile_metadata.csv"
BBOX = [-122.5, 40.7, -122.1, 41.0]  # Example: Shasta-Trinity Forest
START_YEAR, END_YEAR = 2014, 2021
MAX_WORKERS = 8  # Adjust based on your bandwidth and CPU
RETRIES = 3

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === FETCH ITEMS ===
print("📡 Searching NAIP items...")
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                      modifier=planetary_computer.sign_inplace)

search = catalog.search(
    collections=["naip"],
    bbox=BBOX,
    datetime=f"{START_YEAR}-01-01/{END_YEAR}-12-31"
)

items = list(search.get_all_items())
print(f"🔍 Found {len(items)} items")

# === PREPARE DOWNLOAD TASKS ===
tasks = []
metadata_rows = [["item_id", "tile_id", "date", "year", "href", "bbox", "path"]]

for item in items:
    dt = datetime.strptime(item.properties["datetime"], "%Y-%m-%dT%H:%M:%SZ")
    year = dt.year
    if not (START_YEAR <= year <= END_YEAR):
        continue

    item_id = item.id
    tile_id = "_".join(item_id.split("_")[:5])
    output_path = os.path.join(OUTPUT_DIR, f"{item_id}.tif")

    tasks.append((item, item_id, tile_id, dt, year, output_path))

# === DOWNLOAD FUNCTION ===
def download_tile(args):
    item, item_id, tile_id, dt, year, output_path = args

    if os.path.exists(output_path):
        return ("SKIPPED", item_id, tile_id, dt, year, "", item.bbox, output_path)

    for attempt in range(RETRIES):
        try:
            signed = planetary_computer.sign(item.assets["image"])
            r = requests.get(signed.href, timeout=60)
            r.raise_for_status()

            if b"<?xml" in r.content[:100]:
                raise ValueError("XML error (auth issue)")

            with open(output_path, "wb") as f:
                f.write(r.content)

            return ("DOWNLOADED", item_id, tile_id, dt, year, signed.href, item.bbox, output_path)
        except Exception as e:
            print(f"❌ [{item_id}] Attempt {attempt+1} failed: {e}")
            if attempt == RETRIES - 1:
                return ("FAILED", item_id, tile_id, dt, year, "", item.bbox, output_path)

# === PARALLEL DOWNLOAD ===
print(f"🚀 Starting parallel downloads with {MAX_WORKERS} workers...\n")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = [executor.submit(download_tile, t) for t in tasks]
    for future in as_completed(futures):
        result = future.result()
        status, item_id, tile_id, dt, year, href, bbox, path = result

        if status == "DOWNLOADED":
            print(f"✅ Downloaded: {item_id}")
            metadata_rows.append([item_id, tile_id, dt.isoformat(), year, href, bbox, path])
        elif status == "SKIPPED":
            print(f"⏩ Already exists: {item_id}")
            metadata_rows.append([item_id, tile_id, dt.isoformat(), year, href, bbox, path])
        else:
            print(f"❌ Failed completely: {item_id}")

# === SAVE METADATA CSV ===
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(metadata_rows)

print(f"\n📝 Metadata saved to {CSV_PATH}")
