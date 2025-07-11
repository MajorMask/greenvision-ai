from pystac_client import Client
import planetary_computer
import requests
import os
import csv
from datetime import datetime

# === CONFIG ===
OUTPUT_DIR = "../data"
CSV_PATH = "tile_metadata.csv"
BBOX = [-122.5, 40.7, -122.1, 41.0]  # Shasta-Trinity Forest, CA
START_YEAR = 2014
END_YEAR = 2021
MAX_CLOUD_COVER = 20  # (though NAIP is mostly cloud-free)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === INIT ===
catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                      modifier=planetary_computer.sign_inplace)

# === SEARCH ===
search = catalog.search(
    collections=["naip"],
    bbox=BBOX,
    datetime=f"{START_YEAR}-01-01/{END_YEAR}-12-31"
)

items = list(search.get_all_items())
print(f"🛰️ Found {len(items)} NAIP tiles")

# === METADATA CSV INIT ===
csv_rows = [["item_id", "tile_id", "date", "year", "href", "bbox", "path"]]

# === DOWNLOAD LOOP ===
for item in items:
    item_id = item.id
    tile_id = "_".join(item_id.split("_")[:5])
    dt = datetime.strptime(item.properties["datetime"], "%Y-%m-%dT%H:%M:%SZ")
    year = dt.year

    # Skip if outside desired year range
    if not (START_YEAR <= year <= END_YEAR):
        continue

    out_path = os.path.join(OUTPUT_DIR, f"{item_id}.tif")
    asset = item.assets["image"]
    signed_asset = planetary_computer.sign(asset)

    # Already downloaded
    if os.path.exists(out_path):
        print(f"✅ Already downloaded: {item_id}")
    else:
        print(f"⬇️ Downloading {item_id}...")
        try:
            response = requests.get(signed_asset.href, timeout=60)
            response.raise_for_status()
            if b"<?xml" in response.content[:100]:
                raise ValueError("XML response (likely expired link)")

            with open(out_path, "wb") as f:
                f.write(response.content)

            print(f"📁 Saved to {out_path}")
        except Exception as e:
            print(f"❌ Failed to download {item_id}: {e}")
            continue

    # Append metadata
    csv_rows.append([
        item_id,
        tile_id,
        dt.isoformat(),
        year,
        signed_asset.href,
        item.bbox,
        out_path
    ])

# === SAVE CSV ===
with open(CSV_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(csv_rows)

print(f"📝 Metadata saved to {CSV_PATH}")
