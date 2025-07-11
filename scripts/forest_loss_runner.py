import os
import re
import csv
import numpy as np
import torch
import rasterio
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from rasterio.windows import Window
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

# === CONFIG ===
DATA_DIR = "../data"
OUTPUT_DIR = "/outputs"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "forest_loss_summary.csv")
TILE_SIZE = 512
STRIDE = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === DEVICE + MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = deeplabv3_resnet101(weights="DEFAULT").eval().to(device)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# === HELPERS ===

def get_tiles(width, height, tile_size=TILE_SIZE, stride=STRIDE):
    for top in range(0, height, stride):
        for left in range(0, width, stride):
            if top + tile_size > height or left + tile_size > width:
                continue
            yield Window(left, top, tile_size, tile_size), (left, top)

def segment_forest(crop):
    try:
        rgb = np.transpose(crop[:3], (1, 2, 0)).astype(np.float32) / 255.0
        tensor = preprocess(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)['out'][0]
        return (output.argmax(0).cpu().numpy() == 21)  # 21 = trees in COCO+VOC
    except Exception as e:
        print(f"⚠️ Segmentation failed on a crop: {e}")
        return np.zeros((crop.shape[1], crop.shape[2]), dtype=bool)

def analyze_forest_loss(tif_before, tif_after, tile_size=TILE_SIZE):
    try:
        with rasterio.open(tif_before) as src_before, rasterio.open(tif_after) as src_after:
            width, height = src_before.width, src_before.height
            if width < tile_size or height < tile_size:
                print(f"⚠️ Image too small to process: {tif_before}")
                return [], [], 0, 0

            loss_map, coords_map = [], []

            for window, (x, y) in get_tiles(width, height, tile_size):
                before_crop = src_before.read(window=window)
                after_crop = src_after.read(window=window)

                mask_before = segment_forest(before_crop)
                mask_after = segment_forest(after_crop)

                loss = ((mask_before == 1) & (mask_after == 0)).sum()
                total = (mask_before == 1).sum()
                loss_percent = 100 * loss / total if total else 0

                loss_map.append(loss_percent)
                coords_map.append((x, y))

            return loss_map, coords_map, width, height
    except Exception as e:
        print(f"❌ Failed to process pair: {tif_before} and {tif_after}: {e}")
        return [], [], 0, 0

def plot_heatmap(loss_map, coords_map, width, height, tile_id, tile_size=TILE_SIZE):
    grid_w = width // tile_size
    grid_h = height // tile_size
    heatmap = np.zeros((grid_h, grid_w))

    for (x, y), loss in zip(coords_map, loss_map):
        col = x // tile_size
        row = y // tile_size
        heatmap[row, col] = loss

    plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap, cmap="Reds", vmin=0, vmax=100, square=True, cbar_kws={'label': '% Forest Loss'})
    plt.title(f"Forest Loss Heatmap for {tile_id}")
    plt.xlabel("Tile Columns")
    plt.ylabel("Tile Rows")
    plt.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, f"{tile_id}_loss_heatmap.png")
    plt.savefig(output_path)
    plt.close()
    print(f"✅ Saved heatmap: {output_path}")

def extract_tile_id(filename):
    match = re.match(r"(ca_m_\d+_[a-z]{2}_\d+_\d+)", os.path.basename(filename))
    return match.group(1) if match else None

def find_tile_pairs(data_dir):
    all_tifs = glob(os.path.join(data_dir, "*.tif"))
    tile_groups = {}

    for tif in all_tifs:
        tile_id = extract_tile_id(tif)
        if tile_id:
            tile_groups.setdefault(tile_id, []).append(tif)

    tile_pairs = []
    for tile_id, files in tile_groups.items():
        files_sorted = sorted(files)
        if len(files_sorted) >= 2:
            before = files_sorted[0]
            after = files_sorted[-1]
            tile_pairs.append((tile_id, before, after))
    return tile_pairs

# === MAIN RUN ===

if __name__ == "__main__":
    tile_pairs = find_tile_pairs(DATA_DIR)
    print(f"🔍 Found {len(tile_pairs)} tile pairs to process\n")

    with open(SUMMARY_CSV, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["tile_id", "before_file", "after_file", "mean_forest_loss"])

        for tile_id, before_tif, after_tif in tqdm(tile_pairs, desc="🧠 Analyzing Tiles"):
            print(f"\n📍 Processing: {tile_id}")
            loss_map, coords_map, width, height = analyze_forest_loss(before_tif, after_tif)

            if loss_map:
                plot_heatmap(loss_map, coords_map, width, height, tile_id)
                mean_loss = np.mean(loss_map)
                writer.writerow([tile_id, os.path.basename(before_tif), os.path.basename(after_tif), round(mean_loss, 2)])
            else:
                print(f"⚠️ Skipping {tile_id} due to empty or failed result.")

    print(f"\n📝 Saved summary CSV: {SUMMARY_CSV}")
