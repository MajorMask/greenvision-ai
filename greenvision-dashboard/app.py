:import streamlit as st
import os
import pandas as pd
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(layout="wide", page_title="GreenVision: Forest Loss Dashboard")

# === LOAD DATA ===
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
SUMMARY_CSV = os.path.join(OUTPUT_DIR, "forest_loss_summary.csv")

@st.cache_data
def load_summary():
    return pd.read_csv(SUMMARY_CSV)

@st.cache_resource
def load_image(path):
    with rasterio.open(path) as src:
        img = src.read([1, 2, 3])  # RGB channels
    img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min())
    return img

def show_image(img, title):
    st.markdown(f"#### {title}")
    st.image(img, use_column_width=True)

# === UI HEADER ===
st.title("🌳 GreenVision: Forest Loss Detection Dashboard")
st.markdown("Automated segmentation of deforestation using DeepLabV3+ & NAIP data")

# === SIDEBAR SELECTION ===
summary_df = load_summary()
tile_ids = summary_df["tile_id"].tolist()
selected_tile = st.sidebar.selectbox("🔍 Choose a tile", tile_ids)

tile_row = summary_df[summary_df["tile_id"] == selected_tile].iloc[0]
before_file = os.path.join(DATA_DIR, tile_row["before_file"])
after_file  = os.path.join(DATA_DIR, tile_row["after_file"])
heatmap_file = os.path.join(OUTPUT_DIR, f"{selected_tile}_loss_heatmap.png")

# === IMAGE DISPLAY ===
col1, col2 = st.columns(2)
with col1:
    show_image(load_image(before_file), "📸 Before Image")
with col2:
    show_image(load_image(after_file), "🛰️ After Image")

st.markdown("---")

# === HEATMAP ===
st.subheader("🔥 Forest Loss Heatmap")
if os.path.exists(heatmap_file):
    st.image(heatmap_file, caption=f"Tile: {selected_tile} | Mean Loss: {tile_row['mean_forest_loss']}%", use_column_width=True)
else:
    st.warning("No heatmap found for this tile.")

# === SUMMARY TABLE ===
st.subheader("📊 Loss Summary")
st.dataframe(summary_df.style.format({"mean_forest_loss": "{:.2f}"}))

# === DOWNLOADS ===
st.markdown("### 📥 Download Summary")
csv_download = summary_df.to_csv(index=False).encode("utf-8")
st.download_button("Download CSV", csv_download, "forest_loss_summary.csv", "text/csv")

st.markdown("---")
st.caption("Built with ❤️ by GreenVision | DeepLabV3+ | Planetary Computer | Streamlit")
