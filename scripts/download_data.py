from pystac_client import Client
import planetary_computer
import requests

catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                      modifier=planetary_computer.sign_inplace)

search = catalog.search(
    collections=["naip"],
    bbox=[-122.5, 40.7, -122.1, 41.0],
    datetime="2014-01-01/2021-12-31"
)

items = list(search.item_collection())
print(f"Found {len(items)} NAIP tiles")
for item in items:
    print(f"Processing {item.id}")
    asset = item.assets["image"]
    signed_asset = planetary_computer.sign(asset)
    
    # Download the asset
    with open(f"../data/{item.id}.tif", "wb") as f:
        response = requests.get(signed_asset.href)
        f.write(response.content)
    print(f"Downloaded {item.id} to ../data/{item.id}.tif")
