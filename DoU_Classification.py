#!/usr/bin/env python
# coding: utf-8

# In[11]:


# import rasterio
# import geopandas as gpd
# from shapely.geometry import box
# import os

# # Path to the directory containing the TIFF files
# tiff_dir = "/Users/imishaj/Desktop/340W/images/50001"
# output_dir = "/Users/imishaj/Desktop/340W/grids"
# os.makedirs(output_dir, exist_ok=True)

# # Process each TIFF file
# for tiff_file in os.listdir(tiff_dir):
#     if not tiff_file.endswith(".tiff"):
#         continue
    
#     tiff_path = os.path.join(tiff_dir, tiff_file)

#     # Read the TIFF file to get bounds and CRS
#     with rasterio.open(tiff_path) as src:
#         bounds = src.bounds
#         crs = src.crs
    
#     # Reproject bounds to EPSG:3857 (meter-based)
#     gdf_bounds = gpd.GeoDataFrame({"geometry": [box(bounds.left, bounds.bottom, bounds.right, bounds.top)]}, crs=crs)
#     gdf_bounds = gdf_bounds.to_crs("EPSG:3857")
    
#     # Extract the new bounding box
#     minx, miny, maxx, maxy = gdf_bounds.total_bounds
    
#     # Create 1x1 km grid
#     grid_size = 1000  # 1 km in meters
#     grid = []
    
#     for x in range(int(minx), int(maxx), grid_size):
#         for y in range(int(miny), int(maxy), grid_size):
#             grid.append(box(x, y, x + grid_size, y + grid_size))
    
#     # Convert to GeoDataFrame
#     grid_gdf = gpd.GeoDataFrame(geometry=grid, crs="EPSG:3857")
    
#     # Save as shapefile
#     grid_filename = f"grid_1km_{os.path.splitext(tiff_file)[0]}.shp"
#     grid_gdf.to_file(os.path.join(output_dir, grid_filename))
#     print(f"Saved grid: {grid_filename}")


# In[20]:


import rasterio
import geopandas as gpd
from shapely.geometry import box
import os

# Path to the directory containing the TIFF files
tiff_dir = "/Users/imishaj/Desktop/340W/images/50001"
output_dir = "/Users/imishaj/Desktop/340W/grids50001"
os.makedirs(output_dir, exist_ok=True)

# Process each TIFF file
for tiff_file in os.listdir(tiff_dir):
    if not tiff_file.endswith(".tiff"):
        continue
    
    tiff_path = os.path.join(tiff_dir, tiff_file)

    # Read the TIFF file to get bounds and CRS
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        crs = src.crs
    
    # Create the grid in EPSG:3857
    gdf_bounds = gpd.GeoDataFrame({"geometry": [box(bounds.left, bounds.bottom, bounds.right, bounds.top)]}, crs=crs)
    gdf_bounds = gdf_bounds.to_crs("EPSG:3857")
    
    # Extract the new bounding box in EPSG:3857
    minx, miny, maxx, maxy = gdf_bounds.total_bounds
    
    # Create 1x1 km grid in EPSG:3857
    grid_size = 1000  # 1 km in meters
    grid = []
    
    for x in range(int(minx), int(maxx), grid_size):
        for y in range(int(miny), int(maxy), grid_size):
            grid.append(box(x, y, x + grid_size, y + grid_size))
    
    # Convert to GeoDataFrame
    grid_gdf = gpd.GeoDataFrame(geometry=grid, crs="EPSG:3857")
    
    # Reproject back to EPSG:4326
    grid_gdf = grid_gdf.to_crs("EPSG:4326")
    
    # Save as shapefile
    grid_filename = f"grid_1km_{os.path.splitext(tiff_file)[0]}.shp"
    grid_gdf.to_file(os.path.join(output_dir, grid_filename))
    print(f"Saved grid: {grid_filename}")


# In[30]:


#!/usr/bin/env python3
# -------------------------------------------------------------
#  Extract per‑grid features (NDVI, NDBI, Built‑up, GLCM texture)
#  for 1‑km grids generated in grids50001/ against Sentinel‑2 TIFFs
# -------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from skimage.feature import graycomatrix, graycoprops

# ---------- paths ----------
tiff_dir   = "/Users/imishaj/Desktop/340W/images/50001"
grid_dir   = "/Users/imishaj/Desktop/340W/grids50001"   # new grids
output_csv = "/Users/imishaj/Desktop/340W/grid_features_50001.csv"

all_features = []

# ---------- loop over grid shapefiles ----------
for grid_file in os.listdir(grid_dir):
    if not grid_file.endswith(".shp"):
        continue

    grid_path = os.path.join(grid_dir, grid_file)
    date      = grid_file.split("_")[-1].split(".")[0]   # e.g. 2018‑09‑16

    tiff_path = os.path.join(tiff_dir, f"image_{date}.tiff")
    if not os.path.exists(tiff_path):
        print(f"[{date}]  🛈  Missing TIFF → {tiff_path}")
        continue

    # grids already EPSG:4326
    grids = gpd.read_file(grid_path)

    with rasterio.open(tiff_path) as src:
        for idx, row in grids.iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            try:
                window = from_bounds(minx, miny, maxx, maxy,
                                     transform=src.transform)

                # read bands → squeeze to (rows, cols)
                nir  = np.squeeze(src.read(8, window=window)).astype(float)
                red  = np.squeeze(src.read(4, window=window)).astype(float)
                swir = np.squeeze(src.read(11, window=window)).astype(float)

                # empty window?
                if nir.size == 0 or red.size == 0 or swir.size == 0:
                    continue

                # mask no‑data (assume value 0)
                nir  = np.where(nir  == 0, np.nan, nir)
                red  = np.where(red  == 0, np.nan, red)
                swir = np.where(swir == 0, np.nan, swir)

                # NDVI / NDBI / Built‑up images (still 2‑D)
                ndvi_img = (nir - red)  / (nir + red)
                ndbi_img = (swir - nir) / (swir + nir)
                bu_img   = (swir - red) / (swir + red)

                # flatten finite pixels for summary stats
                ndvi = ndvi_img[np.isfinite(ndvi_img)]
                ndbi = ndbi_img[np.isfinite(ndbi_img)]
                bu   = bu_img[np.isfinite(bu_img)]

                if ndvi.size == 0 or ndbi.size == 0 or bu.size == 0:
                    continue

                feat = {
                    "grid_id":   idx,
                    "date":      date,
                    "ndvi_mean": ndvi.mean(),
                    "ndvi_std":  ndvi.std(),
                    "ndbi_mean": ndbi.mean(),
                    "ndbi_std":  ndbi.std(),
                    "bu_mean":   bu.mean(),
                    "bu_std":    bu.std()
                }

                # ---------- texture metrics ----------
                if ndvi_img.shape[0] > 1 and ndvi_img.shape[1] > 1:
                    # scale finite NDVI to 0‑255 8‑bit
                    ndvi_min, ndvi_max = np.nanmin(ndvi_img), np.nanmax(ndvi_img)
                    if ndvi_max > ndvi_min:        # avoid /0
                        ndvi_scaled = 255 * (ndvi_img - ndvi_min) / (ndvi_max - ndvi_min)
                    else:
                        ndvi_scaled = np.zeros_like(ndvi_img)

                    ndvi_scaled = np.nan_to_num(ndvi_scaled).astype(np.uint8)

                    glcm = graycomatrix(ndvi_scaled,
                                        distances=[1],
                                        angles=[0],
                                        levels=256,
                                        symmetric=True,
                                        normed=True)

                    feat.update({
                        "ndvi_contrast":    graycoprops(glcm, "contrast")[0, 0],
                        "ndvi_homogeneity": graycoprops(glcm, "homogeneity")[0, 0],
                        "ndvi_entropy":    -np.sum(glcm * np.log2(glcm + 1e-10))
                    })

                all_features.append(feat)

            except Exception as e:
                print(f"Grid {idx} on {date} → {e}")
                continue

# ---------- save ----------
pd.DataFrame(all_features).to_csv(output_csv, index=False)
print(f"✓  Features saved → {output_csv}  ({len(all_features)} rows)")


# In[32]:


#!/usr/bin/env python3
# -------------------------------------------------------------
#  Feature extractor  •  T1 + T2 indices for 1‑km grids
# -------------------------------------------------------------
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from skimage.feature import graycomatrix, graycoprops

# ---------- paths ----------
tiff_dir   = "/Users/imishaj/Desktop/340W/images/50001"
grid_dir   = "/Users/imishaj/Desktop/340W/grids50001"
output_csv = "/Users/imishaj/Desktop/340W/grid_features_T2_50001.csv"

# ---------- helper ----------
def glcm_metrics(img_2d):
    """Return contrast, homogeneity, entropy for a uint8 image."""
    glcm = graycomatrix(img_2d, distances=[1], angles=[0],
                        levels=256, symmetric=True, normed=True)
    contrast    = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    entropy     = -np.sum(glcm * np.log2(glcm + 1e-10))
    return contrast, homogeneity, entropy

# ---------- extraction ----------
rows = []

for grid_file in os.listdir(grid_dir):
    if not grid_file.endswith(".shp"):
        continue
    grid_path = os.path.join(grid_dir, grid_file)
    date      = grid_file.split("_")[-1].split(".")[0]

    tiff_path = os.path.join(tiff_dir, f"image_{date}.tiff")
    if not os.path.exists(tiff_path):
        print(f"[{date}]  missing TIFF")
        continue

    grids = gpd.read_file(grid_path)          # EPSG:4326

    with rasterio.open(tiff_path) as src:
        for idx, row in grids.iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

            # read bands → squeeze to 2‑D
            nir  = np.squeeze(src.read(8, window=window)).astype(float)
            red  = np.squeeze(src.read(4, window=window)).astype(float)
            swir = np.squeeze(src.read(11, window=window)).astype(float)

            if nir.size == 0 or red.size == 0 or swir.size == 0:
                continue

            nir  = np.where(nir  == 0, np.nan, nir)
            red  = np.where(red  == 0, np.nan, red)
            swir = np.where(swir == 0, np.nan, swir)

            # ---- indices (2‑D) ----
            ndvi_img = (nir - red)  / (nir + red)
            ndbi_img = (swir - nir) / (swir + nir)
            bu_img   = (swir - red) / (swir + red)
            ndwi_img = (nir - swir) / (nir + swir)        # NEW (water)

            # finite pixels for stats
            ndvi = ndvi_img[np.isfinite(ndvi_img)]
            ndbi = ndbi_img[np.isfinite(ndbi_img)]
            bu   = bu_img[np.isfinite(bu_img)]
            ndwi = ndwi_img[np.isfinite(ndwi_img)]

            if ndvi.size == 0 or ndbi.size == 0 or bu.size == 0:
                continue   # still skip empty

            # ---- basic stats ----
            feat = {
                "grid_id":   idx,
                "date":      date,
                # T1 stats
                "ndvi_mean": ndvi.mean(),  "ndvi_std":  ndvi.std(),
                "ndbi_mean": ndbi.mean(),  "ndbi_std":  ndbi.std(),
                "bu_mean":   bu.mean(),    "bu_std":    bu.std(),
                # NEW T2: NDWI stats
                "ndwi_mean": ndwi.mean(),  "ndwi_std":  ndwi.std(),
                # NEW T2: built‑up fraction (NDBI > 0)
                "built_frac": (ndbi > 0).sum() / ndbi.size
            }

            # ---- texture: NDVI (existing) ----
            if ndvi_img.shape[0] > 1 and ndvi_img.shape[1] > 1:
                ndvi_scaled = np.nan_to_num(
                    ((ndvi_img - np.nanmin(ndvi_img)) /
                     (np.nanmax(ndvi_img) - np.nanmin(ndvi_img) + 1e-9) * 255)
                ).astype(np.uint8)
                c, h, e = glcm_metrics(ndvi_scaled)
                feat.update({"ndvi_contrast": c, "ndvi_homogeneity": h, "ndvi_entropy": e})

            # ---- texture: NDBI (NEW) ----
            if ndbi_img.shape[0] > 1 and ndbi_img.shape[1] > 1:
                ndbi_scaled = np.nan_to_num(
                    ((ndbi_img - np.nanmin(ndbi_img)) /
                     (np.nanmax(ndbi_img) - np.nanmin(ndbi_img) + 1e-9) * 255)
                ).astype(np.uint8)
                c, h, e = glcm_metrics(ndbi_scaled)
                feat.update({"ndbi_contrast": c, "ndbi_homogeneity": h, "ndbi_entropy": e})

            rows.append(feat)

# ---------- save ----------
pd.DataFrame(rows).to_csv(output_csv, index=False)
print(f"✓  saved {len(rows)} rows  →  {output_csv}")


# In[48]:


# -------------------------------------------------------------
#  Unsupervised DoU clustering  •  K‑means  (3 clusters)
#  Input :  grid_features_T2_50001.csv
#  Output:  grid_features_T2_DoU.csv  (adds DoU_cluster column)
# -------------------------------------------------------------
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------- 1. load the Tier‑2 feature table ----------
df = pd.read_csv("/Users/imishaj/Desktop/340W/grid_features_T2_50001.csv")

# remove non‑numeric identifier columns
X_raw = df.drop(columns=["grid_id", "date"])

# ---------- 2. standardise features ----------
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# ---------- 3. run K‑means with k = 3 ----------
k = 3
km = KMeans(n_clusters=k, random_state=42, n_init="auto")
clusters = km.fit_predict(X)
df["DoU_cluster"] = clusters          # 0,1,2 numeric labels

# ---------- 4. evaluate clustering ----------
sil = silhouette_score(X, clusters)
print(f"Silhouette score (k={k}): {sil:.3f}")

# back‑transform from z‑space to original units
centroids = pd.DataFrame(km.cluster_centers_, columns=X_raw.columns)
centroids = centroids * X_raw.std(axis=0).values + X_raw.mean(axis=0).values
print("\nCluster centroids (original feature space):")
print(centroids.round(3))

# ---------- 5. map cluster IDs to human labels ----------
# Identify clusters based on built_frac
rank = centroids["built_frac"].sort_values().index
label_map = {rank[0]: "rural",    # lowest built_frac
             rank[1]: "semi",     # middle built_frac
             rank[2]: "urban"}    # highest built_frac
df["DoU_label"] = df["DoU_cluster"].map(label_map)

print("\nLabel counts:")
print(df["DoU_label"].value_counts())

# ---------- 6. save the labelled table ----------
df.to_csv("/Users/imishaj/Desktop/340W/grid_features_T2_DoU.csv", index=False)
print("\n✓ Saved  grid_features_T2_DoU.csv")


# In[ ]:




