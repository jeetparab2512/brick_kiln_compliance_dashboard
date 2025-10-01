import os
import glob
import geopandas as gpd
from shapely.geometry import Point
import pandas as pd

def load_states_geojson_folder(folder_path: str) -> gpd.GeoDataFrame:
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    geojson_files = glob.glob(os.path.join(folder_path, "*.geojson"))
    if not geojson_files:
        raise ValueError(f"No GeoJSON files found in folder: {folder_path}")
    gdfs = []
    for file in geojson_files:
        try:
            gdf = gpd.read_file(file)
            state_col = None
            possible_state_cols = ["state", "state_name", "State_Name", "NAME_1", "ST_NM", "name", "Name", "NAME", "State"]
            for col in possible_state_cols:
                if col in gdf.columns and not gdf[col].isnull().all():
                    state_col = col
                    break
            if state_col:
                gdf["state_name"] = gdf[state_col].astype(str)
            else:
                state_label = os.path.splitext(os.path.basename(file))[0]
                gdf["state_name"] = state_label
            gdfs.append(gdf)
        except Exception:
            continue
    if len(gdfs) == 0:
        raise ValueError(f"No valid GeoJSON files found in folder {folder_path}")
    states_gdf = pd.concat(gdfs, ignore_index=True)
    states_gdf = states_gdf.to_crs("EPSG:4326")
    return states_gdf

def assign_kilns_to_states(kilns: pd.DataFrame, states_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    kilns_gdf = gpd.GeoDataFrame(
        kilns,
        geometry=[Point(xy) for xy in zip(kilns['lon'], kilns['lat'])],
        crs="EPSG:4326"
    )
    states_for_join = states_gdf[['state_name', 'geometry']].copy()
    joined = gpd.sjoin(kilns_gdf, states_for_join, how="left", predicate="within")
    result_data = []
    for idx, row in joined.iterrows():
        kiln_data = {col: row[col] for col in kilns.columns}
        kiln_data['state'] = str(row.get('state_name', "Unknown")) if pd.notna(row.get('state_name')) else "Unknown"
        result_data.append(kiln_data)
    return pd.DataFrame(result_data)