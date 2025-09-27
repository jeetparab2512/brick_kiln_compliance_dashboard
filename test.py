#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import os
from typing import List, Tuple
import glob
import tempfile
import gradio as gr
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import folium
from folium.plugins import MarkerCluster, HeatMap
import shapely.wkt
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import requests
# import markers
# ------------------------------
# Utilities
# ------------------------------

EARTH_RADIUS_KM = 6371.0088

def _to_radians(latlon: np.ndarray) -> np.ndarray:
    return np.radians(latlon.astype(float))

def _balltree_haversine_min_km(a_latlon_deg: np.ndarray, b_latlon_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(a_latlon_deg) == 0 or len(b_latlon_deg) == 0:
        return np.array([]), np.array([], dtype=int)
    a_rad = _to_radians(a_latlon_deg[:, [0,1]])[:, ::-1]
    b_rad = _to_radians(b_latlon_deg[:, [0,1]])[:, ::-1]
    tree = BallTree(b_rad, metric="haversine")
    dist_rad, idx = tree.query(a_rad, k=1)
    dist_km = dist_rad.flatten() * EARTH_RADIUS_KM
    return dist_km, idx.flatten()

def _lines_to_vertices_df(lines_like: pd.DataFrame) -> pd.DataFrame:
    if {"lon", "lat"}.issubset(lines_like.columns):
        return lines_like[["lon", "lat"]].dropna().reset_index(drop=True)
    if "geometry" not in lines_like.columns:
        return pd.DataFrame(columns=["lon", "lat"])
    out = []
    for _, row in lines_like.iterrows():
        geom = row["geometry"]
        if isinstance(geom, str):
            try:
                geom = shapely.wkt.loads(geom)
            except Exception:
                geom = None
        if geom is None:
            continue
        gtype = getattr(geom, "geom_type", "")
        if gtype == "LineString":
            out.extend([(x, y) for x, y in geom.coords])
        elif gtype == "MultiLineString":
            for line in geom.geoms:
                out.extend([(x, y) for x, y in line.coords])
        elif gtype == "Point":
            out.append((geom.x, geom.y))
    return pd.DataFrame(out, columns=["lon", "lat"]).dropna()

def _ensure_cols(df: pd.DataFrame, needed: List[str], name_for_error: str):
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{name_for_error}: missing columns {missing}. Expected at least {needed}.")

def _read_csv(file) -> pd.DataFrame:
    if file is None:
        return pd.DataFrame()
    if isinstance(file, str):
        return pd.read_csv(file)
    if isinstance(file, dict):
        for key in ("path", "name"):
            p = file.get(key)
            if isinstance(p, str) and os.path.exists(p):
                return pd.read_csv(p)
        data = file.get("data")
        if data is not None:
            return pd.read_csv(io.BytesIO(data))
        return pd.DataFrame()
    path = getattr(file, "name", None)
    if isinstance(path, str) and os.path.exists(path):
        return pd.read_csv(path)
    try:
        return pd.read_csv(file)
    except Exception:
        return pd.DataFrame()

def _center_from_points(latlon: np.ndarray) -> Tuple[float, float]:
    if len(latlon) == 0:
        return 28.6, 77.2
    return float(np.mean(latlon[:, 0])), float(np.mean(latlon[:, 1]))

def load_states_geojson_folder(folder_path: str) -> gpd.GeoDataFrame:
    """Load and combine all state GeoJSON files from folder"""
    if not os.path.exists(folder_path):
        raise ValueError(f"Folder path does not exist: {folder_path}")
    
    geojson_files = glob.glob(os.path.join(folder_path, "*.geojson"))
    if not geojson_files:
        raise ValueError(f"No GeoJSON files found in folder: {folder_path}")
    
    gdfs = []
    for file in geojson_files:
        try:
            print(f"Loading: {file}")
            gdf = gpd.read_file(file)
            print(f"Columns in {os.path.basename(file)}: {list(gdf.columns)}")
            state_col = None
            possible_state_cols = ["state", "state_name", "State_Name", "NAME_1", "ST_NM", "name", "Name", "NAME", "State"]
            for col in possible_state_cols:
                if col in gdf.columns and not gdf[col].isnull().all():
                    state_col = col
                    break
            if state_col is None:
                for col in gdf.columns:
                    if col.lower() in [pc.lower() for pc in possible_state_cols] and not gdf[col].isnull().all():
                        state_col = col
                        break
            if state_col:
                gdf["state_name"] = gdf[state_col].astype(str)
            else:
                state_label = os.path.splitext(os.path.basename(file))[0]
                gdf["state_name"] = state_label
            print(f"State names in {os.path.basename(file)}: {gdf['state_name'].unique()}")
            gdfs.append(gdf)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
    if len(gdfs) == 0:
        raise ValueError(f"No valid GeoJSON files found in folder {folder_path}")
    states_gdf = pd.concat(gdfs, ignore_index=True)
    states_gdf = states_gdf.to_crs("EPSG:4326")
    print(f"Total states loaded: {len(states_gdf['state_name'].unique())}")
    print(f"State names: {sorted(states_gdf['state_name'].unique())}")
    return states_gdf

def assign_kilns_to_states(kilns: pd.DataFrame, states_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    print(f"Assigning {len(kilns)} kilns to states...")
    kilns_gdf = gpd.GeoDataFrame(
        kilns,
        geometry=[Point(xy) for xy in zip(kilns['lon'], kilns['lat'])],
        crs="EPSG:4326"
    )
    states_for_join = states_gdf[['state_name', 'geometry']].copy()
    joined = gpd.sjoin(kilns_gdf, states_for_join, how="left", predicate="within")
    result_data = []
    for idx, row in joined.iterrows():
        kiln_data = {}
        for col in kilns.columns:
            kiln_data[col] = row[col]
        if pd.notna(row.get('state_name')):
            kiln_data['state'] = str(row['state_name'])
        else:
            kiln_data['state'] = "Unknown"
        result_data.append(kiln_data)
    result_df = pd.DataFrame(result_data)
    print(f"Kilns assigned to states:")
    if 'state' in result_df.columns:
        state_counts = result_df['state'].value_counts()
        for state, count in state_counts.items():
            print(f"  {state}: {count} kilns")
    else:
        print("Error: 'state' column not found in result")
        print(f"Available columns: {list(result_df.columns)}")
    return result_df

def determine_compliance_category(kiln_violation, hosp_violation, water_violation):
    violations = []
    if kiln_violation:
        violations.append("kiln")
    if hosp_violation:
        violations.append("hospital")
    if water_violation:
        violations.append("water")
    if len(violations) == 0:
        return "Fully Compliant", ""
    elif len(violations) == 1:
        return f"1 Violation ({violations[0]})", violations[0]
    elif len(violations) == 2:
        return f"2 Violations ({', '.join(violations)})", ', '.join(violations)
    else:
        return "3 Violations (all)", ', '.join(violations)

# ------------------------------
# Core: compute compliance
# ------------------------------

def compute_compliance(
    kilns_csv,
    hospitals_csv=None,
    waterways_csv=None,
    kiln_km_thresh: float = 1.0,
    hosp_km_thresh: float = 0.8,
    water_km_thresh: float = 0.5,
    add_heatmap: bool = False,
    cluster_points: bool = True,
    states_geojson_folder: str = "C:/VSCode/Internship/Compliance_Monitoring/welldone/india_shapefiles/"
):
    kilns = _read_csv(kilns_csv)
    _ensure_cols(kilns, ["lat", "lon"], "Kilns CSV")
    print(f"Loaded {len(kilns)} kilns")
    try:
        states_gdf = load_states_geojson_folder(states_geojson_folder)
    except Exception as e:
        print(f"Error loading state boundaries: {e}")
        kilns["state"] = "Unknown"
        states_gdf = None
    else:
        kilns = assign_kilns_to_states(kilns, states_gdf)
    hospitals = _read_csv(hospitals_csv) if hospitals_csv else pd.DataFrame()
    waterways = _read_csv(waterways_csv) if waterways_csv else pd.DataFrame()
    kiln_latlon = kilns[["lat", "lon"]].to_numpy(dtype=float)
    if len(kilns) >= 2:
        rad = _to_radians(kiln_latlon)[:, ::-1]
        tree = BallTree(rad, metric="haversine")
        dist_rad, _ = tree.query(rad, k=2)
        nearest_km = dist_rad[:, 1] * EARTH_RADIUS_KM
    else:
        nearest_km = np.full(len(kilns), np.nan)
    if not hospitals.empty and {"Latitude", "Longitude"}.issubset(hospitals.columns):
        hosp_latlon = hospitals[["Latitude", "Longitude"]].to_numpy(dtype=float)
        hosp_km, _ = _balltree_haversine_min_km(kiln_latlon, hosp_latlon)
    elif not hospitals.empty and {"lat", "lon"}.issubset(hospitals.columns):
        hosp_latlon = hospitals[["lat", "lon"]].to_numpy(dtype=float)
        hosp_km, _ = _balltree_haversine_min_km(kiln_latlon, hosp_latlon)
    else:
        hosp_km = np.full(len(kilns), np.nan)
    if not waterways.empty:
        water_pts = _lines_to_vertices_df(waterways)
        if len(water_pts) > 0:
            water_latlon = water_pts[["lat", "lon"]].to_numpy(dtype=float)
            water_km, _ = _balltree_haversine_min_km(kiln_latlon, water_latlon)
        else:
            water_km = np.full(len(kilns), np.nan)
    else:
        water_km = np.full(len(kilns), np.nan)
    kiln_violations = np.where(
        (kiln_km_thresh is not None and kiln_km_thresh > 0) & ~np.isnan(nearest_km),
        nearest_km < kiln_km_thresh,
        False
    )
    hosp_violations = np.where(
        (hosp_km_thresh is not None and hosp_km_thresh > 0) & ~np.isnan(hosp_km),
        hosp_km < hosp_km_thresh,
        False
    )
    water_violations = np.where(
        (water_km_thresh is not None and water_km_thresh > 0) & ~np.isnan(water_km),
        water_km < water_km_thresh,
        False
    )
    overall_compliant = ~(kiln_violations | hosp_violations | water_violations)
    compliance_categories = []
    violation_details = []
    for i in range(len(kilns)):
        category, details = determine_compliance_category(
            kiln_violations[i], hosp_violations[i], water_violations[i]
        )
        compliance_categories.append(category)
        violation_details.append(details)
    out = kilns.copy()
    out["nearest_kiln_km"] = np.round(nearest_km, 4)
    out["nearest_hospital_km"] = np.round(hosp_km, 4)
    out["nearest_water_km"] = np.round(water_km, 4)
    out["kiln_violation"] = kiln_violations
    out["hospital_violation"] = hosp_violations
    out["water_violation"] = water_violations
    out["overall_compliant"] = overall_compliant
    out["compliance_category"] = compliance_categories
    out["violation_details"] = violation_details
    total = len(out)
    fully_compliant = int(out["overall_compliant"].sum())
    one_violation = int((out["compliance_category"].str.startswith("1 Violation")).sum())
    two_violations = int((out["compliance_category"].str.startswith("2 Violations")).sum())
    three_violations = int((out["compliance_category"] == "3 Violations (all)").sum())
    print(f"Computing detailed state-wise summary...")
    try:
        state_groups = out.groupby("state")
        detailed_summary_data = []
        for state, group in state_groups:
            state_data = {
                'state': state,
                'total_kilns': len(group),
                'fully_compliant': int(group["overall_compliant"].sum()),
                'one_violation': int((group["compliance_category"].str.startswith("1 Violation")).sum()),
                'two_violations': int((group["compliance_category"].str.startswith("2 Violations")).sum()),
                'three_violations': int((group["compliance_category"] == "3 Violations (all)").sum()),
                'kiln_violations': int(group["kiln_violation"].sum()),
                'hospital_violations': int(group["hospital_violation"].sum()),
                'water_violations': int(group["water_violation"].sum())
            }
            detailed_summary_data.append(state_data)
        state_summary_df = pd.DataFrame(detailed_summary_data)
        state_summary_df = state_summary_df.sort_values('total_kilns', ascending=False)
        print("Detailed state summary computed successfully")
    except Exception as e:
        print(f"Error computing detailed state summary: {e}")
        state_summary_df = pd.DataFrame({
            'state': ['Unknown'],
            'total_kilns': [len(out)],
            'fully_compliant': [fully_compliant],
            'one_violation': [one_violation],
            'two_violations': [two_violations],
            'three_violations': [three_violations],
            'kiln_violations': [int(kiln_violations.sum())],
            'hospital_violations': [int(hosp_violations.sum())],
            'water_violations': [int(water_violations.sum())]
        })
    summary_table = "| State | Total | Fully Compliant | 1 Violation | 2 Violations | 3 Violations | Kiln Violations | Hospital Violations | Water Violations |\n"
    summary_table += "|-------|-------|-----------------|-------------|---------------|---------------|-----------------|-------------------|------------------|\n"
    for _, row in state_summary_df.iterrows():
        summary_table += (
            f"| {row['state']} | {row['total_kilns']} | {row['fully_compliant']} | "
            f"{row['one_violation']} | {row['two_violations']} | {row['three_violations']} | "
            f"{row['kiln_violations']} | {row['hospital_violations']} | {row['water_violations']} |\n"
        )
    ctr_lat, ctr_lon = _center_from_points(kiln_latlon)
    m = folium.Map(
        location=[ctr_lat, ctr_lon],
        zoom_start=5,
        control_scale=True,
        tiles="CartoDB positron"
    )
    if states_gdf is not None:
        try:
            folium.GeoJson(
                states_gdf,
                name="State Boundaries",
                style_function=lambda x: {
                    'fillColor': '#00000000',
                    'color': '#333333',
                    'weight': 2,
                    'fillOpacity': 0,
                    'opacity': 0.8
                }
            ).add_to(m)
        except Exception as e:
            print(f"Error adding state boundaries to map: {e}")
    colors = {
        'Fully Compliant': '#16a34a',
        '1 Violation': '#eab308',
        '2 Violations': '#f97316',
        '3 Violations': '#dc2626'
    }
    for state, group_df in out.groupby("state"):
        if state == "Unknown":
            continue
        categories = group_df['compliance_category'].unique()
        for category in categories:
            category_type = category.split(' (')[0] if '(' in category else category
            color = colors.get(category_type, '#6b7280')
            fg = folium.FeatureGroup(
                name=f"{state}: {category} ({len(group_df[group_df['compliance_category'] == category])})",
                show=True
            )
            sub_df = group_df[group_df['compliance_category'] == category]
            for _, r in sub_df.iterrows():
                tooltip_text = (
                    f"<b>State:</b> {state}<br>"
                    f"<b>Status:</b> {r['compliance_category']}<br>"
                    f"<b>Violations:</b> {r['violation_details'] if r['violation_details'] else 'None'}<br>"
                    f"<b>Nearest kiln:</b> {r.get('nearest_kiln_km', 'N/A')} km "
                    f"{'❌' if r['kiln_violation'] else '✅'}<br>"
                    f"<b>Nearest hospital:</b> {r.get('nearest_hospital_km', 'N/A')} km "
                    f"{'❌' if r['hospital_violation'] else '✅'}<br>"
                    f"<b>Nearest water:</b> {r.get('nearest_water_km', 'N/A')} km "
                    f"{'❌' if r['water_violation'] else '✅'}"
                )
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=tooltip_text,
                ).add_to(fg)
            if len(sub_df) > 0:
                m.add_child(fg)
    unknown_kilns = out[out["state"] == "Unknown"]
    if len(unknown_kilns) > 0:
        for category in unknown_kilns['compliance_category'].unique():
            category_type = category.split(' (')[0] if '(' in category else category
            color = colors.get(category_type, '#6b7280')
            fg_unknown = folium.FeatureGroup(
                name=f"Unknown State: {category} ({len(unknown_kilns[unknown_kilns['compliance_category'] == category])})",
                show=True
            )
            sub_df = unknown_kilns[unknown_kilns['compliance_category'] == category]
            for _, r in sub_df.iterrows():
                tooltip_text = (
                    f"<b>State:</b> Unknown<br>"
                    f"<b>Status:</b> {r['compliance_category']}<br>"
                    f"<b>Violations:</b> {r['violation_details'] if r['violation_details'] else 'None'}<br>"
                    f"<b>Nearest kiln:</b> {r.get('nearest_kiln_km', 'N/A')} km "
                    f"{'❌' if r['kiln_violation'] else '✅'}<br>"
                    f"<b>Nearest hospital:</b> {r.get('nearest_hospital_km', 'N/A')} km "
                    f"{'❌' if r['hospital_violation'] else '✅'}<br>"
                    f"<b>Nearest water:</b> {r.get('nearest_water_km', 'N/A')} km "
                    f"{'❌' if r['water_violation'] else '✅'}"
                )
                folium.CircleMarker(
                    location=[r["lat"], r["lon"]],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    tooltip=tooltip_text,
                ).add_to(fg_unknown)
            m.add_child(fg_unknown)
    if add_heatmap and len(out) > 0:
        HeatMap(out[["lat", "lon"]].values.tolist(), name="Kiln density").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    map_html = m._repr_html_()
    summary = (
        f"**Total kilns (India): {total} | Fully Compliant: {fully_compliant} | "
        f"1 Violation: {one_violation} | 2 Violations: {two_violations} | "
        f"3 Violations: {three_violations}**\n\n"
        f"**Compliance Rules:**\n"
        f"- ≥{kiln_km_thresh} km from nearest kiln\n"
        f"- ≥{hosp_km_thresh} km from hospital\n"
        f"- ≥{water_km_thresh} km from water body\n\n"
        f"### Detailed Per-State Summary\n\n"
        f"{summary_table}"
    )
    buf = io.BytesIO()
    out.to_csv(buf, index=False)
    buf.seek(0)
    state_list = ["All States"] + sorted(out["state"].unique().tolist())
    return map_html, summary, buf.read(), out, state_list

def filter_data_by_state(full_data: pd.DataFrame, selected_state: str) -> Tuple[str, bytes]:
    if selected_state is None or selected_state == "All States":
        # Use full data without filtering
        filtered_data = full_data
        title_prefix = "All States"
    else:
        filtered_data = full_data[full_data["state"] == selected_state]
        title_prefix = selected_state
        if len(filtered_data) == 0:
            return f"No data found for {selected_state}", b""
    # Then summarize filtered_data as usual...

    total = len(filtered_data)
    fully_compliant = int(filtered_data["overall_compliant"].sum())
    one_violation = int((filtered_data["compliance_category"].str.startswith("1 Violation")).sum())
    two_violations = int((filtered_data["compliance_category"].str.startswith("2 Violations")).sum())
    three_violations = int((filtered_data["compliance_category"] == "3 Violations (all)").sum())
    kiln_violations = int(filtered_data["kiln_violation"].sum())
    hospital_violations = int(filtered_data["hospital_violation"].sum())
    water_violations = int(filtered_data["water_violation"].sum())
    summary = (
        f"## {title_prefix} - Detailed Compliance Report\n\n"
        f"**Total kilns: {total}**\n\n"
        f"### Compliance Overview\n"
        f"- **Fully Compliant:** {fully_compliant} ({fully_compliant/total*100:.1f}%)\n"
        f"- **1 Violation:** {one_violation} ({one_violation/total*100:.1f}%)\n"
        f"- **2 Violations:** {two_violations} ({two_violations/total*100:.1f}%)\n"
        f"- **3 Violations:** {three_violations} ({three_violations/total*100:.1f}%)\n\n"
        f"### Individual Violation Types\n"
        f"- **Kiln Distance Violations:** {kiln_violations}\n"
        f"- **Hospital Distance Violations:** {hospital_violations}\n"
        f"- **Water Body Distance Violations:** {water_violations}\n\n"
    )
    category_summary = filtered_data['compliance_category'].value_counts().sort_index()
    summary += "### Detailed Breakdown\n\n"
    summary += "| Compliance Category | Count | Percentage |\n"
    summary += "|-------------------|-------|------------|\n"
    for category, count in category_summary.items():
        percentage = count/total*100
        summary += f"| {category} | {count} | {percentage:.1f}% |\n"
    csv_buffer = io.BytesIO()
    filtered_data.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return summary, csv_buffer.read()

# ------------------------------
# Static visualization (Matplotlib) - FIXED VERSION
# ------------------------------

def enhance_axes_styling(ax, title, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=18, fontweight='bold', pad=25, color='#1e293b')
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=14, fontweight='600', color='#475569', labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=14, fontweight='600', color='#475569', labelpad=10)
    ax.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='#cbd5e1')
    ax.set_facecolor('#fefefe')
    for spine in ax.spines.values():
        spine.set_color('#e2e8f0')
        spine.set_linewidth(1)
    ax.tick_params(axis='both', which='major', labelsize=11, colors='#64748b', length=4)

def load_india_boundaries():
    """Load India state boundaries from online source or fallback to simple outline"""
    url = "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            geojson_data = response.json()
            boundaries = []
            for feature in geojson_data['features']:
                geom_type = feature['geometry']['type']
                if geom_type == 'Polygon':
                    boundaries.append(feature['geometry']['coordinates'][0])
                elif geom_type == 'MultiPolygon':
                    for polygon in feature['geometry']['coordinates']:
                        boundaries.append(polygon[0])
            return boundaries
    except Exception as e:
        print(f"Failed to load online boundaries: {e}")
    
    # Fallback: Simple India outline
    return [[(68.0, 37.0), (97.0, 37.0), (97.0, 28.0), (92.0, 21.0), (88.0, 22.0), (85.0, 22.0),
             (83.0, 18.0), (80.0, 13.0), (77.0, 8.0), (73.0, 8.0), (68.0, 23.0), (68.0, 37.0)]]

def plot_state_boundaries(ax, alpha=0.3, linewidth=0.8, color='#2d3748'):
    """Plot India state boundaries on the given axis"""
    boundaries = load_india_boundaries()
    for boundary in boundaries:
        if len(boundary) < 3:
            continue
        lons, lats = zip(*boundary)
        # Ensure closed polygon
        if (lons[0], lats[0]) != (lons[-1], lats[-1]):
            lons = lons + (lons[0],)
            lats = lats + (lats[0],)
        ax.plot(lons, lats, color=color, linewidth=linewidth, alpha=alpha, zorder=1)

def make_detailed_scatter_figure(df: pd.DataFrame, plot_type: str = "pie"):
    """Create detailed visualization figures for compliance data"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.patches import Circle

    STATE_NAME_MAPPING = {
        'ANDAMAN & NICOBAR ISLANDS': 'Andaman and Nicobar Islands',
        'ANDAMAN AND NICOBAR ISLANDS': 'Andaman and Nicobar Islands',
        'ARUNACHAL PRADESH': 'Arunachal Pradesh',
        'ASSAM': 'Assam',
        'BIHAR': 'Bihar',
        'CHANDIGARH': 'Chandigarh',
        'CHHATTISGARH': 'Chhattisgarh',
        'DADRA & NAGAR HAVELI': 'Dadra and Nagar Haveli',
        'DADRA AND NAGAR HAVELI': 'Dadra and Nagar Haveli',
        'DAMAN & DIU': 'Daman and Diu',
        'DAMAN AND DIU': 'Daman and Diu',
        'DELHI': 'Delhi',
        'NCT OF DELHI': 'Delhi',
        'NATIONAL CAPITAL TERRITORY OF DELHI': 'Delhi',
        'GOA': 'Goa',
        'GUJARAT': 'Gujarat',
        'HARYANA': 'Haryana',
        'HIMACHAL PRADESH': 'Himachal Pradesh',
        'JAMMU AND KASHMIR': 'Jammu and Kashmir',
        'JHARKHAND': 'Jharkhand',
        'KARNATAKA': 'Karnataka',
        'KERALA': 'Kerala',
        'LADAKH': 'Ladakh',
        'LAKSHADWEEP': 'Lakshadweep',
        'MADHYA PRADESH': 'Madhya Pradesh',
        'MAHARASHTRA': 'Maharashtra',
        'MANIPUR': 'Manipur',
        'MEGHALAYA': 'Meghalaya',
        'MIZORAM': 'Mizoram',
        'NAGALAND': 'Nagaland',
        'ODISHA': 'Odisha',
        'ORISSA': 'Odisha',
        'PUDUCHERRY': 'Puducherry',
        'PONDICHERRY': 'Puducherry',
        'PUNJAB': 'Punjab',
        'RAJASTHAN': 'Rajasthan',
        'SIKKIM': 'Sikkim',
        'TAMIL NADU': 'Tamil Nadu',
        'TELANGANA': 'Telangana',
        'TRIPURA': 'Tripura',
        'UTTAR PRADESH': 'Uttar Pradesh',
        'UTTARAKHAND': 'Uttarakhand',
        'UTTARANCHAL': 'Uttarakhand',
        'WEST BENGAL': 'West Bengal'
    }

    def standardize_state_names(df):
        """Standardize state names to consistent format"""
        if 'state' not in df.columns:
            return df
        df_copy = df.copy()
        df_copy['state_upper'] = df_copy['state'].str.upper()
        df_copy['state_standardized'] = df_copy['state_upper'].map(STATE_NAME_MAPPING)
        df_copy['state_standardized'] = df_copy['state_standardized'].fillna(df_copy['state'].str.title())
        df_copy['state'] = df_copy['state_standardized']
        return df_copy.drop(['state_upper', 'state_standardized'], axis=1)

    # Color scheme for compliance categories
    violation_colors = {
        'Fully Compliant': '#059669',
        '1 Violation': '#facc15',
        '2 Violations': '#f97316',
        '3 Violations': '#dc2626'
    }
    
    # Marker styles for different compliance categories
    markers = {
        'Fully Compliant': 'o', 
        '1 Violation': '^',
        '2 Violations': 's', 
        '3 Violations': 'X'
    }
    
    # Ensure proper state name standardization
    df = standardize_state_names(df)

    if plot_type == "pie":
        # Create pie chart for compliance distribution
        category_counts = df['compliance_category'].value_counts()
        total = sum(category_counts.values)
        
        # Prepare data for pie chart
        category_info = []
        for i, (category, count) in enumerate(category_counts.items()):
            percentage = (count / total) * 100
            color = violation_colors.get(category.split(' (')[0], '#6b7280')
            category_info.append((category, count, percentage, color, i))
        
        # Sort by percentage for better visual organization
        category_info = sorted(category_info, key=lambda x: x[2])
        color_sorted = [c[3] for c in category_info]
        count_sorted = [c[1] for c in category_info]
        
        fig, ax = plt.subplots(figsize=(9, 7), dpi=120)
        
        # Create donut chart
        wedges, _ = ax.pie(
            count_sorted,
            colors=color_sorted,
            startangle=90,
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=3)
        )
        
        # Add center circle for donut effect
        centre_circle = Circle((0, 0), 0.50, fc='#f8fafc', linewidth=3, edgecolor='#e2e8f0')
        ax.add_artist(centre_circle)
        
        # Add center text with key metrics
        total_kilns = len(df)
        compliance_count = df['overall_compliant'].sum()
        compliance_rate = (compliance_count / total_kilns) * 100 if total_kilns > 0 else 0
        
        ax.text(0, 0.13, f"{total_kilns:,}", ha='center', va='center',
                fontsize=22, fontweight='bold', color='#1e293b')
        ax.text(0, 0, "Total Kilns", ha='center', va='center',
                fontsize=13, color='#64748b')
        ax.text(0, -0.17, f"{compliance_rate:.1f}% Compliant",
                ha='center', va='center', fontsize=15, fontweight='600',
                color=violation_colors['Fully Compliant'])
        
        ax.set_title("Compliance Status Distribution",
                     fontsize=18, fontweight='bold', pad=24, color='#1e293b')
        
        # Create legend
        legend_labels = [f"{c[0]} ({c[1]}) - {c[2]:.1f}%" for c in category_info]
        ax.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        
        plt.tight_layout()
        return fig

    elif plot_type == "scatter":
        # Create geographic scatter plot
        fig, ax = plt.subplots(figsize=(12, 10), dpi=120)
        
        # Standardize state names
        df = standardize_state_names(df)
        
        # Calculate aspect ratio for better visualization
        if 'lat' in df.columns and 'lon' in df.columns and len(df) > 0:
            lat_range = df['lat'].max() - df['lat'].min()
            lon_range = df['lon'].max() - df['lon'].min()
            aspect_ratio = max(0.6, min(1.4, lon_range / lat_range)) if lat_range > 0 else 1
        else:
            aspect_ratio = 1
        
        # Add India state boundaries
        plot_state_boundaries(ax)
        
        # Calculate appropriate marker size based on data density
        total_points = len(df)
        base_size = max(60, min(150, 2000/total_points))
        
        # Plot points by compliance category
        for i, category in enumerate(sorted(df['compliance_category'].unique())):
            cat_type = category.split(' (')[0] if '(' in category else category
            color = violation_colors.get(cat_type, '#6b7280')
            marker = markers.get(cat_type, 'o')
            
            subset = df[df['compliance_category'] == category]
            
            if not subset.empty and 'lat' in subset.columns and 'lon' in subset.columns:
                ax.scatter(subset['lon'], subset['lat'],
                          c=color, marker=marker, s=base_size,
                          label=f"{category} ({len(subset)})",
                          alpha=0.8, edgecolors='white', linewidth=1.5,
                          zorder=5+i)
        
        # Set proper axis limits with padding
        if not df.empty and 'lat' in df.columns and 'lon' in df.columns:
            padding_lon = (df['lon'].max() - df['lon'].min()) * 0.05
            padding_lat = (df['lat'].max() - df['lat'].min()) * 0.05
            ax.set_xlim(df['lon'].min() - padding_lon, df['lon'].max() + padding_lon)
            ax.set_ylim(df['lat'].min() - padding_lat, df['lat'].max() + padding_lat)
        
        # Apply styling
        enhance_axes_styling(ax, "Geographic Distribution of Compliance Status",
                            "Longitude (°)", "Latitude (°)")
        
        # Add legend
        legend = ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                          frameon=True, fancybox=True, shadow=True,
                          title="Compliance Categories", title_fontsize=12)
        legend.get_frame().set_facecolor('#ffffff')
        legend.get_frame().set_alpha(0.95)
        
        plt.tight_layout()
        return fig

    elif plot_type == "bar":
        # Create bar chart for violation types
        violation_data = {
            'Kiln\nDistance': int(df['kiln_violation'].sum()),
            'Hospital\nDistance': int(df['hospital_violation'].sum()),
            'Water\nDistance': int(df['water_violation'].sum())
        }
        
        colors = ['#dc2626', '#f97316', '#eab308']
        x_pos = np.arange(len(violation_data))
        
        fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
        
        bars = ax.bar(x_pos, violation_data.values(),
                      color=colors, alpha=0.9, width=0.7,
                      edgecolor='white', linewidth=2)
        
        # Customize x-axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(violation_data.keys(), fontsize=13, fontweight='600')
        
        # Add value labels on bars
        max_val = max(violation_data.values()) if violation_data.values() else 1
        for i, (bar, value) in enumerate(zip(bars, violation_data.values())):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + max_val * 0.01,
                    f'{value:,}', ha='center', va='bottom',
                    fontweight='bold', fontsize=13, color='#1e293b')
        
        # Set y-axis limit with padding
        ax.set_ylim(0, max_val * 1.15)
        
        # Apply styling
        enhance_axes_styling(ax, "Distribution of Violation Types", 
                            None, "Number of Violations")
        
        plt.tight_layout()
        return fig

    elif plot_type == "state":
        # Create state-wise horizontal stacked bar chart
        state_data = []
        for state, group in df.groupby("state"):
            state_data.append({
                'state': state,
                'fully_compliant': int(group['overall_compliant'].sum()),
                'one_violation': int(group['compliance_category'].str.startswith("1 Violation").sum()),
                'two_violations': int(group['compliance_category'].str.startswith("2 Violations").sum()),
                'three_violations': int((group['compliance_category'] == "3 Violations (all)").sum()),
                'total': len(group)
            })
        
        state_df = pd.DataFrame(state_data).sort_values('total', ascending=True)
        
        # Calculate figure size based on number of states
        n_states = len(state_df)
        y_pos = np.arange(len(state_df))
        bar_height = max(0.6, min(0.9, 15/n_states)) if n_states > 0 else 0.8
        
        fig, ax = plt.subplots(figsize=(12, max(6, n_states * 0.4)), dpi=120)
        
        # Create stacked horizontal bars
        ax.barh(y_pos, state_df['fully_compliant'], bar_height,
                label='Fully Compliant', color=violation_colors['Fully Compliant'],
                alpha=0.9, edgecolor='white', linewidth=0.5)
        
        ax.barh(y_pos, state_df['one_violation'], bar_height,
                left=state_df['fully_compliant'], label='1 Violation',
                color=violation_colors['1 Violation'], alpha=0.9,
                edgecolor='white', linewidth=0.5)
        
        ax.barh(y_pos, state_df['two_violations'], bar_height,
                left=state_df['fully_compliant'] + state_df['one_violation'],
                label='2 Violations', color=violation_colors['2 Violations'],
                alpha=0.9, edgecolor='white', linewidth=0.5)
        
        ax.barh(y_pos, state_df['three_violations'], bar_height,
                left=state_df['fully_compliant'] + state_df['one_violation'] + state_df['two_violations'],
                label='3 Violations', color=violation_colors['3 Violations'],
                alpha=0.9, edgecolor='white', linewidth=0.5)
        
        # Customize y-axis labels (truncate long state names)
        state_labels = [s if len(s) < 20 else s[:17] + '...' for s in state_df['state']]
        ax.set_yticks(y_pos)
        ax.set_yticklabels(state_labels, fontsize=12, fontweight='500')
        
        # Apply styling
        enhance_axes_styling(ax, "State-wise Compliance Breakdown", 
                            "Number of Kilns", None)
        
        # Add legend
        ax.legend(loc='lower right', frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        return fig

    else:
        # Default fallback to pie chart
        return make_detailed_scatter_figure(df, plot_type="pie")

def make_all_compliance_figures(df):
    """Generate all four types of compliance visualization figures"""
    try:
        return (
            make_detailed_scatter_figure(df, plot_type="pie"),
            make_detailed_scatter_figure(df, plot_type="scatter"),
            make_detailed_scatter_figure(df, plot_type="bar"),
            make_detailed_scatter_figure(df, plot_type="state"),
        )
    except Exception as e:
        print(f"Error creating figures: {e}")
        # Return None figures if there's an error
        return None, None, None, None

## Gradio interface
with gr.Blocks(title="Enhanced Brick Kiln Compliance Monitor") as demo:
    gr.Markdown(
        "## Enhanced Compliance Monitoring for Brick Kilns\n"
        "Upload CSVs, set thresholds, and view detailed compliance analysis per state.\n"
        "- **New Features:** Detailed violation categories, state-wise filtering, enhanced CSV output\n"
        "- **Kilns CSV** must include columns: `lat, lon` (WGS84).\n"
        "- Hospitals CSV: `Latitude, Longitude` or `lat, lon`.\n"
        "- Waterways CSV: points (`lat, lon`) or WKT.LineString/MultiLineString in `geometry`.\n"
        "- State boundaries: drop GeoJSON files into folder; set folder path below.\n"
        "- **Compliance Categories:** Fully Compliant, 1 Violation, 2 Violations, 3 Violations"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Data Input")
            use_demo = gr.Checkbox(value=True, label="Use bundled demo data (skip uploads)")
            kilns_csv = gr.File(label="Kilns CSV (required if demo OFF)", file_types=[".csv"])
            hospitals_csv = gr.File(label="Hospitals CSV (optional)", file_types=[".csv"])
            waterways_csv = gr.File(label="Waterways CSV or WKT (optional)", file_types=[".csv"])
            states_geojson_folder = gr.Textbox(
                value="C:/VSCode/Internship/Compliance_Monitoring/welldone/india_shapefiles/",
                label="States GeoJSON Folder"
            )
            gr.Markdown("### Compliance Thresholds (km)")
            kiln_thresh = gr.Number(value=1.0, label="Min distance to nearest kiln (km)")
            hosp_thresh = gr.Number(value=0.8, label="Min distance to hospital (km)")
            water_thresh = gr.Number(value=0.5, label="Min distance to water body (km)")
            gr.Markdown("### Map Options")
            add_heatmap = gr.Checkbox(value=False, label="Add heatmap layer")
            cluster_points = gr.Checkbox(value=True, label="Cluster markers for speed")
            run_btn = gr.Button("Compute & Map", variant="primary")

        with gr.Column(scale=2):
            fmap = gr.HTML(label="Interactive Map with Detailed Categories")
            summary = gr.Markdown(label="Overall Summary")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### State-wise Analysis")
            state_dropdown = gr.Dropdown(
                choices=["All States"],
                value="All States",
                label="Select State for Detailed Analysis",
                interactive=True
            )
            filter_btn = gr.Button("Show State Report", variant="secondary")
            state_plot_btn = gr.Button("Show State Plots", variant="secondary")
        with gr.Column(scale=2):
            state_summary = gr.Markdown(label="State-specific Analysis")

    with gr.Row():
        pie_plot = gr.Plot(label="Compliance Pie/Donut Chart")
    with gr.Row():
        scatter_plot = gr.Plot(label="Geographic Scatter Plot")
    with gr.Row():
        bar_plot = gr.Plot(label="Violation Types Bar Chart")
    with gr.Row():
        state_bar_plot = gr.Plot(label="State-wise Compliance Bar Chart")

    with gr.Row():
        gr.Markdown("### Download Options")
        download_csv = gr.File(label="Download Full Dataset CSV", interactive=False)
        download_state_csv = gr.File(label="Download State-specific CSV", interactive=False)

    def save_bytes_to_tempfile(data_bytes, prefix="tempfile_", suffix=".csv"):
        """Save bytes data to a temporary file and return the path"""
        import tempfile
        import os
        fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
        with os.fdopen(fd, 'wb') as f:
            f.write(data_bytes)
        return path

    global_data = None

    def _run(use_demo_flag, k, h, w, states_folder, kt, ht, wt, heat, cluster):
        """Main function to run compliance analysis"""
        global global_data
        try:
            if use_demo_flag:
                k = "data/kilns_clean.csv"
                h = "data/hospitals.csv" if os.path.exists("data/hospitals.csv") else None
                w = "data/waterways_wkt.csv" if os.path.exists("data/waterways_wkt.csv") else None
            
            map_html, summary_text, csv_bytes, full_data, state_list = compute_compliance(
                k, h, w, float(kt), float(ht), float(wt), bool(heat), bool(cluster),
                states_geojson_folder=states_folder
            )
            
            global_data = full_data
            pie, scatter, bar, state = make_all_compliance_figures(full_data)
            csv_path = save_bytes_to_tempfile(csv_bytes, prefix="full_dataset_")
            
            return (
                map_html,
                summary_text,
                pie,
                scatter,
                bar,
                state,
                gr.update(choices=state_list),
                gr.update(value="All States"),
                csv_path,
                "",
                None,
            )
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return (
                f"<p style='color:red'>{error_msg}</p>",
                error_msg,
                None,
                None,
                None,
                None,
                gr.update(choices=["All States"]),
                gr.update(value="All States"),
                None,
                "",
                None,
            )

    run_btn.click(
        _run,
        inputs=[use_demo, kilns_csv, hospitals_csv, waterways_csv, states_geojson_folder,
                kiln_thresh, hosp_thresh, water_thresh, add_heatmap, cluster_points],
        outputs=[
            fmap,
            summary,
            pie_plot,
            scatter_plot,
            bar_plot,
            state_bar_plot,
            state_dropdown,  # updates choices here
            state_dropdown,  # updates value here
            download_csv,
            state_summary,
            download_state_csv,
        ],
    )

    def _filter_by_state(selected_state):
        """Filter data by selected state and generate report"""
        global global_data
        if global_data is None:
            return "Please run the analysis first.", None
        try:
            if selected_state == "All States":
                # Provide full dataset CSV for "All States"
                state_summary_text, state_csv_bytes = filter_data_by_state(global_data, selected_state=None)
                state_csv_path = save_bytes_to_tempfile(state_csv_bytes, prefix="all_states_data_") if len(state_csv_bytes) > 0 else None
                return state_summary_text, state_csv_path

            state_summary_text, state_csv_bytes = filter_data_by_state(global_data, selected_state)
            state_csv_path = save_bytes_to_tempfile(state_csv_bytes, prefix="state_data_") if len(state_csv_bytes) > 0 else None
            return state_summary_text, state_csv_path
        except Exception as e:
            error_msg = f"Error filtering data: {str(e)}"
            return error_msg, None

    def _state_plot(selected_state):
        """Generate plots for selected state"""
        global global_data
        if global_data is None:
            return None, None, None, None
        
        try:
            if selected_state == "All States":
                return make_all_compliance_figures(global_data)
            
            filtered = global_data[global_data["state"] == selected_state]
            if len(filtered) == 0:
                return None, None, None, None
            
            return make_all_compliance_figures(filtered)
        except Exception as e:
            print(f"Error creating state plots: {e}")
            return None, None, None, None

    filter_btn.click(
        _filter_by_state,
        inputs=[state_dropdown],
        outputs=[state_summary, download_state_csv]
    )

    state_plot_btn.click(
        _state_plot,
        inputs=[state_dropdown],
        outputs=[pie_plot, scatter_plot, bar_plot, state_bar_plot]
    )


if __name__ == "__main__":
    demo.launch()