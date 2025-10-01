import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
from shapely.geometry import Point
from data_utils import _read_csv, _ensure_cols
from geo_utils import load_states_geojson_folder, assign_kilns_to_states

EARTH_RADIUS_KM = 6371.0088

def _to_radians(latlon: np.ndarray) -> np.ndarray:
    return np.radians(latlon.astype(float))

def _balltree_haversine_min_km(a_latlon_deg: np.ndarray, b_latlon_deg: np.ndarray):
    if len(a_latlon_deg) == 0 or len(b_latlon_deg) == 0:
        return np.array([]), np.array([], dtype=int)
    a_rad = _to_radians(a_latlon_deg[:, [0,1]])[:, ::-1]
    b_rad = _to_radians(b_latlon_deg[:, [0,1]])[:, ::-1]
    tree = BallTree(b_rad, metric="haversine")
    dist_rad, idx = tree.query(a_rad, k=1)
    dist_km = dist_rad.flatten() * EARTH_RADIUS_KM
    return dist_km, idx.flatten()

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

def _lines_to_vertices_df(lines_like: pd.DataFrame) -> pd.DataFrame:
    import shapely.wkt
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

def compute_compliance(
    kilns_csv,
    hospitals_csv=None,
    waterways_csv=None,
    kiln_km_thresh: float = 1.0,
    hosp_km_thresh: float = 0.8,
    water_km_thresh: float = 0.5,
    add_heatmap: bool = False,
    cluster_points: bool = True,
    states_geojson_folder: str = "india_shapefiles/"
):
    kilns = _read_csv(kilns_csv)
    _ensure_cols(kilns, ["lat", "lon"], "Kilns CSV")
    try:
        states_gdf = load_states_geojson_folder(states_geojson_folder)
    except Exception:
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
    # ...rest of your summary and map logic remains in app.py...
    return out