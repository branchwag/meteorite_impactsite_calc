import numpy as np
import json

# =============================================================
#  METEORITE STREWN FIELD CALCULATOR
#  Fill in your data below, then run the script.
#  Output: console summary + strewn_field.geojson
# =============================================================

# --- RADAR HITS (lat, lon, altitude in meters) ---
# Add as many as you have. Minimum 1.
RADAR_HITS = [
    {"lat": 34.10, "lon": -118.30, "alt": 50000},
    {"lat": 34.08, "lon": -118.28, "alt": 40000},
    {"lat": 34.06, "lon": -118.26, "alt": 30000},
]

# --- GROUND OBSERVATIONS (optional, used to confirm trajectory) ---
# Leave empty list [] if not available.
GROUND_OBS = [
    {"lat": 34.0522, "lon": -118.2437, "alt": 100, "azimuth": 45.0,  "elevation": 30.0},
    {"lat": 34.1522, "lon": -118.1437, "alt": 150, "azimuth": 220.0, "elevation": 25.0},
]

# --- WIND DATA ---
# Option A: single average wind (simpler)
WIND_MODE = "single"   # set to "single" or "layers"
WIND_SINGLE = {
    "speed_ms":    10.0,   # wind speed in meters per second
    "direction_deg": 270.0 # direction wind is coming FROM (meteorological convention)
                           # 0=from North, 90=from East, 180=from South, 270=from West
}

# Option B: wind layers (more accurate)
# Each layer applies from its altitude down to the next layer's altitude.
WIND_LAYERS = [
    {"alt_m": 50000, "speed_ms": 20.0, "direction_deg": 260.0},
    {"alt_m": 30000, "speed_ms": 15.0, "direction_deg": 270.0},
    {"alt_m": 10000, "speed_ms":  8.0, "direction_deg": 280.0},
    {"alt_m":     0, "speed_ms":  5.0, "direction_deg": 290.0},
]

# --- FRAGMENT MASS RANGES (grams) ---
# Each entry: (label, mass_grams)
# Heavier = lands closer to start of trajectory (higher altitude entry point)
# Lighter = carried further downwind
MASS_CLASSES = [
    ("1kg+",   1000),
    ("500g",    500),
    ("100g",    100),
    ("10g",      10),
    ("1g",        1),
]

# --- STREWN FIELD WIDTH (meters) ---
# Half-width of the ellipse perpendicular to trajectory
STREWN_WIDTH_M = 5000

# =============================================================
#  CONSTANTS & HELPERS
# =============================================================

EARTH_RADIUS = 6_371_000.0

def lla_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    r = EARTH_RADIUS + alt_m
    return np.array([
        r * np.cos(lat) * np.cos(lon),
        r * np.cos(lat) * np.sin(lon),
        r * np.sin(lat)
    ])

def ecef_to_lla(ecef):
    x, y, z = ecef
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    alt = np.sqrt(x**2 + y**2 + z**2) - EARTH_RADIUS
    return lat, lon, alt

def enu_to_ecef_rotation(lat_deg, lon_deg):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    return np.array([
        [-np.sin(lon),               np.cos(lon),              0          ],
        [-np.sin(lat)*np.cos(lon),  -np.sin(lat)*np.sin(lon),  np.cos(lat)],
        [ np.cos(lat)*np.cos(lon),   np.cos(lat)*np.sin(lon),  np.sin(lat)],
    ])

def az_el_to_enu(az_deg, el_deg):
    az, el = np.radians(az_deg), np.radians(el_deg)
    return np.array([np.cos(el)*np.sin(az), np.cos(el)*np.cos(az), np.sin(el)])

def observation_to_plane(obs):
    point = lla_to_ecef(obs["lat"], obs["lon"], obs["alt"])
    R     = enu_to_ecef_rotation(obs["lat"], obs["lon"])
    ray   = R @ az_el_to_enu(obs["azimuth"], obs["elevation"])
    up    = R @ np.array([0.0, 0.0, 1.0])
    n     = np.cross(ray, up)
    return point, n / np.linalg.norm(n)

def wind_vector_enu(speed_ms, direction_deg):
    """Convert meteorological wind (direction it comes FROM) to ENU movement vector."""
    # Wind FROM 270 means air moves toward 90 (east)
    toward_deg = (direction_deg + 180) % 360
    az = np.radians(toward_deg)
    return np.array([np.sin(az), np.cos(az), 0.0])  # east, north, 0

def get_wind_at_altitude(alt_m):
    if WIND_MODE == "single":
        return WIND_SINGLE["speed_ms"], WIND_SINGLE["direction_deg"]
    layers = sorted(WIND_LAYERS, key=lambda x: x["alt_m"], reverse=True)
    for layer in layers:
        if alt_m >= layer["alt_m"]:
            return layer["speed_ms"], layer["direction_deg"]
    return layers[-1]["speed_ms"], layers[-1]["direction_deg"]

# =============================================================
#  TRAJECTORY FROM RADAR HITS
# =============================================================

def fit_trajectory_from_radar(hits):
    """Least-squares line fit through radar hit points in ECEF."""
    points = np.array([lla_to_ecef(h["lat"], h["lon"], h["alt"]) for h in hits])
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    direction = Vt[0]
    # Ensure direction points downward (negative z component roughly)
    if direction[2] > 0:
        direction = -direction
    return centroid, direction / np.linalg.norm(direction)

# =============================================================
#  TRAJECTORY FROM GROUND OBSERVATIONS
# =============================================================

def fit_trajectory_from_observations(obs_list):
    """Intersect planes from ground observations."""
    if len(obs_list) < 2:
        return None, None
    p1, n1 = observation_to_plane(obs_list[0])
    p2, n2 = observation_to_plane(obs_list[1])
    direction = np.cross(n1, n2)
    denom = np.dot(direction, direction)
    if denom < 1e-10:
        return None, None
    A = np.array([n1, n2, direction])
    b = np.array([np.dot(n1, p1), np.dot(n2, p2), 0.0])
    point = np.linalg.solve(A, b)
    return point, direction / np.linalg.norm(direction)

# =============================================================
#  WIND-CORRECTED FRAGMENT LANDING POSITIONS
# =============================================================

def fragment_landing(traj_point, traj_dir, mass_g):
    """
    Walk the trajectory downward, applying wind drift at each step.
    Mass affects terminal velocity: heavier = faster fall = less drift.
    Returns landing ECEF position.
    """
    # Rough terminal velocity scaling: v_t ~ mass^(1/6) * 50 m/s baseline
    terminal_v = 50.0 * (mass_g / 100.0) ** (1/6)

    pos = traj_point.copy()
    dt  = 1.0      # seconds per step
    max_steps = 300_000

    for _ in range(max_steps):
        _, _, alt = ecef_to_lla(pos)
        if alt <= 0:
            return pos

        # Move along trajectory
        pos = pos + traj_dir * terminal_v * dt

        # Apply wind drift in ENU, converted to ECEF
        lat, lon, _ = ecef_to_lla(pos)
        R = enu_to_ecef_rotation(lat, lon)
        spd, dirn = get_wind_at_altitude(alt)
        wind_enu  = wind_vector_enu(spd, dirn)
        pos = pos + R @ (wind_enu * dt)

    return pos  # fallback if never hit ground

def compute_strewn_field(traj_point, traj_dir, ref_lat, ref_lon):
    """
    Compute landing points for each mass class, then build ellipse polygons.
    Returns list of (label, center_lat, center_lon) and GeoJSON features.
    """
    landings = []
    for label, mass_g in MASS_CLASSES:
        ecef = fragment_landing(traj_point, traj_dir, mass_g)
        lat, lon, _ = ecef_to_lla(ecef)
        landings.append((label, mass_g, lat, lon))

    # Build GeoJSON
    features = []

    # Trajectory centerline
    pts = [[l[3], l[2]] for l in landings]  # [lon, lat]
    features.append({
        "type": "Feature",
        "properties": {"name": "Trajectory Centerline", "type": "centerline"},
        "geometry": {"type": "LineString", "coordinates": pts}
    })

    # Ellipse for each mass class
    R = enu_to_ecef_rotation(ref_lat, ref_lon)
    for label, mass_g, clat, clon, in [(l[0], l[1], l[2], l[3]) for l in landings]:
        ellipse_coords = make_ellipse(clat, clon, STREWN_WIDTH_M,
                                      STREWN_WIDTH_M * 1.5, traj_dir, R)
        features.append({
            "type": "Feature",
            "properties": {"name": f"{label} strewn zone", "mass_g": mass_g, "type": "ellipse"},
            "geometry": {"type": "Polygon", "coordinates": [ellipse_coords]}
        })

    # Landing point markers
    for label, mass_g, clat, clon in landings:
        features.append({
            "type": "Feature",
            "properties": {"name": f"{label} landing", "mass_g": mass_g, "type": "point"},
            "geometry": {"type": "Point", "coordinates": [clon, clat]}
        })

    return landings, {"type": "FeatureCollection", "features": features}

def make_ellipse(center_lat, center_lon, semi_minor_m, semi_major_m, traj_dir, R, n_pts=36):
    """Generate a polygon approximating an ellipse at a lat/lon center."""
    # Major axis along trajectory direction projected to ground
    proj = traj_dir.copy()
    proj[2] = 0
    if np.linalg.norm(proj) < 1e-6:
        proj = np.array([1.0, 0.0, 0.0])
    else:
        proj = proj / np.linalg.norm(proj)

    # Minor axis perpendicular
    up   = np.array([0.0, 0.0, 1.0])
    minor_ax = np.cross(proj, up)
    if np.linalg.norm(minor_ax) < 1e-6:
        minor_ax = np.array([0.0, 1.0, 0.0])
    else:
        minor_ax = minor_ax / np.linalg.norm(minor_ax)

    center_ecef = lla_to_ecef(center_lat, center_lon, 0)
    coords = []
    for i in range(n_pts + 1):
        angle = 2 * np.pi * i / n_pts
        offset_ecef = (np.cos(angle) * semi_major_m * proj +
                       np.sin(angle) * semi_minor_m * minor_ax)
        pt = center_ecef + offset_ecef
        lat, lon, _ = ecef_to_lla(pt)
        coords.append([lon, lat])
    return coords

# =============================================================
#  MAIN
# =============================================================

print("=" * 60)
print("  METEORITE STREWN FIELD CALCULATOR")
print("=" * 60)

# Determine trajectory
if len(RADAR_HITS) >= 2:
    print(f"\nFitting trajectory from {len(RADAR_HITS)} radar hits...")
    traj_point, traj_dir = fit_trajectory_from_radar(RADAR_HITS)
    print("Trajectory source: radar (primary)")
elif len(RADAR_HITS) == 1 and len(GROUND_OBS) >= 1:
    print("\nOne radar hit + ground observations — using radar point on obs plane...")
    _, traj_dir = fit_trajectory_from_observations(GROUND_OBS)
    traj_point = lla_to_ecef(RADAR_HITS[0]["lat"], RADAR_HITS[0]["lon"], RADAR_HITS[0]["alt"])
    print("Trajectory source: radar point + ground observation")
else:
    print("\nNo radar hits — using ground observations only...")
    traj_point, traj_dir = fit_trajectory_from_observations(GROUND_OBS)
    print("Trajectory source: ground observations")

if traj_point is None:
    print("ERROR: Could not determine trajectory. Check your inputs.")
    exit(1)

ref_lat, ref_lon, _ = ecef_to_lla(traj_point)
print(f"Trajectory reference point: {ref_lat:.4f}°, {ref_lon:.4f}°")
print(f"Trajectory direction vector: {traj_dir.round(4)}")
print(f"\nWind mode: {WIND_MODE}")

print("\nComputing fragment landing positions...")
landings, geojson = compute_strewn_field(traj_point, traj_dir, ref_lat, ref_lon)

print("\n" + "=" * 60)
print("  STREWN FIELD — FRAGMENT LANDING ESTIMATES")
print("=" * 60)
for label, mass_g, lat, lon in landings:
    print(f"  {label:<8}  {lat:.5f}°,  {lon:.5f}°   "
          f"https://maps.google.com/?q={lat:.5f},{lon:.5f}")

# Export GeoJSON
output_file = "strewn_field.geojson"
with open(output_file, "w") as f:
    json.dump(geojson, f, indent=2)

print(f"\nGeoJSON saved to: {output_file}")
print("Load it at: https://geojson.io  or import into Google Earth / Maps")
print("\nDone.")
