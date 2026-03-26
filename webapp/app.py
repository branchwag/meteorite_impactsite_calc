from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import json

app = Flask(__name__, static_folder="static")

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

def fit_trajectory_from_radar(hits):
    points = np.array([lla_to_ecef(h["lat"], h["lon"], h["alt"]) for h in hits])
    centroid = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - centroid)
    direction = Vt[0]
    if direction[2] > 0:
        direction = -direction
    return centroid, direction / np.linalg.norm(direction)

def fit_trajectory_from_observations(obs_list):
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

def wind_vector_enu(speed_ms, direction_deg):
    toward_deg = (direction_deg + 180) % 360
    az = np.radians(toward_deg)
    return np.array([np.sin(az), np.cos(az), 0.0])

def get_wind_at_altitude(alt_m, wind_mode, wind_single, wind_layers):
    if wind_mode == "single":
        return wind_single["speed_ms"], wind_single["direction_deg"]
    layers = sorted(wind_layers, key=lambda x: x["alt_m"], reverse=True)
    for layer in layers:
        if alt_m >= layer["alt_m"]:
            return layer["speed_ms"], layer["direction_deg"]
    return layers[-1]["speed_ms"], layers[-1]["direction_deg"]

MASS_CLASSES = [("1kg+", 1000), ("500g", 500), ("100g", 100), ("10g", 10), ("1g", 1)]
STREWN_WIDTH_M = 5000

def fragment_landing(traj_point, traj_dir, mass_g, wind_mode, wind_single, wind_layers):
    terminal_v = 50.0 * (mass_g / 100.0) ** (1/6)
    pos = traj_point.copy()
    for _ in range(300_000):
        _, _, alt = ecef_to_lla(pos)
        if alt <= 0:
            return pos
        pos = pos + traj_dir * terminal_v
        lat, lon, _ = ecef_to_lla(pos)
        R = enu_to_ecef_rotation(lat, lon)
        spd, dirn = get_wind_at_altitude(alt, wind_mode, wind_single, wind_layers)
        pos = pos + R @ (wind_vector_enu(spd, dirn))
    return pos

def make_ellipse(center_lat, center_lon, semi_minor_m, semi_major_m, traj_dir, n_pts=36):
    proj = traj_dir.copy(); proj[2] = 0
    proj = proj / np.linalg.norm(proj) if np.linalg.norm(proj) > 1e-6 else np.array([1.0,0,0])
    minor_ax = np.cross(proj, np.array([0.0,0.0,1.0]))
    minor_ax = minor_ax / np.linalg.norm(minor_ax) if np.linalg.norm(minor_ax) > 1e-6 else np.array([0.0,1.0,0.0])
    center_ecef = lla_to_ecef(center_lat, center_lon, 0)
    coords = []
    for i in range(n_pts + 1):
        angle = 2 * np.pi * i / n_pts
        offset = np.cos(angle) * semi_major_m * proj + np.sin(angle) * semi_minor_m * minor_ax
        lat, lon, _ = ecef_to_lla(center_ecef + offset)
        coords.append([lon, lat])
    return coords

def compute_strewn_field(traj_point, traj_dir, wind_mode, wind_single, wind_layers):
    landings = []
    for label, mass_g in MASS_CLASSES:
        ecef = fragment_landing(traj_point, traj_dir, mass_g, wind_mode, wind_single, wind_layers)
        lat, lon, _ = ecef_to_lla(ecef)
        landings.append({"label": label, "mass_g": mass_g, "lat": round(lat, 5), "lon": round(lon, 5)})

    features = []
    features.append({
        "type": "Feature",
        "properties": {"name": "Trajectory Centerline", "type": "centerline"},
        "geometry": {"type": "LineString", "coordinates": [[l["lon"], l["lat"]] for l in landings]}
    })
    for l in landings:
        ellipse = make_ellipse(l["lat"], l["lon"], STREWN_WIDTH_M, STREWN_WIDTH_M * 1.5, traj_dir)
        features.append({
            "type": "Feature",
            "properties": {"name": f'{l["label"]} strewn zone', "mass_g": l["mass_g"], "type": "ellipse"},
            "geometry": {"type": "Polygon", "coordinates": [ellipse]}
        })
        features.append({
            "type": "Feature",
            "properties": {"name": f'{l["label"]} landing', "mass_g": l["mass_g"], "type": "point"},
            "geometry": {"type": "Point", "coordinates": [l["lon"], l["lat"]]}
        })

    return landings, {"type": "FeatureCollection", "features": features}

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()
    try:
        radar_hits  = data.get("radar_hits", [])
        ground_obs  = data.get("ground_obs", [])
        wind_mode   = data.get("wind_mode", "single")
        wind_single = data.get("wind_single", {"speed_ms": 0, "direction_deg": 0})
        wind_layers = data.get("wind_layers", [])

        if len(radar_hits) >= 2:
            traj_point, traj_dir = fit_trajectory_from_radar(radar_hits)
            source = "radar"
        elif len(radar_hits) == 1 and len(ground_obs) >= 1:
            _, traj_dir = fit_trajectory_from_observations(ground_obs)
            traj_point = lla_to_ecef(radar_hits[0]["lat"], radar_hits[0]["lon"], radar_hits[0]["alt"])
            source = "radar+obs"
        elif len(radar_hits) == 1:
            traj_point = lla_to_ecef(radar_hits[0]["lat"], radar_hits[0]["lon"], radar_hits[0]["alt"])
            # Vertical descent — straight down from the radar hit
            traj_dir = -traj_point / np.linalg.norm(traj_point)
            source = "radar-single"
        elif len(ground_obs) >= 2:
            traj_point, traj_dir = fit_trajectory_from_observations(ground_obs)
            source = "observations"
        else:
            return jsonify({"error": "Need at least 1 radar hit or 2 ground observations."}), 400

        if traj_point is None or traj_dir is None:
            return jsonify({"error": "Could not compute trajectory — check that azimuths are not identical."}), 400

        landings, geojson = compute_strewn_field(traj_point, traj_dir, wind_mode, wind_single, wind_layers)

        return jsonify({
            "source": source,
            "trajectory_direction": traj_dir.round(4).tolist(),
            "landings": landings,
            "geojson": geojson,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
