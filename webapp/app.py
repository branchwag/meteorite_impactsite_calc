from flask import Flask, request, jsonify, send_from_directory, Response
import numpy as np
import json
import io
import zipfile
import xml.etree.ElementTree as ET

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

def terminal_velocity(mass_g, density_g_cm3=3.2, Cd=0.8):
    # V = sqrt(8 * D * g * r / (3 * rho_air * Cd)) — from force balance at terminal velocity
    mass_kg = mass_g / 1000.0
    density_kg_m3 = density_g_cm3 * 1000.0
    r_m = (3.0 * mass_kg / (4.0 * np.pi * density_kg_m3)) ** (1.0 / 3.0)
    rho_air = 1.2  # kg/m³, sea level approximation
    g = 9.81
    return np.sqrt(8.0 * density_kg_m3 * g * r_m / (3.0 * rho_air * Cd))

def fragment_landing(traj_point, traj_dir, mass_g, wind_mode, wind_single, wind_layers,
                     density_g_cm3=3.2, Cd=0.8):
    term_v = terminal_velocity(mass_g, density_g_cm3, Cd)
    pos = traj_point.copy()
    for _ in range(300_000):
        _, _, alt = ecef_to_lla(pos)
        if alt <= 0:
            return pos
        pos = pos + traj_dir * term_v
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

def compute_strewn_field(traj_point, traj_dir, wind_mode, wind_single, wind_layers,
                         density_g_cm3=3.2, Cd=0.8):
    landings = []
    for label, mass_g in MASS_CLASSES:
        ecef = fragment_landing(traj_point, traj_dir, mass_g, wind_mode, wind_single, wind_layers,
                                density_g_cm3, Cd)
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
        radar_hits      = data.get("radar_hits", [])
        ground_obs      = data.get("ground_obs", [])
        wind_mode       = data.get("wind_mode", "single")
        wind_single     = data.get("wind_single", {"speed_ms": 0, "direction_deg": 0})
        wind_layers     = data.get("wind_layers", [])
        density_g_cm3   = float(data.get("density_g_cm3", 3.2))
        Cd              = float(data.get("Cd", 0.8))
        if density_g_cm3 <= 0:
            density_g_cm3 = 3.2
        if Cd <= 0:
            Cd = 0.8

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

        landings, geojson = compute_strewn_field(traj_point, traj_dir, wind_mode, wind_single, wind_layers,
                                                  density_g_cm3, Cd)

        return jsonify({
            "source": source,
            "trajectory_direction": traj_dir.round(4).tolist(),
            "landings": landings,
            "geojson": geojson,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _nexrad_resolution_volume(range_km, beamwidth_deg=1.0, gate_m=250.0):
    """Two-way Gaussian beam resolution volume for NEXRAD."""
    import math
    R_m = range_km * 1000.0
    theta = math.radians(beamwidth_deg)
    return (math.pi / (8 * math.log(2))) * theta**2 * R_m**2 * gate_m

@app.route("/dbz", methods=["POST"])
def dbz_estimator():
    data = request.get_json()
    try:
        import math
        n_fragments   = int(data.get("n_fragments", 1))
        diameter_cm   = float(data.get("diameter_cm", 10.0))
        range_km      = float(data.get("range_km", 80.0))
        wavelength_cm = float(data.get("wavelength_cm", 10.0))

        if n_fragments < 1 or diameter_cm <= 0 or range_km <= 0 or wavelength_cm <= 0:
            return jsonify({"error": "All values must be positive."}), 400

        r_m           = (diameter_cm / 100.0) / 2.0
        sigma_per     = math.pi * r_m**2          # m²
        sigma_total   = n_fragments * sigma_per

        dbsm_per   = 10 * math.log10(sigma_per)
        dbsm_total = 10 * math.log10(sigma_total)

        lambda_mm = wavelength_cm * 10.0
        K2        = 0.93  # |K|² for liquid water — standard radar calibration constant
        V_r       = _nexrad_resolution_volume(range_km)

        sigma_total_mm2 = sigma_total * 1e6
        Ze  = (lambda_mm**4 / (math.pi**5 * K2)) * (sigma_total_mm2 / V_r)
        dbz = 10 * math.log10(Ze)

        if dbz < 0:
            context = "Below typical NEXRAD noise floor — marginally detectable at best."
        elif dbz < 10:
            context = "Weak but potentially detectable signal, especially at high altitude where clutter is absent."
        elif dbz < 20:
            context = "Detectable signal, consistent with confirmed radar-tracked meteorite falls (e.g., Hamburg 2018)."
        elif dbz < 35:
            context = "Strong signal — easy NEXRAD detection."
        else:
            context = "Very strong return — comparable to moderate rainfall."

        return jsonify({
            "sigma_per_m2":        round(sigma_per, 6),
            "sigma_total_m2":      round(sigma_total, 6),
            "dbsm_per":            round(dbsm_per, 2),
            "dbsm_total":          round(dbsm_total, 2),
            "dbz":                 round(dbz, 1),
            "resolution_volume_m3": round(V_r),
            "context":             context,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/inverse_dbz", methods=["POST"])
def inverse_dbz():
    data = request.get_json()
    try:
        import math
        dbz           = float(data.get("dbz", 0.0))
        range_km      = float(data.get("range_km", 80.0))
        wavelength_cm = float(data.get("wavelength_cm", 10.0))
        n_fragments   = data.get("n_fragments")   # optional: given n, infer diameter
        diameter_cm   = data.get("diameter_cm")   # optional: given d, infer n

        if range_km <= 0 or wavelength_cm <= 0:
            return jsonify({"error": "Range and wavelength must be positive."}), 400

        lambda_mm = wavelength_cm * 10.0
        K2        = 0.93
        V_r       = _nexrad_resolution_volume(range_km)

        Ze              = 10 ** (dbz / 10.0)          # mm⁶/m³
        sigma_total_mm2 = Ze * V_r * math.pi**5 * K2 / lambda_mm**4
        sigma_total_m2  = sigma_total_mm2 / 1e6
        dbsm_total      = 10 * math.log10(sigma_total_m2) if sigma_total_m2 > 0 else -999

        result = {
            "sigma_total_m2": round(sigma_total_m2, 6),
            "dbsm_total":     round(dbsm_total, 2),
        }

        if n_fragments is not None and int(n_fragments) > 0:
            n  = int(n_fragments)
            sp = sigma_total_m2 / n
            r  = math.sqrt(sp / math.pi)
            result["implied_diameter_cm"] = round(r * 2 * 100, 2)
            result["implied_diameter_in"] = round(r * 2 * 100 / 2.54, 2)

        if diameter_cm is not None and float(diameter_cm) > 0:
            r_m = (float(diameter_cm) / 100.0) / 2.0
            sp  = math.pi * r_m**2
            result["implied_fragment_count"] = round(sigma_total_m2 / sp, 1)

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


ELLIPSE_COLORS = [
    "ff0000ff",  # red - 1kg+
    "ff0088ff",  # orange
    "ff00ddff",  # yellow
    "ff00ff88",  # green
    "ffff4444",  # blue - 1g
]

def geojson_to_kml(geojson):
    kml = ET.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = ET.SubElement(kml, "Document")
    ET.SubElement(doc, "name").text = "Meteorite Strewn Field"

    # Styles for ellipse fills
    for i, color in enumerate(ELLIPSE_COLORS):
        style = ET.SubElement(doc, "Style", id=f"ellipse{i}")
        ls = ET.SubElement(style, "LineStyle")
        ET.SubElement(ls, "color").text = color
        ET.SubElement(ls, "width").text = "2"
        ps = ET.SubElement(style, "PolyStyle")
        ET.SubElement(ps, "color").text = "40" + color[2:]  # semi-transparent fill
        ET.SubElement(ps, "outline").text = "1"

    # Style for centerline
    style = ET.SubElement(doc, "Style", id="centerline")
    ls = ET.SubElement(style, "LineStyle")
    ET.SubElement(ls, "color").text = "ff00ffff"  # yellow
    ET.SubElement(ls, "width").text = "3"

    # Style for landing points
    style = ET.SubElement(doc, "Style", id="landing")
    is_ = ET.SubElement(style, "IconStyle")
    ET.SubElement(is_, "color").text = "ff0000ff"
    ET.SubElement(is_, "scale").text = "1.0"
    icon = ET.SubElement(is_, "Icon")
    ET.SubElement(icon, "href").text = "http://maps.google.com/mapfiles/kml/paddle/red-circle.png"

    ellipse_idx = 0
    for feature in geojson["features"]:
        props = feature["properties"]
        geom = feature["geometry"]
        pm = ET.SubElement(doc, "Placemark")
        ET.SubElement(pm, "name").text = props.get("name", "")

        if geom["type"] == "Point":
            ET.SubElement(pm, "styleUrl").text = "#landing"
            pt = ET.SubElement(pm, "Point")
            lon, lat = geom["coordinates"]
            ET.SubElement(pt, "coordinates").text = f"{lon},{lat},0"

        elif geom["type"] == "LineString":
            ET.SubElement(pm, "styleUrl").text = "#centerline"
            ls = ET.SubElement(pm, "LineString")
            ET.SubElement(ls, "tessellate").text = "1"
            coords = " ".join(f"{c[0]},{c[1]},0" for c in geom["coordinates"])
            ET.SubElement(ls, "coordinates").text = coords

        elif geom["type"] == "Polygon":
            style_id = f"ellipse{ellipse_idx % len(ELLIPSE_COLORS)}"
            ellipse_idx += 1
            ET.SubElement(pm, "styleUrl").text = f"#{style_id}"
            poly = ET.SubElement(pm, "Polygon")
            ET.SubElement(poly, "tessellate").text = "1"
            ob = ET.SubElement(poly, "outerBoundaryIs")
            lr = ET.SubElement(ob, "LinearRing")
            coords = " ".join(f"{c[0]},{c[1]},0" for c in geom["coordinates"][0])
            ET.SubElement(lr, "coordinates").text = coords

    return ET.tostring(kml, encoding="unicode", xml_declaration=True)

@app.route("/download_kmz", methods=["POST"])
def download_kmz():
    data = request.get_json()
    geojson = data.get("geojson")
    if not geojson:
        return jsonify({"error": "No geojson provided"}), 400
    kml_str = geojson_to_kml(geojson)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("doc.kml", kml_str)
    buf.seek(0)
    return Response(buf.read(),
                    mimetype="application/vnd.google-earth.kmz",
                    headers={"Content-Disposition": "attachment; filename=strewn_field.kmz"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
