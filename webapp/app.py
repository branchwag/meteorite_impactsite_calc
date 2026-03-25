from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os

app = Flask(__name__, static_folder="static")

EARTH_RADIUS = 6_371_000.0

def lla_to_ecef(lat_deg, lon_deg, alt_m):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = (EARTH_RADIUS + alt_m) * np.cos(lat) * np.cos(lon)
    y = (EARTH_RADIUS + alt_m) * np.cos(lat) * np.sin(lon)
    z = (EARTH_RADIUS + alt_m) * np.sin(lat)
    return np.array([x, y, z])

def az_el_to_enu(az_deg, el_deg):
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    return np.array([
        np.cos(el) * np.sin(az),
        np.cos(el) * np.cos(az),
        np.sin(el)
    ])

def enu_to_ecef_rotation(lat_deg, lon_deg):
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.array([
        [-np.sin(lon),              np.cos(lon),             0          ],
        [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)],
        [ np.cos(lat)*np.cos(lon),  np.cos(lat)*np.sin(lon), np.sin(lat)],
    ])

def observation_to_plane(obs):
    point = lla_to_ecef(obs["lat"], obs["lon"], obs["alt"])
    R = enu_to_ecef_rotation(obs["lat"], obs["lon"])
    ray_ecef = R @ az_el_to_enu(obs["azimuth"], obs["elevation"])
    up_ecef  = R @ np.array([0.0, 0.0, 1.0])
    normal = np.cross(ray_ecef, up_ecef)
    normal = normal / np.linalg.norm(normal)
    return point, normal

def intersect_planes(p1, n1, p2, n2):
    direction = np.cross(n1, n2)
    denom = np.dot(direction, direction)
    if denom < 1e-10:
        return None, None
    A = np.array([n1, n2, direction])
    b = np.array([np.dot(n1, p1), np.dot(n2, p2), 0.0])
    point = np.linalg.solve(A, b)
    return point, direction / np.linalg.norm(direction)

def ecef_to_lla(ecef):
    x, y, z = ecef
    lon = np.degrees(np.arctan2(y, x))
    lat = np.degrees(np.arctan2(z, np.sqrt(x**2 + y**2)))
    alt = np.sqrt(x**2 + y**2 + z**2) - EARTH_RADIUS
    return lat, lon, alt

def find_ground_impact(point, direction):
    for sign in [1, -1]:
        for t in np.linspace(0, 2_000_000, 500_000):
            pos = point + sign * t * direction
            _, _, alt = ecef_to_lla(pos)
            if alt <= 0:
                return pos
    return None

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

@app.route("/calculate", methods=["POST"])
def calculate():
    data = request.get_json()
    try:
        obs1 = data["obs1"]
        obs2 = data["obs2"]

        p1, n1 = observation_to_plane(obs1)
        p2, n2 = observation_to_plane(obs2)

        line_point, line_dir = intersect_planes(p1, n1, p2, n2)

        if line_point is None:
            return jsonify({"error": "The two planes are parallel — check that your azimuths are different."}), 400

        impact = find_ground_impact(line_point, line_dir)

        if impact is None:
            return jsonify({"error": "Could not find a ground intersection. Check your elevation angles."}), 400

        lat, lon, alt = ecef_to_lla(impact)

        return jsonify({
            "lat": round(lat, 5),
            "lon": round(lon, 5),
            "alt": round(alt, 1),
            "maps_url": f"https://maps.google.com/?q={lat:.5f},{lon:.5f}",
            "direction": line_dir.round(4).tolist(),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
