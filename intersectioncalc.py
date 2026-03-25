import numpy as np

# =============================================================
#  METEORITE TRAJECTORY INTERSECTION CALCULATOR
#  Fill in the observation data below, then run the script.
# =============================================================

# Each observation defines a plane using:
#   - observer position (lat, lon, altitude in meters)
#   - azimuth: compass bearing of the fireball (degrees, 0=North, 90=East)
#   - elevation: angle above horizon where fireball was seen (degrees)
#   - (optional) end elevation: where the fireball disappeared

# --- OBSERVATION 1 ---
obs1 = {
    "lat":       34.0522,   # degrees North
    "lon":      -118.2437,  # degrees East (negative = West)
    "alt":       100.0,     # observer altitude in meters
    "azimuth":   45.0,      # compass bearing of fireball (degrees)
    "elevation": 30.0,      # angle above horizon (degrees)
}

# --- OBSERVATION 2 ---
obs2 = {
    "lat":       34.1522,
    "lon":      -118.1437,
    "alt":       150.0,
    "azimuth":   220.0,
    "elevation": 25.0,
}

# =============================================================
#  CONVERSION HELPERS
# =============================================================

EARTH_RADIUS = 6_371_000.0  # meters

def lla_to_ecef(lat_deg, lon_deg, alt_m):
    """Convert lat/lon/altitude to Earth-Centered Earth-Fixed (ECEF) XYZ."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = (EARTH_RADIUS + alt_m) * np.cos(lat) * np.cos(lon)
    y = (EARTH_RADIUS + alt_m) * np.cos(lat) * np.sin(lon)
    z = (EARTH_RADIUS + alt_m) * np.sin(lat)
    return np.array([x, y, z])

def az_el_to_enu(az_deg, el_deg):
    """Convert azimuth/elevation to a unit vector in ENU (East-North-Up)."""
    az  = np.radians(az_deg)
    el  = np.radians(el_deg)
    e =  np.cos(el) * np.sin(az)
    n =  np.cos(el) * np.cos(az)
    u =  np.sin(el)
    return np.array([e, n, u])

def enu_to_ecef_rotation(lat_deg, lon_deg):
    """Build the rotation matrix from ENU to ECEF at a given lat/lon."""
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    R = np.array([
        [-np.sin(lon),               np.cos(lon),              0           ],
        [-np.sin(lat)*np.cos(lon),  -np.sin(lat)*np.sin(lon),  np.cos(lat) ],
        [ np.cos(lat)*np.cos(lon),   np.cos(lat)*np.sin(lon),  np.sin(lat) ],
    ])
    return R

def observation_to_plane(obs):
    """
    Given an observation dict, return (point, normal) of the plane
    that contains the observer and the observed fireball direction.
    The plane is defined by the observer position and two directions:
      - the fireball ray
      - the local Up vector (so the plane is vertical through the ray)
    """
    point = lla_to_ecef(obs["lat"], obs["lon"], obs["alt"])
    R     = enu_to_ecef_rotation(obs["lat"], obs["lon"])

    # Fireball direction in ECEF
    ray_enu  = az_el_to_enu(obs["azimuth"], obs["elevation"])
    ray_ecef = R @ ray_enu

    # Local up vector in ECEF
    up_enu  = np.array([0.0, 0.0, 1.0])
    up_ecef = R @ up_enu

    # Normal to the plane = cross product of ray and up
    normal = np.cross(ray_ecef, up_ecef)
    normal = normal / np.linalg.norm(normal)

    return point, normal

# =============================================================
#  PLANE INTERSECTION
# =============================================================

def intersect_planes(p1, n1, p2, n2):
    """
    Find the line of intersection of two planes.
    Returns (point_on_line, direction_vector) or None if planes are parallel.
    """
    direction = np.cross(n1, n2)
    denom = np.dot(direction, direction)
    if denom < 1e-10:
        return None, None  # planes are parallel

    # Find a point on the intersection line using least squares
    A = np.array([n1, n2, direction])
    b = np.array([np.dot(n1, p1), np.dot(n2, p2), 0.0])
    point = np.linalg.solve(A, b)

    direction = direction / np.linalg.norm(direction)
    return point, direction

# =============================================================
#  GROUND IMPACT ESTIMATE
# =============================================================

def ecef_to_lla(ecef):
    """Convert ECEF XYZ back to lat/lon/altitude (simple spherical)."""
    x, y, z = ecef
    lon = np.degrees(np.arctan2(y, x))
    hyp = np.sqrt(x**2 + y**2)
    lat = np.degrees(np.arctan2(z, hyp))
    alt = np.sqrt(x**2 + y**2 + z**2) - EARTH_RADIUS
    return lat, lon, alt

def find_ground_impact(point, direction):
    """
    Walk along the intersection line to find where it hits the ground (alt ~ 0).
    Uses simple parametric stepping.
    """
    # Try both directions along the line
    for sign in [1, -1]:
        for t in np.linspace(0, 2_000_000, 500_000):
            pos = point + sign * t * direction
            _, _, alt = ecef_to_lla(pos)
            if alt <= 0:
                return pos
    return None

# =============================================================
#  MAIN
# =============================================================

print("=" * 55)
print("  METEORITE TRAJECTORY INTERSECTION CALCULATOR")
print("=" * 55)

p1, n1 = observation_to_plane(obs1)
p2, n2 = observation_to_plane(obs2)

print(f"\nObserver 1 ECEF (m): {p1.round(1)}")
print(f"Plane 1 normal:      {n1.round(4)}")
print(f"\nObserver 2 ECEF (m): {p2.round(1)}")
print(f"Plane 2 normal:      {n2.round(4)}")

line_point, line_dir = intersect_planes(p1, n1, p2, n2)

if line_point is None:
    print("\nThe two observation planes are parallel — no intersection found.")
    print("Check that your azimuth values are not identical.")
else:
    print(f"\nTrajectory direction vector: {line_dir.round(4)}")

    impact_ecef = find_ground_impact(line_point, line_dir)

    if impact_ecef is not None:
        lat, lon, alt = ecef_to_lla(impact_ecef)
        print("\n" + "=" * 55)
        print("  ESTIMATED GROUND IMPACT")
        print("=" * 55)
        print(f"  Latitude:   {lat:.5f} °")
        print(f"  Longitude:  {lon:.5f} °")
        print(f"  Altitude:   {alt:.1f} m  (should be ~0)")
        print(f"\n  Google Maps link:")
        print(f"  https://maps.google.com/?q={lat:.5f},{lon:.5f}")
    else:
        print("\nCould not find a ground intersection.")
        print("The trajectory may be going upward — check elevation angles.")

print("\nDone.")
