from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import math

app = Flask(__name__, static_folder="static")

EARTH_RADIUS = 6_371_000.0

# ---------- Standard Atmosphere (ISA) -----------------------------------
# Source: spreadsheet "meteor dark flight.xlsx" Spherical tab, rows 0-30 km.
# Columns: altitude_m, air_density_g_per_m3, gravity_m_per_s2
# Rows 31-33 km extrapolated via barometric scale height (~7 200 m).
_ATMO = np.array([
    [0,      1224.977, 9.080665],
    [1000,   1111.624, 9.079242],
    [2000,   1006.476, 9.077818],
    [3000,    909.111, 9.076396],
    [4000,    819.121, 9.074974],
    [5000,    736.110, 9.073552],
    [6000,    659.694, 9.072131],
    [7000,    589.500, 9.070710],
    [8000,    525.168, 9.069290],
    [9000,    466.349, 9.067870],
    [10000,   412.709, 9.066450],
    [11000,   363.921, 9.065031],
    [12000,   310.828, 9.063612],
    [13000,   265.483, 9.062194],
    [14000,   226.753, 9.060777],
    [15000,   193.674, 9.059359],
    [16000,   165.420, 9.057943],
    [17000,   141.288, 9.056526],
    [18000,   120.676, 9.055110],
    [19000,   103.071, 9.053695],
    [20000,    88.035, 9.052280],
    [21000,    74.874, 9.050865],
    [22000,    63.727, 9.049451],
    [23000,    54.280, 9.048037],
    [24000,    46.267, 9.046624],
    [25000,    39.466, 9.045211],
    [26000,    33.688, 9.043799],
    [27000,    28.777, 9.042387],
    [28000,    24.599, 9.040975],
    [29000,    21.042, 9.039564],
    [30000,    18.012, 9.038153],
    [31000,    15.403, 9.036739],
    [32000,    13.179, 9.035325],
    [33000,    11.277, 9.033911],
])

def _atm_density_kg_m3(alt_m):
    alt_m = float(np.clip(alt_m, 0.0, 33000.0))
    return float(np.interp(alt_m, _ATMO[:, 0], _ATMO[:, 1])) / 1000.0

def _atm_gravity(alt_m):
    alt_m = float(np.clip(alt_m, 0.0, 33000.0))
    return float(np.interp(alt_m, _ATMO[:, 0], _ATMO[:, 2]))

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

def wind_vector_enu(speed_ms, direction_deg):
    # Meteorological direction is where the wind blows FROM; drift goes the
    # opposite way. Returns the per-second displacement (metres), so the
    # magnitude must scale with the wind speed.
    toward_deg = (direction_deg + 180) % 360
    az = np.radians(toward_deg)
    return speed_ms * np.array([np.sin(az), np.cos(az), 0.0])

def get_wind_at_altitude(alt_m, wind_mode, wind_single, wind_layers):
    if wind_mode == "single":
        return wind_single["speed_ms"], wind_single["direction_deg"]
    layers = sorted(wind_layers, key=lambda x: x["alt_m"], reverse=True)
    for layer in layers:
        if alt_m >= layer["alt_m"]:
            return layer["speed_ms"], layer["direction_deg"]
    return layers[-1]["speed_ms"], layers[-1]["direction_deg"]

def terminal_velocity(mass_g, density_g_cm3=3.3, Cd=0.8, alt_m=0.0):
    # V = sqrt(8 * D * g * r / (3 * rho_air * Cd)) — from uncle's force-balance derivation
    mass_kg       = mass_g / 1000.0
    density_kg_m3 = density_g_cm3 * 1000.0
    r_m           = (3.0 * mass_kg / (4.0 * np.pi * density_kg_m3)) ** (1.0 / 3.0)
    rho_air       = _atm_density_kg_m3(alt_m)
    g             = _atm_gravity(alt_m)
    return float(np.sqrt(8.0 * density_kg_m3 * g * r_m / (3.0 * rho_air * Cd)))

def fall_time_integral(start_alt_m, end_alt_m, mass_g, density_g_cm3, Cd, n_steps=300):
    """Seconds to fall from start_alt (high) to end_alt (low) using Standard Atmosphere."""
    if start_alt_m <= end_alt_m:
        return 0.0
    alts = np.linspace(start_alt_m, end_alt_m, n_steps + 1)
    vts  = np.array([terminal_velocity(mass_g, density_g_cm3, Cd, h) for h in alts])
    dh   = (start_alt_m - end_alt_m) / n_steps
    return float(np.sum(2.0 * dh / (vts[:-1] + vts[1:])))

def find_mass_for_fall_time(t_obs, start_alt_m, end_alt_m, density_g_cm3, Cd,
                             m_min_g=2.0, m_max_g=10000.0):
    """
    Binary search (log scale) for the meteorite mass (grams) whose fall time
    from start_alt to end_alt matches t_obs seconds.
    Returns (mass_g, diameter_cm, diameter_in) or None if outside [m_min_g, m_max_g].
    """
    t_at_min = fall_time_integral(start_alt_m, end_alt_m, m_min_g, density_g_cm3, Cd)
    t_at_max = fall_time_integral(start_alt_m, end_alt_m, m_max_g, density_g_cm3, Cd)
    # Heavier → faster → less time: t_at_min > t_at_max
    if t_obs > t_at_min or t_obs < t_at_max:
        return None
    lo = math.log(m_min_g)
    hi = math.log(m_max_g)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        m   = math.exp(mid)
        t   = fall_time_integral(start_alt_m, end_alt_m, m, density_g_cm3, Cd)
        if t > t_obs:
            lo = mid   # too slow → need heavier
        else:
            hi = mid   # too fast → need lighter
    mass_g = math.exp((lo + hi) / 2.0)
    density_kg_m3 = density_g_cm3 * 1000.0
    r_m = (3.0 * mass_g / 1000.0 / (4.0 * math.pi * density_kg_m3)) ** (1.0 / 3.0)
    d_cm = r_m * 2.0 * 100.0
    return {
        "mass_g":       round(mass_g, 2),
        "diameter_cm":  round(d_cm, 2),
        "diameter_in":  round(d_cm / 2.54, 2),
    }

def descend_to_altitude(traj_point, traj_dir, mass_g, target_alt_m,
                        wind_mode, wind_single, wind_layers,
                        density_g_cm3=3.3, Cd=0.8):
    """Integrate the wind-corrected descent in 1-second steps until the rock
    reaches target_alt_m. Returns (ecef_position, elapsed_seconds)."""
    pos = traj_point.copy()
    for sec in range(300_000):
        lat, lon, alt = ecef_to_lla(pos)
        if alt <= target_alt_m:
            return pos, sec
        term_v = terminal_velocity(mass_g, density_g_cm3, Cd, alt)
        pos = pos + traj_dir * term_v
        lat, lon, _ = ecef_to_lla(pos)
        R = enu_to_ecef_rotation(lat, lon)
        spd, dirn = get_wind_at_altitude(alt, wind_mode, wind_single, wind_layers)
        pos = pos + R @ (wind_vector_enu(spd, dirn))
    return pos, 300_000

def fragment_landing(traj_point, traj_dir, mass_g, wind_mode, wind_single, wind_layers,
                     density_g_cm3=3.3, Cd=0.8):
    pos, _ = descend_to_altitude(traj_point, traj_dir, mass_g, 0.0,
                                 wind_mode, wind_single, wind_layers,
                                 density_g_cm3, Cd)
    return pos

@app.route("/")
def index():
    return send_from_directory("static", "index.html")

METEORITE_TYPES = [
    {"name": "Ordinary Chondrite (OC)", "density": 3.3},
    {"name": "Carbonaceous (CC)",        "density": 2.7},
    {"name": "Iron",                     "density": 7.2},
]
FALL_MASS_MIN_G = 2.0
FALL_MASS_MAX_G = 10000.0
DARK_FLIGHT_ALT_MIN_M = 18000.0
DARK_FLIGHT_ALT_MAX_M = 33000.0
FALL_TIME_MIN_S = 60.0
FALL_TIME_MAX_S = 600.0

@app.route("/analyze_hit", methods=["POST"])
def analyze_hit():
    data = request.get_json()
    try:
        fireball_end_s  = float(data["fireball_end_s"])   # seconds since midnight UTC
        radar_hit_s     = float(data["radar_hit_s"])       # seconds since midnight UTC
        dark_alt_km     = float(data.get("dark_flight_alt_km", 24.0))
        radar_lat       = float(data["radar_lat"])
        radar_lon       = float(data["radar_lon"])
        radar_alt_m     = float(data["radar_alt_m"])
        radar_dbz       = data.get("radar_dbz")
        radar_range_mi  = data.get("radar_range_mi")
        wind_mode       = data.get("wind_mode", "single")
        wind_single     = data.get("wind_single", {"speed_ms": 0, "direction_deg": 0})
        wind_layers     = data.get("wind_layers", [])
        Cd              = float(data.get("Cd", 0.8))

        dark_alt_m = dark_alt_km * 1000.0
        errors   = []
        warnings = []

        # ---- Sanity checks ----
        if dark_alt_m < DARK_FLIGHT_ALT_MIN_M or dark_alt_m > DARK_FLIGHT_ALT_MAX_M:
            errors.append(
                f"Dark flight altitude {dark_alt_km:.1f} km is outside valid range "
                f"{DARK_FLIGHT_ALT_MIN_M/1000:.0f}–{DARK_FLIGHT_ALT_MAX_M/1000:.0f} km."
            )
        if radar_alt_m >= dark_alt_m:
            errors.append(
                f"Radar hit altitude ({radar_alt_m/1000:.1f} km) must be BELOW dark "
                f"flight altitude ({dark_alt_km:.1f} km)."
            )
        if radar_alt_m < 0:
            errors.append("Radar hit altitude cannot be negative.")

        # Handle day-crossing (e.g. fireball near midnight)
        delta_t = radar_hit_s - fireball_end_s
        if delta_t < -300:      # radar hit looks like it was the previous day
            delta_t += 86400.0

        if delta_t < 0:
            errors.append(
                "Radar hit time is before fireball end time — check your UTC inputs."
            )
        elif delta_t < FALL_TIME_MIN_S:
            errors.append(
                f"Time delta {delta_t:.0f} s < {FALL_TIME_MIN_S:.0f} s minimum — "
                "this radar hit is likely not related to the fireball."
            )
        elif delta_t > FALL_TIME_MAX_S:
            errors.append(
                f"Time delta {delta_t:.0f} s > {FALL_TIME_MAX_S:.0f} s maximum — "
                "this radar hit is likely not related to the fireball."
            )

        if errors:
            return jsonify({"errors": errors, "warnings": warnings,
                            "delta_t_s": round(delta_t, 1) if delta_t >= 0 else None}), 200

        # ---- Type / size determination ----
        type_results = []
        for mt in METEORITE_TYPES:
            match = find_mass_for_fall_time(
                delta_t, dark_alt_m, radar_alt_m, mt["density"], Cd,
                FALL_MASS_MIN_G, FALL_MASS_MAX_G
            )
            if match:
                # Wind-corrected landing from radar hit straight down to ground
                traj_point = lla_to_ecef(radar_lat, radar_lon, radar_alt_m)
                traj_dir   = -traj_point / np.linalg.norm(traj_point)
                landing_ecef = fragment_landing(
                    traj_point, traj_dir, match["mass_g"],
                    wind_mode, wind_single, wind_layers, mt["density"], Cd
                )
                land_lat, land_lon, _ = ecef_to_lla(landing_ecef)
                match["landing_lat"] = round(land_lat, 5)
                match["landing_lon"] = round(land_lon, 5)
                match["type"]        = mt["name"]
                match["density"]     = mt["density"]
            else:
                match = {
                    "type":    mt["name"],
                    "density": mt["density"],
                    "mass_g":  None,
                    "note":    f"No match in {FALL_MASS_MIN_G:.0f} g – {FALL_MASS_MAX_G/1000:.0f} kg range",
                }
            type_results.append(match)

        is_meteorite = any(r.get("mass_g") is not None for r in type_results)
        if not is_meteorite:
            warnings.append(
                "No meteorite type fits this fall time — this radar hit is likely not a meteorite."
            )

        # ---- dBZ consistency check ----
        dbz_checks = []
        if radar_dbz is not None:
            radar_dbz = float(radar_dbz)
            # Range from the radar to the hit drives the resolution volume.
            # Fall back to a typical NEXRAD range only if none was provided.
            range_km = float(radar_range_mi) * 1.60934 if radar_range_mi else 80.0
            for r in type_results:
                if r.get("mass_g") is None:
                    continue
                d_cm = r["diameter_cm"]
                n_frags = 1
                # Estimate dBZ for a single fragment of this size
                r_m       = (d_cm / 100.0) / 2.0
                sigma     = math.pi * r_m**2
                lambda_mm = 100.0  # S-band 10 cm
                K2        = 0.93
                V_r       = _nexrad_resolution_volume(range_km)
                Ze  = (lambda_mm**4 / (math.pi**5 * K2)) * (sigma * 1e6 / V_r)
                est_dbz = 10 * math.log10(Ze) if Ze > 0 else -99
                delta_dbz = abs(est_dbz - radar_dbz)
                dbz_checks.append({
                    "type":      r["type"],
                    "est_dbz":   round(est_dbz, 1),
                    "obs_dbz":   round(radar_dbz, 1),
                    "delta_dbz": round(delta_dbz, 1),
                    "note":      "consistent" if delta_dbz < 10 else "large discrepancy",
                })

        return jsonify({
            "delta_t_s":    round(delta_t, 1),
            "is_meteorite": is_meteorite,
            "type_results": type_results,
            "dbz_checks":   dbz_checks,
            "errors":       errors,
            "warnings":     warnings,
        })

    except KeyError as e:
        return jsonify({"errors": [f"Missing required field: {e}"]}), 400
    except Exception as e:
        return jsonify({"errors": [str(e)]}), 500


def _nexrad_resolution_volume(range_km, beamwidth_deg=1.0, gate_m=250.0):
    """Two-way Gaussian beam resolution volume for NEXRAD."""
    R_m = range_km * 1000.0
    theta = math.radians(beamwidth_deg)
    return (math.pi / (8 * math.log(2))) * theta**2 * R_m**2 * gate_m

@app.route("/dbz", methods=["POST"])
def dbz_estimator():
    data = request.get_json()
    try:
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

        # Geometric cross-section dBZ (valid for D >> lambda, practical for large fragments)
        sigma_total_mm2 = sigma_total * 1e6
        Ze_geo  = (lambda_mm**4 / (math.pi**5 * K2)) * (sigma_total_mm2 / V_r)
        dbz_geo = 10 * math.log10(Ze_geo)

        # Rayleigh backscatter dBZ (Z = n × D^6 / V_r, valid for D << lambda)
        # Uses |K_m|² = |K_w|² assumption (treating fragment as equivalent water drop)
        d_mm       = diameter_cm * 10.0
        Ze_ray     = n_fragments * (d_mm ** 6) / V_r   # mm^6/m^3
        dbz_ray    = 10 * math.log10(Ze_ray) if Ze_ray > 0 else -99
        d_over_lam = (diameter_cm / 100.0) / (wavelength_cm / 100.0)
        if d_over_lam < 0.1:
            regime = "Rayleigh (D/λ < 0.1 — Rayleigh formula is accurate)"
        elif d_over_lam < 1.0:
            regime = "Mie transition (0.1 < D/λ < 1 — geometric formula more reliable)"
        else:
            regime = "Optical/geometric (D/λ > 1 — geometric formula accurate)"

        dbz = dbz_geo  # primary result

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
            "sigma_per_m2":         round(sigma_per, 6),
            "sigma_total_m2":       round(sigma_total, 6),
            "dbsm_per":             round(dbsm_per, 2),
            "dbsm_total":           round(dbsm_total, 2),
            "dbz":                  round(dbz_geo, 1),
            "dbz_rayleigh":         round(dbz_ray, 1),
            "scattering_regime":    regime,
            "resolution_volume_m3": round(V_r),
            "context":              context,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/inverse_dbz", methods=["POST"])
def inverse_dbz():
    data = request.get_json()
    try:
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


def _fmt_utc(seconds):
    """Seconds-since-midnight -> HH:MM:SS (wraps past midnight)."""
    s = int(round(seconds)) % 86400
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

# Altitude levels (m) at which we report a predicted radar position, top-down.
RADAR_PREDICT_LEVELS_M = [20000, 15000, 12000, 10000, 8000, 6000, 4000, 2000, 1000, 0]

@app.route("/predict_radar", methods=["POST"])
def predict_radar():
    """Scenario 2: given a dark-flight point (a radar return high up) and the
    fireball end time, predict where/when the rock should appear on lower radar
    sweeps as it descends — i.e. where to look on radar, down to the ground."""
    data = request.get_json()
    try:
        fireball_end_s = float(data["fireball_end_s"])  # seconds since midnight UTC
        df_time_s      = float(data["df_time_s"])         # dark-flight return time UTC
        df_lat         = float(data["df_lat"])
        df_lon         = float(data["df_lon"])
        df_alt_m       = float(data["df_alt_m"])
        dbz            = data.get("dbz")
        range_mi       = data.get("range_mi")
        wind_mode      = data.get("wind_mode", "single")
        wind_single    = data.get("wind_single", {"speed_ms": 0, "direction_deg": 0})
        wind_layers    = data.get("wind_layers", [])
        density_g_cm3  = float(data.get("density_g_cm3", 3.3))
        Cd             = float(data.get("Cd", 0.8))
        assumed_mass_g = float(data.get("assumed_mass_g", 100.0))  # spec: assume 100 g chondrite

        errors, warnings = [], []

        if df_alt_m < DARK_FLIGHT_ALT_MIN_M or df_alt_m > DARK_FLIGHT_ALT_MAX_M:
            errors.append(
                f"Dark-flight altitude {df_alt_m/1000:.1f} km is outside the valid range "
                f"{DARK_FLIGHT_ALT_MIN_M/1000:.0f}–{DARK_FLIGHT_ALT_MAX_M/1000:.0f} km."
            )

        delta_t = df_time_s - fireball_end_s
        if delta_t < -300:
            delta_t += 86400.0
        if delta_t < 0:
            errors.append(
                "Dark-flight return time is before the fireball end time — check your UTC inputs."
            )
        elif delta_t > FALL_TIME_MAX_S:
            warnings.append(
                f"Dark-flight return is {delta_t:.0f} s after the fireball "
                f"(> {FALL_TIME_MAX_S:.0f} s) — it may not be related to this fireball."
            )

        if errors:
            return jsonify({"errors": errors, "warnings": warnings}), 200

        # Vertical descent straight down from the dark-flight point.
        traj_point = lla_to_ecef(df_lat, df_lon, df_alt_m)
        traj_dir   = -traj_point / np.linalg.norm(traj_point)

        levels = [a for a in RADAR_PREDICT_LEVELS_M if a < df_alt_m] + [0]
        levels = sorted(set(levels), reverse=True)

        predictions = []
        for alt in levels:
            ecef, t_fall = descend_to_altitude(
                traj_point, traj_dir, assumed_mass_g, alt,
                wind_mode, wind_single, wind_layers, density_g_cm3, Cd
            )
            lat, lon, _ = ecef_to_lla(ecef)
            predictions.append({
                "alt_m":      alt,
                "alt_ft":     round(alt / 0.3048),
                "t_fall_s":   round(t_fall, 1),
                "clock_utc":  _fmt_utc(df_time_s + t_fall),
                "lat":        round(lat, 5),
                "lon":        round(lon, 5),
            })

        # Optional: size implied by the observed dBZ (treating the return as a
        # single fragment) — informational only.
        dbz_note = None
        if dbz is not None and range_mi:
            range_km  = float(range_mi) * 1.60934
            lambda_mm = 100.0
            K2        = 0.93
            V_r       = _nexrad_resolution_volume(range_km)
            Ze        = 10 ** (float(dbz) / 10.0)
            sigma_m2  = Ze * V_r * math.pi**5 * K2 / lambda_mm**4 / 1e6
            if sigma_m2 > 0:
                r_m  = math.sqrt(sigma_m2 / math.pi)
                d_cm = r_m * 2 * 100
                dbz_note = (
                    f"Observed {float(dbz):.0f} dBZ at {range_km:.0f} km implies a single "
                    f"fragment ≈ {d_cm:.1f} cm across (geometric cross-section)."
                )

        return jsonify({
            "delta_t_s":      round(delta_t, 1),
            "assumed_mass_g": assumed_mass_g,
            "density_g_cm3":  density_g_cm3,
            "predictions":    predictions,
            "ground":         predictions[-1] if predictions else None,
            "dbz_note":       dbz_note,
            "errors":         errors,
            "warnings":       warnings,
        })

    except KeyError as e:
        return jsonify({"errors": [f"Missing required field: {e}"]}), 400
    except Exception as e:
        return jsonify({"errors": [str(e)]}), 500


if __name__ == "__main__":
    import os
    # HOST defaults to 0.0.0.0 so the Raspberry Pi can serve the LAN in prod.
    # For local dev, run `HOST=127.0.0.1 python app.py` to bind localhost only.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=True)
