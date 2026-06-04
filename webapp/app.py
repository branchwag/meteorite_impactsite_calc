from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import math
import os
import json
import ssl
import urllib.request
import urllib.parse
import urllib.error

app = Flask(__name__, static_folder="static")

EARTH_RADIUS = 6_371_000.0

# ---------- Standard Atmosphere (ISA) -----------------------------------
# Air density from spreadsheet "meteor dark flight.xlsx" Spherical tab,
# rows 0-30 km; 31-33 km extrapolated via barometric scale height (~7 200 m).
# Columns: altitude_m, air_density_g_per_m3.
# NOTE: the spreadsheet's gravity column was wrong (~9.08 at sea level); gravity
# is computed analytically below from the standard value 9.80665 m/s² instead.
_ATMO = np.array([
    [     0,  1224.977],
    [  1000,  1111.624],
    [  2000,  1006.476],
    [  3000,   909.111],
    [  4000,   819.121],
    [  5000,   736.110],
    [  6000,   659.694],
    [  7000,   589.500],
    [  8000,   525.168],
    [  9000,   466.349],
    [ 10000,   412.709],
    [ 11000,   363.921],
    [ 12000,   310.828],
    [ 13000,   265.483],
    [ 14000,   226.753],
    [ 15000,   193.674],
    [ 16000,   165.420],
    [ 17000,   141.288],
    [ 18000,   120.676],
    [ 19000,   103.071],
    [ 20000,    88.035],
    [ 21000,    74.874],
    [ 22000,    63.727],
    [ 23000,    54.280],
    [ 24000,    46.267],
    [ 25000,    39.466],
    [ 26000,    33.688],
    [ 27000,    28.777],
    [ 28000,    24.599],
    [ 29000,    21.042],
    [ 30000,    18.012],
    [ 31000,    15.403],
    [ 32000,    13.179],
    [ 33000,    11.277],
])

STANDARD_GRAVITY = 9.80665  # m/s² at sea level

def _atm_density_kg_m3(alt_m):
    alt_m = float(np.clip(alt_m, 0.0, 33000.0))
    return float(np.interp(alt_m, _ATMO[:, 0], _ATMO[:, 1])) / 1000.0

def _atm_gravity(alt_m):
    # Standard gravity falls off with altitude: g(h) = g0 (R / (R + h))².
    alt_m = float(np.clip(alt_m, 0.0, 33000.0))
    return STANDARD_GRAVITY * (EARTH_RADIUS / (EARTH_RADIUS + alt_m)) ** 2

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

# Per-type plausible mass search ranges (grams), from the uncle's notes:
# iron 5 g–10 kg, CC 2 g–500 g, ordinary chondrite 2 g–10 kg.
METEORITE_TYPES = [
    {"name": "Ordinary Chondrite (OC)", "density": 3.3, "min_g": 2.0, "max_g": 10000.0},
    {"name": "Carbonaceous (CC)",        "density": 2.7, "min_g": 2.0, "max_g":   500.0},
    {"name": "Iron",                     "density": 7.2, "min_g": 5.0, "max_g": 10000.0},
]
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
                mt["min_g"], mt["max_g"]
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
                hi = (f"{mt['max_g']/1000:.0f} kg" if mt["max_g"] >= 1000
                      else f"{mt['max_g']:.0f} g")
                match = {
                    "type":    mt["name"],
                    "density": mt["density"],
                    "mass_g":  None,
                    "note":    f"No match in {mt['min_g']:.0f} g – {hi} range",
                }
            type_results.append(match)

        is_meteorite = any(r.get("mass_g") is not None for r in type_results)
        if not is_meteorite:
            warnings.append(
                "No meteorite type fits this fall time — this radar hit is likely not a meteorite."
            )

        # Part B (number of rocks from the observed dBZ + range) is computed
        # separately via /inverse_dbz once the fragment size is known from A.

        return jsonify({
            "delta_t_s":    round(delta_t, 1),
            "is_meteorite": is_meteorite,
            "type_results": type_results,
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


# ---------- Radiosonde wind sounding (IGRA station list + UWyo data) -----
# The browser can't fetch these sources directly (no CORS), so the Flask
# backend proxies them: find the nearest active radiosonde station to the
# point, pull that station's University of Wyoming sounding for the closest
# synoptic hour, and turn it into altitude wind layers.

IGRA_STATION_URL = "https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt"
UWYO_URL         = "http://weather.uwyo.edu/cgi-bin/sounding"
_STATION_CACHE   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "igra_stations.json")
KNOTS_TO_MS      = 0.514444
_stations_mem    = None  # in-process cache

def _http_get(url, timeout=30):
    """GET a URL as text, tolerating environments with broken cert chains."""
    req = urllib.request.Request(url, headers={"User-Agent": "meteorite-impactsite-calc"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", "replace")
    except (ssl.SSLError, urllib.error.URLError):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
            return r.read().decode("utf-8", "replace")

def _load_stations():
    """Return [{stnm, lat, lon, name, last_year}] for active WMO stations.
    Downloads and caches the IGRA station list on first use."""
    global _stations_mem
    if _stations_mem is not None:
        return _stations_mem
    if os.path.exists(_STATION_CACHE):
        with open(_STATION_CACHE) as f:
            _stations_mem = json.load(f)
        return _stations_mem

    text = _http_get(IGRA_STATION_URL)
    stations = []
    for line in text.splitlines():
        if len(line) < 81:
            continue
        sid = line[0:11]
        # Network char 'M' => station carries a WMO number (= UWyo STNM).
        if sid[2] != "M":
            continue
        try:
            lat = float(line[12:20]); lon = float(line[21:30])
            last_year = int(line[77:81])
        except ValueError:
            continue
        if lat <= -98.0 or lon <= -998.0:   # IGRA missing-value sentinels
            continue
        stations.append({
            "stnm": sid[6:11], "lat": lat, "lon": lon,
            "name": line[41:71].strip(), "last_year": last_year,
        })
    try:
        with open(_STATION_CACHE, "w") as f:
            json.dump(stations, f)
    except OSError:
        pass
    _stations_mem = stations
    return stations

def _haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1); dl = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def _nearest_station(lat, lon, min_last_year=2015):
    best, best_d = None, 1e18
    for s in _load_stations():
        if s["last_year"] < min_last_year:   # skip decommissioned sites
            continue
        d = _haversine_km(lat, lon, s["lat"], s["lon"])
        if d < best_d:
            best, best_d = s, d
    if best is None:
        return None, None
    return best, best_d

def _parse_uwyo_sounding(html):
    """Pull (alt_m, dir_deg, speed_ms) rows out of the first <PRE> data table."""
    lo = html.find("<PRE>")
    hi = html.find("</PRE>", lo + 1)
    if lo < 0 or hi < 0:
        return []
    block = html[lo + 5:hi]
    rows = []
    for line in block.splitlines():
        # Data rows are fixed-width 7-char fields; headers contain letters.
        if len(line) < 56 or not line[0:7].strip().replace(".", "").isdigit():
            continue
        try:
            hght = float(line[7:14])
            drct = line[42:49].strip()
            sknt = line[49:56].strip()
        except ValueError:
            continue
        if not drct or not sknt:   # wind not reported at this level
            continue
        try:
            rows.append((hght, float(drct), float(sknt) * KNOTS_TO_MS))
        except ValueError:
            continue
    return rows

def _thin_layers(rows, min_gap_m=750.0):
    """Reduce dense sounding levels to ~one per min_gap_m, keeping the ends."""
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r[0])
    kept = [rows[0]]
    for r in rows[1:-1]:
        if r[0] - kept[-1][0] >= min_gap_m:
            kept.append(r)
    if rows[-1] is not kept[-1]:
        kept.append(rows[-1])
    return kept

def _synoptic_candidates(hour):
    """Ordered synoptic hours to try, nearest-first (00/12 preferred)."""
    order = sorted([0, 6, 12, 18], key=lambda h: (min(abs(h - hour), 24 - abs(h - hour)),
                                                   0 if h in (0, 12) else 1))
    return order

@app.route("/fetch_wind", methods=["POST"])
def fetch_wind():
    data = request.get_json()
    try:
        lat = float(data["lat"]); lon = float(data["lon"])
        date = str(data["date"])               # YYYY-MM-DD
        hour = int(data.get("hour", 12))
        year, month, day = date.split("-")
        station, dist_km = _nearest_station(lat, lon)
        if station is None:
            return jsonify({"error": "No radiosonde station found."}), 502

        last_err = None
        for h in _synoptic_candidates(hour):
            ddhh = f"{int(day):02d}{h:02d}"
            qs = urllib.parse.urlencode({
                "region": "naconf", "TYPE": "TEXT:LIST",
                "YEAR": year, "MONTH": f"{int(month):02d}",
                "FROM": ddhh, "TO": ddhh, "STNM": station["stnm"],
            })
            try:
                html = _http_get(f"{UWYO_URL}?{qs}")
            except Exception as e:
                last_err = str(e); continue
            rows = _parse_uwyo_sounding(html)
            if rows:
                layers = [{"alt_m": round(a, 1),
                           "direction_deg": round(d, 1),
                           "speed_ms": round(s, 2)} for a, d, s in _thin_layers(rows)]
                return jsonify({
                    "station": {"stnm": station["stnm"], "name": station["name"],
                                "lat": station["lat"], "lon": station["lon"],
                                "distance_km": round(dist_km, 1)},
                    "used_time_utc": f"{year}-{int(month):02d}-{int(day):02d} {h:02d}Z",
                    "n_levels": len(layers),
                    "layers": layers,
                })
        return jsonify({"error": f"No sounding found for station {station['stnm']} "
                                 f"({station['name']}) near {date}. {last_err or ''}".strip(),
                        "station": {"stnm": station["stnm"], "name": station["name"],
                                    "distance_km": round(dist_km, 1)}}), 404
    except KeyError as e:
        return jsonify({"error": f"Missing required field: {e}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # HOST defaults to 0.0.0.0 so the Raspberry Pi can serve the LAN in prod.
    # For local dev, run `HOST=127.0.0.1 python app.py` to bind localhost only.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=True)
