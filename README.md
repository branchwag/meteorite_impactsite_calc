# Meteorite Strewn Field Calculator

A tool for estimating meteorite strewn fields by intersecting observation planes, fitting radar trajectories, applying wind correction, and modeling fragment scatter by mass. Includes a standalone Python script and a local network web app.

---

## Project Structure

```
plane_intersection/
├── intersectioncalc.py   # Standalone CLI calculator
├── README.md
├── .gitignore
└── webapp/
    ├── app.py            # Flask web server
    └── static/
        └── index.html    # Browser UI
```

---

## How It Works

Each observer watches a fireball cross the sky and records their location plus the azimuth and elevation of the fireball. Each observation defines a vertical plane through the sky — the meteorite's trajectory is the line where two planes intersect.

If radar data is available, the trajectory is instead fit directly through the radar hit points using least squares, which is significantly more accurate. Wind correction is then applied per altitude layer to drift each fragment mass class to its estimated landing position, producing a strewn field ellipse for each class.

---

## Inputs

**Radar hits** — lat, lon, altitude (meters) from the middle of each radar return. Two or more hits gives the most accurate trajectory. One hit can be combined with a ground observation.

**Ground observations** — lat, lon, altitude of the observer plus azimuth (compass bearing the fireball traveled toward) and elevation above the horizon in degrees. Used when radar data is unavailable or to supplement a single radar hit.

**Wind data** — either a single average wind speed and direction, or speed and direction at multiple altitude layers (e.g. from a weather balloon sounding at weather.uwyo.edu). Direction follows meteorological convention: the direction the wind is blowing FROM.

---

## Output

- Estimated landing coordinates for five fragment mass classes: 1g, 10g, 100g, 500g, 1kg+
- Strewn field ellipses per mass class
- Trajectory centerline
- GeoJSON file loadable in Google Earth, Google Maps, or geojson.io

---

## Standalone Script

Edit the observation values at the top of `intersectioncalc.py`, then run:

```bash
conda activate meteorite
python intersectioncalc.py
```

Output is printed to the console and saved as `strewn_field.geojson` in the same directory.

---

## Web App (Local Network)

### Setup

```bash
conda activate meteorite
conda install flask
```

### Run

```bash
cd webapp
python app.py
```

If port 5000 is already in use from a previous session:

```bash
pkill -f app.py
python app.py
```

### Access

Find your local IP:

```bash
ip addr show | grep 'inet ' | grep -v 127
```

Use the `wlan0` address, e.g. `192.168.1.213`. Anyone on the same WiFi opens:

```
http://192.168.1.213:5000
```

The GeoJSON download button appears in the results after each calculation.

---

## Dependencies

| Package | Used by      |
|---------|--------------|
| numpy   | Both         |
| flask   | Web app only |

```bash
conda install numpy flask
```

---

## Notes

- Azimuth and elevation are the most error-prone inputs when using ground observations. Camera or dashcam footage gives more accurate angles than eyeballing.
- The Tailscale IP (`tailscale0`) can also reach the web app if the other device is on the same Tailscale network.
- `nohup.out` is gitignored — it is generated when running the server in the background with `nohup python app.py &`.
