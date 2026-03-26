# Meteorite Strewn Field Calculator

A local network web app for estimating meteorite strewn fields by intersecting observation planes, fitting radar trajectories, applying wind correction, and modeling fragment scatter by mass.

---

## Project Structure

```
plane_intersection/
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

If radar data is available, the trajectory is instead fit directly through the radar hit points using least squares, which is significantly more accurate. A single radar hit assumes vertical descent. Wind correction is then applied per altitude layer to drift each fragment mass class to its estimated landing position, producing a strewn field ellipse for each class.

---

## Inputs

**Radar hits** — lat, lon, altitude from the middle of each radar return. Altitude is entered in feet (as provided by the NOAA Weather and Climate Toolkit) and automatically converted to meters. A single hit assumes vertical descent; two or more fits a trajectory line. Ground observations can optionally supplement the radar data.

**Ground observations** (optional) — lat, lon, altitude of the observer plus azimuth (compass bearing the fireball traveled toward) and elevation above the horizon in degrees. Used when radar data is unavailable or to supplement a single radar hit.

**Wind data** — either a single average wind speed and direction, or speed and direction at multiple altitude layers (e.g. from a weather balloon sounding at weather.uwyo.edu). Direction follows meteorological convention: the direction the wind is blowing FROM.

A **Load Sample Data** button is available to pre-fill the form with example inputs for testing.

---

## Output

- Estimated landing coordinates for five fragment mass classes: 1g, 10g, 100g, 500g, 1kg+
- Strewn field ellipses per mass class
- Trajectory centerline
- Google Maps links for each landing point
- GeoJSON file download (loadable in Google Maps or geojson.io)
- KMZ file download (loadable in Google Earth)

---

## Setup

```bash
conda activate meteorite
conda install numpy flask
```

## Run

```bash
cd webapp
python app.py
```

If port 5000 is already in use from a previous session:

```bash
pkill -f app.py
python app.py
```

## Access

Find your local IP:

```bash
ip addr show | grep 'inet ' | grep -v 127
```

Use the `wlan0` address, e.g. `192.168.1.213`. Anyone on the same WiFi opens:

```
http://192.168.1.213:5000
```

---

## Dependencies

| Package | Purpose                    |
|---------|----------------------------|
| numpy   | Coordinate math and linear algebra |
| flask   | Web server                 |

```bash
conda install numpy flask
```

---

## Notes

- Azimuth and elevation are the most error-prone inputs when using ground observations. Camera or dashcam footage gives more accurate angles than eyeballing.
- The Tailscale IP (`tailscale0`) can also reach the web app if the other device is on the same Tailscale network.
- `nohup.out` is gitignored — it is generated when running the server in the background with `nohup python app.py &`.
