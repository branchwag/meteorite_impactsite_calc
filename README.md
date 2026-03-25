# Meteorite Intersection Calculator

A tool for estimating meteorite impact sites by intersecting observation planes from two ground-based witnesses. Includes a standalone Python script and a local network web app.

---

## Project Structure

```
plane_intersection/
├── intersectioncalc.py   # Standalone CLI calculator
└── webapp/
    ├── app.py            # Flask web server
    └── static/
        └── index.html    # Browser UI
```

---

## How It Works

Each observer watches a fireball cross the sky and records:
- Their **location** (latitude, longitude, altitude)
- The **azimuth** — compass direction the fireball was traveling toward (0=N, 90=E, 180=S, 270=W)
- The **elevation** — angle above the horizon in degrees

Each observation defines a vertical **plane** slicing through the sky along that line of sight. The meteorite's actual trajectory is the **line where the two planes intersect**. The script walks that line down to ground level to estimate the impact coordinates.

---

## Standalone Script

Edit the observation values at the top of `intersectioncalc.py`:

```python
obs1 = {
    "lat":       34.0522,
    "lon":      -118.2437,
    "alt":       100.0,      # meters
    "azimuth":   45.0,       # degrees
    "elevation": 30.0,       # degrees
}
```

Run it:

```bash
conda activate meteorite
python intersectioncalc.py
```

Output includes the trajectory direction vector, estimated impact lat/lon, and a Google Maps link.

---

## Web App (Local Network)

The web app lets anyone on the same WiFi network enter observations through a browser — no Python required on their end.

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

### Access

Find your local IP:

```bash
ip addr show | grep 'inet ' | grep -v 127
```

Look for the `wlan0` entry, e.g. `192.168.1.213`. Anyone on the same WiFi opens:

```
http://192.168.1.213:5000
```

---

## Dependencies

| Package | Used by |
|---------|---------|
| numpy   | Both    |
| flask   | Web app only |

Install via conda:

```bash
conda install numpy flask
```

---

## Notes

- Azimuth and elevation are the most error-prone inputs. Camera or dashcam footage will give more accurate angles than eyeballing.
- The altitude field can be left as `0` if unknown — it has minimal effect on the result.
- The Tailscale IP (`tailscale0`) can also be used to reach the web app if the other device is on the same Tailscale network.
