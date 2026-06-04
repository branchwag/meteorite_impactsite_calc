# Meteorite Impact Site Calculator

A local network web app for working radar-tracked meteorite falls. It determines meteorite type and size from fall time, sanity-checks the radar return against the Rayleigh/geometric scattering equations, applies wind correction, and predicts where on radar (and on the ground) to look.

---

## Project Structure

```
meteorite_impactsite_calc/
├── README.md
├── .gitignore
├── meteor dark flight.xlsx   # source spreadsheet (Spherical tab = ISA table)
└── webapp/
    ├── app.py                # Flask web server
    └── static/
        └── index.html        # Browser UI
```

---

## How It Works

The app has two scenarios.

**Scenario 1 — I have a radar hit.** Given the fireball end time and a radar return (time, lat/lon, altitude, dBZ, range), it:
- **A. Type of meteorite** — uses the fall time (radar hit time − fireball end time) and the standard-atmosphere terminal-velocity model to back out which type fits and the implied mass/diameter. Each type has its own plausible mass range (ordinary chondrite 2 g–10 kg, carbonaceous 2 g–500 g, iron 5 g–10 kg); if none fit, the hit is flagged as likely not a meteorite.
- **B. Number of rocks** — Rayleigh and geometric radar scattering, forward (size → dBZ) and inverse (dBZ → size/count). Given the observed dBZ, range, and the fragment size from A, it reports how many rocks of that size are needed to produce the return.
- **C. Ground location** — wind-corrects the descent from the radar hit down to the ground.

**Scenario 2 — I have a dark flight point.** Uses the same inputs as Scenario 1 (fireball end time, dark-flight start altitude, and a radar return point) and runs the same A/B/C analysis. In addition, it predicts where to look on radar: projecting the descent downward (vertical, optionally wind-corrected) for an assumed 100 g chondrite and reporting the lat/lon and clock time the rock passes each lower altitude, down to the ground.

The two scenarios share inputs and the A/B/C calculation — the difference is framing (Scenario 1 starts from a known radar hit; Scenario 2 from a dark-flight solution where the start altitude is known) and the extra radar-hit prediction in Scenario 2.

---

## Inputs

**Radar / dark-flight point** — time (UTC), lat, lon, altitude, dBZ, and range. Altitude is entered in feet (as provided by the NOAA Weather and Climate Toolkit) and auto-converted to meters. Dark-flight altitude must be 18–33 km.

**Fireball end time** (UTC) — from camera footage.

**Wind data** — either a single average wind speed and direction, or speed and direction at multiple altitude layers. Layers can be entered by hand or auto-filled with **Fetch nearest sounding**: the backend finds the closest active radiosonde station to the point (IGRA station list), pulls that station's University of Wyoming sounding for the closest synoptic hour to the entered time, and fills the altitude-layer table (heights in metres, knots converted to m/s). Direction follows meteorological convention: the direction the wind is blowing FROM. Sources: weather.uwyo.edu/upperair/sounding.html and the IGRA station list at ncei.noaa.gov.

**Settings** (right sidebar) — drag coefficient (default 0.8) and per-type densities (OC 3.3, CC 2.7, Iron 7.2 g/cm³).

---

## Output

- **Scenario 1:** meteorite type + mass/diameter per candidate type, a dBZ consistency check, and wind-corrected ground coordinates with Google Maps links.
- **Scenario 2:** the same A/B/C output as Scenario 1, plus a "where to look on radar" table of predicted positions (altitude → fall time → UTC clock time → lat/lon) down to the ground, with Google Maps links.

---

## Setup

```bash
conda activate meteorite
conda install numpy flask
```

## Run

The bind address is controlled by the `HOST` env var (default `0.0.0.0`); `PORT` defaults to `5000`.

**Local dev (this machine):** bind to localhost so the dev instance stays on this computer only and can't be confused with the Pi's prod instance.

```bash
cd webapp
HOST=127.0.0.1 python app.py
# browse http://localhost:5000
```

`debug=True` is on, so edits auto-reload — no need to restart after each change.

**Prod (Raspberry Pi):** bind to all interfaces so other devices on the LAN can reach it.

```bash
cd webapp
python app.py          # HOST defaults to 0.0.0.0
```

If port 5000 is already in use from a previous session:

```bash
pkill -f app.py
```

## Access (prod / LAN)

Find the Pi's local IP:

```bash
ip addr show | grep 'inet ' | grep -v 127
```

Use the `wlan0` address, e.g. `192.168.1.213`. Anyone on the same WiFi opens:

```
http://192.168.1.213:5000
```

Dev and prod can both use port 5000 — they're different hosts, so there's no conflict.

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

- The fireball end time comes from camera/dashcam footage; an accurate UTC time is the most important input for the fall-time calculation.
- Air density and gravity are read from the `Spherical` tab of `meteor dark flight.xlsx` (ISA, 0–30 km, with 31–33 km extrapolated). Terminal velocities and fall times are validated against that spreadsheet.
- The Tailscale IP (`tailscale0`) can also reach the web app if the other device is on the same Tailscale network.
- `nohup.out` is gitignored — it is generated when running the server in the background with `nohup python app.py &`.
