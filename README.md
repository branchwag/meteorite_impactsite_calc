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
- **A. Type of meteorite** — uses the fall time (radar hit time − fireball end time) and the standard-atmosphere terminal-velocity model to back out which type fits (ordinary chondrite, carbonaceous, or iron) and the implied mass/diameter. If none fit the 2 g – 10 kg range, the hit is flagged as likely not a meteorite.
- **B. Number of rocks** — Rayleigh and geometric radar scattering, forward (size → dBZ) and inverse (dBZ → size/count).
- **C. Ground location** — wind-corrects the descent from the radar hit down to the ground.

**Scenario 2 — I have a dark flight point.** Given the fireball end time and a high-altitude dark-flight radar return, it projects the descent downward (vertical, optionally wind-corrected) for an assumed 100 g chondrite and predicts the lat/lon and clock time the rock passes each lower altitude — i.e. where and when to look on lower radar sweeps, down to the ground.

---

## Inputs

**Radar / dark-flight point** — time (UTC), lat, lon, altitude, dBZ, and range. Altitude is entered in feet (as provided by the NOAA Weather and Climate Toolkit) and auto-converted to meters. Dark-flight altitude must be 18–33 km.

**Fireball end time** (UTC) — from camera footage.

**Wind data** — either a single average wind speed and direction, or speed and direction at multiple altitude layers (e.g. from a radiosonde sounding at weather.uwyo.edu/upperair/sounding.html). Direction follows meteorological convention: the direction the wind is blowing FROM.

**Settings** (right sidebar) — drag coefficient (default 0.8) and per-type densities (OC 3.3, CC 2.7, Iron 7.2 g/cm³).

---

## Output

- **Scenario 1:** meteorite type + mass/diameter per candidate type, a dBZ consistency check, and wind-corrected ground coordinates with Google Maps links.
- **Scenario 2:** a table of predicted radar-hit positions (altitude → fall time → UTC clock time → lat/lon) down to the ground, with Google Maps links.

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
