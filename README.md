# Smart Attendance System

QR code + Face recognition + GPS — all 5 phases of the SOP implemented.

## Stack

| Layer | Technology |
|---|---|
| Backend | Flask + SQLite |
| Auth / QR | PyJWT + qrcode |
| Face detection | face-api.js v0.22.2 (SSD MobileNet V1, browser-side) |
| Face recognition | FaceNet 128-d embeddings, cosine similarity (numpy) |
| GPS | Browser Geolocation API + geopy Haversine |
| Map | Leaflet.js (OpenStreetMap) |
| Deployment size | ~85 MB — fits free-tier (500 MB limit) |

## Quick Start

**Windows**
```
run.bat
```

**Linux / Mac**
```
bash run.sh
```

Then open **http://localhost:5000**

## How to Use

### Students
1. **Register** — go to *Register Student*, fill in ID / name / email
2. **Enroll face** — go to *Enroll Face*, enter your Student ID, start camera, blink twice (liveness check), click Capture
3. **Mark attendance** — go to *Mark Attendance*:
   - Scan the QR code shown by your lecturer (or paste the token)
   - Pass the face + liveness check
   - Allow GPS location
   - Click **Mark Me Present**

### Lecturers / Admin
1. Go to **Admin Dashboard → Sessions → New Session**
2. Enter course name, faculty ID, classroom coordinates
3. Click **Create & Generate QR** — a QR modal appears
4. Display the QR on screen; students scan it with their phones
5. Click **Refresh QR** every 10 minutes (tokens expire)
6. Monitor **Live Attendance** in real time
7. Export CSV or view the **Map** of check-in locations

## Attendance Status

| Status | Meaning |
|---|---|
| `present` | All 3 checks passed (QR + face + GPS) |
| `partial` | 1–2 checks passed — flagged for manual review |
| `rejected` | No checks passed |

## Project Structure

```
├── backend/
│   └── app.py          # Flask API — all routes, models, helpers
├── frontend/
│   ├── index.html      # Single-page app
│   └── static/
│       ├── app.js      # All frontend logic
│       ├── style.css   # Styles
│       └── manifest.json
├── database/
│   └── attendance.db   # SQLite (auto-created)
├── init_db.py          # Create tables (run once)
├── requirements.txt
├── run.bat             # Windows start script
├── run.sh              # Linux/Mac start script
└── .env                # Config (secret key, thresholds, etc.)
```

## Configuration (.env)

```
SECRET_KEY=...                  # JWT signing key — change in production
QR_TOKEN_EXPIRY_MINUTES=10      # How long a QR code stays valid
GPS_RADIUS_METRES=100           # Allowed distance from classroom
FACE_THRESHOLD=0.75             # Cosine similarity threshold (0–1, higher = stricter)
```
