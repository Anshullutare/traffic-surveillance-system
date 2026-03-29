# 🚦 Traffic Surveillance System

Real-time traffic monitoring using Computer Vision (YOLOv8 + ByteTrack), Node.js backend, and a live dashboard.

```
traffic-surveillance-system/
├── backend/          ← Node.js  (Express + Socket.IO + Mongoose)
├── frontend/         ← Plain HTML/CSS/JS dashboard
└── python-cv/        ← OpenCV + YOLOv8 + ByteTrack + Socket.IO emitter
```

---

## Architecture

```
[Webcam / Video]
      │
      ▼
[python-cv/main.py]          ← YOLOv8 detection + ByteTrack + lane counting
      │  socket.io  "traffic-data"
      ▼
[backend/server.js]          ← Receives, saves to MongoDB, broadcasts
      │  socket.io  "update-dashboard"
      ▼
[frontend/index.html]        ← Live dashboard, Chart.js history graph
```

---

## Quick Start

### 1. Backend

```bash
cd backend
npm install
node server.js
```

Server runs on **http://localhost:3000**

> **MongoDB**: Start a local MongoDB instance, or set `MONGO_URI` env var.  
> If MongoDB is unavailable the server still works — data just won't persist.

### 2. Frontend

Open `frontend/index.html` in a browser (no build step required).

### 3. Python CV

```bash
cd python-cv
pip install -r requirements.txt

# Webcam
python main.py --source 0

# Video file
python main.py --source /path/to/traffic.mp4
```

---

## Features

| Feature | Status |
|---|---|
| YOLOv8 vehicle detection (car, bike, bus, truck) | ✅ |
| ByteTrack object tracking (unique IDs) | ✅ |
| Virtual lane-line crossing counter | ✅ |
| Traffic density scoring (bike=0.5, car=1, truck/bus=2) | ✅ |
| Traffic signal logic (busier lane gets GREEN) | ✅ |
| Real-time Socket.IO push (1 s interval) | ✅ |
| MongoDB persistence | ✅ |
| REST API: `GET /history` | ✅ |
| Live dashboard with Chart.js history graph | ✅ |
| Signal state (RED/GREEN) visual indicators | ✅ |

---

## API

### `GET /history?limit=60`

Returns last N traffic records from MongoDB.

```json
{
  "success": true,
  "data": [
    {
      "lane1": 12,
      "lane2": 8,
      "density": 18.5,
      "signal": "GREEN_LANE1",
      "timestamp": "2024-01-15T10:30:00.000Z"
    }
  ]
}
```

### Socket.IO Events

| Event | Direction | Payload |
|---|---|---|
| `traffic-data` | Python → Server | `{ lane1, lane2, density, signal }` |
| `update-dashboard` | Server → Frontend | `{ lane1, lane2, density, signal, timestamp }` |
| `history-snapshot` | Server → Frontend (on connect) | Array of last 30 records |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PORT` | `3000` | Backend server port |
| `MONGO_URI` | `mongodb://localhost:27017/traffic_surveillance` | MongoDB connection string |
