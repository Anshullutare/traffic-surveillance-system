"""
Traffic Surveillance System — Python CV Module
================================================
Pipeline:
  Video / Webcam
    → YOLOv8 detection (car, bike, bus, truck)
    → ByteTrack object tracking (unique IDs)
    → Virtual lane-line crossing counter
    → Traffic-signal logic
    → Socket.IO → Node.js backend (every 1 s)
"""

import cv2
import time
import random
import argparse
import collections

import socketio
import numpy as np
from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────

SERVER_URL   = "http://localhost:3000"
EMIT_INTERVAL = 1.0          # seconds between data emissions
CONF_THRESH   = 0.4
IOU_THRESH    = 0.45

# COCO class names we care about  (YOLOv8 uses COCO indices)
VEHICLE_CLASSES = {
    1:  "bike",    # bicycle
    2:  "bike",    # car  (mapped generically for density)
    3:  "bike",    # motorcycle
    5:  "car",     # bus  — we separate below for density
    7:  "truck",   # truck
}

# Override fine-grained for density calc
DENSITY_WEIGHT = {
    "bike":  0.5,
    "car":   1.0,
    "bus":   2.0,
    "truck": 2.0,
}

# Class index → fine-grained name
COCO_FINE = {
    1: "bike",
    2: "car",
    3: "bike",
    5: "bus",
    7: "truck",
}

COCO_COLORS = {
    "bike":  (0,   200, 255),
    "car":   (0,   255, 100),
    "bus":   (255, 150,   0),
    "truck": (200,   0, 255),
}

# ── Socket.IO client ───────────────────────────────────────────────────────────

sio = socketio.Client(reconnection=True, reconnection_attempts=0)

@sio.event
def connect():
    print("[Socket.IO] Connected to Node backend ✓")

@sio.event
def disconnect():
    print("[Socket.IO] Disconnected from Node backend")

@sio.event
def connect_error(data):
    print(f"[Socket.IO] Connection error: {data}")

def try_connect():
    """Try to connect; non-fatal if backend isn't up."""
    try:
        sio.connect(SERVER_URL, transports=["websocket"])
    except Exception as e:
        print(f"[Socket.IO] Could not connect: {e}  (running in offline mode)")

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_center(box):
    """Return (cx, cy) from xyxy box."""
    x1, y1, x2, y2 = box
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def draw_lane_lines(frame, lane1_y, lane2_y, w):
    """Draw two horizontal virtual counting lines."""
    cv2.line(frame, (0, lane1_y), (w, lane1_y), (0, 255, 255), 2)
    cv2.putText(frame, "LANE 1", (10, lane1_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.line(frame, (0, lane2_y), (w, lane2_y), (255, 100, 0), 2)
    cv2.putText(frame, "LANE 2", (10, lane2_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 100, 0), 1)

def draw_hud(frame, lane1_count, lane2_count, density, signal, fps):
    """Overlay stats panel."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (w - 260, 0), (w, 185), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    signal_color = (0, 220, 0) if signal == "GREEN_LANE1" else (0, 80, 220)
    lines = [
        (f"FPS : {fps:.1f}",            (200, 200, 200)),
        (f"Lane 1 : {lane1_count}",     (0, 255, 255)),
        (f"Lane 2 : {lane2_count}",     (255, 150, 80)),
        (f"Density: {density:.2f}",     (180, 255, 180)),
        (f"Signal : {signal}",          signal_color),
    ]
    for i, (text, color) in enumerate(lines):
        cv2.putText(frame, text, (w - 250, 28 + i * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 1, cv2.LINE_AA)

# ── Main ───────────────────────────────────────────────────────────────────────

def main(source=0):
    # Load model
    print("[YOLO] Loading YOLOv8n …")
    model = YOLO("yolov8n.pt")   # downloads automatically on first run
    print("[YOLO] Model loaded ✓")

    try_connect()

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[Video] Source opened — {w}×{h}")

    # Define virtual lane lines at 40 % and 65 % of frame height
    lane1_y = int(h * 0.40)
    lane2_y = int(h * 0.65)

    # Tracking state
    # prev_centers[track_id] = previous cy
    prev_centers: dict[int, int] = {}
    counted_ids:  set[int]       = set()   # IDs that already crossed any line

    lane1_count = 0
    lane2_count = 0
    lane1_density = 0.0
    lane2_density = 0.0

    # Per-second emission
    last_emit   = time.time()
    fps_counter = collections.deque(maxlen=30)
    frame_t     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            # Loop video file
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # ── Detection + Tracking ───────────────────────────────────────────
        results = model.track(
            frame,
            persist=True,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            classes=list(COCO_FINE.keys()),
            tracker="bytetrack.yaml",
            verbose=False,
        )[0]

        # ── Process detections ─────────────────────────────────────────────
        if results.boxes is not None and results.boxes.id is not None:
            boxes   = results.boxes.xyxy.cpu().numpy()
            ids     = results.boxes.id.cpu().numpy().astype(int)
            classes = results.boxes.cls.cpu().numpy().astype(int)
            confs   = results.boxes.conf.cpu().numpy()

            for box, tid, cls, conf in zip(boxes, ids, classes, confs):
                if cls not in COCO_FINE:
                    continue
                label = COCO_FINE[cls]
                color = COCO_COLORS[label]
                x1, y1, x2, y2 = map(int, box)
                cx, cy = get_center(box)

                # Draw bounding box + ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, (cx, cy), 4, color, -1)
                tag = f"{label} #{tid} {conf:.2f}"
                cv2.putText(frame, tag, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                # ── Lane crossing logic ────────────────────────────────────
                if tid in prev_centers and tid not in counted_ids:
                    prev_cy = prev_centers[tid]
                    w_density = DENSITY_WEIGHT.get(label, 1.0)

                    # Crossed lane 1 (moving downward through line)
                    if prev_cy < lane1_y <= cy:
                        lane1_count  += 1
                        lane1_density += w_density
                        counted_ids.add(tid)
                        print(f"  [Lane1] +1 {label} #{tid}  →  total={lane1_count}  density={lane1_density:.1f}")

                    # Crossed lane 2
                    elif prev_cy < lane2_y <= cy:
                        lane2_count  += 1
                        lane2_density += w_density
                        counted_ids.add(tid)
                        print(f"  [Lane2] +1 {label} #{tid}  →  total={lane2_count}  density={lane2_density:.1f}")

                prev_centers[tid] = cy

        # ── Traffic signal logic ───────────────────────────────────────────
        signal = "GREEN_LANE1" if lane1_count >= lane2_count else "GREEN_LANE2"

        # ── Draw overlays ──────────────────────────────────────────────────
        draw_lane_lines(frame, lane1_y, lane2_y, w)
        fps_counter.append(1.0 / max(time.time() - frame_t, 1e-6))
        frame_t = time.time()
        draw_hud(frame, lane1_count, lane2_count,
                 lane1_density + lane2_density,
                 signal, sum(fps_counter) / len(fps_counter))

        cv2.imshow("Traffic Surveillance — CV Module", frame)

        # ── Emit data every EMIT_INTERVAL seconds ─────────────────────────
        now = time.time()
        if now - last_emit >= EMIT_INTERVAL:
            payload = {
                "lane1":   lane1_count,
                "lane2":   lane2_count,
                "density": round(lane1_density + lane2_density, 2),
                "signal":  signal,
            }
            print(f"[Emit] {payload}")
            if sio.connected:
                sio.emit("traffic-data", payload)
            last_emit = now

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if sio.connected:
        sio.disconnect()

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Traffic CV Module")
    parser.add_argument(
        "--source", default="0",
        help="Video source: 0 for webcam, or path to video file (e.g. traffic.mp4)"
    )
    args = parser.parse_args()

    # Allow numeric webcam index
    source = int(args.source) if args.source.isdigit() else args.source
    main(source)
