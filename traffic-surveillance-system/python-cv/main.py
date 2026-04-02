"""
Simple 4-road adaptive traffic signal controller (viva-friendly).

What it does:
1) Detect vehicles in each road camera (YOLO).
2) Count vehicles inside a fixed ROI (signal area).
3) Pick next green road using queue + waiting-time priority.
4) Allocate green time dynamically (short for low queue, longer for high queue).

Run:
  python main_4road_simple.py --road1 traffic.mp4 --road2 traffic2.mp4 --road3 traffic3.mp4 --road4 traffic.mp4
  python main_4road_simple.py --road1 0 --road2 0 --road3 0 --road4 0
"""

import argparse
import time

import cv2
import numpy as np
import socketio
from ultralytics import YOLO

MODEL_NAME = "yolov8s.pt"
CONF_THRESH = 0.30
IOU_THRESH = 0.40
IMGSZ = 640
SERVER_URL = "http://localhost:3000"

MIN_GREEN = 6
MAX_GREEN = 20
STARVATION_LIMIT = 25
EMIT_INTERVAL = 1.0

VEHICLE_CLASSES = [1, 2, 3, 5, 7]  # bike, car, bike, bus, truck
CLASS_NAMES = {1: "bicycle", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
CLASS_COLORS = {
    1: (0, 220, 255),
    2: (0, 255, 120),
    3: (0, 200, 255),
    5: (255, 170, 0),
    7: (220, 0, 255),
}


class Road:
    def __init__(self, idx, source):
        self.idx = idx
        self.source = source
        self.cap = None
        self.frame = None
        self.roi = None
        self.queue = 0
        self.wait_age = 0
        self.signal = "RED"
        self.green_time = 0


def open_source(src):
    v = int(src) if str(src).isdigit() else src
    cap = cv2.VideoCapture(v)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")
    return cap


def build_default_roi(h, w):
    # Fixed trapezoid covering center road perspective.
    return np.array(
        [
            (int(w * 0.15), int(h * 0.25)),
            (int(w * 0.85), int(h * 0.25)),
            (int(w * 0.98), int(h * 0.95)),
            (int(w * 0.02), int(h * 0.95)),
        ],
        dtype=np.int32,
    )


def detect_count_and_draw(result, frame, roi):
    if result.boxes is None:
        return 0
    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()
    count = 0
    for box, cls, conf in zip(boxes, classes, confs):
        if cls not in VEHICLE_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        inside = cv2.pointPolygonTest(roi, (cx, cy), False) >= 0
        if inside:
            count += 1

        # Draw all vehicle detections; highlight those inside ROI.
        color = CLASS_COLORS.get(cls, (180, 180, 180))
        thickness = 2 if inside else 1
        box_color = color if inside else (120, 120, 120)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        cv2.circle(frame, (cx, cy), 3, box_color, -1)
        tag = f"{CLASS_NAMES.get(cls, 'veh')} {conf:.2f}"
        cv2.putText(frame, tag, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, box_color, 1)
    return count


def choose_next_road(roads, active_idx):
    candidates = [r for r in roads if r.queue > 0]
    if not candidates:
        return (active_idx % 4) + 1  # fallback rotation

    # Small-queue priority optimization (safe override):
    # Give quick clearance to very small queues, but only when
    # no road is under heavy congestion.
    SMALL_QUEUE_THRESHOLD = 5
    HEAVY_QUEUE_THRESHOLD = 30

    small_roads = [r for r in candidates if r.queue <= SMALL_QUEUE_THRESHOLD]
    if small_roads:
        max_queue = max(r.queue for r in candidates)
        if max_queue < HEAVY_QUEUE_THRESHOLD:
            # Prefer smaller queue; on ties, prefer road that waited longer.
            return min(small_roads, key=lambda r: (r.queue, -r.wait_age)).idx

    starved = [r for r in candidates if r.wait_age >= STARVATION_LIMIT]
    if starved:
        return max(starved, key=lambda r: r.wait_age).idx

    # Simple, explainable priority:
    # priority = 0.6*queue + 0.3*wait
    return max(candidates, key=lambda r: (0.6 * r.queue + 0.3 * r.wait_age)).idx


def choose_green_time(roads, selected_idx):
    selected = roads[selected_idx - 1]
    base = 5
    per_vehicle = 0.5
    sec = base + per_vehicle * selected.queue

    # Fairness trim if other roads are waiting too long.
    max_other_wait = max((r.wait_age for r in roads if r.idx != selected_idx), default=0)
    sec -= min(4, max_other_wait // 8)
    sec = int(sec)
    return max(MIN_GREEN, min(MAX_GREEN, sec))


def set_active(roads, idx, green_sec):
    for r in roads:
        r.signal = "GREEN" if r.idx == idx else "RED"
        r.green_time = green_sec if r.idx == idx else 0


def draw_overlay(frame, road, active_idx, next_idx):
    top = (0, 120, 0) if road.signal == "GREEN" else (0, 0, 140)
    border = (0, 255, 0) if road.signal == "GREEN" else (0, 0, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 70), top, -1)
    cv2.rectangle(frame, (1, 1), (frame.shape[1] - 2, frame.shape[0] - 2), border, 3)
    cv2.polylines(frame, [road.roi], True, (255, 255, 0), 2)
    cv2.putText(frame, f"Road {road.idx} {road.signal} q={road.queue} wait={road.wait_age}s", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1)
    cv2.putText(frame, f"ACTIVE: Road {active_idx}  NEXT: Road {next_idx}", (8, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (140, 255, 140), 1)


def main(args):
    roads = [Road(1, args.road1), Road(2, args.road2), Road(3, args.road3), Road(4, args.road4)]
    model = YOLO(MODEL_NAME)

    sio = socketio.Client(reconnection=True, reconnection_attempts=0)
    try:
        sio.connect(SERVER_URL, transports=["websocket"])
        sio.emit("clear-session", {})
    except Exception:
        pass

    for r in roads:
        r.cap = open_source(r.source)

    active_idx = 1
    next_idx = 1
    phase_start = time.time()
    phase_sec = MIN_GREEN
    last_emit = time.time()
    last_wait_tick = time.time()

    while True:
        # 1) Read + detect for all roads.
        for r in roads:
            ok, frame = r.cap.read()
            if not ok:
                r.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = r.cap.read()
                if not ok:
                    continue
            r.frame = frame
            if r.roi is None:
                h, w = frame.shape[:2]
                r.roi = build_default_roi(h, w)

            result = model.predict(
                frame,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                imgsz=IMGSZ,
                classes=VEHICLE_CLASSES,
                verbose=False,
            )[0]
            raw_count = detect_count_and_draw(result, frame, r.roi)
            # smooth to avoid flicker
            smoothed = int(0.7 * r.queue + 0.3 * raw_count)
            if raw_count == 0 and r.queue > 0:
                r.queue = max(1, smoothed)
            else:
                r.queue = smoothed

        # 2) Update waits every second.
        now = time.time()
        if now - last_wait_tick >= 1.0:
            for r in roads:
                if r.idx == active_idx and r.signal == "GREEN":
                    r.wait_age = 0
                elif r.queue > 0:
                    r.wait_age += 1
                else:
                    r.wait_age = 0
            last_wait_tick = now

        # 3) Switch when current phase ends or current road empties.
        current = roads[active_idx - 1]
        if (now - phase_start >= phase_sec) or (current.queue == 0):
            active_idx = choose_next_road(roads, active_idx)
            phase_sec = choose_green_time(roads, active_idx)
            phase_start = now
            set_active(roads, active_idx, phase_sec)

        # Predict next road for display.
        next_idx = choose_next_road([r for r in roads if r.idx != active_idx] + [roads[active_idx - 1]], active_idx)

        # 4) Draw + show.
        for r in roads:
            if r.frame is None:
                continue
            display = r.frame.copy()
            draw_overlay(display, r, active_idx, next_idx)
            cv2.imshow(f"Road {r.idx}", display)

        # 5) Emit to backend/dashboard.
        if now - last_emit >= EMIT_INTERVAL:
            payload = {
                "lane1": roads[0].queue,
                "lane2": roads[1].queue,
                "lane3": roads[2].queue,
                "lane4": roads[3].queue,
                "density": float(sum(r.queue for r in roads)),
                "signal": " ".join([f"L{r.idx}:{r.signal}" for r in roads]),
                "signal_l1": roads[0].signal,
                "signal_l2": roads[1].signal,
                "signal_l3": roads[2].signal,
                "signal_l4": roads[3].signal,
                "green_time_l1": roads[0].green_time,
                "green_time_l2": roads[1].green_time,
                "green_time_l3": roads[2].green_time,
                "green_time_l4": roads[3].green_time,
                "active_lane": active_idx,
                "next_lane": next_idx,
                "total_waiting": int(sum(r.wait_age for r in roads if r.idx != active_idx)),
            }
            if sio.connected:
                sio.emit("traffic-data", payload)
            last_emit = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    for r in roads:
        if r.cap is not None:
            r.cap.release()
    cv2.destroyAllWindows()
    if sio.connected:
        sio.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--road1", required=True, help="Road 1 source (camera index or video path)")
    parser.add_argument("--road2", required=True, help="Road 2 source (camera index or video path)")
    parser.add_argument("--road3", required=True, help="Road 3 source (camera index or video path)")
    parser.add_argument("--road4", required=True, help="Road 4 source (camera index or video path)")
    main(parser.parse_args())




