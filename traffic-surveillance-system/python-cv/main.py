import argparse
import time
import random

import cv2
import gymnasium as gym
import numpy as np
import socketio
from gymnasium import spaces
from stable_baselines3 import DQN
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

VEHICLE_CLASSES = [1, 2, 3, 5, 7]

MODEL_SAVE_PATH = "traffic_dqn_model"


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


# ===================== CV HELPERS =====================
def open_source(src):
    v = int(src) if str(src).isdigit() else src
    cap = cv2.VideoCapture(v)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {src}")
    return cap


def build_default_roi(h, w):
    return np.array(
        [
            (int(w * 0.15), int(h * 0.25)),
            (int(w * 0.85), int(h * 0.25)),
            (int(w * 0.98), int(h * 0.95)),
            (int(w * 0.02), int(h * 0.95)),
        ],
        dtype=np.int32,
    )


def detect_count(result, roi):
    if result.boxes is None:
        return 0

    boxes = result.boxes.xyxy.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)

    count = 0
    for box, cls in zip(boxes, classes):
        if cls not in VEHICLE_CLASSES:
            continue

        x1, y1, x2, y2 = map(int, box)
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        inside = cv2.pointPolygonTest(roi, (cx, cy), False) >= 0
        if inside:
            count += 1

    return count


# ===================== RL ENV =====================
class TrafficEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0, high=200, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4)
        self.state = np.zeros(8, dtype=np.float32)

    def reset(self, seed=None, options=None):
        self.state = np.random.randint(0, 10, size=(8,)).astype(np.float32)
        return self.state, {}

    def step(self, action):
        queues = self.state[:4]
        waits = self.state[4:]

        queues[action] = max(0, queues[action] - random.randint(2, 5))

        reward = -np.sum(queues) - 0.5 * np.sum(waits)

        self.state[:4] = queues + np.random.randint(0, 3, size=4)
        self.state[4:] = waits + 1

        return self.state, reward, False, False, {}


def train_rl():
    env = TrafficEnv()

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=5000,
        batch_size=64,
    )

    model.learn(total_timesteps=5000)
    model.save(MODEL_SAVE_PATH)


def load_or_train_rl():
    try:
        return DQN.load(MODEL_SAVE_PATH)
    except Exception:
        train_rl()
        return DQN.load(MODEL_SAVE_PATH)


# ===================== HYBRID DECISION =====================
def safe_rule_override(roads, active_idx, rl_lane):
    candidates = [r for r in roads if r.queue > 0]

    if not candidates:
        return (active_idx % 4) + 1

    starved = [r for r in candidates if r.wait_age >= STARVATION_LIMIT]
    if starved:
        return max(starved, key=lambda r: r.wait_age).idx

    SMALL_QUEUE_THRESHOLD = 5
    small_roads = [r for r in candidates if r.queue <= SMALL_QUEUE_THRESHOLD]
    if small_roads:
        return min(small_roads, key=lambda r: r.queue).idx

    return rl_lane


def choose_green_time(roads, selected_idx):
    selected = roads[selected_idx - 1]
    sec = 5 + 0.5 * selected.queue
    return max(MIN_GREEN, min(MAX_GREEN, int(sec)))


def set_active(roads, idx, green_sec):
    for r in roads:
        r.signal = "GREEN" if r.idx == idx else "RED"
        r.green_time = green_sec if r.idx == idx else 0


# ===================== MAIN =====================
def main(args):
    roads = [Road(1, args.road1), Road(2, args.road2), Road(3, args.road3), Road(4, args.road4)]

    model = YOLO(MODEL_NAME)
    rl_model = load_or_train_rl()

    sio = socketio.Client(reconnection=True, reconnection_attempts=0)
    try:
        sio.connect(SERVER_URL, transports=["websocket"])
    except Exception:
        pass

    for r in roads:
        r.cap = open_source(r.source)

    active_idx = 1
    phase_start = time.time()
    phase_sec = MIN_GREEN
    last_emit = time.time()
    last_wait_tick = time.time()

    while True:
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

            raw_count = detect_count(result, r.roi)
            r.queue = int(0.7 * r.queue + 0.3 * raw_count)

        now = time.time()

        if now - last_wait_tick >= 1:
            for r in roads:
                if r.idx == active_idx:
                    r.wait_age = 0
                elif r.queue > 0:
                    r.wait_age += 1
                else:
                    r.wait_age = 0
            last_wait_tick = now

        current = roads[active_idx - 1]

        if (now - phase_start >= phase_sec) or current.queue == 0:
            state = np.array(
                [r.queue for r in roads] + [r.wait_age for r in roads],
                dtype=np.float32,
            )

            action, _ = rl_model.predict(state, deterministic=True)
            rl_lane = int(action) + 1

            next_lane = safe_rule_override(roads, active_idx, rl_lane)

            active_idx = next_lane
            phase_sec = choose_green_time(roads, active_idx)
            phase_start = now
            set_active(roads, active_idx, phase_sec)

        if now - last_emit >= EMIT_INTERVAL:
            payload = {
                "lane1": roads[0].queue,
                "lane2": roads[1].queue,
                "lane3": roads[2].queue,
                "lane4": roads[3].queue,
                "active_lane": active_idx,
                "mode": "HYBRID_RL",
                "signal_l1": roads[0].signal,
                "signal_l2": roads[1].signal,
                "signal_l3": roads[2].signal,
                "signal_l4": roads[3].signal,
            }
            if sio.connected:
                sio.emit("traffic-data", payload)
            last_emit = now

        for r in roads:
            if r.frame is not None:
                cv2.imshow(f"Road {r.idx}", r.frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--road1", required=True)
    parser.add_argument("--road2", required=True)
    parser.add_argument("--road3", required=True)
    parser.add_argument("--road4", required=True)
    main(parser.parse_args())
