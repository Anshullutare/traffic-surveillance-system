# 🚦 Traffic Surveillance & Adaptive Signal System

A real-time traffic monitoring and adaptive signal control system using Computer Vision (**YOLOv8**), Python, Node.js, and a live dashboard.

---

# 📌 Project Goal

To reduce traffic congestion and unnecessary waiting by:

* Detecting vehicles in real-time
* Dynamically selecting which road gets GREEN signal
* Adjusting signal time based on traffic

---

# 🧠 Simple Idea (Understand this first)

Instead of fixed timers:

👉 Our system decides:

* **Which road should go first**
* **For how long**

Based on:

* Number of vehicles (queue)
* Waiting time

---

# 🏗️ Project Structure

```
traffic-surveillance-system/
├── backend/        → Node.js server (data handling + API)
├── frontend/       → Dashboard (visualization)
└── python-cv/      → Computer Vision + Signal Logic
```

---

# 🔄 System Architecture (Simple Flow)

```
[Camera / Video]
      ↓
[Python (YOLO Detection)]
      ↓
[Signal Decision Logic]
      ↓
[Send Data via Socket.IO]
      ↓
[Node.js Backend]
      ↓
[Frontend Dashboard]
```

---

# 🔁 Detailed Flow (Step-by-Step)

### 1️⃣ Video Input

* Camera or video file is given as input

---

### 2️⃣ Vehicle Detection (Python)

* YOLOv8 detects vehicles:

  * Car
  * Bike
  * Bus
  * Truck

👉 Output:

* Number of vehicles per road (queue)

---

### 3️⃣ Traffic Logic (Core Algorithm)

System calculates:

* Queue (vehicles)
* Waiting time

👉 Priority formula:

```
priority = 0.6 × queue + 0.3 × waiting_time
```

---

### 4️⃣ Smart Features

✔ Dynamic signal selection
✔ Dynamic green time
✔ Small-queue fast clearing
✔ Starvation prevention

---

### 5️⃣ Data Sent to Backend

Python sends:

```
traffic-data (every 1 second)
```

---

### 6️⃣ Backend (Node.js)

* Receives data
* Stores in MongoDB (optional)
* Sends updates to frontend

---

### 7️⃣ Frontend Dashboard

Displays:

* Vehicle count per road
* Active signal
* Next signal
* Waiting time
* Graph (traffic over time)

---

# 📊 Dashboard Explanation (IMPORTANT)

### 🔹 Total Vehicles

Total vehicles across all roads

---

### 🔹 Density

Traffic load (sum of vehicles)

---

### 🔹 Road Cards

Each road shows:

* Signal (RED/GREEN)
* Vehicle count
* Green time

---

### 🔹 Active Road

Currently GREEN road

---

### 🔹 Next Road

Upcoming road

---

### 🔹 Waiting Time

👉 This is VERY IMPORTANT:

**Definition:**

> Total time (in seconds) that vehicles on other roads are waiting

Example:

```
R1 = GREEN
R2 wait = 10s
R3 wait = 15s
R4 wait = 5s

Total Waiting = 30s
```

👉 Lower waiting time = better system

---

# ⚙️ Setup Instructions

---

## 1️⃣ Backend

```
cd backend
npm install
node server.js
```

Runs on:

```
http://localhost:3000
```

---

## 2️⃣ Frontend

Open:

```
frontend/index.html
```

For 4-road:

```
frontend/index.html
```

---

## 3️⃣ Python (Computer Vision)

```
cd python-cv
pip install -r requirements.txt
```

---

### ▶ Run Single Camera

```
python main.py --source 0
```

---

### ▶ Run 4-Road System

```
python main.py \
--road1 traffic.mp4 \
--road2 traffic2.mp4 \
--road3 traffic3.mp4 \
--road4 traffic.mp4
```

---

# 🚦 Key Features

| Feature                  | Description                 |
| ------------------------ | --------------------------- |
| YOLOv8 Detection         | Real-time vehicle detection |
| Adaptive Signal          | Dynamic signal selection    |
| Smart Timing             | Time based on traffic       |
| Waiting-Time Logic       | Prevents starvation         |
| Small Queue Optimization | Fast clearance              |
| Dashboard                | Live visualization          |
| MongoDB                  | Optional storage            |

---

# 🔌 API

### GET /history?limit=60

Returns last traffic records

---

### Socket Events

| Event            | From → To         | Purpose        |
| ---------------- | ----------------- | -------------- |
| traffic-data     | Python → Server   | Send live data |
| update-dashboard | Server → Frontend | Update UI      |
| history-snapshot | Server → Frontend | Load history   |

---

# 🌟 Final Summary

This system improves traffic by:

✔ Giving priority to busy roads
✔ Avoiding unnecessary waiting
✔ Adjusting signal time dynamically
✔ Ensuring fairness for all roads

---

# 🎯 One-Line Explanation (Viva)

> “We developed a real-time adaptive traffic signal system that dynamically allocates signal priority and timing based on vehicle count and waiting time, reducing congestion and improving traffic flow.”

---
