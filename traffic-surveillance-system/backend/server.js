const express = require('express');
const http    = require('http');
const { Server } = require('socket.io');
const cors    = require('cors');
const mongoose = require('mongoose');

// ── App Setup ──────────────────────────────────────────────────────────────────
const app    = express();
const server = http.createServer(app);
const io     = new Server(server, { cors: { origin: '*', methods: ['GET','POST'] } });

app.use(cors());
app.use(express.json());

// ── MongoDB ────────────────────────────────────────────────────────────────────
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/traffic_surveillance';

mongoose.connect(MONGO_URI)
  .then(() => console.log('[MongoDB] Connected ✓'))
  .catch(err => console.warn('[MongoDB] Offline mode:', err.message));

// ── Schema ─────────────────────────────────────────────────────────────────────
const trafficSchema = new mongoose.Schema({
  lane1:          { type: Number, required: true },
  lane2:          { type: Number, required: true },
  lane3:          { type: Number, default: 0 },
  lane4:          { type: Number, default: 0 },
  density:        { type: Number, required: true },
  signal:         { type: String, required: true },
  signal_l1:      { type: String, default: 'RED' },
  signal_l2:      { type: String, default: 'RED' },
  signal_l3:      { type: String, default: 'RED' },
  signal_l4:      { type: String, default: 'RED' },
  green_time_l1:  { type: Number, default: 0 },
  green_time_l2:  { type: Number, default: 0 },
  green_time_l3:  { type: Number, default: 0 },
  green_time_l4:  { type: Number, default: 0 },
  arrival_l1:     { type: Number, default: 0 },
  arrival_l2:     { type: Number, default: 0 },
  arrival_l3:     { type: Number, default: 0 },
  arrival_l4:     { type: Number, default: 0 },
  active_lane:    { type: Number, default: 0 },
  total_waiting:  { type: Number, default: 0 },
  session_id:     { type: String, default: '' },
  timestamp:      { type: Date,   default: Date.now },
});

const TrafficRecord = mongoose.model('TrafficRecord', trafficSchema);

// ── Session tracking ───────────────────────────────────────────────────────────
let currentSessionId = '';

function newSessionId() {
  return `session_${Date.now()}`;
}

// ── REST API ───────────────────────────────────────────────────────────────────
app.get('/health', (req, res) => {
  res.json({ status: 'ok', session: currentSessionId, timestamp: new Date().toISOString() });
});

// Return history for current session only (fresh each run)
app.get('/history', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 60;
    const query = currentSessionId ? { session_id: currentSessionId } : {};
    const records = await TrafficRecord.find(query)
      .sort({ timestamp: -1 })
      .limit(limit)
      .lean();
    res.json({ success: true, session: currentSessionId, data: records.reverse() });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Return all-time history
app.get('/history/all', async (req, res) => {
  try {
    const limit = parseInt(req.query.limit) || 200;
    const records = await TrafficRecord.find()
      .sort({ timestamp: -1 })
      .limit(limit)
      .lean();
    res.json({ success: true, data: records.reverse() });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// Manual DB clear endpoint
app.delete('/history', async (req, res) => {
  try {
    const result = await TrafficRecord.deleteMany({});
    res.json({ success: true, deleted: result.deletedCount });
  } catch (err) {
    res.status(500).json({ success: false, error: err.message });
  }
});

// ── Socket.IO ──────────────────────────────────────────────────────────────────
io.on('connection', (socket) => {
  console.log(`[Socket.IO] Client connected: ${socket.id}`);

  // Send current session snapshot to new frontend
  if (currentSessionId) {
    TrafficRecord.find({ session_id: currentSessionId })
      .sort({ timestamp: -1 }).limit(30).lean()
      .then(records => socket.emit('history-snapshot', records.reverse()))
      .catch(() => {});
  }

  // ── Python CV starts a new session → clear old data ──────────────────────
  socket.on('clear-session', async () => {
    currentSessionId = newSessionId();
    console.log(`\n[Session] New session started: ${currentSessionId}`);
    console.log('[Session] Clearing previous session data from DB …');

    try {
      if (mongoose.connection.readyState === 1) {
        const result = await TrafficRecord.deleteMany({});
        console.log(`[Session] Cleared ${result.deletedCount} old records ✓`);
      }
    } catch (err) {
      console.warn('[Session] Clear failed:', err.message);
    }

    // Notify all frontends to reset their UI
    io.emit('session-reset', { session: currentSessionId });
  });

  // ── Receive traffic data from Python ─────────────────────────────────────
  socket.on('traffic-data', async (data) => {
    const {
      lane1, lane2, lane3, lane4, density,
      signal, signal_l1, signal_l2, signal_l3, signal_l4,
      green_time_l1, green_time_l2, green_time_l3, green_time_l4,
      arrival_l1, arrival_l2, arrival_l3, arrival_l4,
      active_lane, total_waiting,
    } = data;

    console.log(
      `[Traffic] L1=${lane1}(${signal_l1 || '?'} ${green_time_l1 || 0}s) ` +
      `L2=${lane2}(${signal_l2 || '?'} ${green_time_l2 || 0}s) ` +
      `L3=${lane3 || 0}(${signal_l3 || '?'} ${green_time_l3 || 0}s) ` +
      `L4=${lane4 || 0}(${signal_l4 || '?'} ${green_time_l4 || 0}s) ` +
      `density=${density}`
    );

    const outgoing = {
      lane1, lane2, lane3: lane3 || 0, lane4: lane4 || 0, density,
      signal:        signal        || 'L1:RED L2:RED L3:RED L4:RED',
      signal_l1:     signal_l1     || 'RED',
      signal_l2:     signal_l2     || 'RED',
      signal_l3:     signal_l3     || 'RED',
      signal_l4:     signal_l4     || 'RED',
      green_time_l1: green_time_l1 || 0,
      green_time_l2: green_time_l2 || 0,
      green_time_l3: green_time_l3 || 0,
      green_time_l4: green_time_l4 || 0,
      arrival_l1:    arrival_l1    || 0,
      arrival_l2:    arrival_l2    || 0,
      arrival_l3:    arrival_l3    || 0,
      arrival_l4:    arrival_l4    || 0,
      active_lane:   active_lane   || 0,
      total_waiting: total_waiting || 0,
      timestamp:     new Date().toISOString(),
    };

    // Broadcast to frontend
    io.emit('update-dashboard', outgoing);

    // Persist
    try {
      if (mongoose.connection.readyState === 1) {
        await new TrafficRecord({ ...outgoing, session_id: currentSessionId }).save();
      }
    } catch (err) {
      console.warn('[MongoDB] Save failed:', err.message);
    }
  });

  socket.on('disconnect', () => {
    console.log(`[Socket.IO] Client disconnected: ${socket.id}`);
  });
});

// ── Start ──────────────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`\n🚦 Traffic Surveillance Backend`);
  console.log(`   Server  : http://localhost:${PORT}`);
  console.log(`   History : http://localhost:${PORT}/history`);
  console.log(`   All-time: http://localhost:${PORT}/history/all\n`);
});
