const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const mongoose = require('mongoose');

// ── App Setup ─────────────────────────────────────────────────────────────────
const app = express();
const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST'],
  },
});

app.use(cors());
app.use(express.json());

// ── MongoDB Connection ────────────────────────────────────────────────────────
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/traffic_surveillance';

mongoose
  .connect(MONGO_URI)
  .then(() => console.log('[MongoDB] Connected successfully'))
  .catch((err) => console.warn('[MongoDB] Connection failed (data won\'t persist):', err.message));

// ── Traffic Schema & Model ────────────────────────────────────────────────────
const trafficSchema = new mongoose.Schema({
  lane1:     { type: Number, required: true },
  lane2:     { type: Number, required: true },
  density:   { type: Number, required: true },
  signal:    { type: String, required: true, enum: ['GREEN_LANE1', 'GREEN_LANE2'] },
  timestamp: { type: Date, default: Date.now },
});

const TrafficRecord = mongoose.model('TrafficRecord', trafficSchema);

// ── REST API ──────────────────────────────────────────────────────────────────

// Health check
app.get('/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// History endpoint — returns last 60 records (1 minute at 1 req/s)
app.get('/history', async (req, res) => {
  try {
    const limit  = parseInt(req.query.limit)  || 60;
    const records = await TrafficRecord.find()
      .sort({ timestamp: -1 })
      .limit(limit)
      .lean();
    res.json({ success: true, data: records.reverse() });
  } catch (err) {
    console.error('[API] /history error:', err.message);
    res.status(500).json({ success: false, error: 'Failed to fetch history' });
  }
});

// ── Socket.IO ─────────────────────────────────────────────────────────────────
io.on('connection', (socket) => {
  console.log(`[Socket.IO] Client connected: ${socket.id}`);

  // Send last 30 records to newly connected frontend
  TrafficRecord.find()
    .sort({ timestamp: -1 })
    .limit(30)
    .lean()
    .then((records) => {
      socket.emit('history-snapshot', records.reverse());
    })
    .catch(() => {});

  // Receive data from Python CV module
  socket.on('traffic-data', async (data) => {
    const { lane1, lane2, density, signal } = data;

    console.log(
      `[Traffic] lane1=${lane1} | lane2=${lane2} | density=${density.toFixed(2)} | signal=${signal}`
    );

    // Broadcast to all connected frontends
    io.emit('update-dashboard', { lane1, lane2, density, signal, timestamp: new Date().toISOString() });

    // Persist to MongoDB
    try {
      if (mongoose.connection.readyState === 1) {
        const record = new TrafficRecord({ lane1, lane2, density, signal });
        await record.save();
      }
    } catch (err) {
      console.warn('[MongoDB] Save failed:', err.message);
    }
  });

  socket.on('disconnect', () => {
    console.log(`[Socket.IO] Client disconnected: ${socket.id}`);
  });
});

// ── Start Server ──────────────────────────────────────────────────────────────
const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
  console.log(`\n🚦 Traffic Surveillance Backend`);
  console.log(`   Server   : http://localhost:${PORT}`);
  console.log(`   History  : http://localhost:${PORT}/history`);
  console.log(`   Health   : http://localhost:${PORT}/health\n`);
});
