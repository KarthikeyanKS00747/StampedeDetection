'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const STALE_TIMEOUT_MS  = 10_000;
const RECONNECT_DELAY_MS = 1_500;
const CHART_MAX_POINTS  = 60;    // keep last 60 data points on each sparkline

// ── DOM refs ──────────────────────────────────────────────────────────────────
const zonesEl  = document.getElementById('zones');
const emptyEl  = document.getElementById('empty');
const alertsEl = document.getElementById('alerts');
const bannerEl = document.getElementById('banner');

// ── State ─────────────────────────────────────────────────────────────────────
const lastSeen    = {};   // zoneId → ms timestamp
const chartData   = {};   // zoneId → { density: [], velVar: [], times: [] }
let   staleTimer  = null;

// ── Helpers ───────────────────────────────────────────────────────────────────
const esc = s => String(s).replace(/[&<>'"]/g, c =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));

const fmtTime = iso => new Date(iso).toLocaleTimeString([], {
  hour: '2-digit', minute: '2-digit', second: '2-digit'
});

// ── Chart drawing (vanilla canvas sparkline) ──────────────────────────────────
/**
 * Draw a simple sparkline on a <canvas>.
 * @param {HTMLCanvasElement} canvas
 * @param {number[]} values
 * @param {string} color  CSS colour string
 */
function drawSparkline(canvas, values, color) {
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (values.length < 2) return;

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;

  const toX = i  => (i / (values.length - 1)) * W;
  const toY = v  => H - ((v - min) / range) * (H - 8) - 4;

  ctx.beginPath();
  ctx.strokeStyle = color;
  ctx.lineWidth   = 2;
  ctx.lineJoin    = 'round';
  values.forEach((v, i) => {
    if (i === 0) ctx.moveTo(toX(i), toY(v));
    else         ctx.lineTo(toX(i), toY(v));
  });
  ctx.stroke();

  // semi-transparent fill
  ctx.lineTo(toX(values.length - 1), H);
  ctx.lineTo(toX(0), H);
  ctx.closePath();
  ctx.fillStyle = color.replace(')', ', 0.15)').replace('rgb', 'rgba');
  ctx.fill();

  // latest value label
  const last = values[values.length - 1];
  ctx.fillStyle = color;
  ctx.font = '11px system-ui,sans-serif';
  ctx.fillText(last.toFixed(2), 4, 14);
}

function updateCharts(zoneId, densityVals, velVarVals) {
  const densityCanvas = document.getElementById(`chart-density-${zoneId}`);
  const velVarCanvas  = document.getElementById(`chart-velvar-${zoneId}`);
  if (densityCanvas) drawSparkline(densityCanvas, densityVals, 'rgb(92,224,152)');
  if (velVarCanvas)  drawSparkline(velVarCanvas,  velVarVals,  'rgb(255,183,77)');
}

// ── Push a data point for this zone ───────────────────────────────────────────
function pushHistory(zoneId, density, velVar) {
  if (!chartData[zoneId]) chartData[zoneId] = { density: [], velVar: [] };
  const d = chartData[zoneId];
  d.density.push(density);
  d.velVar.push(velVar);
  if (d.density.length > CHART_MAX_POINTS) { d.density.shift(); d.velVar.shift(); }
  updateCharts(zoneId, d.density, d.velVar);
}

// ── Classifier mode badge ─────────────────────────────────────────────────────
function classifierBadge(mode) {
  if (mode === 'trained') {
    return '<span class="mode-tag trained">TRAINED MODEL</span>';
  }
  return '<span class="mode-tag heuristic" title="Not a validated prediction">UNVALIDATED HEURISTIC</span>';
}

// ── Zone rendering ────────────────────────────────────────────────────────────
function renderZone(item) {
  emptyEl.hidden = true;
  const m   = item.metrics;
  const id  = `zone-${item.zone_id}`;
  const cls = item.risk.replaceAll(' ', '-');

  // Preserve existing chart canvas contents across re-renders
  const existingDensity = chartData[item.zone_id]?.density ?? [];
  const existingVelVar  = chartData[item.zone_id]?.velVar  ?? [];

  let box = document.getElementById(id);
  if (!box) {
    box = document.createElement('article');
    box.id = id;
    zonesEl.append(box);
  }

  const sourceTag = m.source === 'simulation'
    ? '<span class="source-tag sim">SIMULATION</span>'
    : `<span class="source-tag">${esc(m.source)}</span>`;

  box.innerHTML = `
    <div class="stale-overlay" hidden><span>STALE — Unknown</span></div>
    <h2>${esc(item.zone_id)} ${sourceTag} ${classifierBadge(item.classifier_mode ?? 'heuristic')}</h2>
    <div class="card risk-card">
      <div class="metric">Current risk</div>
      <div class="risk ${cls}">${esc(item.risk)}</div>
      <p>${esc(item.reasons.join('; '))}</p>
      <div class="last-update">Last update: ${fmtTime(item.timestamp)}</div>
    </div>
    <div class="grid">
      <div class="card"><div class="metric">People</div><div class="value">${m.crowd_count}</div></div>
      <div class="card"><div class="metric">Density / m²</div><div class="value">${m.density.toFixed(2)}</div></div>
      <div class="card"><div class="metric">Density growth</div><div class="value">${m.density_growth_rate.toFixed(2)}</div></div>
      <div class="card"><div class="metric">Avg velocity</div><div class="value">${m.average_velocity.toFixed(1)}</div></div>
      <div class="card"><div class="metric">Velocity var.</div><div class="value">${m.velocity_variance.toFixed(1)}</div></div>
      <div class="card"><div class="metric">Direction var.</div><div class="value">${m.direction_variance.toFixed(2)}</div></div>
      <div class="card"><div class="metric">Congestion</div><div class="value">${m.congestion_index.toFixed(2)}</div></div>
    </div>
    <div class="charts-row">
      <div class="chart-wrap">
        <div class="chart-label">Density / m² <span class="chart-unit">(persons/m²)</span></div>
        <canvas id="chart-density-${esc(item.zone_id)}" width="320" height="72"></canvas>
      </div>
      <div class="chart-wrap">
        <div class="chart-label">Velocity variance <span class="chart-unit">(px²/frame)</span></div>
        <canvas id="chart-velvar-${esc(item.zone_id)}"  width="320" height="72"></canvas>
      </div>
    </div>`;

  // Re-draw chart with preserved + new point
  pushHistory(item.zone_id, m.density, m.velocity_variance);

  lastSeen[item.zone_id] = Date.now();
  clearStale(id);
}

// ── Seed charts from history endpoint ─────────────────────────────────────────
function seedChartsFromHistory(zoneId, history) {
  if (!chartData[zoneId]) chartData[zoneId] = { density: [], velVar: [] };
  const recent = history.slice(-CHART_MAX_POINTS);
  chartData[zoneId].density = recent.map(a => a.metrics.density);
  chartData[zoneId].velVar  = recent.map(a => a.metrics.velocity_variance);
}

// ── Alert log ─────────────────────────────────────────────────────────────────
function prependAlert(ev) {
  const li = document.createElement('li');
  li.className = ev.active ? 'alert-active' : 'alert-cleared';
  li.innerHTML = `<b>${esc(ev.risk)}</b> <span class="alert-time">${fmtTime(ev.timestamp)}</span> — ${esc(ev.message)}`;
  alertsEl.prepend(li);
}

// ── Stale detection ───────────────────────────────────────────────────────────
function markStale(zoneId) {
  const box = document.getElementById(`zone-${zoneId}`);
  if (!box) return;
  box.querySelector('.stale-overlay').hidden = false;
  box.classList.add('stale');
}

function clearStale(domId) {
  const box = document.getElementById(domId);
  if (!box) return;
  const ov = box.querySelector('.stale-overlay');
  if (ov) ov.hidden = true;
  box.classList.remove('stale');
}

function markAllStale() {
  Object.keys(lastSeen).forEach(markStale);
}

function startStaleTimer() {
  if (staleTimer) clearInterval(staleTimer);
  staleTimer = setInterval(() => {
    const now = Date.now();
    Object.entries(lastSeen).forEach(([z, ts]) => {
      if (now - ts > STALE_TIMEOUT_MS) markStale(z);
    });
  }, 2000);
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connect() {
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const ws    = new WebSocket(`${proto}://${location.host}/ws`);

  ws.onopen = () => {
    bannerEl.hidden = true;
    startStaleTimer();
  };

  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    switch (msg.type) {
      case 'init':
        msg.data.forEach(renderZone);
        // Fetch per-zone history to seed the charts properly
        msg.data.forEach(a => {
          fetch(`/api/history/${encodeURIComponent(a.zone_id)}`)
            .then(r => r.json())
            .then(hist => {
              seedChartsFromHistory(a.zone_id, hist);
              // Re-draw charts now that we have historical data
              updateCharts(a.zone_id, chartData[a.zone_id].density, chartData[a.zone_id].velVar);
            })
            .catch(() => {});
        });
        break;
      case 'assessment':
        renderZone(msg.data);
        break;
      case 'alert':
        prependAlert(msg.data);
        break;
      default:
        if (Array.isArray(msg)) msg.forEach(renderZone);
    }
  };

  ws.onclose = () => {
    bannerEl.hidden = false;
    markAllStale();
    setTimeout(connect, RECONNECT_DELAY_MS);
  };

  ws.onerror = () => ws.close();
}

// ── Boot ──────────────────────────────────────────────────────────────────────
fetch('/api/alerts')
  .then(r => r.json())
  .then(xs => xs.forEach(prependAlert))
  .catch(() => {});

connect();
