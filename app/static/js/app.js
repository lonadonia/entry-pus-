const socket = io();

// ── Bot controls ──────────────────────────────────────────────────────────────
const startBtn  = document.getElementById('btn-start');
const stopBtn   = document.getElementById('btn-stop');
const statusBadge = document.getElementById('bot-status');

if (startBtn) {
  startBtn.addEventListener('click', () => {
    fetch('/api/bot/start', {method: 'POST'})
      .then(r => r.json())
      .then(() => { statusBadge.textContent = 'Running'; statusBadge.style.background = '#276749'; });
  });
}

if (stopBtn) {
  stopBtn.addEventListener('click', () => {
    fetch('/api/bot/stop', {method: 'POST'})
      .then(r => r.json())
      .then(() => { statusBadge.textContent = 'Stopping…'; statusBadge.style.background = '#744210'; });
  });
}

// ── Live log (dashboard) ──────────────────────────────────────────────────────
const liveLog = document.getElementById('live-log');

socket.on('log', data => {
  if (!liveLog) return;
  const line = document.createElement('div');
  line.className = `log-line log-${data.level}`;
  line.textContent = `[${data.time}] [${data.level.toUpperCase()}] ${data.message}`;
  liveLog.appendChild(line);
  liveLog.scrollTop = liveLog.scrollHeight;
  // Keep at most 200 lines
  while (liveLog.children.length > 200) {
    liveLog.removeChild(liveLog.firstChild);
  }
});

// ── Contest status updates ────────────────────────────────────────────────────
socket.on('contest_update', data => {
  const badge = document.querySelector(`[data-contest-id="${data.id}"] .badge`);
  if (badge) {
    badge.textContent = data.status;
    badge.className = `badge badge-${data.status}`;
  }
});

// ── Connection state ──────────────────────────────────────────────────────────
socket.on('connect',    () => console.log('[socket] connected'));
socket.on('disconnect', () => console.log('[socket] disconnected'));
