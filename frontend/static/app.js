/* app.js — Smart Attendance System v2 */
'use strict';
const API = '/api';

// ── Utility ──────────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const show = (el, cls='') => { el.classList.remove('hidden'); if(cls) el.className = 'alert ' + cls; };
const hide = el => el.classList.add('hidden');
function msg(el, text, type='') {
  el.textContent = text;
  el.className = 'alert ' + type;
  show(el);
}
async function api(path, opts={}) {
  const r = await fetch(API + path, {
    headers: {'Content-Type':'application/json'},
    ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined
  });
  return r.json();
}

// ── Page navigation (sidebar) ─────────────────────────────────────────────────
document.querySelectorAll('.snav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.snav-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.page').forEach(p => { p.classList.remove('active'); p.classList.add('hidden'); });
    btn.classList.add('active');
    const pg = $('page-' + btn.dataset.page);
    pg.classList.remove('hidden');
    pg.classList.add('active');
    if (btn.dataset.page === 'admin') initAdmin();
    // close mobile sidebar
    $('sidebar').classList.remove('open');
  });
});

// Mobile hamburger
$('hamburger').addEventListener('click', () => $('sidebar').classList.toggle('open'));

// Admin tab bar
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => { p.classList.remove('active'); p.classList.add('hidden'); });
    btn.classList.add('active');
    const panel = $('view-' + btn.dataset.view);
    panel.classList.remove('hidden'); panel.classList.add('active');
    if (btn.dataset.view === 'live') loadLiveAttendance();
    if (btn.dataset.view === 'map') initMap();
    if (btn.dataset.view === 'students') loadStudentsView();
  });
});

// ── face-api.js model loading ─────────────────────────────────────────────────
// Using justadudewhohacks/face-api.js v0.22.2
// Models: ssd_mobilenetv1 (detection) + face_landmark_68 (liveness) + face_recognition (FaceNet 128-d)
const MODEL_URL = 'https://cdn.jsdelivr.net/npm/@vladmandic/face-api@1.7.13/model';
let faceApiReady = false;

async function loadFaceApi() {
  if (faceApiReady) return true;
  try {
    await Promise.all([
      faceapi.nets.ssdMobilenetv1.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL),
      faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_URL),
    ]);
    faceApiReady = true;
    return true;
  } catch(e) {
    console.error('face-api load failed', e);
    return false;
  }
}

// SSD MobileNet options — minConfidence 0.5 is reliable for frontal faces
const SSD_OPTS = () => new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });

// ── Liveness: Eye Aspect Ratio (EAR) blink detection ─────────────────────────
// EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
// landmarks indices for left eye: 36-41, right eye: 42-47
function eyeAspectRatio(pts) {
  const d = (a, b) => Math.hypot(pts[a].x - pts[b].x, pts[a].y - pts[b].y);
  return (d(1,5) + d(2,4)) / (2 * d(0,3));
}
function getEAR(landmarks) {
  const pts = landmarks.positions;
  const leftEAR  = eyeAspectRatio([pts[36],pts[37],pts[38],pts[39],pts[40],pts[41]]);
  const rightEAR = eyeAspectRatio([pts[42],pts[43],pts[44],pts[45],pts[46],pts[47]]);
  return (leftEAR + rightEAR) / 2;
}
const EAR_THRESHOLD = 0.21; // below this = eye closed (blink)

// ── Camera helpers ────────────────────────────────────────────────────────────
function getCameraMode(selectId, defaultMode='user') {
  const el = $(selectId);
  return el ? el.value : defaultMode;
}

async function startCam(videoEl, facingMode = 'user') {
  const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode } });
  videoEl.srcObject = stream;
  await new Promise(res => videoEl.onloadedmetadata = res);
  return stream;
}
function stopCam(stream) {
  if (stream) stream.getTracks().forEach(t => t.stop());
}
async function getDescriptor(videoEl) {
  const det = await faceapi
    .detectSingleFace(videoEl, SSD_OPTS())
    .withFaceLandmarks()
    .withFaceDescriptor();
  return det ? Array.from(det.descriptor) : null;
}

// ── GPS helper ────────────────────────────────────────────────────────────────
function getGPS() {
  return new Promise((res, rej) => {
    if (!navigator.geolocation) return rej(new Error('Geolocation not supported'));
    navigator.geolocation.getCurrentPosition(
      p => res({ lat: p.coords.latitude, lng: p.coords.longitude, acc: p.coords.accuracy }),
      e => rej(e),
      { timeout: 10000 }
    );
  });
}

// ═══════════════════════════════════════════════════════════════════════════════
// PAGE 1 — MARK ATTENDANCE
// ═══════════════════════════════════════════════════════════════════════════════
let attendState = { token: null, descriptor: null, gps: null, sessionInfo: null };

// Step helpers
function goStep(n) {
  [1,2,3,4].forEach(i => {
    const s = $('step'+i), p = $('panel-' + ['qr','face','gps','submit'][i-1]);
    s.classList.remove('active','done');
    p.classList.remove('active'); p.classList.add('hidden');
    if (i < n) s.classList.add('done');
    if (i === n) { s.classList.add('active'); p.classList.remove('hidden'); p.classList.add('active'); }
  });
}

// ── Step 1: QR scan ───────────────────────────────────────────────────────────
let qrStream = null, qrInterval = null;

$('startQrCamera').addEventListener('click', async () => {
  try {
    qrStream = await startCam($('qrVideo'), getCameraMode('qrCameraMode', 'environment'));
    $('startQrCamera').disabled = true;
    $('qrPlaceholder').classList.add('hidden');
    startQrScan();
  } catch(e) {
    msg($('qrResult'), 'Camera error: ' + e.message, 'error');
    show($('qrResult'));
  }
});

function startQrScan() {
  const video = $('qrVideo'), canvas = $('qrCanvas'), ctx = canvas.getContext('2d');
  qrInterval = setInterval(() => {
    if (!video.videoWidth) return;
    canvas.width = video.videoWidth; canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0);
    const img = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const code = jsQR(img.data, img.width, img.height);
    if (code) {
      clearInterval(qrInterval);
      stopCam(qrStream);
      processQrToken(code.data);
    }
  }, 200);
}

$('useManualToken').addEventListener('click', () => {
  const t = $('manualToken').value.trim();
  if (t) processQrToken(t);
});

async function processQrToken(raw) {
  let token = raw;
  try { const u = new URL(raw, location.origin); token = u.searchParams.get('token') || raw; } catch(_){}

  msg($('qrResult'), 'Validating token...', '');
  show($('qrResult'));

  const res = await api('/validate-token', { method:'POST', body:{ token } });
  if (!res.valid) {
    msg($('qrResult'), '✗ ' + res.reason, 'error'); return;
  }
  attendState.token = token;
  attendState.sessionInfo = res;
  msg($('qrResult'), '✓ Session: ' + res.course + ' (Faculty: ' + res.faculty_id + ')', 'success');

  // Go to face step FIRST, then start camera after panel is visible
  setTimeout(() => {
    goStep(2);
    startFaceStep();
  }, 900);
}

// ── Step 2: Face + Liveness ───────────────────────────────────────────────────
let faceStream = null, faceInterval = null;
let blinkCount = 0, lastEAR = 1.0, livenessOk = false;

async function startFaceStep() {
  const statusEl = $('faceStatus');
  blinkCount = 0; lastEAR = 1.0; livenessOk = false;

  // Reset blink dots
  $('blink1').classList.remove('lit');
  $('blink2').classList.remove('lit');
  $('livenessText').textContent = 'Blink twice to confirm liveness';
  $('livenessPrompt').classList.remove('hidden');
  $('captureFaceBtn').classList.add('hidden');
  $('captureFaceBtn').disabled = true;

  msg(statusEl, 'Loading face models...', '');

  const ok = await loadFaceApi();
  if (!ok) { msg(statusEl, 'Could not load face models. Use Skip.', 'error'); return; }

  try {
    faceStream = await startCam($('faceVideo'), getCameraMode('faceCameraMode', 'user'));
    msg(statusEl, 'Look at the camera. Blink twice for liveness check.', '');

    faceInterval = setInterval(async () => {
      if (!faceStream) { clearInterval(faceInterval); return; }

      const det = await faceapi
        .detectSingleFace($('faceVideo'), SSD_OPTS())
        .withFaceLandmarks();

      if (!det) {
        statusEl.className = 'alert';
        statusEl.textContent = 'No face detected — move closer or improve lighting';
        $('captureFaceBtn').disabled = true;
        return;
      }

      // Liveness blink counting via EAR
      if (!livenessOk) {
        const ear = getEAR(det.landmarks);
        if (lastEAR >= EAR_THRESHOLD && ear < EAR_THRESHOLD) {
          blinkCount++;
          $('livenessText').textContent = 'Blink ' + blinkCount + '/2 detected';
          if (blinkCount >= 1) $('blink1').classList.add('lit');
          if (blinkCount >= 2) $('blink2').classList.add('lit');
        }
        lastEAR = ear;
        if (blinkCount >= 2) {
          livenessOk = true;
          $('livenessPrompt').classList.add('hidden');
        }
      }

      if (livenessOk) {
        statusEl.className = 'alert success';
        statusEl.textContent = '✓ Liveness confirmed — click Capture Face';
        $('captureFaceBtn').classList.remove('hidden');
        $('captureFaceBtn').disabled = false;
      } else {
        statusEl.className = 'alert';
        statusEl.textContent = 'Face detected — blink ' + blinkCount + '/2 times';
        $('captureFaceBtn').disabled = true;
      }
    }, 200);

  } catch(e) {
    msg(statusEl, 'Camera error: ' + e.message, 'error');
  }
}

$('captureFaceBtn').addEventListener('click', async () => {
  clearInterval(faceInterval);
  const statusEl = $('faceStatus');
  msg(statusEl, 'Capturing face embedding…');
  const desc = await getDescriptor($('faceVideo'));
  stopCam(faceStream);
  if (!desc) {
    msg(statusEl, 'No face found. Try again or skip.', 'error'); return;
  }
  attendState.descriptor = desc;
  msg(statusEl, '✓ Face captured successfully', 'success');
  setTimeout(() => goStep(3), 600);
});

$('skipFaceBtn').addEventListener('click', () => {
  clearInterval(faceInterval);
  stopCam(faceStream);
  attendState.descriptor = null;
  goStep(3);
});

// ── Step 3: GPS ───────────────────────────────────────────────────────────────
$('getGpsBtn').addEventListener('click', async () => {
  const statusEl = $('gpsStatus');
  msg(statusEl, 'Getting location…');
  show(statusEl);
  try {
    const pos = await getGPS();
    attendState.gps = pos;
    $('gpsLat').textContent = pos.lat.toFixed(6);
    $('gpsLng').textContent = pos.lng.toFixed(6);
    $('gpsAcc').textContent = pos.acc ? pos.acc.toFixed(0) + ' m' : '—';
    msg(statusEl, '✓ Location captured', 'success');
    setTimeout(() => goStep(4), 600);
    buildSummary();
  } catch(e) {
    msg(statusEl, 'GPS error: ' + e.message + '. You can skip.', 'error');
  }
});

$('skipGpsBtn').addEventListener('click', () => {
  attendState.gps = null;
  goStep(4);
  buildSummary();
});

function buildSummary() {
  const g = $('summaryGrid');
  const s = attendState;
  g.innerHTML = `
    <div class="check-item ok">
      <div class="ci-icon">📱</div>
      <div class="ci-label">QR Token</div>
      <div class="ci-val">${s.sessionInfo ? s.sessionInfo.course : '—'}</div>
    </div>
    <div class="check-item ${s.descriptor ? 'ok' : 'skip'}">
      <div class="ci-icon">😊</div>
      <div class="ci-label">Face</div>
      <div class="ci-val">${s.descriptor ? 'Captured' : 'Skipped'}</div>
    </div>
    <div class="check-item ${s.gps ? 'ok' : 'skip'}">
      <div class="ci-icon">📍</div>
      <div class="ci-label">GPS</div>
      <div class="ci-val">${s.gps ? s.gps.lat.toFixed(4)+', '+s.gps.lng.toFixed(4) : 'Skipped'}</div>
    </div>`;
}

// ── Step 4: Submit ────────────────────────────────────────────────────────────
$('submitAttendanceBtn').addEventListener('click', async () => {
  const sid = $('attendStudentId').value.trim();
  if (!sid) { msg($('submitResult'), 'Enter your Student ID', 'error'); show($('submitResult')); return; }
  const s = attendState;
  if (!s.token) { msg($('submitResult'), 'No QR token — go back to Step 1', 'error'); show($('submitResult')); return; }

  $('submitAttendanceBtn').disabled = true;
  $('submitAttendanceBtn').textContent = 'Submitting…';

  const body = {
    token: s.token,
    student_id: sid,
    descriptor: s.descriptor,
    gps_lat: s.gps ? s.gps.lat : null,
    gps_lng: s.gps ? s.gps.lng : null,
    gps_available: !!s.gps,
  };

  const res = await api('/attendance/mark', { method:'POST', body });
  $('submitAttendanceBtn').disabled = false;
  $('submitAttendanceBtn').textContent = '✓ Mark Me Present';

  if (res.error) {
    msg($('submitResult'), '✗ ' + res.error, 'error');
  } else {
    const c = res.checks;
    msg($('submitResult'),
      `✓ ${res.message}\n` +
      `QR: ${c.qr?'✓':'✗'}  Face: ${c.face?'✓':'✗'}  GPS: ${c.gps?'✓':'✗'}` +
      (c.gps_distance_m != null ? `  (${c.gps_distance_m}m away)` : ''),
      res.attendance.status === 'present' ? 'success' : 'warning'
    );
    // Reset for next use
    attendState = { token:null, descriptor:null, gps:null, sessionInfo:null };
    goStep(1);
  }
  show($('submitResult'));
});

// ═══════════════════════════════════════════════════════════════════════════════
// PAGE 2 — ENROLL FACE
// ═══════════════════════════════════════════════════════════════════════════════
let enrollStream = null;

$('startEnrollCamera').addEventListener('click', async () => {
  const statusEl = $('enrollFaceStatus');
  msg(statusEl, 'Loading face models (SSD MobileNet + FaceNet)...');
  const ok = await loadFaceApi();
  if (!ok) { msg(statusEl, 'Could not load face models.', 'error'); return; }
  try {
    enrollStream = await startCam($('enrollVideo'), getCameraMode('enrollCameraMode', 'user'));
    $('startEnrollCamera').disabled = true;
    // Live feedback loop
    const feedbackLoop = setInterval(async () => {
      if (!enrollStream) { clearInterval(feedbackLoop); return; }
      const det = await faceapi.detectSingleFace($('enrollVideo'), SSD_OPTS());
      if (det) {
        statusEl.className = 'alert success';
        statusEl.textContent = '✓ Face detected — click Capture & Enroll';
        $('captureEnrollBtn').classList.remove('hidden');
        $('captureEnrollBtn').disabled = false;
      } else {
        statusEl.className = 'alert';
        statusEl.textContent = 'No face detected — adjust position or lighting';
        $('captureEnrollBtn').disabled = true;
      }
    }, 300);
  } catch(e) {
    msg(statusEl, 'Camera error: ' + e.message, 'error');
  }
});

$('captureEnrollBtn').addEventListener('click', async () => {
  const statusEl = $('enrollFaceStatus');
  const sid = $('enrollStudentId').value.trim();
  if (!sid) { msg(statusEl, 'Enter your Student ID first', 'error'); return; }

  msg(statusEl, 'Detecting face…');
  const desc = await getDescriptor($('enrollVideo'));
  if (!desc) { msg(statusEl, 'No face detected. Adjust lighting and try again.', 'error'); return; }

  stopCam(enrollStream);
  $('captureEnrollBtn').disabled = true;

  // Find student by student_id
  const students = await api('/students');
  const student = students.find(s => s.student_id === sid);
  if (!student) {
    msg($('enrollResult'), 'Student ID not found. Register first.', 'error');
    show($('enrollResult'));
    $('captureEnrollBtn').disabled = false;
    return;
  }

  const res = await api(`/students/${student.id}/enroll-face`, {
    method: 'POST', body: { descriptor: desc }
  });
  $('captureEnrollBtn').disabled = false;
  if (res.error) {
    msg($('enrollResult'), '✗ ' + res.error, 'error');
  } else {
    msg($('enrollResult'), '✓ Face enrolled! You can now mark attendance.', 'success');
  }
  show($('enrollResult'));
});

// ═══════════════════════════════════════════════════════════════════════════════
// PAGE 3 — ADMIN DASHBOARD
// ═══════════════════════════════════════════════════════════════════════════════
let adminMap = null, adminMapMarkers = [];

function initAdmin() {
  loadStats();
  loadSessionsView();
  populateSessionSelects();
}

// Sidebar navigation — handled by tab-bar listeners above

// ── Stats ─────────────────────────────────────────────────────────────────────
async function loadStats() {
  const s = await api('/stats');
  $('statStudents').textContent = s.total_students;
  $('statSessions').textContent = s.total_sessions;
  $('statActive').textContent   = s.active_sessions;
  $('statPresent').textContent  = s.present_today;
  $('statPartial').textContent  = s.partial_today;
  $('statDate').textContent     = s.date;
}
$('refreshStats').addEventListener('click', loadStats);

// ── Sessions ──────────────────────────────────────────────────────────────────
$('openCreateSession').addEventListener('click', () => {
  $('createSessionForm').classList.toggle('hidden');
});
$('cancelCreateSession').addEventListener('click', () => {
  $('createSessionForm').classList.add('hidden');
});
$('useMyLocationBtn').addEventListener('click', async () => {
  try {
    const pos = await getGPS();
    $('sLat').value = pos.lat.toFixed(6);
    $('sLng').value = pos.lng.toFixed(6);
  } catch(e) { alert('GPS error: ' + e.message); }
});

$('createSessionBtn').addEventListener('click', async () => {
  const body = {
    course: $('sCourse').value.trim(),
    faculty_id: $('sFaculty').value.trim(),
    location_name: $('sLocName').value.trim(),
    location_lat: parseFloat($('sLat').value) || null,
    location_lng: parseFloat($('sLng').value) || null,
  };
  if (!body.course || !body.faculty_id) { alert('Course and Faculty ID are required'); return; }
  const res = await api('/sessions/create', { method:'POST', body });
  if (res.error) { alert(res.error); return; }
  $('createSessionForm').classList.add('hidden');
  loadSessionsView();
  populateSessionSelects();
  openQrModal(res.session);
});

async function loadSessionsView() {
  const sessions = await api('/sessions');
  const el = $('sessionsList');
  if (!sessions.length) { el.innerHTML = '<p style="color:var(--muted)">No sessions yet.</p>'; return; }
  el.innerHTML = sessions.map(s => `
    <div class="session-card">
      <div class="session-info">
        <h4>${s.course}</h4>
        <p>Faculty: ${s.faculty_id} &nbsp;|&nbsp; ${s.location_name || 'No location'} &nbsp;|&nbsp; ${new Date(s.start_time).toLocaleString()}</p>
      </div>
      <div class="session-actions">
        <span class="badge ${s.is_active ? 'badge-green' : 'badge-gray'}">${s.is_active ? 'Active' : 'Closed'}</span>
        ${s.is_active ? `<button class="btn btn-outline" onclick="showQr(${s.id})">Show QR</button>` : ''}
        ${s.is_active ? `<button class="btn btn-danger" onclick="closeSession(${s.id})">Close</button>` : ''}
      </div>
    </div>`).join('');
}

async function showQr(id) {
  const s = await api(`/sessions/${id}`);
  openQrModal(s);
}
window.showQr = showQr;

async function closeSession(id) {
  if (!confirm('Close this session? Students will no longer be able to mark attendance.')) return;
  await api(`/sessions/${id}/close`, { method:'POST' });
  loadSessionsView();
  loadStats();
}
window.closeSession = closeSession;

// ── QR Modal ──────────────────────────────────────────────────────────────────
let currentQrSessionId = null;
function openQrModal(session) {
  currentQrSessionId = session.id;
  $('qrModalTitle').textContent = session.course;
  $('qrExpiryMins').textContent = '10';
  $('qrModalImg').src = 'data:image/png;base64,' + session.qr_image_b64;
  $('qrModal').classList.remove('hidden');
}
$('closeQrModal').addEventListener('click', () => $('qrModal').classList.add('hidden'));
$('qrModal').addEventListener('click', e => {
  if (!e.target.closest('.modal-box')) $('qrModal').classList.add('hidden');
});
$('refreshQrBtn').addEventListener('click', async () => {
  if (!currentQrSessionId) return;
  const res = await api(`/sessions/${currentQrSessionId}/refresh-qr`, { method:'POST' });
  if (res.session) $('qrModalImg').src = 'data:image/png;base64,' + res.session.qr_image_b64;
});

// ── Live Attendance ───────────────────────────────────────────────────────────
async function populateSessionSelects() {
  const sessions = await api('/sessions');
  const opts = sessions.map(s => `<option value="${s.id}">${s.course} (${new Date(s.start_time).toLocaleDateString()})</option>`).join('');
  $('liveSessionSelect').innerHTML = '<option value="">— All Today —</option>' + opts;
  $('mapSessionSelect').innerHTML  = '<option value="">— All Today —</option>' + opts;
}

async function loadLiveAttendance() {
  const sid = $('liveSessionSelect').value;
  const records = sid
    ? await api(`/attendance/session/${sid}`)
    : await api('/attendance/today');
  renderLiveTable(records);
}
$('refreshLive').addEventListener('click', loadLiveAttendance);
$('liveSessionSelect').addEventListener('change', loadLiveAttendance);

function statusBadge(s) {
  const map = { present:'badge-green', partial:'badge-yellow', rejected:'badge-red', pending:'badge-gray' };
  return `<span class="badge ${map[s]||'badge-gray'}">${s}</span>`;
}

function renderLiveTable(records) {
  if (!records.length) {
    $('liveTable').innerHTML = '<p style="color:var(--muted);padding:16px">No records yet.</p>'; return;
  }
  $('liveTable').innerHTML = `
    <div class="table-wrap"><table>
      <thead><tr>
        <th>Student</th><th>Course</th><th>Time</th>
        <th>QR</th><th>Face</th><th>GPS</th><th>Status</th>
      </tr></thead>
      <tbody>${records.map(r => `<tr>
        <td>${r.student_name || r.student_id}</td>
        <td>${r.course || '—'}</td>
        <td>${new Date(r.timestamp).toLocaleTimeString()}</td>
        <td>${r.qr_verified ? '✓' : '✗'}</td>
        <td>${r.face_verified ? '✓' : (r.face_confidence != null ? '✗' : '—')}</td>
        <td>${r.gps_status === 'ok' ? '✓' : r.gps_status === 'out_of_range' ? '⚠ '+r.gps_distance_m+'m' : r.gps_status}</td>
        <td>${statusBadge(r.status)}</td>
      </tr>`).join('')}</tbody>
    </table></div>`;
}

$('exportCsvBtn').addEventListener('click', () => {
  const sid = $('liveSessionSelect').value;
  window.location = API + '/attendance/export/csv' + (sid ? '?session_id='+sid : '');
});

// ── Map ───────────────────────────────────────────────────────────────────────
async function initMap() {
  if (!adminMap) {
    adminMap = L.map('attendanceMap').setView([20, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '© OpenStreetMap contributors'
    }).addTo(adminMap);
  }
  adminMapMarkers.forEach(m => m.remove());
  adminMapMarkers = [];

  const sid = $('mapSessionSelect').value;
  const records = sid
    ? await api(`/attendance/session/${sid}`)
    : await api('/attendance/today');

  const pts = records.filter(r => r.gps_lat && r.gps_lng);
  if (!pts.length) return;

  pts.forEach(r => {
    const color = r.status === 'present' ? 'green' : r.status === 'partial' ? 'orange' : 'red';
    const m = L.circleMarker([r.gps_lat, r.gps_lng], { radius:8, color, fillOpacity:0.8 })
      .bindPopup(`<b>${r.student_name}</b><br>${r.course}<br>${r.status}<br>${new Date(r.timestamp).toLocaleString()}`)
      .addTo(adminMap);
    adminMapMarkers.push(m);
  });

  const bounds = L.latLngBounds(pts.map(r => [r.gps_lat, r.gps_lng]));
  adminMap.fitBounds(bounds, { padding:[40,40] });
}
$('mapSessionSelect').addEventListener('change', initMap);

// ── Students ──────────────────────────────────────────────────────────────────
async function loadStudentsView() {
  const students = await api('/students');
  const el = $('studentsList');
  if (!students.length) { el.innerHTML = '<p style="color:var(--muted)">No students registered.</p>'; return; }
  el.innerHTML = `<div class="table-wrap"><table>
    <thead><tr><th>ID</th><th>Name</th><th>Email</th><th>Face Enrolled</th><th>Registered</th><th>Actions</th></tr></thead>
    <tbody>${students.map(s => `<tr>
      <td>${s.student_id}</td>
      <td>${s.name}</td>
      <td>${s.email}</td>
      <td>${s.has_face ? '<span class="badge badge-green">Yes</span>' : '<span class="badge badge-gray">No</span>'}</td>
      <td>${new Date(s.created_at).toLocaleDateString()}</td>
      <td><button class="btn btn-danger btn-sm delete-student-btn" data-student-id="${s.id}" data-student-label="${s.student_id}">Remove</button></td>
    </tr>`).join('')}</tbody>
  </table></div>`;
}

$('studentsList').addEventListener('click', async event => {
  const btn = event.target.closest('.delete-student-btn');
  if (!btn) return;
  const studentId = btn.dataset.studentId;
  const studentLabel = btn.dataset.studentLabel;
  if (!studentId) return;
  if (!confirm(`Remove student ${studentLabel}? This will also delete their attendance history.`)) return;
  const res = await api(`/students/${studentId}`, { method: 'DELETE' });
  if (res.error) {
    alert('Could not remove student: ' + res.error);
  } else {
    alert('Student removed successfully.');
    loadStudentsView();
  }
});

// ═══════════════════════════════════════════════════════════════════════════════
// PAGE 4 — REGISTER STUDENT
// ═══════════════════════════════════════════════════════════════════════════════
$('registerStudentBtn').addEventListener('click', async () => {
  const body = {
    student_id: $('regId').value.trim(),
    name: $('regName').value.trim(),
    email: $('regEmail').value.trim(),
  };
  if (!body.student_id || !body.name || !body.email) {
    msg($('registerResult'), 'All fields are required', 'error'); show($('registerResult')); return;
  }
  const res = await api('/students/register', { method:'POST', body });
  if (res.error) {
    msg($('registerResult'), '✗ ' + res.error, 'error');
  } else {
    msg($('registerResult'), `✓ ${res.student.name} registered! Now go to Enroll Face.`, 'success');
    $('regId').value = ''; $('regName').value = ''; $('regEmail').value = '';
  }
  show($('registerResult'));
});

// ── Init ──────────────────────────────────────────────────────────────────────
// Check for ?token= in URL (student scanned QR with phone)
window.addEventListener('DOMContentLoaded', () => {
  const params = new URLSearchParams(location.search);
  const token = params.get('token');
  if (token) {
    // Student opened the /attend?token=xxx URL from QR scan
    document.querySelector('.snav-btn[data-page="attend"]').click();
    $('manualToken').value = token;
    processQrToken(token);
  }
});
