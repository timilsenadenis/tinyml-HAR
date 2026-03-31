"""
serve.py  —  IMU Data Collection + Inference Server
────────────────────────────────────────────────────
Ports:
  HTTPS (phone app) : 5443
  WSS  (websocket)  : 5444

Install:
    pip install websockets numpy scipy cryptography

Run:
    python serve.py                     # collection mode
    python serve.py --model model.pth   # inference mode
"""

import asyncio, ssl, json, argparse, os, socket, tempfile, datetime, csv
import numpy as np
from collections import deque
from scipy import stats as scipy_stats
from pathlib import Path

import websockets

try:
    import torch
    import torch.nn as nn
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    print("[warn] PyTorch not found — inference disabled")

# ── config ────────────────────────────────────────────────────────────────────
HTTP_PORT   = 5443
WS_PORT     = 5444
WINDOW_SIZE = 128
STEP_SIZE   = 64          # 50% overlap
DATA_FILE   = "imu_data.csv"

LABELS = {
    0: "WALKING",
    1: "WALKING UPSTAIRS",
    2: "WALKING DOWNSTAIRS",
    3: "SITTING",
    4: "STANDING",
    5: "LAYING"
}

# ── global collection state (shared across connections) ───────────────────────
class CollectionState:
    def __init__(self):
        self.recording   = False
        self.paused      = False
        self.current_label = None
        self.window_count  = {i: 0 for i in range(6)}
        self.total_windows = 0

    def start(self, label_id):
        self.recording     = True
        self.paused        = False
        self.current_label = label_id

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False

    def stop(self):
        self.recording     = False
        self.paused        = False
        self.current_label = None

    def is_active(self):
        return self.recording and not self.paused

CSTATE = CollectionState()

# ── CSV writer ────────────────────────────────────────────────────────────────
def save_window(label_id: int, window: np.ndarray):
    """Append one (128×6) window as a flat row with label to CSV."""
    file_exists = Path(DATA_FILE).exists()
    with open(DATA_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            # header: label, ax_0..ax_127, ay_0..ay_127, ... gz_127
            cols = ["label"]
            for ch in ["ax","ay","az","gx","gy","gz"]:
                cols += [f"{ch}_{i}" for i in range(WINDOW_SIZE)]
            writer.writerow(cols)
        row = [label_id] + window.flatten().tolist()
        writer.writerow(row)

    CSTATE.window_count[label_id] += 1
    CSTATE.total_windows += 1

def get_csv_stats():
    counts = {LABELS[i]: CSTATE.window_count[i] for i in range(6)}
    return counts

# ── model ─────────────────────────────────────────────────────────────────────
if TORCH_OK:
    class CNN1D(nn.Module):
        def __init__(self, n_channels=6, n_classes=6):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64), nn.ReLU(),
            )
            self.classifier = nn.Sequential(
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, n_classes)
            )
        def forward(self, x):
            # x: (B, 128, 6) → (B, 6, 128) for Conv1d
            x = x.permute(0, 2, 1)
            x = self.encoder(x)
            x = x.mean(dim=2)          # global average pooling
            return self.classifier(x)

def load_model(path):
    if not TORCH_OK or not path: return None
    try:
        m = CNN1D()
        m.load_state_dict(torch.load(path, map_location="cpu"))
        m.eval()
        print(f"[✓] Model loaded: {path}")
        return m
    except Exception as e:
        print(f"[!] Model load failed: {e}"); return None

def predict(model, window: np.ndarray):
    if model is None: return None, 0.0
    with torch.no_grad():
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0)  # (1,128,6)
        p = torch.softmax(model(x), dim=1).squeeze()
        c = int(torch.argmax(p))
    return LABELS.get(c, str(c)), float(p[c])

# ── self-signed TLS cert ──────────────────────────────────────────────────────
def make_cert():
    from cryptography import x509
    from cryptography.x509.oid import NameOID
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    import ipaddress

    local_ip = socket.gethostbyname(socket.gethostname())
    key  = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, local_ip)])
    now  = datetime.datetime.now(datetime.timezone.utc)

    cert = (
        x509.CertificateBuilder()
        .subject_name(name).issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName([
            x509.DNSName("localhost"),
            x509.IPAddress(ipaddress.ip_address(local_ip)),
        ]), critical=False)
        .sign(key, hashes.SHA256())
    )

    tmp      = tempfile.mkdtemp()
    cert_pem = os.path.join(tmp, "cert.pem")
    key_pem  = os.path.join(tmp, "key.pem")
    with open(cert_pem, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))
    with open(key_pem, "wb") as f:
        f.write(key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        ))

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(cert_pem, key_pem)
    return ctx, local_ip

# ── embedded HTML ─────────────────────────────────────────────────────────────
def build_html(local_ip: str) -> bytes:
    wss_url = f"wss://{local_ip}:{WS_PORT}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0"/>
<title>IMU Collector</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     background:#0a0a0a;color:#fff;min-height:100vh;
     display:flex;flex-direction:column;align-items:center;padding:18px 14px;
     padding-bottom:40px}}
h1{{font-size:1.3rem;font-weight:700;margin-bottom:2px}}
.sub{{font-size:.75rem;color:#666;margin-bottom:18px}}
.card{{background:#161616;border:1px solid #242424;border-radius:14px;
       padding:14px;width:100%;max-width:440px;margin-bottom:12px}}
.ct{{font-size:.62rem;text-transform:uppercase;letter-spacing:.09em;
     color:#444;margin-bottom:10px;font-weight:600}}

/* status row */
.sr{{display:flex;align-items:center;gap:8px;margin-bottom:12px}}
.dot{{width:9px;height:9px;border-radius:50%;background:#333;flex-shrink:0}}
.dot.conn{{background:#22c55e;box-shadow:0 0 5px #22c55e}}
.dot.live{{background:#3b82f6;box-shadow:0 0 5px #3b82f6;animation:pulse 1s infinite}}
.dot.rec{{background:#ef4444;box-shadow:0 0 5px #ef4444;animation:pulse .7s infinite}}
.dot.pause{{background:#f59e0b;box-shadow:0 0 5px #f59e0b}}
.dot.err{{background:#ef4444}}
@keyframes pulse{{0%,100%{{opacity:1}}50%{{opacity:.25}}}}
#st{{font-size:.8rem;color:#888}}

/* activity grid */
.ag{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.ab{{background:#111;border:2px solid #222;border-radius:11px;
     padding:12px 8px;text-align:center;cursor:pointer;transition:all .15s;
     user-select:none;-webkit-tap-highlight-color:transparent}}
.ab:active{{transform:scale(.96)}}
.ab.sel{{border-color:#6366f1;background:#1e1b4b}}
.ab .icon{{font-size:1.5rem;margin-bottom:4px}}
.ab .name{{font-size:.72rem;font-weight:600;color:#ccc;line-height:1.2}}
.ab .count{{font-size:.65rem;color:#555;margin-top:3px}}
.ab.sel .name{{color:#a5b4fc}}
.ab.sel .count{{color:#6366f1}}

/* big control buttons */
.ctrl-row{{display:flex;gap:8px;width:100%;max-width:440px;margin-bottom:10px}}
.btn{{flex:1;padding:14px 8px;border:none;border-radius:12px;font-size:.95rem;
      font-weight:700;cursor:pointer;transition:background .15s;
      -webkit-tap-highlight-color:transparent}}
.btn:disabled{{background:#1e1e1e;color:#444;cursor:not-allowed}}
.btn-rec{{background:#dc2626;color:#fff}}
.btn-rec:hover:not(:disabled){{background:#b91c1c}}
.btn-pause{{background:#d97706;color:#fff}}
.btn-pause:hover:not(:disabled){{background:#b45309}}
.btn-resume{{background:#059669;color:#fff}}
.btn-resume:hover:not(:disabled){{background:#047857}}
.btn-stop{{background:#374151;color:#fff}}
.btn-stop:hover:not(:disabled){{background:#1f2937}}
.btn-conn{{background:#2563eb;color:#fff;width:100%;max-width:440px;
           padding:13px;border:none;border-radius:12px;font-size:.95rem;
           font-weight:700;cursor:pointer;margin-bottom:10px}}
.btn-conn:hover{{background:#1d4ed8}}

/* sensor mini display */
.sg{{display:grid;grid-template-columns:1fr 1fr;gap:8px}}
.sb{{background:#0f0f0f;border-radius:9px;padding:10px}}
.sl{{font-size:.6rem;text-transform:uppercase;letter-spacing:.07em;
     color:#444;margin-bottom:6px}}
.ar{{display:flex;justify-content:space-between;font-size:.75rem;margin-bottom:2px}}
.ak{{color:#555}}.av{{font-variant-numeric:tabular-nums}}
.av.x{{color:#f87171}}.av.y{{color:#4ade80}}.av.z{{color:#60a5fa}}

/* stats */
.tr{{display:flex;justify-content:space-between;font-size:.78rem;
     color:#555;padding:3px 0;border-bottom:1px solid #1a1a1a}}
.tr:last-child{{border-bottom:none}}
.tr span{{color:#e2e8f0;font-weight:600}}

/* progress bar */
.prog-wrap{{background:#111;border-radius:6px;height:8px;overflow:hidden;margin-top:6px}}
.prog-bar{{height:100%;background:#6366f1;border-radius:6px;
           transition:width .3s;width:0%}}

/* predicted activity */
#actbox{{text-align:center;padding:6px}}
#alab{{font-size:.62rem;text-transform:uppercase;letter-spacing:.1em;color:#444;margin-bottom:4px}}
#aval{{font-size:1.6rem;font-weight:800;color:#a78bfa;min-height:40px}}
#aconf{{font-size:.7rem;color:#444;margin-top:2px}}

/* log */
#log{{width:100%;max-width:440px;background:#0d0d0d;border:1px solid #1a1a1a;
      border-radius:10px;padding:8px 11px;font-size:.68rem;
      height:80px;overflow-y:auto;font-family:monospace;color:#444}}
.e{{margin-bottom:1px}}.ok{{color:#4ade80}}.er{{color:#f87171}}.in{{color:#93c5fd}}.warn{{color:#fbbf24}}
</style>
</head>
<body>

<h1>📡 IMU Collector</h1>
<p class="sub">Collect training data &amp; run live inference</p>

<!-- connection status -->
<div class="sr"><div class="dot" id="dot"></div><div id="st">Not connected</div></div>
<button class="btn-conn" id="btnConn" onclick="connectWS()">Connect to Server</button>

<!-- activity selector -->
<div class="card">
  <div class="ct">1 — Select Activity</div>
  <div class="ag" id="actGrid">
    <div class="ab" id="act0" onclick="selAct(0)">
      <div class="icon">🚶</div>
      <div class="name">Walking</div>
      <div class="count" id="cnt0">0 windows</div>
    </div>
    <div class="ab" id="act1" onclick="selAct(1)">
      <div class="icon">⬆️</div>
      <div class="name">Upstairs</div>
      <div class="count" id="cnt1">0 windows</div>
    </div>
    <div class="ab" id="act2" onclick="selAct(2)">
      <div class="icon">⬇️</div>
      <div class="name">Downstairs</div>
      <div class="count" id="cnt2">0 windows</div>
    </div>
    <div class="ab" id="act3" onclick="selAct(3)">
      <div class="icon">🪑</div>
      <div class="name">Sitting</div>
      <div class="count" id="cnt3">0 windows</div>
    </div>
    <div class="ab" id="act4" onclick="selAct(4)">
      <div class="icon">🧍</div>
      <div class="name">Standing</div>
      <div class="count" id="cnt4">0 windows</div>
    </div>
    <div class="ab" id="act5" onclick="selAct(5)">
      <div class="icon">🛌</div>
      <div class="name">Laying</div>
      <div class="count" id="cnt5">0 windows</div>
    </div>
  </div>
</div>

<!-- progress toward 100 windows target -->
<div class="card">
  <div class="ct">Collection Progress (target: 100 windows each)</div>
  <div id="progRows"></div>
</div>

<!-- record controls -->
<div class="card">
  <div class="ct">2 — Record Controls</div>
  <div class="ctrl-row">
    <button class="btn btn-rec"    id="btnRec"    onclick="startRec()"   disabled>⏺ Record</button>
    <button class="btn btn-pause"  id="btnPause"  onclick="pauseRec()"   disabled>⏸ Pause</button>
    <button class="btn btn-resume" id="btnResume" onclick="resumeRec()"  disabled>▶ Resume</button>
    <button class="btn btn-stop"   id="btnStop"   onclick="stopRec()"    disabled>⏹ Stop</button>
  </div>
  <div style="font-size:.72rem;color:#555;text-align:center;margin-top:2px">
    For short staircases: Record → go up/down → Pause → walk back → Resume → repeat
  </div>
</div>

<!-- live sensor readings -->
<div class="card">
  <div class="ct">Live Sensor</div>
  <div class="sg">
    <div class="sb">
      <div class="sl">Accel (m/s²)</div>
      <div class="ar"><span class="ak">X</span><span class="av x" id="ax">—</span></div>
      <div class="ar"><span class="ak">Y</span><span class="av y" id="ay">—</span></div>
      <div class="ar"><span class="ak">Z</span><span class="av z" id="az">—</span></div>
    </div>
    <div class="sb">
      <div class="sl">Gyro (°/s)</div>
      <div class="ar"><span class="ak">X</span><span class="av x" id="gx">—</span></div>
      <div class="ar"><span class="ak">Y</span><span class="av y" id="gy">—</span></div>
      <div class="ar"><span class="ak">Z</span><span class="av z" id="gz">—</span></div>
    </div>
  </div>
</div>

<!-- stream stats -->
<div class="card">
  <div class="ct">Stream Stats</div>
  <div class="tr">Packets sent   <span id="pc">0</span></div>
  <div class="tr">Sample rate    <span id="hz">— Hz</span></div>
  <div class="tr">Windows saved  <span id="wc">0</span></div>
  <div class="tr">Dropped        <span id="dr">0</span></div>
</div>

<!-- inference result (shown when model loaded) -->
<div class="card" id="actbox">
  <div id="alab">Live Prediction</div>
  <div id="aval">—</div>
  <div id="aconf"></div>
</div>

<div id="log"></div>

<script>
const WSS_URL = "{wss_url}";
const ACTIVITIES = ["Walking","Upstairs","Downstairs","Sitting","Standing","Laying"];
const TARGET = 100;

let ws=null, streaming=false, iid=null, rid=null;
let pkt=0, drop=0, rc=0, lts=Date.now();
const HZ=50, MS=1000/HZ;
let snap={{ax:0,ay:0,az:0,gx:0,gy:0,gz:0}};
let selActivity = null;
let recState = 'idle'; // idle | recording | paused
let counts = [0,0,0,0,0,0];

// ── init progress bars ──────────────────────────────────────────────────────
function initProgress() {{
  const el = document.getElementById('progRows');
  el.innerHTML = ACTIVITIES.map((a,i) => `
    <div style="margin-bottom:8px">
      <div style="display:flex;justify-content:space-between;
                  font-size:.7rem;color:#666;margin-bottom:3px">
        <span>${{a}}</span><span id="pcnt${{i}}">0 / ${{TARGET}}</span>
      </div>
      <div class="prog-wrap"><div class="prog-bar" id="pbar${{i}}"></div></div>
    </div>`).join('');
}}
initProgress();

function updateProgress(data) {{
  if (!data.counts) return;
  ACTIVITIES.forEach((a,i) => {{
    const n = data.counts[i] || 0;
    counts[i] = n;
    document.getElementById(`cnt${{i}}`).textContent = n + ' windows';
    document.getElementById(`pcnt${{i}}`).textContent = n + ' / ' + TARGET;
    document.getElementById(`pbar${{i}}`).style.width = Math.min(100, n/TARGET*100) + '%';
  }});
  document.getElementById('wc').textContent = data.total || 0;
}}

// ── logging ──────────────────────────────────────────────────────────────────
function lg(m, t='in') {{
  const el=document.getElementById('log'), d=document.createElement('div');
  d.className='e '+t;
  d.textContent=new Date().toLocaleTimeString()+' '+m;
  el.appendChild(d); el.scrollTop=el.scrollHeight;
}}
function dot(c,t) {{
  document.getElementById('dot').className='dot '+c;
  document.getElementById('st').textContent=t;
}}

// ── activity selection ────────────────────────────────────────────────────────
function selAct(id) {{
  if (recState === 'recording') {{ lg('Stop recording before switching activity','warn'); return; }}
  document.querySelectorAll('.ab').forEach(e => e.classList.remove('sel'));
  document.getElementById('act'+id).classList.add('sel');
  selActivity = id;
  document.getElementById('btnRec').disabled = (ws?.readyState !== 1);
  lg('Selected: ' + ACTIVITIES[id], 'in');
}}

// ── sensor permission ─────────────────────────────────────────────────────────
async function reqSensors() {{
  if (typeof DeviceMotionEvent !== 'undefined' &&
      typeof DeviceMotionEvent.requestPermission === 'function') {{
    try {{
      if (await DeviceMotionEvent.requestPermission() !== 'granted') {{
        lg('Motion permission denied','er'); return false;
      }}
    }} catch(e) {{ lg('Permission error: '+e,'er'); return false; }}
  }}
  window.addEventListener('devicemotion', onMotion, {{passive:true}});
  lg('Sensors attached ✓','ok');
  return true;
}}

function onMotion(e) {{
  const a=e.accelerationIncludingGravity||e.acceleration, g=e.rotationRate;
  if(a){{snap.ax=+(a.x||0); snap.ay=+(a.y||0); snap.az=+(a.z||0);}}
  if(g){{snap.gx=+(g.alpha||0); snap.gy=+(g.beta||0); snap.gz=+(g.gamma||0);}}
  ['ax','ay','az','gx','gy','gz'].forEach(k=>{{
    document.getElementById(k).textContent=snap[k].toFixed(3);
  }});
}}

// ── WebSocket ─────────────────────────────────────────────────────────────────
async function connectWS() {{
  if(ws) ws.close();
  lg('Connecting...'); dot('','Connecting...');
  ws = new WebSocket(WSS_URL);

  ws.onopen = async () => {{
    lg('Connected ✓','ok');
    dot('conn','Connected');
    document.getElementById('btnConn').textContent = '↺ Reconnect';
    const ok = await reqSensors();
    if (!ok) return;
    // start streaming immediately
    streaming=true; pkt=0; drop=0; rc=0; lts=Date.now();
    iid = setInterval(send, MS);
    rid = setInterval(()=>{{
      const e=(Date.now()-lts)/1000;
      document.getElementById('hz').textContent=(rc/e).toFixed(1)+' Hz';
      rc=0; lts=Date.now();
    }}, 1000);
    dot('live','Streaming');
    lg('Streaming at 50 Hz ▶','ok');
    if (selActivity !== null) document.getElementById('btnRec').disabled = false;
  }};

  ws.onclose = () => {{
    lg('Disconnected','er'); dot('err','Disconnected');
    stopRec(); streaming=false;
    if(iid){{ clearInterval(iid); iid=null; }}
    if(rid){{ clearInterval(rid); rid=null; }}
    document.getElementById('btnRec').disabled = true;
  }};

  ws.onerror = () => {{ lg('Connection failed','er'); dot('err','Error'); }};

  ws.onmessage = (e) => {{
    try {{
      const d = JSON.parse(e.data);
      if (d.type === 'stats')   updateProgress(d);
      if (d.type === 'saved')   {{ lg('✓ Window saved — ' + ACTIVITIES[d.label] + ' (' + d.total + ' total)','ok'); }}
      if (d.type === 'predict') {{
        document.getElementById('aval').textContent  = d.activity;
        document.getElementById('aconf').textContent = d.confidence != null
          ? (d.confidence*100).toFixed(1)+'% confidence' : '';
      }}
      if (d.type === 'log')     lg('[server] '+d.msg, d.level||'in');
    }} catch(_) {{}}
  }};
}}

function send() {{
  if (!ws || ws.readyState !== 1) {{
    drop++; document.getElementById('dr').textContent=drop; return;
  }}
  const payload = {{ts:Date.now(), ...snap, rec: recState==='recording' ? selActivity : null}};
  ws.send(JSON.stringify(payload));
  pkt++; rc++;
  document.getElementById('pc').textContent=pkt;
}}

// ── record controls ───────────────────────────────────────────────────────────
function startRec() {{
  if (selActivity === null) {{ lg('Select an activity first!','warn'); return; }}
  recState = 'recording';
  dot('rec', '⏺ Recording — ' + ACTIVITIES[selActivity]);
  document.getElementById('btnRec').disabled    = true;
  document.getElementById('btnPause').disabled  = false;
  document.getElementById('btnResume').disabled = true;
  document.getElementById('btnStop').disabled   = false;
  // lock activity selection during recording
  document.querySelectorAll('.ab').forEach(e=>e.style.pointerEvents='none');
  ws.send(JSON.stringify({{cmd:'start', label: selActivity}}));
  lg('⏺ Recording ' + ACTIVITIES[selActivity],'ok');
}}

function pauseRec() {{
  recState = 'paused';
  dot('pause','⏸ Paused — walk back to start');
  document.getElementById('btnPause').disabled  = true;
  document.getElementById('btnResume').disabled = false;
  ws.send(JSON.stringify({{cmd:'pause'}}));
  lg('⏸ Paused — walk back, then Resume','warn');
}}

function resumeRec() {{
  recState = 'recording';
  dot('rec','⏺ Recording — ' + ACTIVITIES[selActivity]);
  document.getElementById('btnPause').disabled  = false;
  document.getElementById('btnResume').disabled = true;
  ws.send(JSON.stringify({{cmd:'resume'}}));
  lg('▶ Resumed recording','ok');
}}

function stopRec() {{
  if (recState === 'idle') return;
  recState = 'idle';
  dot('live','Streaming');
  document.getElementById('btnRec').disabled    = (selActivity === null || ws?.readyState !== 1);
  document.getElementById('btnPause').disabled  = true;
  document.getElementById('btnResume').disabled = true;
  document.getElementById('btnStop').disabled   = true;
  document.querySelectorAll('.ab').forEach(e=>e.style.pointerEvents='auto');
  if (ws?.readyState === 1) ws.send(JSON.stringify({{cmd:'stop'}}));
  lg('⏹ Stopped','in');
}}
</script>
</body>
</html>""".encode("utf-8")

# ── WebSocket handler ─────────────────────────────────────────────────────────
class ClientState:
    def __init__(self):
        self.buf    = deque(maxlen=WINDOW_SIZE)
        self.since  = 0
        self.recording  = False
        self.paused     = False
        self.label      = None

    def is_active(self):
        return self.recording and not self.paused

async def ws_handler(websocket, model):
    addr = websocket.remote_address
    print(f"[+] Phone connected: {addr}")
    cs = ClientState()

    # send current stats on connect
    await websocket.send(json.dumps({
        "type": "stats",
        "counts": CSTATE.window_count,
        "total": CSTATE.total_windows
    }))

    try:
        async for msg in websocket:
            try:
                data = json.loads(msg)
            except:
                continue

            # ── handle commands ───────────────────────────────────────────
            cmd = data.get("cmd")
            if cmd == "start":
                cs.recording = True
                cs.paused    = False
                cs.label     = int(data["label"])
                cs.buf.clear()
                cs.since = 0
                print(f"  [REC] Start — {LABELS[cs.label]}")
                continue

            if cmd == "pause":
                cs.paused = True
                print(f"  [REC] Paused")
                continue

            if cmd == "resume":
                cs.paused = False
                print(f"  [REC] Resumed")
                continue

            if cmd == "stop":
                cs.recording = False
                cs.paused    = False
                print(f"  [REC] Stopped — {LABELS.get(cs.label,'?')} | "
                      f"total={CSTATE.total_windows}")
                await websocket.send(json.dumps({
                    "type":   "stats",
                    "counts": CSTATE.window_count,
                    "total":  CSTATE.total_windows
                }))
                continue

            # ── handle sensor sample ──────────────────────────────────────
            sample = [data.get(k, 0) for k in ("ax","ay","az","gx","gy","gz")]
            cs.buf.append(sample)
            cs.since += 1

            if len(cs.buf) < WINDOW_SIZE:
                continue

            window = np.array(cs.buf, dtype=np.float32)  # (128, 6)

            # ── save window if recording ──────────────────────────────────
            if cs.is_active() and cs.since >= STEP_SIZE:
                cs.since = 0
                save_window(cs.label, window)
                await websocket.send(json.dumps({
                    "type":  "saved",
                    "label": cs.label,
                    "total": CSTATE.total_windows
                }))
                await websocket.send(json.dumps({
                    "type":   "stats",
                    "counts": CSTATE.window_count,
                    "total":  CSTATE.total_windows
                }))

            # ── run inference if model loaded ─────────────────────────────
            if model is not None and cs.since >= STEP_SIZE:
                cs.since = 0
                activity, conf = predict(model, window)
                await websocket.send(json.dumps({
                    "type":       "predict",
                    "activity":   activity,
                    "confidence": round(conf, 4)
                }))

    except websockets.exceptions.ConnectionClosed:
        print(f"[-] Disconnected: {addr}")

# ── HTTP handler ──────────────────────────────────────────────────────────────
async def http_handler(reader, writer):
    global _html_cache
    try:
        await asyncio.wait_for(reader.read(2048), timeout=5)
        body = _html_cache
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
            b"Content-Length: " + str(len(body)).encode() +
            b"\r\nConnection: close\r\n\r\n" + body
        )
        await writer.drain()
    except:
        pass
    finally:
        try: writer.close()
        except: pass

_html_cache = b""

# ── main ──────────────────────────────────────────────────────────────────────
async def main(model_path):
    global _html_cache
    ssl_ctx, local_ip = make_cert()
    model = load_model(model_path)
    _html_cache = build_html(local_ip)

    http_srv = await asyncio.start_server(
        http_handler, "0.0.0.0", HTTP_PORT, ssl=ssl_ctx
    )

    print(f"\n{'='*58}")
    print(f"  IMU Data Collection + Inference Server")
    print(f"{'='*58}")
    print(f"\n  STEP 1 — On your phone (same WiFi) open:")
    print(f"\n      https://{local_ip}:{HTTP_PORT}\n")
    print(f"  STEP 2 — Accept the certificate warning")
    print(f"  STEP 3 — Tap 'Connect to Server'")
    print(f"  STEP 4 — Select activity → Record → Pause/Resume → Stop")
    print(f"\n  Data saves to : {Path(DATA_FILE).resolve()}")
    print(f"  Model         : {'loaded ✓' if model else 'not loaded'}")
    print(f"  WS port       : {WS_PORT}")
    print(f"{'='*58}\n")

    async with websockets.serve(
        lambda ws: ws_handler(ws, model),
        "0.0.0.0", WS_PORT, ssl=ssl_ctx
    ):
        async with http_srv:
            await asyncio.gather(
                http_srv.serve_forever(),
                asyncio.Future()
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="Path to trained model .pth")
    args = parser.parse_args()
    try:
        asyncio.run(main(args.model))
    except KeyboardInterrupt:
        print(f"\nStopped. Total windows saved: {CSTATE.total_windows}")
        print(f"Data file: {Path(DATA_FILE).resolve()}")
