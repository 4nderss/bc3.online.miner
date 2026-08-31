/* bc3.online miner GUI — styr mining-processen och visar dess JSON-ström.
   All formatering följer användarens locale (Intl utan hårdkodad locale). */

// Tauri-API:t når vi via window.__TAURI__ (kräver withGlobalTauri i
// tauri.conf.json). Saknas det vill vi se det i UI:t — inte tyst dö och
// lämna alla knappar döda.
const tauri = window.__TAURI__;
if (!tauri) {
  document.addEventListener("DOMContentLoaded", () => {
    const box = document.getElementById("log");
    if (box) {
      box.textContent =
        "Fatal: the Tauri API is unavailable (withGlobalTauri). The UI cannot control the miner.";
    }
  });
  throw new Error("window.__TAURI__ missing");
}
const { invoke } = tauri.core;
const { listen } = tauri.event;

// Visa oväntade JS-fel i aktivitetsloggen i stället för att dö tyst.
window.addEventListener("error", (e) => {
  const box = document.getElementById("log");
  if (box) {
    const line = document.createElement("div");
    line.className = "l-bad";
    line.textContent = "UI error: " + (e.message || e.error);
    box.append(line);
  }
});

const el = (id) => document.getElementById(id);
const num2 = new Intl.NumberFormat(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const numInt = new Intl.NumberFormat(undefined);
const timeFmt = new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit", second: "2-digit" });

let mining = false;
let mode = "pplns";   // payout: pplns | solo
let hw = "gpu";       // hårdvara: gpu | cpu | dual
const spark = [];

// ---------- Formattering ----------
function fmtHashrate(h) {
  if (!isFinite(h) || h <= 0) return "0 H/s";
  const units = [["TH/s", 1e12], ["GH/s", 1e9], ["MH/s", 1e6], ["kH/s", 1e3]];
  for (const [u, f] of units) if (h >= f) return num2.format(h / f) + " " + u;
  return numInt.format(Math.round(h)) + " H/s";
}
function fmtDuration(s) {
  if (!isFinite(s) || s <= 0) return "—";
  const d = Math.floor(s / 86400), h = Math.floor((s % 86400) / 3600);
  const m = Math.floor((s % 3600) / 60), sec = Math.floor(s % 60);
  if (d > 0) return `${numInt.format(d)}d ${h}h`;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${sec}s`;
  return `${sec}s`;
}
function fmtDiff(d) {
  if (!isFinite(d) || d <= 0) return "—";
  const units = [["G", 1e9], ["M", 1e6], ["k", 1e3]];
  for (const [u, f] of units) if (d >= f) return num2.format(d / f) + u;
  return num2.format(d);
}
function setStat(id, value) {
  const e = el(id);
  if (!e || e.textContent === value) return;
  e.textContent = value;
  e.classList.remove("bump");
  void e.offsetWidth;
  e.classList.add("bump");
}

// ---------- Logg ----------
function log(text, cls) {
  const line = document.createElement("div");
  const t = document.createElement("span");
  t.className = "t";
  t.textContent = timeFmt.format(new Date());
  const body = document.createElement("span");
  if (cls) body.className = cls;
  body.textContent = text;
  line.append(t, body);
  const box = el("log");
  box.append(line);
  while (box.children.length > 300) box.firstChild.remove();
  box.scrollTop = box.scrollHeight;
}

// ---------- Hashrate-sparkline ----------
function drawSpark() {
  const canvas = el("spark");
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth, h = 48;
  if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.scale(dpr, dpr);
  }
  ctx.clearRect(0, 0, w, h);
  if (spark.length < 2) return;
  const max = Math.max(...spark) * 1.15 || 1;
  const step = w / (spark.length - 1);
  const style = getComputedStyle(document.documentElement);
  const data = style.getPropertyValue("--data").trim();

  ctx.beginPath();
  spark.forEach((v, i) => {
    const x = i * step, y = h - (v / max) * (h - 4) - 2;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.strokeStyle = data;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.stroke();

  ctx.lineTo(w, h);
  ctx.lineTo(0, h);
  ctx.closePath();
  ctx.fillStyle = data + "22";
  ctx.fill();
}
window.addEventListener("resize", drawSpark);

// ---------- Temperatur ----------
// Färgen följer värmen: normalt → varmt (>75) → hett (>85).
function tempClass(c) {
  if (c == null) return "";
  if (c >= 85) return "bad";
  if (c >= 75) return "warm";
  return "ok";
}
function renderTemps(ev) {
  const g = ev.gpu_temp_c, c = ev.cpu_temp_c;
  const main = g != null ? g : c;
  const e = el("temp");
  setStat("temp", main != null ? `${numInt.format(main)}°C` : "—");
  e.className = "stat-value " + tempClass(main);

  const parts = [];
  if (g != null) parts.push(`GPU ${numInt.format(g)}°C`);
  if (c != null) parts.push(`CPU ${numInt.format(c)}°C`);
  if (ev.gpu_power_w != null) parts.push(`${num2.format(ev.gpu_power_w)} W`);
  if (ev.gpu_fan_pct != null) parts.push(`fan ${numInt.format(ev.gpu_fan_pct)}%`);
  el("temp-sub").textContent = parts.length ? parts.join(" · ") : "not available";
}

// ---------- Status ----------
function setStatus(state, text) {
  const dot = el("status-dot");
  dot.className = "dot" + (state ? " " + state : "");
  el("status-text").textContent = text;
}

// ---------- Inställningar (sparas mellan körningar) ----------
const SETTINGS_KEY = "bc3-miner-settings";
function loadSettings() {
  try {
    const s = JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}");
    if (s.address) el("address").value = s.address;
    if (s.rig) el("rig").value = s.rig;
    if (s.mode) selectMode(s.mode);
    if (s.hw) selectHw(s.hw);
    if (s.intensity) el("intensity").value = s.intensity;
    updateIntensityLabel();
  } catch (_) {}
}
function saveSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify({
      address: el("address").value.trim(),
      rig: el("rig").value.trim(),
      mode,
      hw,
      intensity: el("intensity").value,
    }));
  } catch (_) {}
}

function updateIntensityLabel() {
  el("intensity-value").textContent = el("intensity").value + "%";
}
el("intensity").addEventListener("input", updateIntensityLabel);
el("intensity").addEventListener("change", saveSettings);

/// Grovkoll så Start inte kan tryckas i onödan — poolen gör den slutgiltiga
/// valideringen vid authorize.
///
/// BC3 använder Bitcoins alla adressformat. Den här funktionen krävde
/// tidigare prefixet "bc1" och stängde därmed av Start helt för alla med en
/// legacy-adress (1… eller 3…), trots att poolen accepterar dem. Att vara
/// för strikt här är just vad som bröt det — hellre släppa igenom något
/// tveksamt och låta poolen säga ifrån.
function addressLooksValid(addr) {
  const a = (addr || "").trim();
  if (!a) return false;
  // bech32/bech32m (segwit v0 och taproot). BIP173 tillåter en helt versal
  // adress, så prefixet måste matchas skiftlägesokänsligt.
  if (/^bc1[a-z0-9]{20,87}$/i.test(a)) return true;
  // Legacy base58check: P2PKH (1…) och P2SH (3…). Base58 saknar 0, O, I och l.
  if (/^[13][1-9A-HJ-NP-Za-km-z]{25,39}$/.test(a)) return true;
  return false;
}

/// Start är avstängd tills adressen ser rimlig ut (och alltid aktiv för Stop).
function updateStartEnabled() {
  const ok = addressLooksValid(el("address").value);
  el("toggle").disabled = !mining && !ok;
  const err = el("address-error");
  const typed = el("address").value.trim().length > 0;
  if (!ok && typed) {
    err.textContent = "Enter a BC3 address — bc1…, 1… or 3…";
    el("address").classList.add("invalid");
  } else {
    err.textContent = "";
    el("address").classList.remove("invalid");
  }
}
el("address").addEventListener("input", updateStartEnabled);
el("address").addEventListener("change", saveSettings);

/// Markera en knapp i en radiogrupp (attribut `key` = "mode" eller "hw").
function selectIn(key, value) {
  document.querySelectorAll(`.mode[data-${key}]`).forEach((b) => {
    const on = b.dataset[key] === value;
    b.classList.toggle("active", on);
    b.setAttribute("aria-checked", on ? "true" : "false");
  });
}
function selectMode(m) {
  mode = m;
  selectIn("mode", m);
}
function selectHw(h) {
  // "dual" fanns i en tidigare version — sparade inställningar kan ha kvar
  // det, och mätning visade att det gav lägre hashrate än enbart GPU.
  hw = h === "gpu" || h === "cpu" ? h : "gpu";
  selectIn("hw", hw);
}
document.querySelectorAll(".mode").forEach((b) => {
  b.addEventListener("click", () => {
    if (mining) return; // valen byts inte mitt i en körning
    if (b.dataset.mode) selectMode(b.dataset.mode);
    else if (b.dataset.hw) selectHw(b.dataset.hw);
    saveSettings();
  });
});

// ---------- Start/stopp ----------
function setRunning(on) {
  mining = on;
  const btn = el("toggle");
  btn.textContent = on ? "Stop mining" : "Start mining";
  btn.classList.toggle("running", on);
  ["address", "rig", "intensity"].forEach((id) => (el(id).disabled = on));
  document.querySelectorAll(".mode").forEach((b) => (b.disabled = on));
  updateStartEnabled();
}

async function toggle() {
  if (mining) {
    await invoke("stop_mining").catch((e) => log(String(e), "l-bad"));
    return;
  }
  el("address-error").textContent = "";
  el("address").classList.remove("invalid");
  spark.length = 0;
  drawSpark();
  try {
    await invoke("start_mining", {
      opts: {
        address: el("address").value.trim(),
        rig: el("rig").value.trim(),
        mode,
        pool: "",
        hardware: hw,
        intensity: Number(el("intensity").value) || 100,
      },
    });
    setRunning(true);
    saveSettings();
  } catch (e) {
    const msg = String(e);
    el("address-error").textContent = msg;
    el("address").classList.add("invalid");
    log(msg, "l-bad");
  }
}
el("toggle").addEventListener("click", toggle);
el("clear-log").addEventListener("click", () => (el("log").textContent = ""));
el("block-close").addEventListener("click", () =>
  el("block-overlay").classList.add("hidden")
);

// ---------- Händelser från minern ----------
listen("miner-event", ({ payload: ev }) => {
  switch (ev.type) {
    case "startup":
      el("backend").textContent = ev.gpus.length ? "GPU" : "CPU";
      el("device").textContent = ev.gpus.length
        ? ev.gpus.join(", ")
        : `${ev.cpu_threads} threads`;
      log(`miner ${ev.version} started — ${ev.gpus.length ? ev.gpus.join(", ") : "CPU only"}`);
      break;
    case "status":
      if (ev.state === "mining") setStatus("mining", "Mining");
      else if (ev.state === "connecting") setStatus("connecting", "Connecting");
      else if (ev.state === "error") setStatus("error", "Error");
      log(ev.message, ev.state === "error" ? "l-bad" : null);
      break;
    case "stats":
      setStat("hashrate", fmtHashrate(ev.hashrate));
      // Visa uppdelningen bara när båda backends faktiskt bidrar.
      if (ev.hashrate_gpu > 0 && ev.hashrate_cpu > 0) {
        el("hashrate-sub").textContent =
          `GPU ${fmtHashrate(ev.hashrate_gpu)} · CPU ${fmtHashrate(ev.hashrate_cpu)}`;
      }
      setStat("accepted", numInt.format(ev.accepted));
      setStat("rejected", numInt.format(ev.rejected));
      setStat("eta", ev.eta_secs ? fmtDuration(ev.eta_secs) : "—");
      setStat("best-share", ev.best_share > 0 ? fmtDiff(ev.best_share) : "—");
      setStat("blocks", numInt.format(ev.blocks || 0));
      // Hur nära ett block bästa sharen var (nätverkssvårigheten = 100 %).
      if (ev.best_share > 0 && ev.network_difficulty > 0) {
        const pct = (ev.best_share / ev.network_difficulty) * 100;
        el("best-share-sub").textContent =
          (pct >= 1 ? num2.format(pct) : pct.toPrecision(2)) + "% of a block";
      }
      el("netdiff").textContent = "network difficulty " + fmtDiff(ev.network_difficulty);
      // Höjden poolen jobbar på — kvitto på att vi är i synk.
      if (ev.job_height > 0) {
        setStat("job-height", "#" + numInt.format(ev.job_height));
        el("job-height-sub").textContent = "in sync with the pool";
        el("job-height-card").classList.add("live");
      }
      renderTemps(ev);
      spark.push(ev.hashrate);
      while (spark.length > 60) spark.shift();
      drawSpark();
      break;
    case "share":
      log(ev.accepted ? "share accepted" : "share rejected",
          ev.accepted ? "l-ok" : "l-bad");
      break;
    case "newblockheight": {
      setStat("job-height", "#" + numInt.format(ev.height));
      el("job-height-sub").textContent = "in sync with the pool";
      el("job-height-card").classList.add("live");
      log("pool started on block #" + numInt.format(ev.height), "l-accent");
      break;
    }
    case "block":
      el("block-hash").textContent = ev.hash;
      el("block-overlay").classList.remove("hidden");
      log("BLOCK FOUND: " + ev.hash, "l-accent");
      break;
  }
});

listen("miner-log", ({ payload }) => log(payload.text));

listen("miner-stopped", ({ payload: code }) => {
  setRunning(false);
  setStatus(null, "Idle");
  setStat("hashrate", "—");
  setStat("job-height", "—");
  el("job-height-sub").textContent = "waiting for a job";
  el("job-height-card").classList.remove("live");
  log(code === 0 || code == null ? "miner stopped" : `miner exited (code ${code})`,
      code ? "l-bad" : null);
});

// ---------- Hårdvarudetektering ----------
// Fråga minern vad som finns, så knapparna visar kortnamn/kärnor direkt.
async function probeHardware() {
  try {
    const p = await invoke("probe_hardware");
    const gpuBtn = document.querySelector('.mode[data-hw="gpu"]');
    const dualBtn = document.querySelector('.mode[data-hw="dual"]');
    if (p.gpus && p.gpus.length) {
      // "CUDA #0: NVIDIA GeForce RTX 3050 Ti Laptop GPU" → kortnamnet.
      const short = p.gpus[0].replace(/^[A-Z]+ #\d+:\s*/, "");
      el("hw-gpu-sub").textContent = short;
      gpuBtn.title = p.gpus.join("\n");
    } else {
      el("hw-gpu-sub").textContent = "No GPU detected";
      gpuBtn.disabled = true;
      dualBtn.disabled = true;
      if (hw !== "cpu") selectHw("cpu");
    }
    if (p.cpu_cores) {
      el("hw-cpu-sub").textContent = `All ${numInt.format(p.cpu_cores)} cores`;
    }
  } catch (e) {
    el("hw-gpu-sub").textContent = "Detection failed";
    log("hardware probe failed: " + e, "l-bad");
  }
}


// ---------- Update check ----------
// We only ASK GitHub whether a newer release exists and show a link. The miner
// never updates itself: it holds the user's payout address, so a binary that
// downloads and executes code would let anyone who compromised the release key
// silently redirect everyone's rewards. Antivirus heuristics flag that shape of
// behaviour too, which would cost us false positives on top of the risk.
const RELEASES_API =
  "https://api.github.com/repos/4nderss/bc3.online.miner/releases/latest";
const RELEASES_PAGE = "https://github.com/4nderss/bc3.online.miner/releases/latest";
const UPDATE_INTERVAL_MS = 6 * 60 * 60 * 1000;
const UPDATE_DISMISSED_KEY = "bc3-miner-update-dismissed";

// "v1.10.2" -> [1, 10, 2]. Any pre-release suffix is dropped, so 1.2.0-rc1 and
// 1.2.0 compare equal and we never nag someone running a release candidate.
function parseVersion(v) {
  const m = String(v || "").trim().replace(/^v/i, "").match(/^(\d+)(?:\.(\d+))?(?:\.(\d+))?/);
  return m ? [+m[1], +(m[2] || 0), +(m[3] || 0)] : null;
}

function isNewer(candidate, current) {
  const a = parseVersion(candidate), b = parseVersion(current);
  if (!a || !b) return false;
  for (let i = 0; i < 3; i++) {
    if (a[i] !== b[i]) return a[i] > b[i];
  }
  return false;
}

function showUpdateBanner(latest, current) {
  el("update-version").textContent = latest;
  el("update-current").textContent = current;
  el("update-banner").hidden = false;
}

// Reachable only if the webview has no way to hand a URL to the system
// browser. Replaces the button with the address itself, pre-selected.
function showUpdateUrlFallback() {
  const btn = el("update-open");
  const box = document.createElement("input");
  box.type = "text";
  box.readOnly = true;
  box.className = "update-url";
  box.value = RELEASES_PAGE;
  btn.replaceWith(box);
  box.focus();
  box.select();
}

async function checkForUpdate() {
  try {
    const current = await tauri.app.getVersion();
    const res = await fetch(RELEASES_API, {
      headers: { Accept: "application/vnd.github+json" },
    });
    if (!res.ok) return;                     // rate limit, outage - stay quiet
    const rel = await res.json();
    const latest = String(rel.tag_name || "").replace(/^v/i, "");
    if (!isNewer(latest, current)) return;

    // Dismissing hides THIS version only; a later release speaks up again.
    let dismissed = null;
    try {
      dismissed = localStorage.getItem(UPDATE_DISMISSED_KEY);
    } catch (e) { /* private mode - just show it */ }
    if (dismissed === latest) return;

    showUpdateBanner(latest, current);
  } catch (e) {
    // Offline rigs are normal. An update check must never be able to disrupt
    // mining, so every failure here is silent by design.
  }
}

function initUpdateCheck() {
  el("update-close").addEventListener("click", () => {
    try {
      localStorage.setItem(UPDATE_DISMISSED_KEY, el("update-version").textContent);
    } catch (e) { /* nothing to remember it with - fine */ }
    el("update-banner").hidden = true;
  });

  // Open in the system browser, never in the webview: navigating the webview
  // away would take the running miner's UI with it.
  el("update-open").addEventListener("click", async () => {
    try {
      if (tauri.shell && tauri.shell.open) return void (await tauri.shell.open(RELEASES_PAGE));
      if (tauri.opener && tauri.opener.openUrl) return void (await tauri.opener.openUrl(RELEASES_PAGE));
      throw new Error("no opener API");
    } catch (e) {
      // Never leave the button doing nothing visible: if the host cannot open
      // a browser for us, put the URL on screen so it can be copied by hand.
      showUpdateUrlFallback();
      log("open the release page manually: " + RELEASES_PAGE, "l-accent");
    }
  });

  checkForUpdate();
  setInterval(checkForUpdate, UPDATE_INTERVAL_MS);
}

// ---------- Init ----------
loadSettings();
setStatus(null, "Idle");
updateStartEnabled();
probeHardware();
invoke("is_mining").then((on) => on && setRunning(true)).catch(() => {});
initUpdateCheck();
