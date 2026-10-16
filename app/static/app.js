"use strict";

const $ = (id) => document.getElementById(id);
const video = $("video"), uploaded = $("uploaded"), overlay = $("overlay");
const ctx = overlay.getContext("2d");
const grab = document.createElement("canvas");

const SMOOTHING = 10;         // nombre d'images pour le lissage temporel (~2 s)
const INTERVAL_MS = 120;      // pause entre deux requetes
const MAX_WIDTH = 640;        // resolution envoyee au serveur

let threshold = 0.5;
let stream = null;
let running = false;
let mode = "webcam";
let history = [];
let lastFrameTs = 0;

// ---------- Modele ----------
async function loadModelInfo() {
  try {
    const r = await fetch("/model");
    if (!r.ok) throw new Error(r.status);
    const m = await r.json();
    threshold = m.threshold;
    $("meterThr").style.left = `${threshold * 100}%`;
    $("metricsTitle").textContent = `Performances — ${m.mode} (test ${
      m.test_set === "webcam" ? "webcam, session inédite" : "NUAA session 3"})`;
    for (const dd of document.querySelectorAll("#metrics dd")) {
      const v = m.test[dd.dataset.k];
      if (v == null) continue;
      dd.textContent = dd.dataset.k === "auc" ? v.toFixed(4) : `${(v * 100).toFixed(2)} %`;
    }
    setStatus(`Modèle ${m.version} prêt`, "ok");
  } catch (e) {
    setStatus("Modèle indisponible", "error");
  }
}

function setStatus(text, kind) {
  const s = $("status");
  s.textContent = text;
  s.className = `status status--${kind}`;
}

// ---------- Verdict ----------
function showVerdict(result) {
  const box = $("verdict");
  if (!result || !result.face_found) {
    history = [];
    box.dataset.state = "none";
    $("verdictLabel").textContent = "Aucun visage";
    $("verdictHint").textContent = "Placez un visage bien éclairé, de face, devant la caméra.";
    $("meterFill").style.width = "0";
    $("scoreText").textContent = "score —";
    $("details").textContent = "";
    return;
  }
  const d = result.details || {};
  $("details").textContent = d.lbp != null
    ? `Dernière image — CNN : ${d.cnn.toFixed(3)} · texture LBP : ${d.lbp.toFixed(3)}`
    : "";
  history.push(result.spoof_score);
  if (history.length > SMOOTHING) history.shift();
  const score = history.reduce((a, b) => a + b, 0) / history.length;
  const isSpoof = score >= threshold;
  box.dataset.state = isSpoof ? "spoof" : "real";
  $("verdictLabel").textContent = isSpoof ? "ATTAQUE" : "VISAGE RÉEL";
  $("verdictHint").textContent = isSpoof
    ? "Photo imprimée ou écran détecté : accès refusé."
    : "Visage vivant détecté : accès autorisé.";
  $("meterFill").style.width = `${score * 100}%`;
  $("scoreText").textContent = `score ${score.toFixed(3)}`;
  return isSpoof;
}

// ---------- Dessin de la boite ----------
let lastDraw = null;

function drawBox(result, source, mirrored) {
  lastDraw = result ? [result, source, mirrored] : null;
  const rect = overlay.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  overlay.width = rect.width * dpr;
  overlay.height = rect.height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!result || !result.face_found) return;

  const sw = source.videoWidth || source.naturalWidth;
  const sh = source.videoHeight || source.naturalHeight;
  const scale = Math.min(rect.width / sw, rect.height / sh) * (sw / result.frame_w);
  const offX = (rect.width - sw * Math.min(rect.width / sw, rect.height / sh)) / 2;
  const offY = (rect.height - sh * Math.min(rect.width / sw, rect.height / sh)) / 2;
  const b = result.box;
  const bx = mirrored ? result.frame_w - b.x - b.w : b.x;
  const x = offX + bx * scale, y = offY + b.y * scale, w = b.w * scale, h = b.h * scale;

  const color = getComputedStyle(document.documentElement)
    .getPropertyValue(result.is_spoof ? "--spoof" : "--real").trim();
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.strokeRect(x, y, w, h);
  const tag = result.is_spoof ? "ATTAQUE" : "RÉEL";
  ctx.font = "600 14px system-ui, sans-serif";
  const tw = ctx.measureText(tag).width + 14;
  ctx.fillStyle = color;
  ctx.fillRect(x - 1.5, y - 24, tw, 24);
  ctx.fillStyle = "#fff";
  ctx.fillText(tag, x + 6, y - 7);
}

// ---------- Envoi au serveur ----------
async function analyse(source) {
  const sw = source.videoWidth || source.naturalWidth;
  const sh = source.videoHeight || source.naturalHeight;
  const k = Math.min(1, MAX_WIDTH / sw);
  grab.width = Math.round(sw * k);
  grab.height = Math.round(sh * k);
  grab.getContext("2d").drawImage(source, 0, 0, grab.width, grab.height);
  const blob = await new Promise((res) => grab.toBlob(res, "image/jpeg", 0.9));
  const form = new FormData();
  form.append("file", blob, "frame.jpg");
  const r = await fetch("/predict", { method: "POST", body: form });
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return { ...(await r.json()), frame_w: grab.width };
}

async function loop() {
  while (running) {
    const t0 = performance.now();
    try {
      const result = await analyse(video);
      result.is_spoof = showVerdict(result);
      drawBox(result, video, true);
      const fps = 1000 / (t0 - lastFrameTs || 1);
      lastFrameTs = t0;
      $("perf").textContent = result.face_found
        ? `inférence ${result.latency_ms} ms · ${fps.toFixed(1)} img/s`
        : `${fps.toFixed(1)} img/s`;
    } catch (e) {
      setStatus("Erreur de communication avec le serveur", "error");
    }
    await new Promise((res) => setTimeout(res, INTERVAL_MS));
  }
}

// ---------- Webcam ----------
async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 }, audio: false });
  } catch (e) {
    alert("Accès à la caméra refusé ou indisponible.");
    return;
  }
  video.srcObject = stream;
  await video.play();
  $("placeholder").hidden = true;
  $("toggleCam").textContent = "Arrêter la caméra";
  $("toggleCam").className = "btn btn--stop";
  running = true;
  history = [];
  loop();
}

function stopCamera() {
  running = false;
  if (stream) stream.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  $("toggleCam").textContent = "Démarrer la caméra";
  $("toggleCam").className = "btn btn--primary";
  $("placeholder").hidden = false;
  $("perf").textContent = "";
  drawBox(null);
  showVerdict(null);
}

$("toggleCam").addEventListener("click", () => (running ? stopCamera() : startCamera()));

// ---------- Image importee ----------
$("fileInput").addEventListener("change", async (ev) => {
  const file = ev.target.files[0];
  if (!file) return;
  uploaded.src = URL.createObjectURL(file);
  await uploaded.decode();
  uploaded.hidden = false;
  $("placeholder").hidden = true;
  history = [];
  try {
    const result = await analyse(uploaded);
    result.is_spoof = showVerdict(result);
    drawBox(result, uploaded, false);
    $("perf").textContent = result.face_found ? `inférence ${result.latency_ms} ms` : "";
  } catch (e) {
    setStatus("Erreur de communication avec le serveur", "error");
  }
});

// ---------- Onglets ----------
for (const tab of document.querySelectorAll(".tab")) {
  tab.addEventListener("click", () => {
    mode = tab.dataset.mode;
    document.querySelectorAll(".tab").forEach((t) => {
      t.classList.toggle("is-active", t === tab);
      t.setAttribute("aria-selected", t === tab);
    });
    const webcam = mode === "webcam";
    if (!webcam && running) stopCamera();
    $("webcamControls").hidden = !webcam;
    $("uploadControls").hidden = webcam;
    video.hidden = !webcam;
    uploaded.hidden = webcam || !uploaded.src;
    $("placeholder").hidden = !webcam ? Boolean(uploaded.src) : false;
    drawBox(null);
    showVerdict(null);
  });
}

// Au redimensionnement, on redessine la derniere boite au lieu de l'effacer
window.addEventListener("resize", () => {
  if (lastDraw) drawBox(...lastDraw);
});

// Lien de demonstration : /#src=/static/... ouvre l'onglet "image" et analyse cette image
// (meme origine uniquement).
async function analyseFromHash() {
  const m = location.hash.match(/^#src=(\/static\/[\w.-]+)$/);
  if (!m) return;
  document.body.classList.add("no-anim");     // rendu immediat (captures d'ecran)
  document.querySelector('.tab[data-mode="upload"]').click();
  uploaded.src = m[1];
  await uploaded.decode();
  uploaded.hidden = false;
  $("placeholder").hidden = true;
  const result = await analyse(uploaded);
  result.is_spoof = showVerdict(result);
  drawBox(result, uploaded, false);
  $("perf").textContent = result.face_found ? `inférence ${result.latency_ms} ms` : "";
}

loadModelInfo().then(analyseFromHash);
showVerdict(null);
