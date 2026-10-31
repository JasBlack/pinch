import * as THREE from 'three';
import { FilesetResolver, HandLandmarker, FaceLandmarker, PoseLandmarker } from '@mediapipe/tasks-vision';
import { EFFECTS, VERTEX, buildFragment } from './shaders.js?v=14';

// show startup failures in the status screen instead of leaving a silent, dead page
const showFatal = msg => { const el = document.getElementById('status'); if (el) el.textContent = msg; };
addEventListener('error', e => showFatal(`error: ${e.message}`));
addEventListener('unhandledrejection', e => console.warn('unhandled:', e.reason));

// ---------- state ----------
const state = {
  mode: 'hands',            // 'hands' | 'face' | 'both'
  trackingSource: 'webcam', // 'webcam' | 'drive' — what the trackers watch
  effect: 'pull',           // mesh warp type for hands/mouse: 'pull' | 'inflate' | 'twist' | 'ripple'
  faceEffect: 'inflate',    // warp kernel for face points — inflate reads bulbous/organic
  material: 'clay',         // 'clay' (holds shape) | 'jelly' (springs back) | 'liquid' (ripples)
  smoothing: 0.35,          // 0..1 settle amount (diffusion / damping depending on material)
  mouthBlow: true,          // open mouth inflates the canvas at its position
  palmHeal: false,          // open flat hand smooths the warp back out (opt-in)
  micReact: false,          // microphone level modulates shader amount
  micGain: 0.5,
  soundOut: false,          // canvas groans when stretched
  attract: false,           // idle attract mode: ghost performers when nobody is tracked
  shader: 'none',           // id of the active fragment shader from shaders.js
  amt: 0.5,                 // 0..1 intensity of the active shader
  brushFrac: 0.16,          // brush radius as fraction of min(viewW, viewH)
  pinchScale: 1,            // multiplies the pinch grab/release thresholds
  faceAmp: 3.0,             // amplification of facial deltas
  autoReset: false,
  resetDelaySec: 10,
  showPreview: true,
  showCursors: true,
  camReady: false,
  driveReady: false,
  handsReady: false,
  faceReady: false,
  poseReady: false,
  facing: 'user',           // 'user' (front, mirrored) | 'environment' (rear)
};

const SEG_X = 128, SEG_Y = 72;
const VERTS = (SEG_X + 1) * (SEG_Y + 1);
const EDGE_MARGIN_FRAC = 0.12;   // border band where warp fades to zero (edges stay pinned)
const IMAGE_SECONDS = 10;        // how long a still image stays before the playlist advances

// Facial landmarks used as warp handles (MediaPipe FaceMesh indices):
// forehead, nose tip, chin, mouth corners, upper/lower lip, below-lip, eye outer
// corners, upper eyelids, eyebrows, cheeks
const FACE_POINTS = [10, 1, 152, 61, 291, 13, 14, 17, 33, 263, 159, 386, 105, 334, 234, 454];
const EYE_L = 33, EYE_R = 263; // used to scale brushes to the face's on-screen size

// ---------- three.js scaffold ----------
// AA off (fullscreen quad, nothing to antialias) and DPR capped: fragment shaders
// are fill-rate bound, and 2x DPR quadruples the pixels for no visible gain here
const renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance' });
// phones: touch-first and fill-rate poor, so they render at 1x and default to a lighter render scale
const IS_PHONE = matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820;
const BASE_DPR = Math.min(devicePixelRatio, IS_PHONE ? 1 : 1.5);
renderer.setPixelRatio(BASE_DPR);
document.getElementById('stage').appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);

let geometry = null;
let basePos = null;                             // undeformed vertex xy
let edgeW = null;                               // per-vertex 0..1 weight, 0 at plane borders
const persistOff = new Float32Array(VERTS * 2); // accumulated warp offsets (hands + face, persistent)
const velOff = new Float32Array(VERTS * 2);     // per-vertex velocity for jelly/liquid materials
let meshDirty = true;                           // vertices changed since last GPU upload
let hasWarp = false;                            // any non-zero offsets at all
let commitAll = true;                           // next commit must touch the whole grid
let brushActivity = 0;                          // recent stroke energy, drives the groan synth

// Bounding box (in grid rows/cols) of the region with non-zero offsets. All the
// per-vertex loops (brush, diffusion, decay, commit) run only inside it — a
// typical warp covers a fraction of the grid, so this is a 5-20x CPU saving.
const COLS = SEG_X + 1, ROWS = SEG_Y + 1;
let warpBox = null;
let gridMap = null; // world xy -> grid col/row mapping, rebuilt on resize

function expandBox(c0, c1, r0, r1) {
  c0 = Math.max(0, c0); c1 = Math.min(SEG_X, c1);
  r0 = Math.max(0, r0); r1 = Math.min(SEG_Y, r1);
  if (!warpBox) warpBox = { c0, c1, r0, r1 };
  else {
    warpBox.c0 = Math.min(warpBox.c0, c0); warpBox.c1 = Math.max(warpBox.c1, c1);
    warpBox.r0 = Math.min(warpBox.r0, r0); warpBox.r1 = Math.max(warpBox.r1, r1);
  }
}

// Texture is sampled raw (no color-space conversion in either direction), so every
// shader is an exact pass-through at its identity settings.
// One ShaderMaterial per effect, compiled lazily on first selection.
const materialCache = new Map();

function makeUniforms() {
  return {
    uMap: { value: null },
    uHistory: { value: null },
    uPrev: { value: null },
    uMirror: { value: 0 },
    uAmt: { value: 0.5 },
    uTime: { value: 0 },
    uAspect: { value: 16 / 9 },
    uHead: { value: 0 },
    uTexel: { value: new THREE.Vector2(1 / 1280, 1 / 720) },
    uResolution: { value: new THREE.Vector2(1, 1) },
  };
}

function getMaterial(shaderId) {
  if (!materialCache.has(shaderId)) {
    const def = EFFECTS.find(e => e.id === shaderId) || EFFECTS[0];
    materialCache.set(shaderId, new THREE.ShaderMaterial({
      uniforms: makeUniforms(),
      vertexShader: VERTEX,
      fragmentShader: buildFragment(def.body),
    }));
  }
  return materialCache.get(shaderId);
}

let textureMirror = false;

function setShader(id) {
  state.shader = id;
  mesh.material = getMaterial(id);
}

const mesh = new THREE.Mesh(undefined, getMaterial('none'));
scene.add(mesh);

// pinch cursors
const cursorGeo = new THREE.RingGeometry(6, 9, 32);
const cursors = {};
function getCursor(id) {
  if (!cursors[id]) {
    const c = new THREE.Mesh(cursorGeo, new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 }));
    c.position.z = 1;
    c.visible = false;
    scene.add(c);
    cursors[id] = c;
  }
  return cursors[id];
}

let texAspect = 16 / 9;

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h);
  camera.left = -w / 2; camera.right = w / 2;
  camera.top = h / 2; camera.bottom = -h / 2;
  camera.updateProjectionMatrix();

  // plane covers the window like CSS object-fit: cover
  let pw, ph;
  if (w / h > texAspect) { pw = w; ph = w / texAspect; }
  else { ph = h; pw = h * texAspect; }

  const old = geometry;
  geometry = new THREE.PlaneGeometry(pw, ph, SEG_X, SEG_Y);
  geometry.setAttribute('aDisp', new THREE.BufferAttribute(new Float32Array(VERTS), 1));
  geometry.setAttribute('aGrad', new THREE.BufferAttribute(new Float32Array(VERTS * 2), 2));
  basePos = new Float32Array(VERTS * 2);
  edgeW = new Float32Array(VERTS);
  const margin = EDGE_MARGIN_FRAC * Math.min(pw, ph);
  const p = geometry.attributes.position.array;
  for (let i = 0; i < VERTS; i++) {
    const x = p[i * 3], y = p[i * 3 + 1];
    basePos[i * 2] = x;
    basePos[i * 2 + 1] = y;
    const d = Math.min(x + pw / 2, pw / 2 - x, y + ph / 2, ph / 2 - y);
    const t = Math.min(Math.max(d / margin, 0), 1);
    edgeW[i] = t * t * (3 - 2 * t); // smoothstep: pinned at borders
  }
  mesh.geometry = geometry;
  gridMap = {
    x0: basePos[0],
    y0: basePos[1],
    stepX: basePos[2] - basePos[0],
    stepY: basePos[COLS * 2 + 1] - basePos[1],
  };
  meshDirty = true;
  commitAll = true;
  if (old) old.dispose();
}
addEventListener('resize', resize);
resize();

// ---------- textures / sources ----------
const camVideo = document.createElement('video');
camVideo.playsInline = true; camVideo.muted = true;
const fileVideo = document.createElement('video');
fileVideo.playsInline = true; fileVideo.muted = true;
// hidden video whose motion drives the trackers instead of the webcam
const driveVideo = document.createElement('video');
driveVideo.playsInline = true; driveVideo.muted = true; driveVideo.loop = true;

// surface decode failures (e.g. ProRes/HEVC in a .mov that the browser can't play)
// .mov is a container: H.264 and (on a Mac, in Chrome / Safari) HEVC play; ProRes plays only in Safari
function unplayable(name) {
  const mov = /\.mov$/i.test(name || '');
  return `can't play ${name || 'this file'} in this browser — ` +
    (mov ? 'probably ProRes or HEVC: try Safari, or export it as H.264 mp4' : 'codec unsupported: try an H.264 mp4');
}
fileVideo.addEventListener('error', () => {
  if (!fileVideo.getAttribute('src')) return;
  mediaError = unplayable(playlist[plIndex]?.name);
  setStatus();
});
driveVideo.addEventListener('error', () =>
  setStatus('cannot play driving video — codec unsupported (try H.264 mp4)'));

// The webcam behaves like a mirror; a driving video is watched as-is.
function trackingIsMirrored() { return activeTrackVideo() === camVideo && state.facing === 'user'; }
function activeTrackVideo() {
  return (state.trackingSource === 'drive' && state.driveReady) ? driveVideo : camVideo;
}
function trackSourceReady() {
  return activeTrackVideo() === driveVideo ? state.driveReady : state.camReady;
}

function loadDrivingVideo(file) {
  if (driveVideo.src) URL.revokeObjectURL(driveVideo.src);
  driveVideo.src = URL.createObjectURL(file);
  driveVideo.play().then(() => {
    state.driveReady = true;
    state.trackingSource = 'drive';
    document.getElementById('trackInput').value = 'drive';
    resetFaceTracking();
    setStatus(`driving video: ${file.name}`);
  }).catch(err => setStatus(`driving video failed: ${err.name}`));
}

let activeTexture = null;
let fallbackCanvas = null, fallbackCtx = null;

function setTexture(tex, { mirror }) {
  if (activeTexture) activeTexture.dispose();
  activeTexture = tex;
  textureMirror = mirror;
}

function useWebcamSource() {
  stopPlaylistPlayback();
  if (!state.camReady) { useFallbackPattern(); renderPlaylistUI(); setStatus(); return; }
  texAspect = camVideo.videoWidth / camVideo.videoHeight;
  setTexture(new THREE.VideoTexture(camVideo), { mirror: state.facing === 'user' });
  resize();
  renderPlaylistUI();
  setStatus();
}

function useFallbackPattern() {
  fallbackCanvas = document.createElement('canvas');
  fallbackCanvas.width = 1280; fallbackCanvas.height = 720;
  fallbackCtx = fallbackCanvas.getContext('2d');
  texAspect = 16 / 9;
  setTexture(new THREE.CanvasTexture(fallbackCanvas), { mirror: false });
  resize();
}

function drawFallback(t) {
  const c = fallbackCtx, W = 1280, H = 720;
  const g = c.createLinearGradient(0, 0, W, H);
  g.addColorStop(0, '#222326');
  g.addColorStop(1, '#121314');
  c.fillStyle = g; c.fillRect(0, 0, W, H);
  const lg = c.createLinearGradient(0, H, W, 0); // violet → green, like the icon
  lg.addColorStop(0, '#b36bff'); lg.addColorStop(1, '#39ff88');
  c.strokeStyle = lg; c.globalAlpha = 0.6 + 0.15 * Math.sin(t * 0.002); c.lineWidth = 2;
  for (let x = 0; x <= W; x += 80) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, H); c.stroke(); }
  for (let y = 0; y <= H; y += 80) { c.beginPath(); c.moveTo(0, y); c.lineTo(W, y); c.stroke(); }
  c.globalAlpha = 1;
  c.fillStyle = 'rgba(228,228,225,.85)';
  for (let x = 80; x < W; x += 160)
    for (let y = 80; y < H; y += 160) { c.beginPath(); c.arc(x, y, 14, 0, 7); c.fill(); }
  c.font = '500 28px "IBM Plex Mono", ui-monospace, monospace'; c.textAlign = 'center';
  c.letterSpacing = '2px';
  c.fillStyle = '#e4e4e1';
  c.fillText(IS_PHONE ? 'no camera · drag with a finger' : 'no camera · drag with the mouse', W / 2, H / 2 + 10);
  c.letterSpacing = '0px';
  if (activeTexture) activeTexture.needsUpdate = true;
}

// what to tell the person when the camera can't start, by getUserMedia error name
const CAM_HELP = {
  NotAllowedError: 'camera blocked — allow it in the address bar (and on a Mac: System Settings → Privacy → Camera), then press webcam',
  SecurityError: 'camera blocked — the page must be opened over https',
  NotReadableError: 'camera busy — close other apps or tabs using it, then press webcam',
  AbortError: 'camera busy — close other apps or tabs using it, then press webcam',
  NotFoundError: 'no camera found — plug one in, then press webcam',
  OverconstrainedError: 'camera rejected the requested size — press webcam to retry',
  // (facing switches go through the same path)
};
let camStarting = false;
let camError = '';   // shown in the status screen until the camera starts
let mediaError = ''; // why the current playlist item can't play
async function startWebcam() {
  if (camStarting) return;
  camStarting = true;
  setStatus('asking for the camera…');
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw Object.assign(new Error('no getUserMedia'), { name: 'SecurityError' });
    camVideo.srcObject?.getTracks().forEach(t => t.stop());
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: state.facing }, audio: false,
      });
    } catch (err) {
      // some cameras / browsers refuse the size hints: retry with whatever the camera offers
      if (err.name !== 'OverconstrainedError' && err.name !== 'NotReadableError') throw err;
      stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: state.facing }, audio: false });
    }
    camVideo.srcObject = stream;
    await camVideo.play();
    if (!camVideo.videoWidth) await new Promise(r => camVideo.addEventListener('loadedmetadata', r, { once: true }));
    state.camReady = true;
    camError = '';
    if (!playlistActive) useWebcamSource();
    setStatus();
  } catch (err) {
    console.warn('webcam unavailable:', err);
    if (!playlistActive) useFallbackPattern();
    camError = `${CAM_HELP[err.name] || `no webcam (${err.name || err})`} · ${IS_PHONE ? 'finger' : 'mouse'} drag still works`;
    setStatus();
  } finally {
    camStarting = false;
  }
}

// ---------- playlist ----------
const playlist = [];  // {name, url, isImage}
let plIndex = 0;
let plMode = 'all';       // 'single' | 'all' | 'pingpong' | 'random'
let plLoop = true;        // loop current video (single) / whole playlist (all)
let pingReverse = false;  // ping-pong is currently playing backwards
let fixingDuration = false; // true while probing a webm with Infinity duration
let playlistActive = false;
let imageTimer = null;

const VIDEO_EXT = /\.(mov|mp4|m4v|webm|mkv|avi|ogv)$/i;
const IMAGE_EXT = /\.(png|jpe?g|gif|webp|bmp|avif)$/i;
// the OS doesn't always report a MIME type (e.g. .mov on Linux) — fall back to extension
function isVideoFile(f) { return f.type.startsWith('video/') || VIDEO_EXT.test(f.name); }
function isImageFile(f) { return f.type.startsWith('image/') || (!isVideoFile(f) && IMAGE_EXT.test(f.name)); }

function addFiles(files) {
  let added = false;
  for (const f of files) {
    if (!isVideoFile(f) && !isImageFile(f)) { setStatus(`skipped ${f.name} — not a video or image`); continue; }
    playlist.push({ name: f.name, url: URL.createObjectURL(f), isImage: isImageFile(f) });
    added = true;
  }
  if (added && !playlistActive) playItem(playlist.length > 0 ? 0 : plIndex);
  renderPlaylistUI();
}

function stopPlaylistPlayback() {
  playlistActive = false;
  stopReverseScrub();
  fileVideo.pause();
  clearTimeout(imageTimer);
  imageTimer = null;
}

function playItem(i) {
  if (!playlist.length) return;
  plIndex = ((i % playlist.length) + playlist.length) % playlist.length;
  playlistActive = true;
  stopReverseScrub();
  clearTimeout(imageTimer);
  const item = playlist[plIndex];
  mediaError = '';
  fallbackCtx = null;

  if (item.isImage) {
    fileVideo.pause();
    const img = new Image();
    img.onload = () => {
      if (!playlistActive || playlist[plIndex] !== item) return;
      texAspect = img.naturalWidth / img.naturalHeight;
      const tex = new THREE.Texture(img);
      tex.needsUpdate = true;
      setTexture(tex, { mirror: false });
      resize();
    };
    img.src = item.url;
    // stills "end" after a fixed hold, but only in modes that move on
    if (plMode === 'all' || plMode === 'random')
      imageTimer = setTimeout(handleMediaEnded, IMAGE_SECONDS * 1000);
  } else {
    const start = () => {
      fileVideo.play().then(() => {
        if (!playlistActive || playlist[plIndex] !== item) return;
        texAspect = fileVideo.videoWidth / fileVideo.videoHeight;
        setTexture(new THREE.VideoTexture(fileVideo), { mirror: false });
        resize();
      }).catch(err => {
        console.warn('video play failed:', err);
        if (err.name !== 'AbortError') { mediaError = unplayable(item.name); setStatus(); }
      });
    };
    if (fileVideo.src !== item.url) {
      fileVideo.src = item.url;
      fileVideo.addEventListener('loadedmetadata', function onMeta() {
        fileVideo.removeEventListener('loadedmetadata', onMeta);
        if (!isFinite(fileVideo.duration)) {
          // MediaRecorder webms report Infinity until forced to the end once;
          // without a real duration, seeking (ping-pong) and 'ended' are unreliable
          fixingDuration = true;
          fileVideo.addEventListener('seeked', function onSeeked() {
            fileVideo.removeEventListener('seeked', onSeeked);
            fileVideo.currentTime = 0;
            fixingDuration = false;
            start();
          });
          fileVideo.currentTime = 1e7;
        } else start();
      });
    } else {
      fileVideo.currentTime = 0;
      start();
    }
  }
  renderPlaylistUI();
  setStatus();
}

fileVideo.addEventListener('ended', handleMediaEnded);

// Reverse playback for ping-pong: negative playbackRate isn't supported, so we
// chain backwards seeks — each new seek is issued as soon as the previous lands.
let reverseHandler = null;
function startReverseScrub() {
  pingReverse = true;
  const FRAME = 1 / 30;
  let lastIssue = performance.now();
  reverseHandler = () => {
    if (!pingReverse) { stopReverseScrub(); return; }
    const target = fileVideo.currentTime - FRAME;
    if (target <= 0.02) {
      stopReverseScrub();
      fileVideo.currentTime = 0;
      fileVideo.play();
      return;
    }
    // pace to real time: don't issue the next seek before this frame's wall-clock slot
    const wait = Math.max(0, FRAME * 1000 - (performance.now() - lastIssue));
    setTimeout(() => {
      if (!pingReverse) return;
      lastIssue = performance.now();
      fileVideo.currentTime = target;
    }, wait);
  };
  fileVideo.addEventListener('seeked', reverseHandler);
  fileVideo.currentTime = Math.max(0, (fileVideo.duration || 0) - FRAME);
}
function stopReverseScrub() {
  pingReverse = false;
  if (reverseHandler) { fileVideo.removeEventListener('seeked', reverseHandler); reverseHandler = null; }
}

function handleMediaEnded() {
  if (!playlistActive || !playlist.length || fixingDuration) return;
  // the duration probe can queue a spurious 'ended' that lands after the flag
  // clears — only act when the video is genuinely sitting at its end
  const cur = playlist[plIndex];
  if (cur && !cur.isImage && !fileVideo.ended) return;
  const n = playlist.length;
  switch (plMode) {
    case 'single':
      if (plLoop) playItem(plIndex);
      break; // no loop: hold the last frame
    case 'all':
      if (plIndex < n - 1) playItem(plIndex + 1);
      else if (plLoop) playItem(0);
      break; // no loop: stop after the last item
    case 'pingpong':
      // reverse-scrub back to the start, then play forward again;
      // unseekable files fall back to restarting from the beginning
      if (playlist[plIndex].isImage) break;
      if (isFinite(fileVideo.duration)) startReverseScrub();
      else playItem(plIndex);
      break;
    case 'random': {
      let next = plIndex;
      if (n > 1) while (next === plIndex) next = Math.floor(Math.random() * n);
      playItem(next);
      break;
    }
  }
}

function removeItem(i) {
  const [item] = playlist.splice(i, 1);
  if (item) URL.revokeObjectURL(item.url);
  if (!playlist.length) {
    useWebcamSource(); // stops playlist playback and falls back to test pattern if no cam
    return;
  }
  if (playlistActive) {
    if (i < plIndex) plIndex--;
    else if (i === plIndex) { playItem(Math.min(plIndex, playlist.length - 1)); return; }
  }
  renderPlaylistUI();
}

function reorderItem(from, to) {
  if (from === to) return;
  const current = playlist[plIndex];
  const [moved] = playlist.splice(from, 1);
  playlist.splice(to, 0, moved);
  plIndex = Math.max(0, playlist.indexOf(current));
  renderPlaylistUI();
}

const playlistEl = document.getElementById('playlist');
let dragIdx = null;

function renderPlaylistUI() {
  playlistEl.innerHTML = '';
  playlist.forEach((item, i) => {
    const div = document.createElement('div');
    div.className = 'item' + (playlistActive && i === plIndex ? ' current' : '');
    div.title = item.name;
    div.draggable = true;
    div.addEventListener('dragstart', e => {
      dragIdx = i;
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', String(i));
    });
    div.addEventListener('dragover', e => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      div.classList.add('dragover');
    });
    div.addEventListener('dragleave', () => div.classList.remove('dragover'));
    div.addEventListener('drop', e => {
      e.preventDefault();
      e.stopPropagation(); // don't let the page-level file-drop handler see this
      div.classList.remove('dragover');
      if (dragIdx !== null) reorderItem(dragIdx, i);
      dragIdx = null;
    });
    div.addEventListener('dragend', () => { dragIdx = null; });
    const name = document.createElement('span');
    name.className = 'name';
    name.textContent = `${i + 1}. ${item.name}`;
    name.addEventListener('click', () => playItem(i));
    const rm = document.createElement('button');
    rm.className = 'rm';
    rm.textContent = '×';
    rm.title = 'Remove from playlist';
    rm.addEventListener('click', ev => { ev.stopPropagation(); removeItem(i); });
    div.append(name, rm);
    playlistEl.appendChild(div);
  });
}

// ---------- mediapipe ----------
let handLandmarker = null, faceLandmarker = null, poseLandmarker = null;
let visionFileset = null;
let lastHands = null, lastFace = null, lastPose = null;
let lastPresence = performance.now(); // last time a person (or the mouse) was seen

const wantHandsFor = m => m === 'hands' || m === 'both' || m === 'all';
const wantFaceFor = m => m === 'face' || m === 'both' || m === 'all';
const wantPoseFor = m => m === 'body' || m === 'all';

async function initTrackers() {
  try {
    visionFileset = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
    await ensureTrackersForMode();
  } catch (err) {
    console.warn('tracker init failed:', err);
    setStatus('tracking unavailable — mouse drag still works');
  }
}

// GPU first; some phones / browsers can't run the GPU delegate, so retry on the CPU
async function createTask(Task, options) {
  try { return await Task.createFromOptions(visionFileset, options); }
  catch (err) {
    console.warn('GPU delegate failed, using CPU:', err);
    return Task.createFromOptions(visionFileset, { ...options, baseOptions: { ...options.baseOptions, delegate: 'CPU' } });
  }
}

async function ensureTrackersForMode() {
  if (!visionFileset) return;
  const wantHands = wantHandsFor(state.mode);
  const wantFace = wantFaceFor(state.mode);
  const wantPose = wantPoseFor(state.mode);
  if (wantPose && !poseLandmarker) {
    poseLandmarker = await createTask(PoseLandmarker, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numPoses: 1,
    });
    state.poseReady = true;
    setStatus();
  }
  if (wantHands && !handLandmarker) {
    handLandmarker = await createTask(HandLandmarker, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numHands: 2,
    });
    state.handsReady = true;
  }
  if (wantFace && !faceLandmarker) {
    faceLandmarker = await createTask(FaceLandmarker, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU',
      },
      runningMode: 'VIDEO',
      numFaces: 1,
    });
    state.faceReady = true;
  }
  setStatus();
}

// ---------- warp engine ----------
function brushRadius() { return state.brushFrac * Math.min(innerWidth, innerHeight); }

// Displace vertices near (px,py). Edge vertices are pinned. Only the grid patch
// the brush can reach is visited (margin covers already-displaced vertices).
function applyBrush(px, py, dx, dy, radius, out, mode = state.effect) {
  const s2 = 2 * radius * radius;
  const cutoff = 9 * radius * radius;
  const m = Math.hypot(dx, dy);
  const effect = mode;
  const theta0 = dx * 0.012; // twist angle driven by horizontal drag

  const gm = gridMap;
  const reach = 3 * radius + 140;
  let ca = Math.floor((px - reach - gm.x0) / gm.stepX);
  let cb = Math.ceil((px + reach - gm.x0) / gm.stepX);
  if (ca > cb) { const t = ca; ca = cb; cb = t; }
  const ta = (py - reach - gm.y0) / gm.stepY, tb = (py + reach - gm.y0) / gm.stepY;
  let ra = Math.floor(Math.min(ta, tb));
  let rb = Math.ceil(Math.max(ta, tb));
  ca = Math.max(0, ca); cb = Math.min(SEG_X, cb);
  ra = Math.max(0, ra); rb = Math.min(SEG_Y, rb);
  if (ca > cb || ra > rb) return;

  for (let r = ra; r <= rb; r++) {
  const rowBase = r * COLS;
  for (let c = ca; c <= cb; c++) {
    const i = rowBase + c;
    const cx = basePos[i * 2] + persistOff[i * 2];
    const cy = basePos[i * 2 + 1] + persistOff[i * 2 + 1];
    const rx = cx - px, ry = cy - py;
    const d2 = rx * rx + ry * ry;
    if (d2 > cutoff) continue;
    const w = Math.exp(-d2 / s2) * edgeW[i];
    if (w < 1e-4) continue;

    let ox = 0, oy = 0;
    if (effect === 'pull') {
      ox = dx * w; oy = dy * w;
    } else if (effect === 'inflate') {
      const d = Math.sqrt(d2) + 1e-6;
      ox = (rx / d) * m * w; oy = (ry / d) * m * w;
    } else if (effect === 'twist') {
      const th = theta0 * w;
      const c = Math.cos(th), s = Math.sin(th);
      ox = (rx * c - ry * s) - rx;
      oy = (rx * s + ry * c) - ry;
    } else if (effect === 'ripple') {
      const d = Math.sqrt(d2) + 1e-6;
      const ring = Math.cos(d / (radius * 0.3));
      ox = (rx / d) * m * ring * w; oy = (ry / d) * m * ring * w;
    }
    out[i * 2] += ox;
    out[i * 2 + 1] += oy;
  }
  }
  expandBox(ca, cb, ra, rb);
  meshDirty = true;
  hasWarp = true;
  brushActivity += m;
}

// Per-frame warp-field dynamics, windowed to the warped region (grows 1 cell/step).
// clay:   Jacobi diffusion — creases melt into smooth blobs, shape persists
// jelly:  spring-mass — deformation wobbles and bounces back to flat
// liquid: wave equation — every touch sends ripples propagating outward
const relaxTmp = new Float32Array(VERTS * 2);
let dynFrame = 0;
function stepDynamics() {
  if (!warpBox) return;
  dynFrame++;
  const mat = state.material;
  const s = state.smoothing;
  if (mat === 'clay' && (s === 0 || (dynFrame & 1))) { checkEnergy(); return; } // clay relaxes at 30Hz
  expandBox(warpBox.c0 - 1, warpBox.c1 + 1, warpBox.r0 - 1, warpBox.r1 + 1);
  const { c0, c1, r0, r1 } = warpBox;
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      const i = r * COLS + c;
      const l = c > 0 ? i - 1 : i, rt = c < SEG_X ? i + 1 : i;
      const u = r > 0 ? i - COLS : i, d = r < SEG_Y ? i + COLS : i;
      relaxTmp[i * 2] = (persistOff[l * 2] + persistOff[rt * 2] + persistOff[u * 2] + persistOff[d * 2]) * 0.25;
      relaxTmp[i * 2 + 1] = (persistOff[l * 2 + 1] + persistOff[rt * 2 + 1] + persistOff[u * 2 + 1] + persistOff[d * 2 + 1]) * 0.25;
    }
  }
  const clayK = s * 0.44;
  const jellyStiff = 0.02 + s * 0.05, jellyDamp = 1 - (0.04 + s * 0.06);
  const liqSpeed = 0.30, liqDamp = 1 - (0.004 + s * 0.03);
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      const i = r * COLS + c;
      const ix = i * 2, iy = ix + 1;
      if (edgeW[i] < 1e-3) { persistOff[ix] = 0; persistOff[iy] = 0; velOff[ix] = 0; velOff[iy] = 0; continue; }
      const lapx = relaxTmp[ix] - persistOff[ix];
      const lapy = relaxTmp[iy] - persistOff[iy];
      if (mat === 'clay') {
        persistOff[ix] += lapx * clayK;
        persistOff[iy] += lapy * clayK;
      } else if (mat === 'jelly') {
        velOff[ix] = (velOff[ix] + lapx * 0.35 - persistOff[ix] * jellyStiff) * jellyDamp;
        velOff[iy] = (velOff[iy] + lapy * 0.35 - persistOff[iy] * jellyStiff) * jellyDamp;
        persistOff[ix] += velOff[ix];
        persistOff[iy] += velOff[iy];
      } else { // liquid
        velOff[ix] = (velOff[ix] + lapx * liqSpeed - persistOff[ix] * 0.001) * liqDamp;
        velOff[iy] = (velOff[iy] + lapy * liqSpeed - persistOff[iy] * 0.001) * liqDamp;
        persistOff[ix] += velOff[ix];
        persistOff[iy] += velOff[iy];
      }
    }
  }
  meshDirty = true;
  checkEnergy();
}

// every ~half second, see whether the warp has fully settled and can be freed
function checkEnergy() {
  if ((dynFrame & 31) !== 0 || !warpBox) return;
  const { c0, c1, r0, r1 } = warpBox;
  let m = 0;
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      const i = r * COLS + c;
      m = Math.max(m, Math.abs(persistOff[i * 2]), Math.abs(persistOff[i * 2 + 1]),
                   Math.abs(velOff[i * 2]), Math.abs(velOff[i * 2 + 1]));
    }
  }
  if (m < 0.3) resetWarp();
}

// palm-heal brush: multiplicatively irons the warp (and its velocity) back to flat
function healBrush(px, py, radius) {
  if (!warpBox || !gridMap) return;
  const gm = gridMap;
  const reach = 3 * radius + 140;
  let ca = Math.max(0, Math.floor((px - reach - gm.x0) / gm.stepX));
  let cb = Math.min(SEG_X, Math.ceil((px + reach - gm.x0) / gm.stepX));
  const ta = (py - reach - gm.y0) / gm.stepY, tb = (py + reach - gm.y0) / gm.stepY;
  let ra = Math.max(0, Math.floor(Math.min(ta, tb)));
  let rb = Math.min(SEG_Y, Math.ceil(Math.max(ta, tb)));
  if (ca > cb || ra > rb) return;
  const s2 = 2 * radius * radius;
  for (let r = ra; r <= rb; r++) {
    for (let c = ca; c <= cb; c++) {
      const i = r * COLS + c;
      const cx = basePos[i * 2] + persistOff[i * 2];
      const cy = basePos[i * 2 + 1] + persistOff[i * 2 + 1];
      const d2 = (cx - px) ** 2 + (cy - py) ** 2;
      if (d2 > 9 * radius * radius) continue;
      const f = 1 - Math.exp(-d2 / s2) * 0.10;
      persistOff[i * 2] *= f; persistOff[i * 2 + 1] *= f;
      velOff[i * 2] *= f; velOff[i * 2 + 1] *= f;
    }
  }
  meshDirty = true;
}

function commitVertices() {
  if (!meshDirty) return; // nothing changed: skip the loop and the GPU re-upload
  meshDirty = false;
  const p = geometry.attributes.position;
  const disp = geometry.attributes.aDisp;
  const dispScale = 1 / (0.25 * Math.min(innerWidth, innerHeight));
  let c0 = 0, c1 = SEG_X, r0 = 0, r1 = SEG_Y;
  if (!commitAll && warpBox) ({ c0, c1, r0, r1 } = warpBox);
  commitAll = false;
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      const i = r * COLS + c;
      const ox = persistOff[i * 2];
      const oy = persistOff[i * 2 + 1];
      p.array[i * 3] = basePos[i * 2] + ox;
      p.array[i * 3 + 1] = basePos[i * 2 + 1] + oy;
      disp.array[i] = Math.min(Math.hypot(ox, oy) * dispScale, 1);
    }
  }
  // displacement gradient for the relief-light shader (central differences)
  const grad = geometry.attributes.aGrad;
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      const i = r * COLS + c;
      const l = c > 0 ? i - 1 : i, rt = c < SEG_X ? i + 1 : i;
      const u = r > 0 ? i - COLS : i, d = r < SEG_Y ? i + COLS : i;
      grad.array[i * 2] = disp.array[rt] - disp.array[l];
      grad.array[i * 2 + 1] = disp.array[u] - disp.array[d];
    }
  }
  p.needsUpdate = true;
  disp.needsUpdate = true;
  grad.needsUpdate = true;
}

function resetWarp() {
  persistOff.fill(0);
  velOff.fill(0);
  warpBox = null;
  meshDirty = true;
  commitAll = true;
  hasWarp = false;
}

// auto-reset: ease back to flat once idle for resetDelaySec
let lastInteraction = performance.now();
function maybeAutoReset(now) {
  if (!state.autoReset || !hasWarp || !warpBox) return;
  if (now - lastInteraction < state.resetDelaySec * 1000) return;
  let maxAbs = 0;
  const { c0, c1, r0, r1 } = warpBox;
  for (let r = r0; r <= r1; r++) {
    for (let c = c0; c <= c1; c++) {
      const i = r * COLS + c;
      persistOff[i * 2] *= 0.94;
      persistOff[i * 2 + 1] *= 0.94;
      const a = Math.max(Math.abs(persistOff[i * 2]), Math.abs(persistOff[i * 2 + 1]));
      if (a > maxAbs) maxAbs = a;
    }
  }
  meshDirty = true;
  if (maxAbs < 0.4) resetWarp();
}

// landmark [0..1] camera coords -> world px (webcam is mirrored, driving video is not)
function lmToWorld(lx, ly) {
  const sx = trackingIsMirrored() ? 1 - lx : lx;
  return {
    x: (sx - 0.5) * innerWidth,
    y: (0.5 - ly) * innerHeight,
  };
}

// ---------- pointers (hands + mouse share one model) ----------
const pointers = {}; // id -> {x, y, px, py, pinched, lastSeen}
const POINTER_TIMEOUT_MS = 300;

function updatePointer(id, x, y, pinched, now) {
  let pt = pointers[id];
  if (!pt) pt = pointers[id] = { x, y, px: x, py: y, pinched: false };
  pt.px = pt.x; pt.py = pt.y;
  pt.x = x; pt.y = y;
  if (pinched && !pt.pinched) { pt.px = x; pt.py = y; } // fresh grab: no jump
  pt.pinched = pinched;
  pt.lastSeen = now;

  const c = getCursor(id);
  c.visible = state.showCursors;
  c.position.set(x, y, 1);
  c.material.color.set(pinched ? 0x39ff88 : 0xe4e4e1);
  c.scale.setScalar(pinched ? 1.4 : 1);

  if (pt.pinched) {
    lastInteraction = now;
    const dx = pt.x - pt.px, dy = pt.y - pt.py;
    if (dx || dy) applyBrush(pt.x, pt.y, dx, dy, brushRadius(), persistOff);
  }
}

function expirePointers(now) {
  for (const id of Object.keys(pointers)) {
    const pt = pointers[id];
    if (now - pt.lastSeen > POINTER_TIMEOUT_MS) {
      pt.pinched = false;
      getCursor(id).visible = false;
    }
  }
}

// Pinch: EXACTLY the v1 metric — thumb-tip to index-tip screen distance
// normalized by palm length, 0.35 grab / 0.48 release, no filtering —
// with state.pinchScale as the only addition (Pinch slider multiplier).
const PINCH_ON_BASE = 0.35, PINCH_HYST = 0.13;
const pinchState = [false, false];

function processHands(result, now) {
  lastHands = result;
  const n = result.landmarks.length;
  if (n > 0) lastPresence = now;
  for (let i = 0; i < n && i < 2; i++) {
    const lm = result.landmarks[i];
    const thumb = lm[4], index = lm[8];
    const handSize = Math.hypot(lm[0].x - lm[9].x, lm[0].y - lm[9].y) || 1e-6;
    const d = Math.hypot(thumb.x - index.x, thumb.y - index.y) / handSize;
    const on = PINCH_ON_BASE * state.pinchScale;
    const off = on + PINCH_HYST * state.pinchScale;
    if (!pinchState[i] && d < on) pinchState[i] = true;
    else if (pinchState[i] && d > off) pinchState[i] = false;

    const mid = lmToWorld((thumb.x + index.x) / 2, (thumb.y + index.y) / 2);
    updatePointer('hand' + i, mid.x, mid.y, pinchState[i], now);

    // open flat palm irons the warp back out — strict gate so pinch attempts
    // never trigger it: EVERY finger must be extended (a curled index while
    // preparing a pinch cancels it) and thumb-index must be clearly apart
    if (state.palmHeal && !pinchState[i] && hasWarp && d > off * 1.4) {
      let minExt = Infinity;
      for (const tip of [8, 12, 16, 20]) {
        const ext = Math.hypot(lm[tip].x - lm[0].x, lm[tip].y - lm[0].y) / handSize;
        if (ext < minExt) minExt = ext;
      }
      if (minExt > 1.5) {
        const palm = lmToWorld(lm[9].x, lm[9].y);
        healBrush(palm.x, palm.y, brushRadius() * 1.6);
        lastInteraction = now;
        const c = getCursor('hand' + i);
        c.material.color.set(0x66aaff); // blue = healing
        c.scale.setScalar(1.8);
      }
    }
  }
}

// body points push the canvas like slow heavy fingers — walking past leaves wakes
const POSE_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
let posePts = null;

function processPose(result, now) {
  lastPose = result;
  const lms = result.landmarks?.[0];
  if (!lms) { posePts = null; return; }
  lastPresence = now;

  const raw = POSE_POINTS.map(idx => lmToWorld(lms[idx].x, lms[idx].y));
  if (!posePts) { posePts = raw.map(p => ({ ...p })); return; }

  const prev = posePts.map(p => ({ ...p }));
  for (let k = 0; k < raw.length; k++) {
    posePts[k].x += (raw[k].x - posePts[k].x) * 0.5;
    posePts[k].y += (raw[k].y - posePts[k].y) * 0.5;
  }
  const shL = lmToWorld(lms[11].x, lms[11].y), shR = lmToWorld(lms[12].x, lms[12].y);
  const radius = Math.max(40, Math.hypot(shR.x - shL.x, shR.y - shL.y) * 0.7);

  for (let k = 0; k < posePts.length; k++) {
    let dx = (posePts[k].x - prev[k].x) * 1.5;
    let dy = (posePts[k].y - prev[k].y) * 1.5;
    const mag = Math.hypot(dx, dy);
    if (mag < 0.6) continue;
    if (mag > 30) { dx *= 30 / mag; dy *= 30 / mag; }
    lastInteraction = now;
    applyBrush(posePts[k].x, posePts[k].y, dx, dy, radius, persistOff);
  }
}

// ---------- face morph ----------
// Each face landmark acts like a permanently-pinched finger: the canvas region
// under it follows it frame to frame, persistently — like hand pinching, but
// with its own kernel (default Inflate → bulbous). Landmarks are low-pass
// filtered so tracking jitter can't etch sharp creases into the mesh, and
// rigid head translation is subtracted so individual features do the morphing.
let smoothPts = null;
const FACE_EMA = 0.45;   // landmark low-pass factor (lower = smoother, laggier)
const FACE_MAX_STEP = 25; // px cap per frame — one bad detection can't spike the mesh

function resetFaceTracking() { smoothPts = null; }

function processFace(result, now) {
  lastFace = result;
  const lms = result.faceLandmarks?.[0];
  if (!lms) { smoothPts = null; return; }
  lastPresence = now;

  const raw = FACE_POINTS.map(idx => lmToWorld(lms[idx].x, lms[idx].y));
  if (!smoothPts) {
    smoothPts = raw.map(p => ({ ...p }));
    return;
  }

  const prev = smoothPts.map(p => ({ ...p }));
  let mdx = 0, mdy = 0;
  for (let k = 0; k < raw.length; k++) {
    smoothPts[k].x += (raw[k].x - smoothPts[k].x) * FACE_EMA;
    smoothPts[k].y += (raw[k].y - smoothPts[k].y) * FACE_EMA;
    mdx += smoothPts[k].x - prev[k].x;
    mdy += smoothPts[k].y - prev[k].y;
  }
  mdx /= raw.length; mdy /= raw.length; // mean delta ≈ rigid head translation

  // brush size follows the face's size on screen — big soft brushes blend into
  // one continuous organic surface instead of separate sharp pokes
  const eyeL = lmToWorld(lms[EYE_L].x, lms[EYE_L].y);
  const eyeR = lmToWorld(lms[EYE_R].x, lms[EYE_R].y);
  const radius = Math.max(30, Math.hypot(eyeR.x - eyeL.x, eyeR.y - eyeL.y) * 0.7);

  for (let k = 0; k < smoothPts.length; k++) {
    let dx = (smoothPts[k].x - prev[k].x - mdx) * state.faceAmp;
    let dy = (smoothPts[k].y - prev[k].y - mdy) * state.faceAmp;
    const mag = Math.hypot(dx, dy);
    if (mag < 0.35) continue; // ignore residual jitter
    if (mag > FACE_MAX_STEP) { dx *= FACE_MAX_STEP / mag; dy *= FACE_MAX_STEP / mag; }
    lastInteraction = now;
    applyBrush(smoothPts[k].x, smoothPts[k].y, dx, dy, radius, persistOff, state.faceEffect);
  }

  // mouth blow: opening wide inflates the canvas at the mouth's position
  if (state.mouthBlow) {
    const up = lmToWorld(lms[13].x, lms[13].y);
    const lo = lmToWorld(lms[14].x, lms[14].y);
    const eyeDist = Math.hypot(eyeR.x - eyeL.x, eyeR.y - eyeL.y);
    const openRatio = Math.hypot(up.x - lo.x, up.y - lo.y) / Math.max(eyeDist, 1);
    if (openRatio > 0.30) {
      const strength = Math.min((openRatio - 0.30) * eyeDist * 0.25, 8);
      applyBrush((up.x + lo.x) / 2, (up.y + lo.y) / 2, strength, 0, eyeDist * 0.9, persistOff, 'inflate');
      lastInteraction = now;
    }
  }
}

// ---------- camera preview overlay ----------
const previewCanvas = document.getElementById('camPreview');
const previewCtx = previewCanvas.getContext('2d');

function drawPreview() {
  const src = activeTrackVideo();
  if (!state.showPreview || !trackSourceReady()) {
    previewCanvas.classList.add('hidden');
    return;
  }
  previewCanvas.classList.remove('hidden');
  const mirrored = trackingIsMirrored();
  const px = x => (mirrored ? 1 - x : x);
  const w = 240, h = Math.round(240 / (src.videoWidth / src.videoHeight || 16 / 9));
  if (previewCanvas.width !== w || previewCanvas.height !== h) {
    previewCanvas.width = w; previewCanvas.height = h;
  }
  const c = previewCtx;
  c.save();
  if (mirrored) { c.scale(-1, 1); c.drawImage(src, -w, 0, w, h); }
  else c.drawImage(src, 0, 0, w, h);
  c.restore();

  const wantHands = wantHandsFor(state.mode);
  const wantFace = wantFaceFor(state.mode);

  if (wantHands && lastHands) {
    lastHands.landmarks.forEach((lm, hi) => {
      const pinched = pinchState[hi];
      c.fillStyle = 'rgba(255,255,255,.8)';
      for (const p of lm) {
        c.beginPath(); c.arc(px(p.x) * w, p.y * h, 1.5, 0, 7); c.fill();
      }
      const t = lm[4], ix = lm[8];
      c.strokeStyle = c.fillStyle = pinched ? '#39ff88' : '#b36bff';
      c.lineWidth = 2;
      c.beginPath();
      c.moveTo(px(t.x) * w, t.y * h);
      c.lineTo(px(ix.x) * w, ix.y * h);
      c.stroke();
      for (const p of [t, ix]) { c.beginPath(); c.arc(px(p.x) * w, p.y * h, 3.5, 0, 7); c.fill(); }
    });
  }
  if (wantFace && lastFace?.faceLandmarks?.[0]) {
    const lms = lastFace.faceLandmarks[0];
    c.fillStyle = '#e4e4e1';
    for (const idx of FACE_POINTS) {
      const p = lms[idx];
      c.beginPath(); c.arc(px(p.x) * w, p.y * h, 3, 0, 7); c.fill();
    }
  }
  if (wantPoseFor(state.mode) && lastPose?.landmarks?.[0]) {
    const lms = lastPose.landmarks[0];
    c.fillStyle = '#39ff88';
    for (const idx of POSE_POINTS) {
      const p = lms[idx];
      c.beginPath(); c.arc(px(p.x) * w, p.y * h, 3, 0, 7); c.fill();
    }
  }
}

// ---------- projector window ----------
let projWin = null, projStream = null;

function openProjector() {
  if (projWin && !projWin.closed) { projWin.focus(); return; }
  projWin = window.open('', 'trackmorph-projector', 'width=960,height=540');
  if (!projWin) { setStatus('projector popup blocked — allow popups for this page'); return; }
  const doc = projWin.document;
  doc.title = 'track-morph — projector';
  doc.body.style.cssText = 'margin:0;background:#000;overflow:hidden;';
  const v = doc.createElement('video');
  v.muted = true; v.autoplay = true; v.playsInline = true;
  v.style.cssText = 'width:100vw;height:100vh;object-fit:contain;cursor:none;';
  if (!projStream) projStream = renderer.domElement.captureStream(60);
  v.srcObject = projStream;
  doc.body.appendChild(v);
  const hint = doc.createElement('div');
  hint.textContent = 'drag this window to the projector display, then double-click for fullscreen';
  hint.style.cssText = 'position:fixed;top:14px;left:50%;transform:translateX(-50%);color:#6f7074;font:10px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.04em;transition:opacity 1s;';
  doc.body.appendChild(hint);
  setTimeout(() => { hint.style.opacity = '0'; }, 6000);
  v.addEventListener('dblclick', () => doc.documentElement.requestFullscreen());
}

// ---------- mouse / touch fallback ----------
// every finger (or the mouse) is its own grab, so two fingers can pull two places at once
const downPointers = new Set();
const pointerKey = e => e.pointerType === 'mouse' ? 'mouse' : 'touch' + e.pointerId;
const cv = renderer.domElement;
cv.style.touchAction = 'none'; // no scrolling, pinch-zoom or pull-to-refresh while sculpting
cv.addEventListener('pointerdown', e => {
  downPointers.add(e.pointerId);
  cv.setPointerCapture?.(e.pointerId);
  feedPointer(e);
});
cv.addEventListener('pointermove', e => { if (downPointers.has(e.pointerId) || e.pointerType === 'mouse') feedPointer(e); });
const releasePointer = e => {
  if (!downPointers.delete(e.pointerId)) return;
  feedPointer(e);
  if (e.pointerType !== 'mouse') getCursor(pointerKey(e)).visible = false;
};
cv.addEventListener('pointerup', releasePointer);
cv.addEventListener('pointercancel', releasePointer);
function feedPointer(e) {
  const x = e.clientX - innerWidth / 2;
  const y = innerHeight / 2 - e.clientY;
  const down = downPointers.has(e.pointerId);
  const id = pointerKey(e);
  lastPresence = performance.now();
  updatePointer(id, x, y, down, performance.now());
  getCursor(id).visible = down && state.showCursors;
}

// ---------- souvenir clip recording ----------
let recording = false;
let pendingClip = null; // phones: a finished clip waiting for a tap to share
function shareClip() {
  const file = pendingClip;
  pendingClip = null;
  document.querySelectorAll('.recLabel').forEach(el => { el.textContent = el.dataset.label; });
  navigator.share({ files: [file] })
    .then(() => setStatus('clip shared'))
    .catch(err => setStatus(err.name === 'AbortError' ? 'clip discarded' : `share failed: ${err.name}`));
}
function recordClip(seconds = 10) {
  if (pendingClip) { shareClip(); return; }
  if (recording) return;
  recording = true;
  $('recordBtn').classList.add('on'); $('ledRec').classList.add('on');
  const stream = renderer.domElement.captureStream(30);
  const mimeType = ['video/webm;codecs=vp9', 'video/webm', 'video/mp4;codecs=avc1', 'video/mp4']
    .find(t => window.MediaRecorder?.isTypeSupported?.(t));
  if (!mimeType) {
    recording = false;
    $('recordBtn').classList.remove('on'); $('ledRec').classList.remove('on');
    setStatus("this browser can't record the canvas");
    return;
  }
  const ext = mimeType.startsWith('video/mp4') ? 'mp4' : 'webm';
  const rec = new MediaRecorder(stream, { mimeType });
  const chunks = [];
  rec.ondataavailable = e => chunks.push(e.data);
  rec.onstop = () => {
    recording = false;
    $('recordBtn').classList.remove('on'); $('ledRec').classList.remove('on');
    const blob = new Blob(chunks, { type: mimeType.split(';')[0] });
    const name = `pinch-${new Date().toISOString().replace(/[:.]/g, '-')}.${ext}`;
    // phones: hand the clip to the share sheet (save to Photos, send…); elsewhere download it
    const file = new File([blob], name, { type: blob.type });
    // the share sheet needs a fresh tap, so the record keys turn into "save clip" until then
    if (IS_PHONE && navigator.canShare?.({ files: [file] })) {
      pendingClip = file;
      document.querySelectorAll('.recLabel').forEach(el => { el.textContent = 'save clip'; });
      setStatus('clip ready · tap save clip');
      return;
    }
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
    setStatus('clip saved to downloads');
  };
  rec.start();
  let left = seconds;
  setStatus(`recording… ${left}s`);
  const cd = setInterval(() => {
    left--;
    if (left <= 0) { clearInterval(cd); rec.stop(); }
    else setStatus(`recording… ${left}s`);
  }, 1000);
}

// ---------- presets: save/load scene setups, auto-cycle for shows ----------
const PRESET_FIELDS = ['mode', 'effect', 'faceEffect', 'material', 'smoothing', 'shader',
  'amt', 'brushFrac', 'pinchScale', 'faceAmp', 'autoReset', 'resetDelaySec',
  'mouthBlow', 'palmHeal'];
const PRESET_COUNT = 6;
let cycleOn = false, lastCycle = 0, cycleIdx = 0;

function savePreset(n) {
  const data = {};
  for (const k of PRESET_FIELDS) data[k] = state[k];
  localStorage.setItem('tm_preset_' + n, JSON.stringify(data));
  renderPresetButtons();
  setStatus(`preset ${n + 1} saved`);
}

function loadPreset(n) {
  const raw = localStorage.getItem('tm_preset_' + n);
  if (!raw) return;
  const data = JSON.parse(raw);
  for (const k of PRESET_FIELDS) if (k in data) state[k] = data[k];
  setShader(state.shader);
  ensureTrackersForMode();
  syncUI();
  setStatus(`preset ${n + 1} loaded`);
}

function syncUI() {
  const $ = id => document.getElementById(id);
  $('modeSel').value = state.mode;
  $('effectSel').value = state.effect;
  $('faceEffectSel').value = state.faceEffect;
  $('materialSel').value = state.material;
  $('smoothing').value = state.smoothing;
  $('shaderSel').value = state.shader;
  $('amtSlider').value = state.amt;
  $('brushSize').value = state.brushFrac;
  $('pinchDist').value = state.pinchScale;
  $('pinchDistVal').textContent = `×${state.pinchScale.toFixed(2)}`;
  $('faceAmp').value = state.faceAmp;
  $('autoReset').checked = state.autoReset;
  $('resetDelay').value = state.resetDelaySec;
  $('resetDelayVal').textContent = `${state.resetDelaySec}s`;
  $('mouthBlow').checked = state.mouthBlow;
  $('palmHeal').checked = state.palmHeal;
}

function renderPresetButtons() {
  const row = document.getElementById('presetRow');
  row.innerHTML = '';
  for (let n = 0; n < PRESET_COUNT; n++) {
    const b = document.createElement('button');
    b.textContent = String(n + 1);
    b.className = 'preset' + (localStorage.getItem('tm_preset_' + n) ? ' saved' : '');
    b.title = 'click: load · shift+click: save';
    b.addEventListener('click', e => e.shiftKey ? savePreset(n) : loadPreset(n));
    row.appendChild(b);
  }
}

function maybeCyclePresets(now) {
  if (!cycleOn || now - lastCycle < 20000) return;
  lastCycle = now;
  const saved = [];
  for (let n = 0; n < PRESET_COUNT; n++) if (localStorage.getItem('tm_preset_' + n)) saved.push(n);
  if (saved.length < 2) return;
  cycleIdx = (cycleIdx + 1) % saved.length;
  loadPreset(saved[cycleIdx]);
}

// ---------- UI ----------
const statusEl = document.getElementById('status');
const ledCam = document.getElementById('ledCam'), ledTrack = document.getElementById('ledTrack');
function setStatus(extra) {
  const lines = [];
  lines.push(`cam ${state.camReady ? 'live' : 'off'} · hands ${state.handsReady ? 'on' : '—'} · face ${state.faceReady ? 'on' : '—'} · body ${state.poseReady ? 'on' : '—'}`);
  if (attractActive) lines.push('attract mode active — ghost performers');
  if (state.trackingSource === 'drive' && state.driveReady) lines.push('tracking: driving video');
  if (!state.camReady && camError) lines.push(camError);
  if (playlistActive && playlist.length)
    lines.push(`playing ${plIndex + 1}/${playlist.length}: ${playlist[plIndex].name}`);
  if (playlistActive && mediaError) lines.push(mediaError);
  if (extra) lines.push(extra);
  statusEl.textContent = lines.join('\n');
  ledCam.classList.toggle('on', state.camReady || (state.trackingSource === 'drive' && state.driveReady));
  ledTrack.classList.toggle('on', state.handsReady || state.faceReady || state.poseReady);
}

const $ = id => document.getElementById(id);

$('modeSel').addEventListener('change', e => {
  state.mode = e.target.value;
  if (state.mode !== 'hands') resetFaceTracking();
  ensureTrackersForMode();
});
$('brushSize').addEventListener('input', e => { state.brushFrac = +e.target.value; });
$('pinchDist').addEventListener('input', e => {
  state.pinchScale = +e.target.value;
  $('pinchDistVal').textContent = `×${state.pinchScale.toFixed(2)}`;
});
$('faceAmp').addEventListener('input', e => { state.faceAmp = +e.target.value; });
$('effectSel').addEventListener('change', e => { state.effect = e.target.value; });
$('faceEffectSel').addEventListener('change', e => { state.faceEffect = e.target.value; });
$('materialSel').addEventListener('change', e => {
  state.material = e.target.value;
  velOff.fill(0); // fresh dynamics when switching physics
});
$('smoothing').addEventListener('input', e => { state.smoothing = +e.target.value; });
$('mouthBlow').addEventListener('change', e => { state.mouthBlow = e.target.checked; });
$('palmHeal').addEventListener('change', e => { state.palmHeal = e.target.checked; });
$('micReact').addEventListener('change', e => {
  state.micReact = e.target.checked;
  if (state.micReact && !analyser) enableMic();
});
$('micGain').addEventListener('input', e => { state.micGain = +e.target.value; });
$('soundOut').addEventListener('change', e => {
  state.soundOut = e.target.checked;
  if (state.soundOut) enableSound();
});
$('attract').addEventListener('change', e => { state.attract = e.target.checked; });
$('recordBtn').addEventListener('click', () => recordClip(10));
$('presetCycle').addEventListener('change', e => { cycleOn = e.target.checked; });
renderPresetButtons();

// populate shader dropdown grouped by category
{
  const sel = $('shaderSel');
  const cats = [...new Set(EFFECTS.map(e => e.cat))];
  for (const cat of cats) {
    const group = document.createElement('optgroup');
    group.label = cat;
    for (const e of EFFECTS.filter(x => x.cat === cat)) {
      const opt = document.createElement('option');
      opt.value = e.id;
      opt.textContent = e.label;
      group.appendChild(opt);
    }
    sel.appendChild(group);
  }
  sel.value = state.shader;
  sel.addEventListener('change', e => setShader(e.target.value));
}
$('amtSlider').addEventListener('input', e => { state.amt = +e.target.value; });
$('autoReset').addEventListener('change', e => { state.autoReset = e.target.checked; lastInteraction = performance.now(); });
$('resetDelay').addEventListener('input', e => {
  state.resetDelaySec = +e.target.value;
  $('resetDelayVal').textContent = `${state.resetDelaySec}s`;
});
$('showPreview').addEventListener('change', e => { state.showPreview = e.target.checked; });
// ---------- phone controls ----------
async function flipCamera() {
  state.facing = state.facing === 'user' ? 'environment' : 'user';
  state.camReady = false;
  await startWebcam();
  resetFaceTracking();
}
$('flipBtn').addEventListener('click', flipCamera);
$('qFlip').addEventListener('click', flipCamera);
$('qReset').addEventListener('click', () => resetWarp());
$('qRec').addEventListener('click', () => recordClip(10));
$('qMore').addEventListener('click', () => {
  const open = $('ui').classList.toggle('open');
  $('qMore').classList.toggle('on', open);
});
// H / "hide ui": the controls, the top bar and the hint go; the artwork stays
function toggleUi() {
  for (const id of ['ui', 'hint', 'topbar']) $(id).classList.toggle('hidden');
}
$('tbOpen').addEventListener('click', () => $('fileInput').click());
$('tbReset').addEventListener('click', () => resetWarp());
$('tbHide').addEventListener('click', toggleUi);
$('tbFull').addEventListener('click', () => document.documentElement.requestFullscreen?.());
if (!IS_PHONE) {
  // computers: the title and the status lights live in the top bar
  const card = document.querySelector('#ui > header.panel');
  $('topbar').querySelector('.l').append(...card.children);
  document.body.classList.add('topbar');
}
if (IS_PHONE) {
  document.body.classList.add('phone');
  $('renderScale').value = 0.75;
  queueMicrotask(() => $('renderScale').dispatchEvent(new Event('input')));
}

$('renderScale').addEventListener('input', e => {
  renderer.setPixelRatio(BASE_DPR * +e.target.value);
  renderer.setSize(innerWidth, innerHeight);
});
$('showCursors').addEventListener('change', e => {
  state.showCursors = e.target.checked;
  if (!state.showCursors) Object.values(cursors).forEach(c => { c.visible = false; });
});
$('resetBtn').addEventListener('click', resetWarp);
$('projectorBtn').addEventListener('click', openProjector);
$('webcamBtn').addEventListener('click', () => { if (state.camReady) useWebcamSource(); else { stopPlaylistPlayback(); startWebcam(); } });
$('playlistMode').addEventListener('change', e => {
  plMode = e.target.value;
  stopReverseScrub();
  // if the current media already finished (or is a held image), kick off the new mode
  if (playlistActive && playlist.length) {
    const cur = playlist[plIndex];
    if (cur?.isImage) playItem(plIndex);
    else if (fileVideo.ended) handleMediaEnded();
  }
});
$('plLoop').addEventListener('change', e => {
  plLoop = e.target.checked;
  if (plLoop && playlistActive && fileVideo.ended && !pingReverse) handleMediaEnded();
});

// tracking input: webcam vs driving video
$('trackInput').addEventListener('change', e => {
  if (e.target.value === 'drive' && !state.driveReady) {
    e.target.value = 'webcam';
    $('driveInput').click(); // pick a file first; loading switches automatically
    return;
  }
  state.trackingSource = e.target.value;
  if (e.target.value === 'webcam') driveVideo.pause();
  else driveVideo.play();
  resetFaceTracking();
  setStatus();
});
$('loadDriveBtn').addEventListener('click', () => $('driveInput').click());
$('driveInput').addEventListener('change', () => {
  if ($('driveInput').files[0]) loadDrivingVideo($('driveInput').files[0]);
  $('driveInput').value = '';
});

const fileInput = $('fileInput');
$('loadVideoBtn').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => { addFiles(fileInput.files); fileInput.value = ''; });
addEventListener('dragover', e => e.preventDefault());
addEventListener('drop', e => {
  e.preventDefault();
  if (e.dataTransfer.files?.length) addFiles(e.dataTransfer.files);
});

addEventListener('keydown', e => {
  if (e.key === 'r' || e.key === 'R') resetWarp();
  if (e.key === 'f' || e.key === 'F') document.documentElement.requestFullscreen?.();
  if (e.key === 's' || e.key === 'S') recordClip(10);
  if (e.key === 'h' || e.key === 'H') toggleUi();
  const digit = parseInt(e.key, 10);
  if (digit >= 1 && digit <= PRESET_COUNT) {
    e.shiftKey ? savePreset(digit - 1) : loadPreset(digit - 1);
  }
  // shift+digit produces symbols on most layouts; map those too
  const shiftMap = { '!': 0, '@': 1, '#': 2, '$': 3, '%': 4, '^': 5 };
  if (e.key in shiftMap) savePreset(shiftMap[e.key]);
});

// ---------- attract mode: ghost performers when the space is empty ----------
const ATTRACT_IDLE_MS = 45000;
let ghostPrev = [null, null];
let attractActive = false;

function runAttract(now) {
  attractActive = true;
  const t = now * 0.001;
  for (let g = 0; g < 2; g++) {
    const px = Math.sin(t * (0.21 + g * 0.07) + g * 2.3) * innerWidth * 0.30;
    const py = Math.sin(t * (0.16 + g * 0.05) + g * 4.1) * innerHeight * 0.28;
    const prev = ghostPrev[g];
    if (prev) {
      applyBrush(px, py, (px - prev.x) * 0.6, (py - prev.y) * 0.6,
        brushRadius() * 1.2, persistOff, g ? 'inflate' : 'pull');
    }
    ghostPrev[g] = { x: px, y: py };
    if (state.showCursors) {
      const c = getCursor('ghost' + g);
      c.visible = true; c.position.set(px, py, 1);
      c.material.color.set(0xb36bff);
    }
  }
}
function stopAttract() {
  if (!attractActive) return;
  attractActive = false;
  ghostPrev = [null, null];
  getCursor('ghost0').visible = false;
  getCursor('ghost1').visible = false;
}

// ---------- rolling frame history for the time-smear shader ----------
const HIST_N = 16, HIST_COLS = 4, HIST_W = 320, HIST_H = 180;
let histCanvas = null, histCtx = null, histTex = null, histHead = 0, histLast = 0;

function updateHistory(now) {
  if (state.shader !== 'timewarp' || !activeTexture?.image) return;
  if (now - histLast < 80) return; // ~12.5 fps of history ≈ 1.3 s reach
  histLast = now;
  if (!histCanvas) {
    histCanvas = document.createElement('canvas');
    histCanvas.width = HIST_W * HIST_COLS;
    histCanvas.height = HIST_H * (HIST_N / HIST_COLS);
    histCtx = histCanvas.getContext('2d');
    histTex = new THREE.CanvasTexture(histCanvas);
    histTex.flipY = false;
    histTex.generateMipmaps = false;
    histTex.minFilter = THREE.LinearFilter;
  }
  histHead = (histHead + 1) % HIST_N;
  const cx = (histHead % HIST_COLS) * HIST_W;
  const cy = Math.floor(histHead / HIST_COLS) * HIST_H;
  try { histCtx.drawImage(activeTexture.image, cx, cy, HIST_W, HIST_H); } catch { /* source not ready */ }
  histTex.needsUpdate = true;
}

// ---------- feedback texture for the trails shader ----------
let prevFrameTex = null, prevFrameSize = { w: 0, h: 0 };
const copyOrigin = new THREE.Vector2(0, 0);

function updateTrails() {
  if (state.shader !== 'trails') return;
  const size = renderer.getDrawingBufferSize(new THREE.Vector2());
  if (!prevFrameTex || prevFrameSize.w !== size.x || prevFrameSize.h !== size.y) {
    prevFrameTex?.dispose();
    prevFrameTex = new THREE.FramebufferTexture(size.x, size.y);
    prevFrameSize = { w: size.x, h: size.y };
  }
  renderer.copyFramebufferToTexture(copyOrigin, prevFrameTex);
}

// ---------- audio: mic in (reactivity) and synth out (the canvas groans) ----------
let audioCtx = null, analyser = null, micBuf = null, micLevel = 0;
let groanOsc = null, groanOsc2 = null, groanGain = null, groanFilter = null;
let activityEma = 0;

async function enableMic() {
  try {
    audioCtx ??= new AudioContext();
    await audioCtx.resume();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const src = audioCtx.createMediaStreamSource(stream);
    analyser = audioCtx.createAnalyser();
    analyser.fftSize = 512;
    micBuf = new Uint8Array(analyser.fftSize);
    src.connect(analyser);
    setStatus('mic live');
  } catch (err) {
    state.micReact = false;
    const el = document.getElementById('micReact'); if (el) el.checked = false;
    setStatus(`mic unavailable (${err.name})`);
  }
}

function enableSound() {
  audioCtx ??= new AudioContext();
  audioCtx.resume();
  if (groanOsc) return;
  groanGain = audioCtx.createGain(); groanGain.gain.value = 0;
  groanFilter = audioCtx.createBiquadFilter();
  groanFilter.type = 'lowpass'; groanFilter.frequency.value = 220; groanFilter.Q.value = 6;
  groanOsc = audioCtx.createOscillator(); groanOsc.type = 'sawtooth'; groanOsc.frequency.value = 55;
  groanOsc2 = audioCtx.createOscillator(); groanOsc2.type = 'sawtooth'; groanOsc2.frequency.value = 55.7;
  groanOsc.connect(groanFilter); groanOsc2.connect(groanFilter);
  groanFilter.connect(groanGain); groanGain.connect(audioCtx.destination);
  groanOsc.start(); groanOsc2.start();
}

function updateAudio(now) {
  if (analyser && state.micReact) {
    analyser.getByteTimeDomainData(micBuf);
    let sum = 0;
    for (let i = 0; i < micBuf.length; i++) { const v = (micBuf[i] - 128) / 128; sum += v * v; }
    micLevel += (Math.sqrt(sum / micBuf.length) - micLevel) * 0.2;
  } else micLevel = 0;

  activityEma += (brushActivity - activityEma) * 0.15;
  brushActivity = 0;
  if (groanGain && state.soundOut) {
    const t = audioCtx.currentTime;
    const g = Math.min(0.22, activityEma * 0.004);
    groanGain.gain.setTargetAtTime(g, t, 0.08);
    const f = 45 + Math.min(120, activityEma * 1.5) + Math.sin(now * 0.0021) * 8;
    groanOsc.frequency.setTargetAtTime(f, t, 0.1);
    groanOsc2.frequency.setTargetAtTime(f * 1.013, t, 0.1);
    groanFilter.frequency.setTargetAtTime(140 + activityEma * 4, t, 0.1);
  } else if (groanGain) {
    groanGain.gain.setTargetAtTime(0, audioCtx.currentTime, 0.05);
  }
}

// ---------- main loop ----------
let lastVideoTime = -1;
let frameCount = 0;
let detIdx = 0;
let detEma = 0;       // rolling average of detection cost in ms
let detCounter = 0;

function tick(now) {
  requestAnimationFrame(tick);
  frameCount++;

  if (fallbackCtx) drawFallback(now);

  const trackVideo = activeTrackVideo();
  if (trackSourceReady() && trackVideo.readyState >= 2 && trackVideo.currentTime !== lastVideoTime) {
    lastVideoTime = trackVideo.currentTime;
    try {
      // hands run EVERY camera frame — pinch responsiveness is the product,
      // so it is never throttled or made to share slots with other detectors
      if (wantHandsFor(state.mode) && handLandmarker)
        processHands(handLandmarker.detectForVideo(trackVideo, now), now);

      // face/body are heavier and less latency-critical: they alternate with
      // each other and back off when detection is slow on this machine
      const wantF = wantFaceFor(state.mode) && faceLandmarker;
      const wantP = wantPoseFor(state.mode) && poseLandmarker;
      if (wantF || wantP) {
        const skipN = detEma > 24 ? 3 : detEma > 12 ? 2 : 1;
        if (++detCounter % skipN === 0) {
          const list = [];
          if (wantF) list.push('f');
          if (wantP) list.push('p');
          detIdx = (detIdx + 1) % list.length;
          const t0 = performance.now();
          if (list[detIdx] === 'f') processFace(faceLandmarker.detectForVideo(trackVideo, now), now);
          else processPose(poseLandmarker.detectForVideo(trackVideo, now), now);
          detEma = detEma * 0.9 + (performance.now() - t0) * 0.1;
        }
      }
    } catch (err) { console.warn(err); }
  }

  if (state.attract && now - lastPresence > ATTRACT_IDLE_MS) runAttract(now);
  else stopAttract();

  expirePointers(now);
  maybeAutoReset(now);
  if (hasWarp) stepDynamics();
  commitVertices();
  updateHistory(now);
  updateAudio(now);
  maybeCyclePresets(now);

  const u = mesh.material.uniforms;
  u.uMap.value = activeTexture;
  u.uMirror.value = textureMirror ? 1 : 0;
  u.uAmt.value = Math.min(1, state.amt + micLevel * state.micGain * 3);
  u.uTime.value = now * 0.001;
  u.uAspect.value = texAspect;
  u.uHistory.value = histTex;
  u.uHead.value = histHead;
  u.uPrev.value = prevFrameTex;
  renderer.getDrawingBufferSize(u.uResolution.value);
  const img = activeTexture?.image;
  const tw = img?.videoWidth || img?.naturalWidth || img?.width || 1280;
  const th = img?.videoHeight || img?.naturalHeight || img?.height || 720;
  u.uTexel.value.set(1 / tw, 1 / th);

  if ((frameCount & 3) === 0) drawPreview(); // preview PIP at ~15fps is plenty
  renderer.render(scene, camera);
  updateTrails(); // must run right after render, while the frame is still in the buffer
}

startWebcam();
Promise.resolve().then(initTrackers).catch(err => {
  console.warn('trackers failed:', err);
  setStatus(`trackers failed to load (${err.message || err}) · camera and mouse still work`);
});
requestAnimationFrame(tick);

// debug handle for automated smoke tests
window.__trackMorph = {
  persistOff, velOff, state, playlist, playItem, removeItem, setShader, EFFECTS,
  reorderItem, handleMediaEnded, loadDrivingVideo, fileVideo, driveVideo,
  savePreset, loadPreset, recordClip, healBrush, resetWarp,
  get plIndex() { return plIndex; },
  get plMode() { return plMode; },
  get plLoop() { return plLoop; },
  get pingReverse() { return pingReverse; },
  get attractActive() { return attractActive; },
  get hasWarp() { return hasWarp; },
  get warpBox() { return warpBox; },
  get dynFrame() { return dynFrame; },
  setPresence(t) { lastPresence = t; },
  step(n = 1) { for (let k = 0; k < n; k++) stepDynamics(); commitVertices(); },
};
