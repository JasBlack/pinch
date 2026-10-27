# pinch

**Live demo: [pinch.jas.black](https://pinch.jas.black)** — open in Chrome and allow camera access.

A live camera tracks the viewer's hands and face; the tracked points warp a playing
video in real time, like pulling on a rubber canvas.

- **Pinch** (thumb + index) grabs the nearest region of the video canvas and drags
  it until you release. The deformation is persistent — the video keeps playing in
  its morphed state.
- **Face mode** — 16 facial landmarks (brows, eyelids, eye corners, nose, mouth
  corners, lips, chin, cheeks, forehead) each act like a permanently-pinched
  finger: the canvas region under each point follows it as it moves, and the
  deformation persists — exactly like hand pinching. Rigid head movement is
  subtracted, so raising an eyebrow or opening your mouth morphs just that
  region. Brush size scales with how large the face is on screen. No
  calibration needed.
- Mouse click-drag works as a fallback pinch, so you can test without tracking.

## Features

- **Pinned edges** — the borders of the video always stay fixed to the frame
  boundary; the warp fades out in a band near the edges.
- **Warp types** — Pull (drag), Inflate (push outward), Twist (rotate), Ripple
  (concentric rings). Hands/mouse and face have independent kernels ("Warp
  type" vs "Face warp"); the face defaults to Inflate for bulbous, organic forms.
- **Materials** — Clay (holds its shape, slowly softens), Jelly (deformations
  wobble and spring back to flat), Liquid (every touch sends ripples). The
  Smoothing slider tunes settle speed / damping per material.
- **Open-palm heal** — a flat open hand irons the warp back out (pinch sculpts,
  palm erases). Toggleable.
- **Mouth blow** — opening your mouth wide inflates the canvas at its position.
- **Body mode** — MediaPipe Pose: shoulders, elbows, wrists, hips, knees and
  ankles each push the canvas, so people walking past leave wakes.
  "Everything" runs hands + face + body, round-robined across frames.
- **Time & light shaders** — *Time smear* (warped regions sample a rolling
  1.3 s frame history: pulling drags that area into the past), *Feedback
  trails* (motion leaves luminous ghosts), *Relief light* (bulges are lit and
  shaded like physical relief).
- **Audio** — *Mic reactive* (room loudness modulates the shader amount) and
  *Canvas sound* (a synth groans as the mesh is stretched).
- **Attract mode** — after 45 s with nobody tracked, ghost performers gently
  play with the canvas; stops the instant a person (or the mouse) appears.
- **Record 10s clip** (button or `S`) — saves a .webm of the artwork.
- **Presets** — 6 slots. Click loads, shift+click saves (also keys 1–6 /
  shift+1–6). Auto-cycle rotates through saved presets every 20 s.
- **Shader library** — 60+ fragment shader effects in `shaders.js`, selectable
  from the Shader dropdown with an Amount slider, grouped into: Glitch & damage
  (VHS, block mosh, TV static, row shred…), Color (duotones, thermal, hue spin,
  solarize…), Distortion (kaleidoscope, swirl, fisheye, underwater…), Stylize
  (halftone, LED wall, neon edges, crosshatch, blueprint…), Motion & blur, and
  Warp-reactive (effects that respond to how far the mesh is deformed — heat,
  glow, fracture, x-ray…). Each is a small GLSL snippet; add your own by
  appending to the `EFFECTS` list.
- **Auto reset** — optional timer (1–120 s): after that long with no interaction,
  the canvas eases back to flat. Leave it off and the morph is stuck forever.
- **Projector output** — "Open projector window" opens a second window mirroring
  the artwork (via `canvas.captureStream`); drag it to the external display and
  double-click for fullscreen. Turn off "Points" so cursor rings don't project.
- **Camera preview + points** — toggleable picture-in-picture of the camera with
  hand/face landmarks drawn, and toggleable on-canvas cursor rings.
- **Driving video** — the trackers can watch a video file instead of the webcam:
  Tracking → Input → Driving video (or "Load driving video…"). Hands/faces in
  that footage then morph the canvas, so the piece can perform without a live
  camera. The webcam is treated as a mirror; a driving video is read as-is.
- **Playlist** — load multiple videos/images (button or drag-drop). Drag items
  to reorder. Click an item to jump to it, or its × to remove it (removing the
  playing item advances; emptying the list falls back to the webcam). Modes:
  - **Play single** — plays the current file once and holds the last frame.
  - **Play all** — plays each file once, in order, then stops.
  - **Ping-pong** — plays the clip forward to the end, then backwards to the
    start (real reverse playback via paced scrubbing), then forward again,
    endlessly — no visible seam.
  - **Random** — picks a random other file when the current one finishes.
  - **Loop** checkbox — with Play single, restarts the same file endlessly;
    with Play all, wraps from the last file back to the first.
  Images hold for 10 s in Play all / Random. "Webcam" switches the display
  source back to the live camera.

## Stack

- [MediaPipe Tasks Vision](https://developers.google.com/mediapipe) — `HandLandmarker`
  (21 points/hand, 2 hands) and `FaceLandmarker` (468 points), GPU-delegated, in-browser.
- [Three.js](https://threejs.org) — the video is a `VideoTexture` on a densely
  subdivided plane (128×72 segments); warping displaces vertices with a gaussian
  brush. Everything loads from CDN; no build step.

## Run

```sh
python3 -m http.server 8123
# then open http://localhost:8123 in Chrome and allow camera access
```

(Any static server works. A server is needed because the page is an ES module;
camera access requires localhost or https.)

## Controls

| Action | Control |
|---|---|
| Grab & drag canvas | pinch thumb+index, or mouse click-drag |
| Switch tracking mode | Mode dropdown: Hands / Face / Hands + face / Body / Everything |
| Load videos/images | "Load files…" button, or drop files on the page |
| Reset warp | `R` or Reset button |
| Hide/show the UI | `H` |
| Fullscreen | `F` |

## Performance

Renders at a capped device-pixel-ratio (1.5) with antialiasing off — fragment
shaders are fill-rate bound and this is invisible on a fullscreen video. The
mesh only re-uploads to the GPU on frames where the warp actually changed, the
camera preview redraws at ~15 fps, and in "Both" tracking mode the hand and
face detectors alternate frames. If a machine still struggles (heavy shaders on
a big projector resolution), lower the **Render scale** slider in the View
section — 0.75 is usually indistinguishable.

## Tuning ideas

- `SEG_X/SEG_Y` in `main.js` — mesh density
- `PINCH_ON/PINCH_OFF` — pinch sensitivity (hysteresis)
- `FACE_POINTS` — which facial landmarks act as warp handles
- Brush falloff in `applyBrush()` — swap the gaussian for other kernels
  (bulge, twist, shear) for different pull characters

## License

[MIT](LICENSE)
