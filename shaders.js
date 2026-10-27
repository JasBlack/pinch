// Shader effect library. Each effect defines `vec4 effect(vec2 uv)` and has access to:
//   uMap (video), uAmt (Amount slider 0..1), uTime (seconds), uAspect (texture w/h),
//   uTexel (1/resolution), vDisp (0..1 local mesh deformation), tex(uv), hash, luma,
//   pal (cosine palette), rot, hueShift.

export const VERTEX = /* glsl */`
attribute float aDisp;
attribute vec2 aGrad;
varying vec2 vUv;
varying float vDisp;
varying vec2 vGrad;
void main() {
  vUv = uv;
  vDisp = aDisp;
  vGrad = aGrad;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}`;

const HEAD = /* glsl */`
precision highp float;
uniform sampler2D uMap;
uniform sampler2D uHistory;
uniform sampler2D uPrev;
uniform float uMirror, uAmt, uTime, uAspect, uHead;
uniform vec2 uTexel, uResolution;
varying vec2 vUv;
varying float vDisp;
varying vec2 vGrad;

vec2 muv(vec2 uv) { return vec2(mix(uv.x, 1.0 - uv.x, uMirror), uv.y); }
vec4 tex(vec2 uv) { return texture2D(uMap, muv(clamp(uv, 0.001, 0.999))); }
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
vec3 pal(float t, float phase) { return 0.5 + 0.5 * cos(6.28318 * (t + phase + vec3(0.0, 0.33, 0.67))); }
vec2 rot(vec2 p, float a) { float c = cos(a), s = sin(a); return vec2(p.x*c - p.y*s, p.x*s + p.y*c); }
vec3 hueShift(vec3 c, float a) {
  const mat3 toYIQ = mat3(0.299, 0.596, 0.211, 0.587, -0.274, -0.523, 0.114, -0.322, 0.312);
  const mat3 toRGB = mat3(1.0, 1.0, 1.0, 0.956, -0.272, -1.106, 0.621, -0.647, 1.703);
  vec3 yiq = toYIQ * c;
  float h = atan(yiq.z, yiq.y) + a;
  float l = length(yiq.yz);
  yiq.yz = vec2(cos(h), sin(h)) * l;
  return clamp(toRGB * yiq, 0.0, 1.0);
}
`;

const MAIN = /* glsl */`
void main() { gl_FragColor = vec4(effect(vUv).rgb, 1.0); }
`;

export function buildFragment(body) { return HEAD + body + MAIN; }

const fx = (id, label, cat, body) => ({ id, label, cat, body });

export const EFFECTS = [
  fx('none', 'None (clean)', 'Clean', `
    vec4 effect(vec2 uv) { return tex(uv); }`),

  // ---------- glitch & damage ----------
  fx('rgbshift', 'RGB shift', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      vec2 o = vec2(0.02 * uAmt * (0.3 + vDisp), 0.0);
      return vec4(tex(uv + o).r, tex(uv).g, tex(uv - o).b, 1.0);
    }`),
  fx('scanglitch', 'Scanline glitch', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      float band = floor(uv.y * 48.0);
      float n = fract(sin(band * 91.7 + floor(uTime * 9.0) * 13.1) * 43758.5453);
      uv.x += (n - 0.5) * 0.15 * uAmt * (0.2 + vDisp);
      vec2 s = vec2(0.02, 0.0) * uAmt * (0.2 + vDisp);
      return vec4(tex(uv + s).r, tex(uv).g, tex(uv - s).b, 1.0);
    }`),
  fx('blockmosh', 'Block mosh', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      vec2 g = vec2(14.0, 10.0);
      vec2 b = floor(uv * g);
      float r = hash(b + floor(uTime * 5.0));
      if (r > 1.0 - 0.35 * uAmt) {
        uv += (vec2(hash(b + 1.3), hash(b + 2.7)) - 0.5) * 0.25 * uAmt;
        vec2 s = vec2(0.03 * uAmt, 0.0);
        return vec4(tex(uv + s).r, tex(uv).g, tex(uv - s).b, 1.0);
      }
      return tex(uv);
    }`),
  fx('rowshred', 'Row shred', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      float row = floor(uv.y * 60.0);
      float act = step(0.6, hash(vec2(row, floor(uTime * 2.0))));
      uv.x += (hash(vec2(row, floor(uTime * 10.0))) - 0.5) * 0.3 * uAmt * act;
      return tex(uv);
    }`),
  fx('vhs', 'VHS', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      float n = hash(uv * vec2(300.0, 200.0) + fract(uTime) * 60.0);
      float band = fract(uv.y - uTime * 0.13);
      float track = smoothstep(0.05, 0.0, abs(band - 0.5) - 0.44);
      uv.x += track * 0.06 * uAmt + (n - 0.5) * 0.004 * uAmt;
      vec3 c = tex(uv).rgb;
      c = mix(c, vec3(luma(c)), 0.3 * uAmt);
      c += (n - 0.5) * 0.15 * uAmt;
      float lines = sin(uv.y * 400.0) * 0.5 + 0.5;
      c *= mix(1.0, 0.85 + 0.15 * lines, uAmt);
      return vec4(c, 1.0);
    }`),
  fx('interlace', 'Interlace tear', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      float odd = step(0.5, fract(uv.y * 134.0));
      uv.x += (odd - 0.5) * 0.04 * uAmt * (0.3 + vDisp);
      return tex(uv);
    }`),
  fx('staticmix', 'TV static', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      vec3 n = vec3(hash(uv * 431.0 + fract(uTime) * 17.0),
                    hash(uv * 397.0 - fract(uTime) * 13.0),
                    hash(uv * 355.0 + fract(uTime) * 7.0));
      return vec4(mix(tex(uv).rgb, n, uAmt * (0.25 + 0.75 * vDisp)), 1.0);
    }`),
  fx('grain', 'Film grain', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      float g = hash(uv * 800.0 + fract(uTime) * 43.0) - 0.5;
      vec3 c = tex(uv).rgb + g * 0.35 * uAmt;
      c *= 1.0 + 0.06 * uAmt * sin(uTime * 24.0);
      return vec4(c, 1.0);
    }`),
  fx('chromabend', 'Chroma bleed', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      vec3 c;
      c.g = tex(uv).g;
      c.r = tex(uv + vec2(0.0, 0.015 * uAmt)).r;
      c.b = tex(uv - vec2(0.0, 0.022 * uAmt)).b;
      return vec4(c, 1.0);
    }`),
  fx('ghost', 'Ghost trails', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec2 o = vec2(0.03, 0.015) * uAmt;
      c = max(c, tex(uv - o).rgb * 0.7);
      c = max(c, tex(uv - 2.0 * o).rgb * 0.45);
      return vec4(c, 1.0);
    }`),
  fx('anaglyph', 'Anaglyph 3D', 'Glitch & damage', `
    vec4 effect(vec2 uv) {
      vec2 o = vec2(0.02 * uAmt, 0.0);
      return vec4(tex(uv - o).r, tex(uv + o).g, tex(uv + o).b, 1.0);
    }`),

  // ---------- color ----------
  fx('invert', 'Invert', 'Color', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      return vec4(mix(c, 1.0 - c, uAmt), 1.0);
    }`),
  fx('invertwarp', 'Invert where warped', 'Color', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      return vec4(mix(c, 1.0 - c, clamp(vDisp * 2.0, 0.0, 1.0) * uAmt), 1.0);
    }`),
  fx('posterize', 'Posterize', 'Color', `
    vec4 effect(vec2 uv) {
      float n = mix(12.0, 3.0, uAmt);
      vec3 c = floor(tex(uv).rgb * n) / n;
      return vec4(c, 1.0);
    }`),
  fx('threshold', 'Threshold B/W', 'Color', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec3 bw = vec3(smoothstep(0.45, 0.55, luma(c)));
      return vec4(mix(c, bw, uAmt), 1.0);
    }`),
  fx('duotone', 'Duotone (ember)', 'Color', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec3 d = mix(vec3(0.08, 0.0, 0.25), vec3(1.0, 0.45, 0.1), luma(c));
      return vec4(mix(c, d, uAmt), 1.0);
    }`),
  fx('neonduo', 'Duotone (neon)', 'Color', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec3 d = mix(vec3(1.0, 0.0, 0.55), vec3(0.0, 0.95, 1.0), luma(c));
      return vec4(mix(c, d, uAmt), 1.0);
    }`),
  fx('huespin', 'Hue spin', 'Color', `
    vec4 effect(vec2 uv) {
      return vec4(hueShift(tex(uv).rgb, uTime * uAmt * 2.0), 1.0);
    }`),
  fx('huedisp', 'Hue by warp', 'Color', `
    vec4 effect(vec2 uv) {
      return vec4(hueShift(tex(uv).rgb, vDisp * uAmt * 6.28), 1.0);
    }`),
  fx('vaporwave', 'Vaporwave', 'Color', `
    vec4 effect(vec2 uv) {
      vec2 o = vec2(0.006 * uAmt, 0.0);
      vec3 c = vec3(tex(uv + o).r, tex(uv).g, tex(uv - o).b);
      c = hueShift(c, 0.9 * uAmt);
      c = mix(c, c * vec3(1.15, 0.85, 1.2) + vec3(0.06, 0.0, 0.12), uAmt * 0.7);
      return vec4(c, 1.0);
    }`),
  fx('thermal', 'Thermal cam', 'Color', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec3 t = pal(luma(c) * 0.7 + 0.02, 0.55);
      return vec4(mix(c, t, uAmt), 1.0);
    }`),
  fx('sepia', 'Sepia', 'Color', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec3 s = vec3(dot(c, vec3(0.393, 0.769, 0.189)),
                    dot(c, vec3(0.349, 0.686, 0.168)),
                    dot(c, vec3(0.272, 0.534, 0.131)));
      return vec4(mix(c, s, uAmt), 1.0);
    }`),
  fx('solarize', 'Solarize', 'Color', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec3 s = mix(c, 1.0 - c, step(0.5, c));
      return vec4(mix(c, s, uAmt), 1.0);
    }`),
  fx('bitcrush', 'Bitcrush dither', 'Color', `
    vec4 effect(vec2 uv) {
      float n = mix(10.0, 3.0, uAmt);
      float d = (hash(floor(uv * vec2(320.0, 180.0))) - 0.5) / n;
      vec3 c = floor((tex(uv).rgb + d) * n) / n;
      return vec4(c, 1.0);
    }`),

  // ---------- distortion ----------
  fx('pixelate', 'Pixelate', 'Distortion', `
    vec4 effect(vec2 uv) {
      float n = mix(300.0, 18.0, uAmt);
      vec2 g = vec2(n * uAspect, n);
      return tex((floor(uv * g) + 0.5) / g);
    }`),
  fx('pixelmelt', 'Pixelate where warped', 'Distortion', `
    vec4 effect(vec2 uv) {
      float k = clamp(vDisp * 2.0 * uAmt, 0.0, 1.0);
      float n = mix(400.0, 22.0, k);
      vec2 g = vec2(n * uAspect, n);
      return tex((floor(uv * g) + 0.5) / g);
    }`),
  fx('waves', 'Waves', 'Distortion', `
    vec4 effect(vec2 uv) {
      uv.x += sin(uv.y * 20.0 + uTime * 2.0) * 0.03 * uAmt;
      uv.y += cos(uv.x * 15.0 + uTime * 1.7) * 0.02 * uAmt;
      return tex(uv);
    }`),
  fx('rings', 'Ripple rings', 'Distortion', `
    vec4 effect(vec2 uv) {
      vec2 p = (uv - 0.5) * vec2(uAspect, 1.0);
      float d = length(p);
      vec2 dir = p / max(d, 1e-4);
      uv += dir * sin(d * 40.0 - uTime * 4.0) * 0.01 * uAmt / vec2(uAspect, 1.0);
      return tex(uv);
    }`),
  fx('swirl', 'Swirl', 'Distortion', `
    vec4 effect(vec2 uv) {
      vec2 p = (uv - 0.5) * vec2(uAspect, 1.0);
      float d = length(p);
      p = rot(p, uAmt * 4.0 * exp(-d * 4.0) + 0.3 * uAmt * sin(uTime * 0.7));
      return tex(p / vec2(uAspect, 1.0) + 0.5);
    }`),
  fx('barrel', 'Barrel bulge', 'Distortion', `
    vec4 effect(vec2 uv) {
      vec2 p = (uv - 0.5) * vec2(uAspect, 1.0);
      p *= 1.0 + 1.2 * uAmt * dot(p, p);
      return tex(p / vec2(uAspect, 1.0) + 0.5);
    }`),
  fx('fisheye', 'Fisheye', 'Distortion', `
    vec4 effect(vec2 uv) {
      vec2 p = (uv - 0.5) * vec2(uAspect, 1.0);
      p *= 1.0 - 0.6 * uAmt * dot(p, p);
      return tex(p / vec2(uAspect, 1.0) + 0.5);
    }`),
  fx('kaleido', 'Kaleidoscope', 'Distortion', `
    vec4 effect(vec2 uv) {
      vec2 p = (uv - 0.5) * vec2(uAspect, 1.0);
      float seg = 6.28318 / 6.0;
      float a = mod(atan(p.y, p.x) + uTime * 0.1 * uAmt, seg);
      a = abs(a - seg * 0.5);
      vec2 k = vec2(cos(a), sin(a)) * length(p);
      vec2 kuv = k / vec2(uAspect, 1.0) + 0.5;
      return mix(tex(uv), tex(kuv), uAmt);
    }`),
  fx('mirrorx', 'Mirror X', 'Distortion', `
    vec4 effect(vec2 uv) {
      vec2 m = vec2(0.5 - abs(uv.x - 0.5), uv.y);
      return mix(tex(uv), tex(m), uAmt);
    }`),
  fx('quadmirror', 'Quad mirror', 'Distortion', `
    vec4 effect(vec2 uv) {
      vec2 m = 0.5 - abs(uv - 0.5);
      return mix(tex(uv), tex(m), uAmt);
    }`),
  fx('tiles', 'Tile repeat', 'Distortion', `
    vec4 effect(vec2 uv) {
      float n = mix(1.0, 4.0, uAmt);
      return tex(fract(uv * n));
    }`),
  fx('zoompulse', 'Zoom pulse', 'Distortion', `
    vec4 effect(vec2 uv) {
      float s = 1.0 + 0.15 * uAmt * sin(uTime * 3.0);
      return tex((uv - 0.5) / s + 0.5);
    }`),
  fx('spinfx', 'Pendulum spin', 'Distortion', `
    vec4 effect(vec2 uv) {
      vec2 p = (uv - 0.5) * vec2(uAspect, 1.0);
      p = rot(p, 0.3 * uAmt * sin(uTime * 0.8));
      return tex(p / vec2(uAspect, 1.0) + 0.5);
    }`),
  fx('jelly', 'Jelly wobble', 'Distortion', `
    vec4 effect(vec2 uv) {
      uv += vec2(sin(uv.y * 10.0 + uTime * 3.0), cos(uv.x * 12.0 + uTime * 2.4))
            * 0.02 * uAmt * (0.3 + vDisp);
      return tex(uv);
    }`),
  fx('underwater', 'Underwater', 'Distortion', `
    vec4 effect(vec2 uv) {
      uv += 0.012 * uAmt * vec2(sin(uv.y * 25.0 + uTime * 2.0) + sin(uv.y * 41.0 - uTime * 3.3),
                                sin(uv.x * 30.0 + uTime * 2.7));
      vec3 c = tex(uv).rgb * mix(vec3(1.0), vec3(0.6, 0.9, 1.0), uAmt * 0.6);
      float caustic = pow(0.5 + 0.5 * sin(uv.x * 30.0 + uTime * 2.0) * sin(uv.y * 24.0 - uTime * 1.6), 6.0);
      c += caustic * 0.35 * uAmt;
      return vec4(c, 1.0);
    }`),

  // ---------- stylize ----------
  fx('crt', 'CRT scanlines', 'Stylize', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      float line = 0.5 + 0.5 * sin(uv.y * 800.0);
      float m = mod(floor(uv.x * 800.0), 3.0);
      vec3 mask = vec3(step(m, 0.5), step(0.5, m) * step(m, 1.5), step(1.5, m));
      c = mix(c, c * (0.6 + 0.4 * line) * (mask * 1.5 + 0.4), uAmt);
      vec2 p = uv - 0.5;
      c *= 1.0 - uAmt * 0.8 * dot(p, p);
      return vec4(c, 1.0);
    }`),
  fx('halftone', 'Halftone dots', 'Stylize', `
    vec4 effect(vec2 uv) {
      float n = 90.0;
      vec2 g = uv * vec2(n * uAspect, n);
      vec2 cell = (floor(g) + 0.5) / vec2(n * uAspect, n);
      float l = luma(tex(cell).rgb);
      float d = length(fract(g) - 0.5);
      vec3 dots = vec3(1.0 - smoothstep(sqrt(l) * 0.55 - 0.06, sqrt(l) * 0.55, d));
      return vec4(mix(tex(uv).rgb, dots, uAmt), 1.0);
    }`),
  fx('dotmatrix', 'LED wall', 'Stylize', `
    vec4 effect(vec2 uv) {
      float n = 70.0;
      vec2 g = uv * vec2(n * uAspect, n);
      vec2 cell = (floor(g) + 0.5) / vec2(n * uAspect, n);
      vec3 c = tex(cell).rgb;
      float d = length(fract(g) - 0.5);
      vec3 led = c * (1.0 - smoothstep(0.28, 0.42, d)) * 1.4;
      return vec4(mix(tex(uv).rgb, led, uAmt), 1.0);
    }`),
  fx('mosaicsq', 'Luma blocks', 'Stylize', `
    vec4 effect(vec2 uv) {
      float n = 60.0;
      vec2 g = uv * vec2(n * uAspect, n);
      vec2 cell = (floor(g) + 0.5) / vec2(n * uAspect, n);
      float l = luma(tex(cell).rgb);
      vec2 f = abs(fract(g) - 0.5);
      float sq = step(max(f.x, f.y), l * 0.55);
      return vec4(mix(tex(uv).rgb, tex(cell).rgb * sq * 1.3, uAmt), 1.0);
    }`),
  fx('edges', 'Edge detect', 'Stylize', `
    vec4 effect(vec2 uv) {
      vec2 e = uTexel * 1.5;
      float tl = luma(tex(uv + vec2(-e.x, -e.y)).rgb), t = luma(tex(uv + vec2(0.0, -e.y)).rgb), tr = luma(tex(uv + vec2(e.x, -e.y)).rgb);
      float l  = luma(tex(uv + vec2(-e.x, 0.0)).rgb),                                            r  = luma(tex(uv + vec2(e.x, 0.0)).rgb);
      float bl = luma(tex(uv + vec2(-e.x, e.y)).rgb),  b = luma(tex(uv + vec2(0.0, e.y)).rgb),  br = luma(tex(uv + vec2(e.x, e.y)).rgb);
      float gx = tr + 2.0 * r + br - tl - 2.0 * l - bl;
      float gy = bl + 2.0 * b + br - tl - 2.0 * t - tr;
      float g = clamp(length(vec2(gx, gy)) * 2.0, 0.0, 1.0);
      return vec4(mix(tex(uv).rgb, vec3(g), uAmt), 1.0);
    }`),
  fx('neonedges', 'Neon edges', 'Stylize', `
    vec4 effect(vec2 uv) {
      vec2 e = uTexel * 1.5;
      float gx = luma(tex(uv + vec2(e.x, 0.0)).rgb) - luma(tex(uv - vec2(e.x, 0.0)).rgb);
      float gy = luma(tex(uv + vec2(0.0, e.y)).rgb) - luma(tex(uv - vec2(0.0, e.y)).rgb);
      float g = clamp(length(vec2(gx, gy)) * 6.0, 0.0, 1.0);
      vec3 neon = pal(luma(tex(uv).rgb) + uTime * 0.1, 0.0) * g;
      return vec4(mix(tex(uv).rgb, tex(uv).rgb * 0.15 + neon, uAmt), 1.0);
    }`),
  fx('emboss', 'Emboss', 'Stylize', `
    vec4 effect(vec2 uv) {
      float a = luma(tex(uv - uTexel * 2.0).rgb);
      float b = luma(tex(uv + uTexel * 2.0).rgb);
      vec3 em = vec3(0.5 + (a - b) * 3.0);
      return vec4(mix(tex(uv).rgb, em, uAmt), 1.0);
    }`),
  fx('toon', 'Toon', 'Stylize', `
    vec4 effect(vec2 uv) {
      vec2 e = uTexel * 1.5;
      float gx = luma(tex(uv + vec2(e.x, 0.0)).rgb) - luma(tex(uv - vec2(e.x, 0.0)).rgb);
      float gy = luma(tex(uv + vec2(0.0, e.y)).rgb) - luma(tex(uv - vec2(0.0, e.y)).rgb);
      float edge = 1.0 - clamp(length(vec2(gx, gy)) * 8.0, 0.0, 1.0);
      vec3 c = floor(tex(uv).rgb * 5.0) / 5.0;
      return vec4(mix(tex(uv).rgb, c * edge, uAmt), 1.0);
    }`),
  fx('contour', 'Contour lines', 'Stylize', `
    vec4 effect(vec2 uv) {
      float l = luma(tex(uv).rgb);
      float bands = fract(l * 8.0);
      float line = 1.0 - smoothstep(0.0, 0.12, bands) * smoothstep(1.0, 0.88, bands);
      vec3 c = mix(tex(uv).rgb, vec3(1.0 - line) * tex(uv).rgb + vec3(line) * vec3(0.1, 1.0, 0.6), uAmt);
      return vec4(c, 1.0);
    }`),
  fx('crosshatch', 'Crosshatch', 'Stylize', `
    vec4 effect(vec2 uv) {
      float l = luma(tex(uv).rgb);
      vec2 p = uv * vec2(uAspect, 1.0) * 400.0;
      float v = 1.0;
      if (l < 0.8) v *= 0.45 + 0.55 * step(0.5, fract((p.x + p.y) * 0.05));
      if (l < 0.6) v *= 0.45 + 0.55 * step(0.5, fract((p.x - p.y) * 0.05));
      if (l < 0.4) v *= 0.45 + 0.55 * step(0.5, fract(p.x * 0.07));
      if (l < 0.2) v *= 0.45 + 0.55 * step(0.5, fract(p.y * 0.07));
      return vec4(mix(tex(uv).rgb, vec3(v), uAmt), 1.0);
    }`),
  fx('oilslick', 'Oil slick', 'Stylize', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec3 slick = pal(luma(c) * 2.0 + uTime * 0.15 + vDisp, 0.0);
      return vec4(mix(c, c * 0.4 + slick * 0.7, uAmt * (0.4 + 0.6 * vDisp)), 1.0);
    }`),
  fx('blueprint', 'Blueprint', 'Stylize', `
    vec4 effect(vec2 uv) {
      vec2 e = uTexel * 1.5;
      float gx = luma(tex(uv + vec2(e.x, 0.0)).rgb) - luma(tex(uv - vec2(e.x, 0.0)).rgb);
      float gy = luma(tex(uv + vec2(0.0, e.y)).rgb) - luma(tex(uv - vec2(0.0, e.y)).rgb);
      float g = clamp(length(vec2(gx, gy)) * 6.0, 0.0, 1.0);
      vec2 gridp = fract(uv * vec2(40.0 * uAspect, 40.0));
      float grid = 0.08 * (step(gridp.x, 0.02) + step(gridp.y, 0.03));
      vec3 bp = vec3(0.02, 0.09, 0.35) + vec3(g) + grid;
      return vec4(mix(tex(uv).rgb, bp, uAmt), 1.0);
    }`),

  // ---------- motion & blur ----------
  fx('smear', 'Motion smear', 'Motion & blur', `
    vec4 effect(vec2 uv) {
      vec3 acc = vec3(0.0);
      for (int i = 0; i < 8; i++) {
        acc += tex(uv - vec2(float(i) * 0.008 * uAmt, 0.0)).rgb;
      }
      return vec4(acc / 8.0, 1.0);
    }`),
  fx('radialblur', 'Radial blur', 'Motion & blur', `
    vec4 effect(vec2 uv) {
      vec2 dir = (uv - 0.5) * 0.06 * uAmt;
      vec3 acc = vec3(0.0);
      for (int i = 0; i < 8; i++) {
        acc += tex(uv - dir * float(i)).rgb;
      }
      return vec4(acc / 8.0, 1.0);
    }`),
  fx('dream', 'Dream haze', 'Motion & blur', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec3 blur = (tex(uv + uTexel * 6.0).rgb + tex(uv - uTexel * 6.0).rgb
                 + tex(uv + vec2(uTexel.x, -uTexel.y) * 6.0).rgb
                 + tex(uv + vec2(-uTexel.x, uTexel.y) * 6.0).rgb) * 0.25;
      vec3 soft = mix(c, blur, 0.7) + blur * blur * 0.4;
      soft = mix(soft, vec3(luma(soft)), 0.25) + vec3(0.04, 0.02, 0.06);
      return vec4(mix(c, soft, uAmt), 1.0);
    }`),

  // ---------- time & light ----------
  fx('timewarp', 'Time smear (warp = past)', 'Time & light', `
    vec4 effect(vec2 uv) {
      vec2 m = muv(uv);
      // deformation reaches back into a 16-frame rolling history (4x4 atlas)
      float back = clamp(vDisp * uAmt * 2.0, 0.0, 0.999) * 15.0;
      float idx = mod(uHead - floor(back) + 16.0, 16.0);
      vec2 cell = vec2(mod(idx, 4.0), floor(idx / 4.0));
      vec2 cuv = clamp(vec2(m.x, 1.0 - m.y), 0.004, 0.996);
      return texture2D(uHistory, (cell + cuv) / 4.0);
    }`),
  fx('trails', 'Feedback trails', 'Time & light', `
    vec4 effect(vec2 uv) {
      vec3 cur = tex(uv).rgb;
      vec2 suv = gl_FragCoord.xy / uResolution;
      vec3 prev = texture2D(uPrev, suv).rgb;
      return vec4(max(cur, prev * (0.78 + 0.21 * uAmt)), 1.0);
    }`),
  fx('relief', 'Relief light', 'Time & light', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec3 n = normalize(vec3(-vGrad * (30.0 + uAmt * 90.0), 1.0));
      vec3 L = normalize(vec3(0.35, 0.5, 0.75));
      float diff = max(dot(n, L), 0.0);
      vec3 H = normalize(L + vec3(0.0, 0.0, 1.0));
      float spec = pow(max(dot(n, H), 0.0), 24.0) * uAmt;
      return vec4(c * (0.45 + 0.7 * diff) + spec * 0.6, 1.0);
    }`),

  // ---------- warp-reactive ----------
  fx('heatwarp', 'Heat where warped', 'Warp-reactive', `
    vec4 effect(vec2 uv) {
      float k = clamp(vDisp * 1.5, 0.0, 1.0) * uAmt;
      uv.x += sin(uv.y * 60.0 + uTime * 8.0) * 0.006 * k;
      vec3 c = tex(uv).rgb;
      c = mix(c, c * vec3(1.3, 0.5, 0.2) + vec3(0.45, 0.12, 0.0) * vDisp, k);
      return vec4(c, 1.0);
    }`),
  fx('dispglow', 'Glow where warped', 'Warp-reactive', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      c += pal(vDisp * 1.5 + uTime * 0.2, 0.0) * clamp(vDisp * 2.0, 0.0, 1.0) * uAmt;
      return vec4(c, 1.0);
    }`),
  fx('dispview', 'Warp heat map', 'Warp-reactive', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      return vec4(mix(c, pal(clamp(vDisp, 0.0, 1.0) * 0.7, 0.55), uAmt * clamp(vDisp * 4.0 + 0.1, 0.0, 1.0)), 1.0);
    }`),
  fx('fracture', 'Fracture where warped', 'Warp-reactive', `
    vec4 effect(vec2 uv) {
      float k = clamp(vDisp * 2.0, 0.0, 1.0) * uAmt;
      vec2 g = vec2(30.0, 20.0);
      vec2 b = floor(uv * g);
      uv += (vec2(hash(b), hash(b + 7.7)) - 0.5) * 0.06 * k;
      vec2 s = vec2(0.02 * k, 0.0);
      return vec4(tex(uv + s).r, tex(uv).g, tex(uv - s).b, 1.0);
    }`),
  fx('xray', 'X-ray where warped', 'Warp-reactive', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      vec3 x = 1.0 - vec3(luma(c));
      x *= vec3(0.7, 0.9, 1.1);
      return vec4(mix(c, x, clamp(vDisp * 2.5, 0.0, 1.0) * uAmt), 1.0);
    }`),
  fx('liquidchrome', 'Liquid chrome', 'Warp-reactive', `
    vec4 effect(vec2 uv) {
      float k = (0.2 + vDisp) * uAmt;
      uv += 0.03 * k * vec2(sin(uv.y * 18.0 + uTime * 2.2), cos(uv.x * 16.0 + uTime * 1.8));
      vec3 c = tex(uv).rgb;
      float l = luma(c);
      vec3 chrome = mix(vec3(0.05, 0.06, 0.1), vec3(0.9, 0.95, 1.05), smoothstep(0.2, 0.8, l));
      chrome += pal(l * 3.0, 0.1) * 0.15;
      return vec4(mix(c, chrome, uAmt * (0.3 + 0.7 * clamp(vDisp * 2.0, 0.0, 1.0))), 1.0);
    }`),
  fx('burnedges', 'Burn at warp edge', 'Warp-reactive', `
    vec4 effect(vec2 uv) {
      vec3 c = tex(uv).rgb;
      float band = smoothstep(0.05, 0.25, vDisp) * smoothstep(0.75, 0.35, vDisp);
      c = mix(c, vec3(1.0, 0.4, 0.05), band * uAmt);
      c *= 1.0 - 0.75 * smoothstep(0.5, 1.0, vDisp) * uAmt;
      return vec4(c, 1.0);
    }`),
  fx('shockwave', 'Shockwave rings', 'Warp-reactive', `
    vec4 effect(vec2 uv) {
      float k = smoothstep(0.02, 0.2, vDisp);
      vec3 c = tex(uv).rgb;
      c += sin(vDisp * 40.0 - uTime * 6.0) * 0.2 * uAmt * k;
      return vec4(c, 1.0);
    }`),
];
