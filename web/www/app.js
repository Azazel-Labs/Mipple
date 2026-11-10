// mipple web demo: UI, worker scheduling and the comparison viewer. No build step, no framework.
import { Viewer } from "./viewer.js";

const LAMBDAS = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 12, 16];

const SAMPLES = [
  { id: "castle_brick_01_diff", name: "Castle brick" },
  { id: "blue_painted_planks_diff", name: "Painted planks" },
  { id: "pine_bark_diff", name: "Pine bark" },
  { id: "white_rough_plaster_diff", name: "Rough plaster" },
  { id: "wood_table_worn_diff", name: "Worn wood" },
  { id: "cobblestone_floor_13_nor", name: "Cobblestone normals", normal: true },
];

const FORMAT_NOTES = {
  bc1: "4 bits/texel · RGB, opaque · albedo and bulk color",
  bc4: "4 bits/texel · one channel (R, shown as gray) · roughness, masks, height",
  bc5: "8 bits/texel · two channels (R, G) · tangent-space normal maps",
  bc7: "8 bits/texel · RGB or RGBA · best color quality. With RDO (λ > 0) it runs the effort-8 search first, which takes tens of seconds per 512² image in the browser",
};

const DEFAULTS = {
  format: "bc1",
  lambda: 1,
  effort: 5,
  entropy: "auto",
  mips: false,
  normal: false,
  smoothScale: 5,
  smoothMaxStd: 14,
  maxRmsRatio: 10,
  minRms: 3,
  zstd: 19,
  maxSize: 1024,
};

const $ = (id) => document.getElementById(id);
const nf = new Intl.NumberFormat("en-US");

const state = {
  settings: { ...DEFAULTS },
  colorFormat: "bc1", // restored when leaving a normal map
  source: null, // { name, fileName, bytes: ArrayBuffer | null, blob: Blob | null, sampleId }
  image: null, // { width, height, rgba, sourceWidth, sourceHeight }
  prev: null, // previous result on the same image, for the deltas
  downloadUrl: null,
};

// ---------------------------------------------------------------------------------------------
// Worker: one at a time; terminated and respawned to cancel a stale encode.

class Cancelled extends Error {}

class Engine {
  constructor(module) {
    this.module = module;
    this.seq = 0;
    this.spawn();
  }

  spawn() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.pending = new Map();
    this.worker.onmessage = ({ data }) => {
      const p = this.pending.get(data.id);
      if (!p) return;
      this.pending.delete(data.id);
      if (data.ok) p.resolve(data.result);
      else p.reject(new Error(data.error));
    };
    this.worker.onerror = (e) => {
      for (const p of this.pending.values()) p.reject(new Error(e.message || "worker failed"));
      this.pending.clear();
    };
    this.ready = this.call({ type: "init", module: this.module });
    this.ready.catch(() => {}); // a cancel during init is not an error; callers see their own
  }

  call(msg, transfer = []) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...msg, id }, transfer);
    });
  }

  get busy() {
    return this.pending.size > 0;
  }

  cancel() {
    this.worker.terminate();
    for (const p of this.pending.values()) p.reject(new Cancelled());
    this.spawn();
  }
}

async function compileModule() {
  const url = new URL("./pkg/mipple_web_bg.wasm", import.meta.url);
  try {
    return await WebAssembly.compileStreaming(fetch(url));
  } catch {
    // Servers that send the wrong MIME type for .wasm.
    const r = await fetch(url);
    if (!r.ok) throw new Error(`could not load ${url.pathname} (${r.status})`);
    return WebAssembly.compile(await r.arrayBuffer());
  }
}

let engine = null;
let job = null; // the encode or decode in flight: { token }
let encodeTimer = 0;
let busyTimer = 0;

// ---------------------------------------------------------------------------------------------
// Viewer

const viewer = new Viewer($("view"), $("canvas"), {
  onZoom: (pct) => ($("zoom-level").textContent = pct),
});

function setTags() {
  const s = state.settings;
  const left = $("tag-left");
  const right = $("tag-right");
  if (viewer.mode === "diff") {
    left.textContent = `|original − ${s.format.toUpperCase()}| × ${viewer.gain}`;
    right.hidden = true;
    return;
  }
  left.textContent = "Original";
  right.hidden = !viewer.hasDecoded;
  right.textContent = `${s.format.toUpperCase()} · λ ${s.lambda} · e${s.effort}`;
}

// ---------------------------------------------------------------------------------------------
// Busy state

function setBusy(text) {
  clearInterval(busyTimer);
  const busy = $("busy");
  const status = $("status");
  if (!text) {
    busy.hidden = true;
    $("stats").classList.remove("stale");
    return;
  }
  const t0 = performance.now();
  const tick = () => {
    const s = (performance.now() - t0) / 1000;
    $("busy-text").textContent = s >= 0.4 ? `${text} ${s.toFixed(1)} s` : text;
  };
  tick();
  busyTimer = setInterval(tick, 100);
  busy.hidden = false;
  status.textContent = text.replace("…", "").toLowerCase();
  status.className = "pill busy-pill";
  $("stats").classList.add("stale");
}

function setStatus(text, cls = "") {
  $("status").textContent = text;
  $("status").className = `pill ${cls}`;
}

function showPlaceholder(text, error = false) {
  const p = $("placeholder");
  p.hidden = !text;
  p.textContent = text || "";
  p.classList.toggle("error", error);
}

// ---------------------------------------------------------------------------------------------
// Loading images

function cancelJob() {
  clearTimeout(encodeTimer);
  if (job && engine?.busy) engine.cancel();
  job = null;
}

async function browserDecode(blob, maxSize) {
  const bmp = await createImageBitmap(blob, { premultiplyAlpha: "none", colorSpaceConversion: "none" });
  const c = document.createElement("canvas");
  c.width = bmp.width;
  c.height = bmp.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  const d = ctx.getImageData(0, 0, c.width, c.height);
  const rgba = new Uint8Array(d.data.buffer);
  return engine.call({ type: "fit", rgba, width: d.width, height: d.height, maxSize }, [rgba.buffer]);
}

async function decodeSource() {
  const src = state.source;
  cancelJob();
  writeHash();
  const token = (job = {});
  setBusy("Decoding…");
  try {
    await engine.ready;
    if (!src.bytes) {
      const r = await fetch(`samples/${src.sampleId}.png`);
      if (!r.ok) throw new Error(`could not load ${src.fileName} (${r.status})`);
      src.bytes = await r.arrayBuffer();
    }
    const maxSize = state.settings.maxSize;
    let img;
    try {
      const bytes = src.bytes.slice(0);
      img = await engine.call({ type: "decode", bytes, maxSize }, [bytes]);
    } catch (err) {
      if (err instanceof Cancelled || !src.blob) throw err;
      img = await browserDecode(src.blob, maxSize); // WebP, AVIF, GIF, BMP...
    }
    if (job !== token) return;
    job = null;
    state.image = img;
    state.prev = null;
    showPlaceholder("");
    viewer.setImages(img.rgba, null, img.width, img.height);
    setTags();
    encodeNow();
  } catch (err) {
    if (err instanceof Cancelled || job !== token) return;
    job = null;
    setBusy(null);
    setStatus("error", "error");
    showPlaceholder(`Could not open ${src.name}: ${err.message}`, true);
  }
}

function selectSample(sample) {
  const s = state.settings;
  if (sample.normal && !s.normal) {
    state.colorFormat = s.format === "bc5" ? state.colorFormat : s.format;
    s.normal = true;
    s.format = "bc5";
  } else if (!sample.normal && s.normal) {
    s.normal = false;
    s.format = state.colorFormat;
  }
  state.source = {
    name: sample.name,
    fileName: `${sample.id}.png`,
    sampleId: sample.id,
    bytes: null,
    blob: null,
  };
  markTile(sample.id);
  syncControls();
  decodeSource();
}

function openFile(file) {
  if (!file) return;
  state.source = { name: file.name, fileName: file.name, sampleId: null, bytes: null, blob: file };
  const tile = $("upload-tile");
  tile.querySelector("img")?.remove();
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.alt = "";
  img.onload = () => tile.classList.add("has-image");
  img.onerror = () => {
    img.remove();
    tile.classList.remove("has-image");
  };
  img.src = url;
  tile.prepend(img);
  tile.title = file.name;
  markTile("upload");
  file
    .arrayBuffer()
    .then((b) => {
      if (state.source?.blob !== file) return;
      state.source.bytes = b;
      decodeSource();
    })
    .catch((err) => showPlaceholder(`Could not read ${file.name}: ${err.message}`, true));
}

function markTile(id) {
  for (const t of document.querySelectorAll(".tile")) {
    const on = t.dataset.sample === id || (id === "upload" && t.id === "upload-tile");
    t.setAttribute("aria-checked", on ? "true" : "false");
  }
}

// ---------------------------------------------------------------------------------------------
// Encoding

function scheduleEncode(delay = 220) {
  clearTimeout(encodeTimer);
  writeHash();
  if (!state.image) return;
  encodeTimer = setTimeout(encodeNow, delay);
}

function quoteArg(s) {
  return /^[\w.\-/]+$/.test(s) ? s : `"${s.replace(/"/g, '\\"')}"`;
}

async function encodeNow() {
  clearTimeout(encodeTimer);
  const img = state.image;
  if (!img || !engine) return;
  if (job && engine.busy) engine.cancel(); // a stale encode: drop it
  const token = (job = {});
  const s = state.settings;
  const settings = { ...s, lambda: s.lambda, inputName: quoteArg(state.source.fileName) };
  setBusy("Encoding…");
  try {
    const rgba = img.rgba.slice();
    const r = await engine.call(
      { type: "encode", rgba, width: img.width, height: img.height, settings },
      [rgba.buffer],
    );
    if (job !== token) return;
    job = null;
    setBusy(null);
    showResult(r, s);
  } catch (err) {
    if (err instanceof Cancelled || job !== token) return;
    job = null;
    setBusy(null);
    setStatus("error", "error");
    $("s-detail").textContent = `Encode failed: ${err.message}`;
  }
}

function fmtBytes(n) {
  if (n < 1024) return `${nf.format(n)} B`;
  return `${(n / 1024).toFixed(n < 10240 ? 2 : 1)} KiB`;
}

function fmtMs(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function setDelta(id, cur, prev, { rel = false, digits = 2, higherIsBetter = true, unit = "" } = {}) {
  const el = $(id);
  if (prev == null || !Number.isFinite(cur) || !Number.isFinite(prev)) {
    el.textContent = "";
    return;
  }
  const d = rel ? ((cur - prev) / prev) * 100 : cur - prev;
  const shown = d.toFixed(digits);
  if (Number(shown) === 0) {
    el.textContent = "±0";
    el.className = "delta same";
    return;
  }
  el.textContent = `${d > 0 ? "+" : "−"}${Math.abs(d).toFixed(digits)}${rel ? "%" : unit}`;
  el.className = `delta ${d > 0 === higherIsBetter ? "good" : "bad"}`;
}

function showResult(r, s) {
  viewer.setImages(r.original, r.decoded, r.width, r.height);
  setTags();
  setStatus(`${fmtMs(r.totalMs)}`);

  const texels = r.width * r.height;
  const bpt = (r.storedBytes * 8) / texels;
  const prev = state.prev;
  $("s-bpt").textContent = bpt.toFixed(3);
  setDelta("d-bpt", bpt, prev?.bpt, { rel: true, digits: 1, higherIsBetter: false });
  $("s-size").textContent =
    s.zstd > 0
      ? `${nf.format(r.storedBytes)} B after Zstd-${s.zstd} · level 0, ${r.width}×${r.height}`
      : `${nf.format(r.storedBytes)} B of raw blocks (Zstd off) · ${r.width}×${r.height}`;

  $("s-psnr").textContent = Number.isFinite(r.psnr) ? `${r.psnr.toFixed(2)} dB` : "∞";
  setDelta("d-psnr", r.psnr, prev?.psnr, { digits: 2, unit: "" });
  $("s-rmse").textContent = r.rmse.toFixed(3);
  setDelta("d-rmse", r.rmse, prev?.rmse, { digits: 3, higherIsBetter: false });
  $("s-ssim").textContent = r.ssim.toFixed(4);
  setDelta("d-ssim", r.ssim, prev?.ssim, { digits: 4 });
  $("normal-stat").hidden = !Number.isFinite(r.normalDeg);
  $("s-normal").textContent = Number.isFinite(r.normalDeg) ? `${r.normalDeg.toFixed(2)}°` : "–";
  $("s-time").textContent = fmtMs(r.baseMs + r.rdoMs);
  $("s-rdo").textContent = s.lambda > 0 ? `${((100 * r.rdoChanged) / Math.max(1, r.blocks)).toFixed(1)}%` : "off";
  $("s-raw").textContent = fmtBytes(r.rawBytes);
  $("s-file").textContent = fmtBytes(r.fileBytes);

  const img = state.image;
  const parts = [
    `${s.format.toUpperCase()}${!s.normal && s.format !== "bc4" && s.format !== "bc5" ? " sRGB" : ""}`,
    `${r.levels} level${r.levels > 1 ? "s" : ""}`,
    `base ${fmtMs(r.baseMs)} + RDO ${fmtMs(r.rdoMs)}, ${fmtMs(r.totalMs)} with Zstd and metrics on one thread`,
  ];
  if (img.sourceWidth !== img.width || img.sourceHeight !== img.height) {
    parts.push(`downscaled from ${img.sourceWidth}×${img.sourceHeight}; the CLI needs the resized copy to match`);
  }
  $("s-detail").textContent = parts.join(" · ");

  if (state.downloadUrl) URL.revokeObjectURL(state.downloadUrl);
  state.downloadUrl = URL.createObjectURL(new Blob([r.ktx2], { type: "image/ktx2" }));
  const base = state.source.fileName.replace(/\.[^.]+$/, "");
  const dl = $("download");
  dl.href = state.downloadUrl;
  dl.download = `${base}_${s.format}_l${s.lambda}_e${s.effort}.ktx2`;
  dl.setAttribute("aria-disabled", "false");
  dl.querySelector("span").textContent = `Download .ktx2 (${fmtBytes(r.fileBytes)})`;
  // One nowrap span per argument, so a line never breaks inside "--no-mips".
  $("cmd").replaceChildren(
    ...r.command.split(" ").flatMap((arg, i) => {
      const s = document.createElement("span");
      s.className = "arg";
      s.textContent = arg;
      return i ? [" ", s] : [s];
    }),
  );

  state.prev = { bpt, psnr: r.psnr, rmse: r.rmse, ssim: r.ssim };
  const root = document.documentElement.dataset;
  root.encoded = String(Number(root.encoded ?? 0) + 1); // a hook for automated page checks
}

// ---------------------------------------------------------------------------------------------
// Controls

const RANGES = [
  // [element id, settings key, value -> slider, slider -> value, label]
  ["lambda", "lambda", (v) => nearestLambda(v), (i) => LAMBDAS[i], (v) => (v === 0 ? "0 · off" : `${v}`)],
  ["effort", "effort", (v) => v, (i) => i, (v) => `${v}`],
  ["smooth-scale", "smoothScale", (v) => v, (i) => i, (v) => `${v}×`],
  ["smooth-max-std", "smoothMaxStd", (v) => v, (i) => i, (v) => `${v}`],
  ["max-rms-ratio", "maxRmsRatio", (v) => v, (i) => i, (v) => `${v}×`],
  ["min-rms", "minRms", (v) => v, (i) => i, (v) => `${v}`],
];

function nearestLambda(v) {
  let best = 0;
  for (let i = 0; i < LAMBDAS.length; i++) {
    if (Math.abs(LAMBDAS[i] - v) < Math.abs(LAMBDAS[best] - v)) best = i;
  }
  return best;
}

function paintRange(el) {
  const pct = ((el.value - el.min) / (el.max - el.min)) * 100;
  el.style.setProperty("--fill", `${pct}%`);
}

function syncControls() {
  const s = state.settings;
  for (const [id, key, toSlider, , label] of RANGES) {
    const el = $(id);
    el.value = toSlider(s[key]);
    $(`${id}-out`).textContent = label(s[key]);
    paintRange(el);
  }
  for (const b of $("format").querySelectorAll("button")) b.setAttribute("aria-pressed", b.dataset.format === s.format);
  for (const b of $("entropy").querySelectorAll("button")) {
    b.setAttribute("aria-pressed", b.dataset.entropy === s.entropy);
    b.disabled = s.format !== "bc7";
  }
  $("entropy-field").classList.toggle("disabled", s.format !== "bc7");
  $("format-note").textContent = FORMAT_NOTES[s.format];
  $("mips").checked = s.mips;
  $("normal").checked = s.normal;
  $("zstd").value = String(s.zstd);
  $("max-size").value = String(s.maxSize);
  setTags();
}

function bindControls() {
  for (const [id, key, , fromSlider, label] of RANGES) {
    const el = $(id);
    el.max = id === "lambda" ? LAMBDAS.length - 1 : el.max;
    el.addEventListener("input", () => {
      state.settings[key] = fromSlider(Number(el.value));
      $(`${id}-out`).textContent = label(state.settings[key]);
      paintRange(el);
      setTags();
      scheduleEncode();
    });
  }
  $("format").addEventListener("click", (e) => {
    const f = e.target.closest("button")?.dataset.format;
    if (!f || f === state.settings.format) return;
    state.settings.format = f;
    if (!state.settings.normal) state.colorFormat = f;
    syncControls();
    scheduleEncode(0);
  });
  $("entropy").addEventListener("click", (e) => {
    const v = e.target.closest("button")?.dataset.entropy;
    if (!v || v === state.settings.entropy) return;
    state.settings.entropy = v;
    syncControls();
    scheduleEncode(0);
  });
  $("mips").addEventListener("change", (e) => {
    state.settings.mips = e.target.checked;
    scheduleEncode(0);
  });
  $("normal").addEventListener("change", (e) => {
    state.settings.normal = e.target.checked;
    scheduleEncode(0);
  });
  $("zstd").addEventListener("change", (e) => {
    state.settings.zstd = Number(e.target.value);
    scheduleEncode(0);
  });
  $("max-size").addEventListener("change", (e) => {
    state.settings.maxSize = Number(e.target.value);
    writeHash();
    if (state.source) decodeSource();
  });
  $("reset").addEventListener("click", () => {
    const keep = { format: state.settings.format, normal: state.settings.normal, maxSize: state.settings.maxSize };
    state.settings = { ...DEFAULTS, ...keep };
    syncControls();
    scheduleEncode(0);
  });

  $("mode").addEventListener("click", (e) => {
    const m = e.target.closest("button")?.dataset.mode;
    if (!m) return;
    for (const b of $("mode").querySelectorAll("button")) b.setAttribute("aria-pressed", b.dataset.mode === m);
    $("diff-gain-wrap").hidden = m !== "diff";
    viewer.setMode(m);
    setTags();
  });
  $("diff-gain").addEventListener("change", (e) => {
    viewer.setGain(Number(e.target.value));
    setTags();
  });
  $("grid").addEventListener("click", () => toggleGrid());
  $("zoom-in").addEventListener("click", () => viewer.zoomStep(1));
  $("zoom-out").addEventListener("click", () => viewer.zoomStep(-1));
  $("zoom-level").addEventListener("click", () => viewer.toggleFit());
  $("copy-cmd").addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText($("cmd").textContent);
      $("copy-cmd").textContent = "Copied";
    } catch {
      $("copy-cmd").textContent = "Select it";
    }
    setTimeout(() => ($("copy-cmd").textContent = "Copy"), 1400);
  });

  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.target instanceof Element && e.target.closest("input, select, textarea")) return;
    const k = e.key;
    if (k === "0") viewer.fit();
    else if (k >= "1" && k <= "4") viewer.setScale(2 ** (Number(k) - 1));
    else if (k === "+" || k === "=") viewer.zoomStep(1);
    else if (k === "-" || k === "_") viewer.zoomStep(-1);
    else if (k === "g" || k === "G") toggleGrid();
    else return;
    e.preventDefault();
  });
}

function toggleGrid() {
  const on = !viewer.grid;
  viewer.setGrid(on);
  $("grid").setAttribute("aria-pressed", on);
}

function buildGallery() {
  const g = $("gallery");
  const upload = $("upload-tile");
  for (const sample of SAMPLES) {
    const b = document.createElement("button");
    b.type = "button";
    b.className = "tile";
    b.dataset.sample = sample.id;
    b.setAttribute("role", "radio");
    b.setAttribute("aria-checked", "false");
    b.title = `${sample.name} (Poly Haven, CC0)`;
    b.innerHTML = `<img src="samples/thumbs/${sample.id}.jpg" alt="" width="96" height="96" loading="lazy"><span class="tile-name"></span>`;
    b.querySelector(".tile-name").textContent = sample.name;
    b.addEventListener("click", () => selectSample(sample));
    g.insertBefore(b, upload);
  }
  const input = $("file");
  input.addEventListener("change", () => openFile(input.files[0]));
  upload.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      input.click();
    }
  });

  // Drop anywhere on the page.
  let depth = 0;
  const overlay = $("drop-overlay");
  const hasFiles = (e) => [...(e.dataTransfer?.types ?? [])].includes("Files");
  window.addEventListener("dragenter", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    overlay.hidden = false;
  });
  window.addEventListener("dragover", (e) => {
    if (hasFiles(e)) e.preventDefault();
  });
  window.addEventListener("dragleave", () => {
    depth = Math.max(0, depth - 1);
    if (depth === 0) overlay.hidden = true;
  });
  window.addEventListener("drop", (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    overlay.hidden = true;
    openFile(e.dataTransfer.files[0]);
  });
  // Paste an image from the clipboard.
  window.addEventListener("paste", (e) => {
    const f = [...(e.clipboardData?.files ?? [])].find((x) => x.type.startsWith("image/"));
    if (f) openFile(f);
  });
}

// ---------------------------------------------------------------------------------------------
// Settings in the URL hash, so a view can be shared.

const HASH_KEYS = {
  f: "format",
  l: "lambda",
  e: "effort",
  en: "entropy",
  m: "mips",
  n: "normal",
  ss: "smoothScale",
  sm: "smoothMaxStd",
  rr: "maxRmsRatio",
  mr: "minRms",
  z: "zstd",
  max: "maxSize",
};

function writeHash() {
  const p = new URLSearchParams();
  if (state.source?.sampleId) p.set("img", state.source.sampleId);
  for (const [k, key] of Object.entries(HASH_KEYS)) {
    const v = state.settings[key];
    if (v !== DEFAULTS[key]) p.set(k, typeof v === "boolean" ? Number(v) : v);
  }
  const h = p.toString();
  history.replaceState(null, "", h ? `#${h}` : location.pathname + location.search);
}

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const s = state.settings;
  for (const [k, key] of Object.entries(HASH_KEYS)) {
    if (!p.has(k)) continue;
    const raw = p.get(k);
    const d = DEFAULTS[key];
    if (typeof d === "boolean") s[key] = raw === "1";
    else if (typeof d === "number") s[key] = Number.isFinite(Number(raw)) ? Number(raw) : d;
    else s[key] = raw;
  }
  if (!FORMAT_NOTES[s.format]) s.format = DEFAULTS.format;
  if (!["auto", "on", "off"].includes(s.entropy)) s.entropy = DEFAULTS.entropy;
  s.lambda = LAMBDAS[nearestLambda(s.lambda)];
  s.effort = Math.min(9, Math.max(0, Math.round(s.effort)));
  if (!s.normal && s.format !== "bc4" && s.format !== "bc5") state.colorFormat = s.format;
  return SAMPLES.find((x) => x.id === p.get("img")) ?? SAMPLES[0];
}

// ---------------------------------------------------------------------------------------------

async function main() {
  buildGallery();
  bindControls();
  const first = readHash();
  syncControls();
  try {
    if (typeof WebAssembly !== "object") throw new Error("this browser has no WebAssembly");
    engine = new Engine(await compileModule());
    await engine.ready;
  } catch (err) {
    setStatus("error", "error");
    showPlaceholder(`The encoder could not start: ${err.message}`, true);
    return;
  }
  // A shared link keeps its own format and normal settings; otherwise the normal map gets BC5.
  const hashed = new URLSearchParams(location.hash.slice(1));
  const s = state.settings;
  if (first.normal && !hashed.has("n")) {
    s.normal = true;
    if (!hashed.has("f")) s.format = "bc5";
  }
  state.source = { name: first.name, fileName: `${first.id}.png`, sampleId: first.id, bytes: null, blob: null };
  markTile(first.id);
  syncControls();
  decodeSource();
}

main();
