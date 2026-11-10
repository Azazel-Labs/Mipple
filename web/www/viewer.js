// The comparison canvas: split (draggable divider), side by side, or amplified difference, with
// nearest-neighbor zoom, pan, wheel and pinch zoom, and a 4x4 block grid.

const STEPS = [0.125, 0.25, 0.5, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32];

function toCanvas(rgba, w, h) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  const px = new Uint8ClampedArray(rgba.buffer, rgba.byteOffset, w * h * 4);
  c.getContext("2d").putImageData(new ImageData(px, w, h), 0, 0);
  return c;
}

function diffCanvas(a, b, w, h, gain) {
  const out = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < out.length; i += 4) {
    out[i] = Math.abs(a[i] - b[i]) * gain;
    out[i + 1] = Math.abs(a[i + 1] - b[i + 1]) * gain;
    out[i + 2] = Math.abs(a[i + 2] - b[i + 2]) * gain;
    out[i + 3] = 255;
  }
  return toCanvas(out, w, h);
}

export class Viewer {
  constructor(container, canvas, { onZoom } = {}) {
    this.el = container;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onZoom = onZoom ?? (() => {});
    this.mode = "split";
    this.gain = 16;
    this.grid = false;
    this.split = 0.5; // fraction of the view width
    this.w = 0;
    this.h = 0;
    this.orig = null;
    this.dec = null;
    this.diff = null;
    this.rgbaA = null;
    this.rgbaB = null;
    this.scale = 1;
    this.fitted = true;
    this.cx = 0; // image point at the pane center
    this.cy = 0;
    this.pointers = new Map();
    this.drag = null;
    this.frame = 0;
    new ResizeObserver(() => this.resize()).observe(container);
    this.bindInput();
    this.resize();
  }

  get hasDecoded() {
    return !!this.dec;
  }

  setImages(original, decoded, w, h) {
    const sameSize = w === this.w && h === this.h;
    this.w = w;
    this.h = h;
    this.rgbaA = original;
    this.rgbaB = decoded;
    this.orig = toCanvas(original, w, h);
    this.dec = decoded ? toCanvas(decoded, w, h) : null;
    this.diff = null;
    if (!sameSize || this.fitted) this.fit(false);
    this.draw();
  }

  setMode(m) {
    this.mode = m;
    if (this.fitted) this.fit(false);
    this.draw();
  }

  setGain(g) {
    this.gain = g;
    this.diff = null;
    this.draw();
  }

  setGrid(on) {
    this.grid = on;
    this.draw();
  }

  // ---- geometry (CSS pixels) ----

  pane() {
    const W = this.el.clientWidth;
    const H = this.el.clientHeight;
    return this.mode === "side" ? { w: W / 2, h: H } : { w: W, h: H };
  }

  fitScale() {
    if (!this.w) return 1;
    const p = this.pane();
    const s = Math.min(p.w / this.w, p.h / this.h);
    return s >= 1 ? Math.floor(s) : s; // whole-number upscales keep texels square
  }

  fit(redraw = true) {
    this.scale = this.fitScale();
    this.fitted = true;
    this.cx = this.w / 2;
    this.cy = this.h / 2;
    this.report();
    if (redraw) this.draw();
  }

  toggleFit() {
    if (this.fitted && this.scale !== 1) this.setScale(1);
    else this.fit();
  }

  setScale(s, anchor) {
    if (!this.w) return;
    s = Math.min(STEPS.at(-1), Math.max(Math.min(STEPS[0], this.fitScale()), s));
    if (anchor) {
      // Keep the image point under the anchor fixed.
      const ix = this.cx + anchor.dx / this.scale;
      const iy = this.cy + anchor.dy / this.scale;
      this.cx = ix - anchor.dx / s;
      this.cy = iy - anchor.dy / s;
    }
    this.scale = s;
    this.fitted = false;
    this.clamp();
    this.report();
    this.draw();
  }

  zoomStep(dir) {
    const s = this.scale;
    const next = dir > 0 ? STEPS.find((x) => x > s * 1.001) : STEPS.findLast((x) => x < s / 1.001);
    this.setScale(next ?? s);
  }

  clamp() {
    this.cx = Math.min(this.w, Math.max(0, this.cx));
    this.cy = Math.min(this.h, Math.max(0, this.cy));
  }

  report() {
    const pct = `${Math.round(this.scale * 100)}%`;
    this.onZoom(this.fitted ? `Fit ${pct}` : pct);
  }

  resize() {
    const dpr = window.devicePixelRatio || 1;
    const W = this.el.clientWidth;
    const H = this.el.clientHeight;
    this.canvas.width = Math.max(1, Math.round(W * dpr));
    this.canvas.height = Math.max(1, Math.round(H * dpr));
    if (this.fitted) this.fit(false);
    this.draw();
  }

  // ---- drawing ----

  draw() {
    if (this.frame) return;
    this.frame = requestAnimationFrame(() => {
      this.frame = 0;
      this.paint();
    });
  }

  paint() {
    const { ctx, canvas } = this;
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!this.orig) return;
    const W = this.el.clientWidth;
    const H = this.el.clientHeight;
    ctx.scale(dpr, dpr);
    ctx.imageSmoothingEnabled = this.scale < 1;
    ctx.imageSmoothingQuality = "high";

    if (this.mode === "diff" && this.dec) {
      this.diff ??= diffCanvas(this.rgbaA, this.rgbaB, this.w, this.h, this.gain);
      this.blit(this.diff, 0, 0, W, H);
    } else if (this.mode === "side" && this.dec) {
      this.blit(this.orig, 0, 0, W / 2, H);
      this.blit(this.dec, W / 2, 0, W / 2, H);
      ctx.fillStyle = "rgba(255,255,255,.85)";
      ctx.fillRect(W / 2 - 1, 0, 2, H);
    } else {
      this.blit(this.orig, 0, 0, W, H);
      if (this.dec) {
        const x = Math.round(W * this.split);
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, 0, W - x, H);
        ctx.clip();
        this.blit(this.dec, 0, 0, W, H);
        ctx.restore();
        this.handle(x, H);
      }
    }
  }

  // Draws `src` into the pane at (px, py, pw, ph) with the shared transform; clipped to the pane.
  blit(src, px, py, pw, ph) {
    const { ctx, scale } = this;
    const ox = px + pw / 2 - this.cx * scale; // screen x of image x = 0
    const oy = py + ph / 2 - this.cy * scale;
    // The visible source rectangle in whole texels, so huge zooms stay cheap.
    const sx0 = Math.max(0, Math.floor((px - ox) / scale));
    const sy0 = Math.max(0, Math.floor((py - oy) / scale));
    const sx1 = Math.min(this.w, Math.ceil((px + pw - ox) / scale));
    const sy1 = Math.min(this.h, Math.ceil((py + ph - oy) / scale));
    if (sx1 <= sx0 || sy1 <= sy0) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(px, py, pw, ph);
    ctx.clip();
    ctx.drawImage(
      src,
      sx0,
      sy0,
      sx1 - sx0,
      sy1 - sy0,
      ox + sx0 * scale,
      oy + sy0 * scale,
      (sx1 - sx0) * scale,
      (sy1 - sy0) * scale,
    );
    if (this.grid && scale * 4 >= 8) this.blockGrid(ox, oy, sx0, sy0, sx1, sy1);
    ctx.restore();
  }

  blockGrid(ox, oy, sx0, sy0, sx1, sy1) {
    const { ctx, scale } = this;
    const dpr = window.devicePixelRatio || 1;
    ctx.beginPath();
    const snap = (v) => (Math.round(v * dpr) + 0.5) / dpr;
    for (let x = Math.ceil(sx0 / 4) * 4; x <= sx1; x += 4) {
      const X = snap(ox + x * scale);
      ctx.moveTo(X, oy + sy0 * scale);
      ctx.lineTo(X, oy + sy1 * scale);
    }
    for (let y = Math.ceil(sy0 / 4) * 4; y <= sy1; y += 4) {
      const Y = snap(oy + y * scale);
      ctx.moveTo(ox + sx0 * scale, Y);
      ctx.lineTo(ox + sx1 * scale, Y);
    }
    ctx.lineWidth = 1 / dpr;
    ctx.strokeStyle = "rgba(255,255,255,.35)";
    ctx.stroke();
  }

  handle(x, H) {
    const { ctx } = this;
    ctx.fillStyle = "rgba(255,255,255,.9)";
    ctx.fillRect(x - 1, 0, 2, H);
    ctx.beginPath();
    ctx.arc(x, H / 2, 14, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(20,20,28,.72)";
    ctx.fill();
    ctx.lineWidth = 2;
    ctx.strokeStyle = "rgba(255,255,255,.9)";
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - 4, H / 2 - 5);
    ctx.lineTo(x - 8, H / 2);
    ctx.lineTo(x - 4, H / 2 + 5);
    ctx.moveTo(x + 4, H / 2 - 5);
    ctx.lineTo(x + 8, H / 2);
    ctx.lineTo(x + 4, H / 2 + 5);
    ctx.stroke();
  }

  // ---- input ----

  local(e) {
    const r = this.canvas.getBoundingClientRect();
    return { x: e.clientX - r.left, y: e.clientY - r.top };
  }

  onHandle(p) {
    return this.mode === "split" && this.dec && Math.abs(p.x - this.el.clientWidth * this.split) <= 16;
  }

  // Offset of a point from its pane center, for anchored zoom.
  anchor(p) {
    const W = this.el.clientWidth;
    const H = this.el.clientHeight;
    const pw = this.mode === "side" ? W / 2 : W;
    const px = this.mode === "side" && p.x >= pw ? pw : 0;
    return { dx: p.x - px - pw / 2, dy: p.y - H / 2 };
  }

  bindInput() {
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => {
      if (!this.orig) return;
      c.setPointerCapture(e.pointerId);
      const p = this.local(e);
      this.pointers.set(e.pointerId, p);
      if (this.pointers.size === 2) {
        const [a, b] = [...this.pointers.values()];
        this.drag = {
          kind: "pinch",
          dist: Math.hypot(a.x - b.x, a.y - b.y),
          scale: this.scale,
          mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 },
        };
      } else if (this.onHandle(p)) {
        this.drag = { kind: "split" };
      } else {
        this.drag = { kind: "pan", last: p };
        c.classList.add("panning");
      }
    });
    c.addEventListener("pointermove", (e) => {
      const p = this.local(e);
      if (!this.pointers.has(e.pointerId)) {
        c.classList.toggle("on-handle", !!this.onHandle(p));
        return;
      }
      this.pointers.set(e.pointerId, p);
      const d = this.drag;
      if (!d) return;
      if (d.kind === "split") {
        this.split = Math.min(1, Math.max(0, p.x / this.el.clientWidth));
        this.draw();
      } else if (d.kind === "pan") {
        this.cx -= (p.x - d.last.x) / this.scale;
        this.cy -= (p.y - d.last.y) / this.scale;
        d.last = p;
        this.fitted = false;
        this.clamp();
        this.report();
        this.draw();
      } else if (d.kind === "pinch" && this.pointers.size >= 2) {
        const [a, b] = [...this.pointers.values()];
        const dist = Math.hypot(a.x - b.x, a.y - b.y);
        this.setScale((d.scale * dist) / Math.max(1, d.dist), this.anchor(d.mid));
      }
    });
    const end = (e) => {
      this.pointers.delete(e.pointerId);
      if (this.pointers.size === 0) {
        this.drag = null;
        c.classList.remove("panning");
      } else if (this.drag?.kind === "pinch") {
        const [p] = this.pointers.values();
        this.drag = { kind: "pan", last: p };
      }
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener(
      "wheel",
      (e) => {
        if (!this.orig) return;
        e.preventDefault();
        const k = Math.exp(-Math.sign(e.deltaY) * Math.min(Math.abs(e.deltaY), 100) * 0.0035);
        this.setScale(this.scale * k, this.anchor(this.local(e)));
      },
      { passive: false },
    );
    c.addEventListener("dblclick", () => this.toggleFit());
  }
}
