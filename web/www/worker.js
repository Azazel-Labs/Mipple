// Runs mipple (WebAssembly) off the main thread. The page terminates and respawns this worker to
// cancel a stale encode, so it holds no state beyond the instantiated module and its WebGPU device.
import init, { Settings, decodeImage, encode, fitImage, initGpu } from "./pkg/mipple_web.js";

let ready = null;

function imageMessage(img) {
  const rgba = img.rgba;
  const msg = {
    width: img.width,
    height: img.height,
    sourceWidth: img.source_width,
    sourceHeight: img.source_height,
    rgba,
  };
  img.free();
  return [msg, [rgba.buffer]];
}

async function runEncode({ rgba, width, height, settings: o }) {
  const s = new Settings(o.format, o.entropy);
  s.effort = o.effort;
  s.lambda = o.lambda;
  s.smooth_scale = o.smoothScale;
  s.smooth_std = o.smoothStd;
  s.max_rms_ratio = o.maxRmsRatio;
  s.min_rms = o.minRms;
  s.zstd = o.zstd;
  s.mips = o.mips;
  s.normal = o.normal;
  s.gpu = o.gpu ?? true;
  const command = s.command(o.inputName);
  const t0 = performance.now();
  // encode consumes the settings.
  const e = await encode(rgba, width, height, s);
  const totalMs = performance.now() - t0;
  const ktx2 = e.take_ktx2();
  const original = e.take_original();
  const decoded = e.take_decoded();
  const msg = {
    command,
    totalMs,
    width: e.width,
    height: e.height,
    levels: e.levels,
    rawBytes: e.raw_bytes,
    storedBytes: e.stored_bytes,
    fileBytes: e.file_bytes,
    psnr: e.psnr,
    rmse: e.rmse,
    ssim: e.ssim,
    normalDeg: e.normal_deg,
    baseMs: e.base_ms,
    rdoMs: e.rdo_ms,
    rdoChanged: e.rdo_changed,
    blocks: e.blocks,
    gpu: e.gpu ?? null,
    ktx2,
    original,
    decoded,
  };
  e.free();
  return [msg, [ktx2.buffer, original.buffer, decoded.buffer]];
}

self.onmessage = async ({ data }) => {
  const { id, type } = data;
  try {
    if (type === "init") {
      // The WebGPU device opens with the module, so the first encode does not wait for it.
      ready ??= init({ module_or_path: data.module }).then(() => initGpu());
      const gpu = await ready;
      self.postMessage({ id, ok: true, result: { gpu: gpu ?? null } });
      return;
    }
    await ready;
    let out;
    if (type === "decode") {
      out = imageMessage(decodeImage(new Uint8Array(data.bytes), data.maxSize));
    } else if (type === "fit") {
      out = imageMessage(fitImage(data.rgba, data.width, data.height, data.maxSize));
    } else if (type === "encode") {
      out = await runEncode(data);
    } else {
      throw new Error(`unknown request ${type}`);
    }
    self.postMessage({ id, ok: true, result: out[0] }, out[1]);
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err?.message ?? err) });
  }
};
