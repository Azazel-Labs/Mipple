//! Optional GPU compute backend (cargo feature `gpu`, built on wgpu).
//!
//! GPU base encoding is the default ([`crate::pipeline::EncodeOptions::gpu`]). This module finds an
//! adapter, reports what it supports, and answers per format whether a GPU encoder can run there.
//! Formats get compute encoders one at a time; until a format has one (or on hardware without
//! compute shaders), [`Gpu::encodes`] is false and [`crate::pipeline::encode`] uses the CPU
//! encoder for it. Without the feature, or on wasm32 for now, [`Gpu::shared`] is always `None`.

use crate::format::{BlockOpts, Format};
use crate::surface::Surface;
use std::sync::OnceLock;

/// Formats with a GPU encoder. Each one is added with its compute kernel.
const ENCODERS: &[Format] = &[];

/// What an adapter is and what it supports.
#[derive(Clone, Debug)]
pub struct GpuInfo {
    /// Adapter name, e.g. "NVIDIA GeForce RTX 4070".
    pub name: String,
    /// Graphics API wgpu drives it through (Vulkan, Metal, Dx12, Gl).
    pub backend: String,
    /// Discrete, integrated, virtual or CPU (software rasterizers such as llvmpipe or WARP).
    pub device_type: String,
    /// Driver name and version, when the API reports them.
    pub driver: String,
    /// Compute shaders: every GPU encoder needs them (some GLES/WebGL2 devices lack them).
    pub compute: bool,
    /// The device samples BC1-BC7 textures, for decoding and checking output on the GPU.
    pub bc_textures: bool,
}

/// An open GPU device.
pub struct Gpu {
    info: GpuInfo,
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    #[expect(dead_code, reason = "used by the per-format encoders as they land")]
    device: wgpu::Device,
    #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
    #[expect(dead_code, reason = "used by the per-format encoders as they land")]
    queue: wgpu::Queue,
}

impl Gpu {
    /// The process-wide device, opened on first use. `None` without the `gpu` feature, without a
    /// usable adapter, or if the device fails to open.
    pub fn shared() -> Option<&'static Gpu> {
        static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
        GPU.get_or_init(Gpu::open).as_ref()
    }

    /// Opens the highest-performance adapter, or returns `None` (see [`Gpu::shared`]).
    pub fn open() -> Option<Gpu> {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            imp::open()
        }
        #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
        {
            None
        }
    }

    /// Whether this build has a GPU encoder for `format` at all. Checked before opening a device,
    /// so formats without one never pay for adapter discovery.
    pub fn has_encoder(format: Format) -> bool {
        Self::built() && ENCODERS.contains(&format)
    }

    /// Whether this build includes the GPU backend at all.
    pub const fn built() -> bool {
        cfg!(all(feature = "gpu", not(target_arch = "wasm32")))
    }

    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// Whether this device can base-encode `format`. False sends the format to the CPU encoder.
    pub fn encodes(&self, format: Format) -> bool {
        self.info.compute && ENCODERS.contains(&format)
    }

    /// Base-encodes `surface` into raw blocks (`format.block_bytes()` each, raster order), or
    /// returns `None` when [`Gpu::encodes`] is false for `format`.
    pub fn encode_surface(&self, surface: &Surface, format: Format, opts: BlockOpts) -> Option<Vec<u8>> {
        if !self.encodes(format) {
            return None;
        }
        // Each format's compute encoder is dispatched from here as it lands.
        let _ = (surface, opts);
        None
    }
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod imp {
    use super::{Gpu, GpuInfo};

    pub(super) fn open() -> Option<Gpu> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .ok()?;
        let a = adapter.get_info();
        let compute = adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::COMPUTE_SHADERS);
        let bc_textures = adapter.features().contains(wgpu::Features::TEXTURE_COMPRESSION_BC);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("mipple"),
            required_features: if bc_textures {
                wgpu::Features::TEXTURE_COMPRESSION_BC
            } else {
                wgpu::Features::empty()
            },
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()?;
        let driver = [a.driver.as_str(), a.driver_info.as_str()]
            .iter()
            .filter(|s| !s.is_empty())
            .copied()
            .collect::<Vec<_>>()
            .join(" ");
        Some(Gpu {
            info: GpuInfo {
                name: a.name,
                backend: format!("{:?}", a.backend),
                device_type: format!("{:?}", a.device_type),
                driver,
                compute,
                bc_textures,
            },
            device,
            queue,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With or without hardware, every format either has a GPU encoder or falls back cleanly.
    #[test]
    fn falls_back_without_an_encoder() {
        let Some(gpu) = Gpu::shared() else {
            return;
        };
        let surf = crate::synth::foliage(16);
        for format in Format::ALL {
            let out = gpu.encode_surface(&surf, format, BlockOpts::from(0u8));
            assert_eq!(out.is_some(), gpu.encodes(format), "{format}");
        }
    }
}
