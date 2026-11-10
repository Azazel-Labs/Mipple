//! Optional GPU compute backend (cargo feature `gpu`, built on wgpu).
//!
//! This module finds an adapter, reports what it supports, and runs the compute encoders. Each
//! kernel is a port of the CPU encoder for its format and writes the same blocks (the tests compare
//! them block for block), so the backend only changes speed. [`crate::pipeline::encode`] uses a
//! GPU encoder by default where it is the faster one ([`Gpu::preferred`]), everywhere with
//! [`crate::pipeline::GpuMode::On`]; formats without one, and hardware without compute shaders,
//! use the CPU. Without the feature, or on wasm32 for now, [`Gpu::shared`] is always `None`.
//!
//! Kernels:
//! - `bc1.wgsl`: BC1 efforts 0 and 1 (the line fit and least-squares refits of
//!   `crate::bc1::batch`, which the CPU defines in exact integers). Higher efforts use the CPU.
//!   The CPU encoder is faster here too (4096 x 4096, 20 AVX2 threads vs an Intel Arc iGPU:
//!   9 vs 71 ms at effort 0, 18 vs 97 ms at effort 1).
//! - `bc4.wgsl`: BC4 and BC5, every effort (a port of `crate::bc4::batch`). The CPU encoder is
//!   faster: it is memory-bound at a few ms per megapixel on a desktop CPU, under the GPU's
//!   upload, dispatch and readback alone (1024 x 1553 BC5, 20 threads vs an Intel Arc iGPU:
//!   1-3 ms vs 14 ms at efforts 0-3, 24 vs 22 ms at effort 9).
//! - `bc7.wgsl`: BC7 effort 0 without entropy reduction: mode 6 from the exact line fit of
//!   `crate::bc7::fast` (covariance, power iterations and endpoints in integers). Blocks with
//!   alpha are re-encoded on the CPU, as the CPU encoder sends them to its scalar path, and BC7
//!   RDO keeps its bases on the CPU (they start at effort 8). The CPU is faster here as well
//!   (4096 x 4096: 14 vs 59 ms on an Intel Arc iGPU).

use crate::format::{BlockOpts, Format};
use crate::surface::Surface;
use std::sync::OnceLock;

/// Formats with a GPU encoder. Each one is added with its compute kernel.
const ENCODERS: &[Format] = &[Format::Bc1, Format::Bc4, Format::Bc5, Format::Bc7];

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
    dev: imp::Device,
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

    /// Whether this build has a GPU kernel for `format` at `opts` (see [`Gpu::kernel_scope`]).
    pub fn has_kernel(format: Format, opts: BlockOpts) -> bool {
        Self::has_encoder(format) && covers(format, opts)
    }

    /// The efforts a format's kernel covers, for messages.
    pub fn kernel_scope(format: Format) -> &'static str {
        match format {
            Format::Bc1 => "efforts 0-1",
            Format::Bc7 => "effort 0 without entropy reduction",
            _ => "every effort",
        }
    }

    /// Whether [`crate::pipeline::GpuMode::Auto`] picks the GPU encoder for `format` at `opts`:
    /// only where it beats the CPU encoder (see the kernel list in the module docs).
    pub fn preferred(format: Format, opts: BlockOpts) -> bool {
        let _ = opts;
        match format {
            Format::Bc1 | Format::Bc4 | Format::Bc5 | Format::Bc7 => false,
        }
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

    /// Base-encodes `surface` into raw blocks (`format.block_bytes()` each, raster order), the
    /// same blocks as the CPU encoder. `None` when [`Gpu::encodes`] is false for `format`, when
    /// the surface exceeds the device's buffer limits, or when the device fails.
    pub fn encode_surface(&self, surface: &Surface, format: Format, opts: BlockOpts) -> Option<Vec<u8>> {
        if !self.encodes(format) || !covers(format, opts) || surface.width == 0 || surface.height == 0 {
            return None;
        }
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            match format {
                Format::Bc1 => self.dev.bc1(surface, opts.effort),
                Format::Bc4 => self.dev.bc4(surface, 1, opts.effort),
                Format::Bc5 => self.dev.bc4(surface, 2, opts.effort),
                Format::Bc7 => self.dev.bc7(surface, opts),
            }
        }
        #[cfg(not(all(feature = "gpu", not(target_arch = "wasm32"))))]
        {
            let _ = opts;
            None
        }
    }
}

/// Whether the format's kernel covers `opts`: every BC4/BC5 effort, BC1 efforts 0-1, and BC7
/// effort 0 without entropy reduction (mode 6).
fn covers(format: Format, opts: BlockOpts) -> bool {
    match format {
        Format::Bc1 => opts.effort <= 1,
        Format::Bc7 => opts.effort == 0 && !opts.reduce_entropy,
        _ => true,
    }
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
mod imp {
    use super::{Gpu, GpuInfo};
    use crate::format::{BlockOpts, Format};
    use crate::surface::Surface;
    use std::sync::OnceLock;
    use wgpu::util::DeviceExt;

    /// Invocations per dispatch. Splitting long encodes keeps each dispatch short, well inside the
    /// OS GPU watchdog (TDR) at the slowest efforts. Small under test, so tests cover the split.
    const CHUNK: u32 = if cfg!(test) { 1 << 12 } else { 1 << 18 };
    /// Threads per workgroup in every kernel (`@workgroup_size(64)`).
    const GROUP: u32 = 64;

    /// The device, its queue, and the compute pipelines built so far.
    pub(super) struct Device {
        device: wgpu::Device,
        queue: wgpu::Queue,
        bc1: OnceLock<wgpu::ComputePipeline>,
        bc4: OnceLock<wgpu::ComputePipeline>,
        bc7: OnceLock<wgpu::ComputePipeline>,
    }

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
            dev: Device {
                device,
                queue,
                bc1: OnceLock::new(),
                bc4: OnceLock::new(),
                bc7: OnceLock::new(),
            },
        })
    }

    impl Device {
        fn pipeline(&self, label: &str, source: &str) -> wgpu::ComputePipeline {
            let module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            })
        }

        /// BC1 blocks of `surface` at effort 0 or 1.
        pub(super) fn bc1(&self, surface: &Surface, effort: u8) -> Option<Vec<u8>> {
            let pipeline = self.bc1.get_or_init(|| {
                // The single-color tables as WGSL constants: (hi, lo) levels as hi | lo << 8.
                let (t5, t6) = crate::bc1::single_color_tables();
                let table = |name: &str, t: &crate::bc1::MatchTable| {
                    let v: Vec<String> = t
                        .iter()
                        .map(|&(a, b)| format!("{}u", u32::from(a) | u32::from(b) << 8))
                        .collect();
                    format!(
                        "const {name} = array<u32, 256>({});
",
                        v.join(", ")
                    )
                };
                let source = table("MATCH5", t5) + &table("MATCH6", t6) + include_str!("bc1.wgsl");
                self.pipeline("bc1", &source)
            });
            let params = |first: u32, count: u32| {
                [
                    surface.width,
                    surface.height,
                    surface.blocks_x(),
                    count,
                    u32::from(effort),
                    first,
                    0,
                    0,
                ]
            };
            let words = self.run(pipeline, surface, surface.block_count() as u32, 2, params)?;
            Some(words.iter().flat_map(|w| w.to_le_bytes()).collect())
        }

        /// BC4 (`channels == 1`) or BC5 (`channels == 2`) blocks of `surface`.
        pub(super) fn bc4(&self, surface: &Surface, channels: u32, effort: u8) -> Option<Vec<u8>> {
            let pipeline = self.bc4.get_or_init(|| self.pipeline("bc4", include_str!("bc4.wgsl")));
            let radius = crate::bc4::batch::RADIUS[effort.min(9) as usize];
            let blocks = surface.block_count() as u32;
            let params = |first: u32, count: u32| {
                [
                    surface.width,
                    surface.height,
                    surface.blocks_x(),
                    count,
                    channels,
                    effort as u32,
                    radius as u32,
                    first,
                ]
            };
            let words = self.run(pipeline, surface, blocks * channels, 2, params)?;
            Some(words.iter().flat_map(|w| w.to_le_bytes()).collect())
        }

        /// BC7 blocks of `surface` at `opts` (effort 0: mode 6). Blocks with alpha are re-encoded
        /// on the CPU by the scalar encoder, as the CPU's fast encoder does.
        pub(super) fn bc7(&self, surface: &Surface, opts: BlockOpts) -> Option<Vec<u8>> {
            use rayon::prelude::*;
            let cfg = crate::bc7::fast_cfg(opts)?;
            let pipeline = self.bc7.get_or_init(|| self.pipeline("bc7", include_str!("bc7.wgsl")));
            let params = |first: u32, count: u32| {
                [
                    surface.width,
                    surface.height,
                    surface.blocks_x(),
                    count,
                    u32::from(cfg.power),
                    first,
                    0,
                    0,
                ]
            };
            let words = self.run(pipeline, surface, surface.block_count() as u32, 4, params)?;
            let mut out: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
            let bw = surface.blocks_x();
            let wts = Format::Bc7.error_weights();
            out.par_chunks_mut(16).enumerate().for_each(|(i, o)| {
                let b = surface.read_block(i as u32 % bw, i as u32 / bw);
                if b.iter().any(|p| p[3] != 255) {
                    o.copy_from_slice(&crate::bc7::encode(&b, &wts, opts));
                }
            });
            Some(out)
        }

        /// Runs a kernel over `count` invocations, each writing `words_per` output words, with
        /// the surface's RGBA8 texels at binding 1 and the output at binding 2. `params(first, n)`
        /// is the uniform (binding 0) of the dispatch covering invocations `first..first + n`.
        fn run(
            &self,
            pipeline: &wgpu::ComputePipeline,
            surface: &Surface,
            count: u32,
            words_per: u32,
            params: impl Fn(u32, u32) -> [u32; 8],
        ) -> Option<Vec<u32>> {
            let limits = self.device.limits();
            let px_bytes = surface.pixels.as_flattened();
            let out_bytes = u64::from(count) * u64::from(words_per) * 4;
            let max_binding = limits.max_storage_buffer_binding_size;
            if px_bytes.len() as u64 > max_binding || out_bytes > max_binding || out_bytes > limits.max_buffer_size {
                return None;
            }
            let device = &self.device;
            let px = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("texels"),
                contents: px_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });
            let out = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("blocks"),
                size: out_bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let readback = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("readback"),
                size: out_bytes,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let layout = pipeline.get_bind_group_layout(0);
            let max_groups = limits.max_compute_workgroups_per_dimension;
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("encode") });
            let mut first = 0;
            while first < count {
                let n = (count - first).min(CHUNK);
                let p = params(first, n);
                let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("params"),
                    contents: &p.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<_>>(),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                let bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: uniform.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: px.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: out.as_entire_binding(),
                        },
                    ],
                });
                // Kernels flatten (x, y) as gid.x + gid.y * num_workgroups.x * 64.
                let groups = n.div_ceil(GROUP);
                let (gx, gy) = if groups <= max_groups {
                    (groups, 1)
                } else {
                    (max_groups, groups.div_ceil(max_groups))
                };
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &bind, &[]);
                pass.dispatch_workgroups(gx, gy, 1);
                drop(pass);
                first += n;
            }
            encoder.copy_buffer_to_buffer(&out, 0, &readback, 0, out_bytes);
            self.queue.submit([encoder.finish()]);
            let slice = readback.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            device.poll(wgpu::PollType::wait_indefinitely()).ok()?;
            rx.recv().ok()?.ok()?;
            let words = {
                let view = slice.get_mapped_range().ok()?;
                view.as_chunks::<4>().0.iter().map(|b| u32::from_le_bytes(*b)).collect()
            };
            readback.unmap();
            Some(words)
        }
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

    /// Test surfaces: every synthetic image, one cropped to a size that is not a multiple of 4
    /// (edge blocks repeat their last row and column), and two Kodak photos when fetched.
    fn surfaces() -> Vec<(String, Surface)> {
        let mut out: Vec<(String, Surface)> = crate::synth::set()
            .iter()
            .map(|(name, make)| (name.to_string(), make(64)))
            .collect();
        let big = crate::synth::bricks(64);
        let (w, h) = (61u32, 37u32);
        let pixels = (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .map(|(x, y)| big.pixels[(y * 64 + x) as usize])
            .collect();
        for name in ["kodim01", "kodim23"] {
            let path = format!("test-images/kodak/{name}.png");
            if let Ok(surf) = Surface::load(std::path::Path::new(&path)) {
                out.push((name.into(), surf));
            }
        }
        // Block kinds the photos rarely cover: solid colors (all 256 values per channel),
        // two colors, full-range gray ramps (the widest axis), pure noise, 0/255 extremes.
        let mut seed = 0x9e37_79b9u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let (sw, sh) = (128u32, 96u32);
        let mut kinds = Surface::new(sw, sh);
        for by in 0..sh / 4 {
            for bx in 0..sw / 4 {
                let k = (by * (sw / 4) + bx) as usize;
                let (c0, c1) = (rnd().to_le_bytes(), rnd().to_le_bytes());
                for i in 0..16u32 {
                    let r = rnd().to_le_bytes();
                    let p = match k % 6 {
                        0 => [(k / 6 % 256) as u8, (k * 7 % 256) as u8, (k * 13 % 256) as u8, 255],
                        1 => {
                            if r[0] & 1 == 0 {
                                c0
                            } else {
                                c1
                            }
                        }
                        2 => {
                            let v = (i * 17) as u8;
                            [v, v, v, 255]
                        }
                        3 => r,
                        4 => [r[0], r[1], r[2], 255].map(|v| if v & 0x80 != 0 { 255 } else { 0 }),
                        _ => [c0[0].wrapping_add(r[0] & 3), c0[1].wrapping_add(r[1] & 3), c0[2], r[3]],
                    };
                    kinds.set(bx * 4 + (i & 3), by * 4 + i / 4, p);
                }
            }
        }
        out.push(("block kinds".into(), kinds));
        out.push((
            "bricks 61x37".into(),
            Surface {
                width: w,
                height: h,
                pixels,
            },
        ));
        out
    }

    /// The GPU encoder writes exactly the CPU encoder's blocks, for every format it encodes and
    /// every effort. Skips without a usable GPU.
    #[test]
    fn matches_the_cpu_encoder() {
        let Some(gpu) = Gpu::shared() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let mut bad = Vec::new();
        for format in ENCODERS.iter().copied().filter(|&f| gpu.encodes(f)) {
            for effort in (0..=9u8).filter(|&e| covers(format, BlockOpts::from(e))) {
                for (name, surf) in surfaces() {
                    let opts = BlockOpts::from(effort);
                    let cpu = match format {
                        Format::Bc1 => crate::bc1::encode_surface(&surf, effort),
                        Format::Bc4 => crate::bc4::encode_surface(&surf, 1, effort),
                        Format::Bc5 => crate::bc4::encode_surface(&surf, 2, effort),
                        Format::Bc7 => crate::bc7::fast::encode_surface(
                            &surf,
                            crate::bc7::fast_cfg(opts).expect("plain BC7 has a fast configuration"),
                            opts,
                        ),
                    };
                    let got = gpu.encode_surface(&surf, format, opts).expect("GPU encode failed");
                    assert_eq!(got.len(), cpu.len(), "{format} e{effort} {name}");
                    let n = format.block_bytes();
                    let diff: Vec<usize> = (0..cpu.len() / n)
                        .filter(|&i| got[i * n..(i + 1) * n] != cpu[i * n..(i + 1) * n])
                        .collect();
                    if let Some(&i) = diff.first() {
                        bad.push(format!(
                            "{format} effort {effort} {name}: {} of {} blocks differ, first {i}: gpu {:02x?} cpu {:02x?}",
                            diff.len(),
                            cpu.len() / n,
                            &got[i * n..(i + 1) * n],
                            &cpu[i * n..(i + 1) * n],
                        ));
                    }
                }
            }
        }
        assert!(
            bad.is_empty(),
            "GPU blocks differ from the CPU's:
{}",
            bad.join(
                "
"
            )
        );
    }

    /// Through the whole pipeline (mips, RDO), forcing the GPU gives the same texture as the CPU.
    #[test]
    fn pipeline_gpu_matches_cpu() {
        use crate::pipeline::{EncodeOptions, GpuMode, encode};
        let Some(gpu) = Gpu::shared() else {
            return;
        };
        let surf = crate::synth::normal_rocks(64);
        for format in ENCODERS.iter().copied().filter(|&f| gpu.encodes(f)) {
            for lambda in [0.0, 1.0] {
                let mut opts = EncodeOptions::new(format);
                opts.rdo.lambda = lambda;
                // Covered by every kernel; BC7 RDO always encodes its base on the CPU.
                opts.effort = match format {
                    Format::Bc7 => 0, // the kernel covers effort 0
                    _ => 1,
                };
                opts.gpu = GpuMode::On;
                let (on, report) = encode(&surf, &opts);
                let rdo_on_cpu = format == Format::Bc7 && lambda > 0.0;
                assert_eq!(report.gpu.is_some(), !rdo_on_cpu, "{format} λ {lambda}: GPU use");
                opts.gpu = GpuMode::Off;
                let (off, report) = encode(&surf, &opts);
                assert!(report.gpu.is_none());
                assert_eq!(on, off, "{format} λ {lambda}");
            }
        }
    }
}
