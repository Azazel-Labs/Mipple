//! Alpha as transparency on a foliage cutout with busy colour left under its mask: smaller files,
//! visible texels as good as with alpha as data, and no hidden colour in the mips.

use mipple::alpha::AlphaMode;
use mipple::format::Format;
use mipple::pipeline::{self, EncodeOptions};
use mipple::surface::Surface;

/// Zstd-19 size of every level, and the RGB PSNR of the top level's visible texels.
fn encode(src: &Surface, alpha: AlphaMode) -> (usize, f64, pipeline::EncodedTexture) {
    let opts = EncodeOptions {
        alpha,
        ..EncodeOptions::new(Format::Bc7)
    };
    let (tex, _) = pipeline::encode(src, &opts);
    let raw: Vec<u8> = tex.levels.iter().flat_map(|l| l.data.iter().copied()).collect();
    let size = zstd::bulk::compress(&raw, 19).unwrap().len();
    let dec = pipeline::decode_level(&tex, 0, false);
    let (mut sse, mut n) = (0f64, 0f64);
    for (a, b) in src.pixels.iter().zip(&dec.pixels) {
        if a[3] != 0 {
            for c in 0..4 {
                let d = f64::from(a[c]) - f64::from(b[c]);
                sse += d * d;
            }
            n += 4.0;
        }
    }
    (size, 10.0 * (255.0f64 * 255.0 * n / sse).log10(), tex)
}

#[test]
fn transparency_frees_hidden_colour() {
    let src = mipple::synth::cutout(256);
    let (data, data_psnr, _) = encode(&src, AlphaMode::Data);
    let (tr, tr_psnr, tex) = encode(&src, AlphaMode::Auto);
    println!("alpha data {data} B {data_psnr:.2} dB, transparency {tr} B {tr_psnr:.2} dB");
    assert!((tr as f64) < 0.8 * data as f64, "{tr} vs {data} bytes");
    assert!(tr_psnr > data_psnr - 0.1, "{tr_psnr:.2} vs {data_psnr:.2} dB");
    // The visible texels of the first mip carry leaf colour only: no background magenta-ish
    // noise, so their blue stays near the leaves' (below 0.35 * 255 plus encoding error).
    let m1 = pipeline::decode_level(&tex, 1, false);
    for p in m1.pixels.iter().filter(|p| p[3] >= 128) {
        assert!(p[2] < 110, "{p:?}");
    }
}
